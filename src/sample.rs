//! Zero-allocation `WaveData` sample views and owned M4A PCM samples.

use crate::error::{Error, Result};

/// Size of a Nintendo M4A `WaveData` header in bytes.
pub const WAVE_DATA_HEADER_SIZE: usize = 16;

/// Number of decoded 8-bit PCM samples per Pokemon Gen 3 4-bit DPCM block.
pub const DPCM_BLOCK_SAMPLES: usize = 64;

/// Encoded byte length of a single Pokemon Gen 3 4-bit DPCM block (1 initial byte + 32 nibble bytes).
pub const DPCM_BLOCK_BYTES: usize = 33;

/// Exact 16-entry quadratic signed delta table used by Game Freak's Pokemon Gen 3
/// (`BPRE` / `BPEE` / `AXPE` / `AXVE`) 4-bit DPCM `SoundMainRAM` decoder.
pub const DPCM_DELTA_TABLE: [i8; 16] = [
    0, 1, 4, 9, 16, 25, 36, 49, -64, -49, -36, -25, -16, -9, -4, -1,
];

/// Compute the compressed payload size in bytes for `sample_count` Pokemon Gen 3 DPCM samples.
#[inline]
#[must_use]
pub const fn dpcm_encoded_len(sample_count: usize) -> usize {
    sample_count.div_ceil(DPCM_BLOCK_SAMPLES) * DPCM_BLOCK_BYTES
}

/// Decode a single 33-byte Pokemon Gen 3 4-bit quadratic DPCM block into 64 signed 8-bit PCM samples.
///
/// Matches the ARM `SampleFetch` routine (`0x081DC71C` in *Pokemon FireRed* `BPRE`) bit-for-bit:
/// - Byte `0`: initial signed 8-bit sample (`out[0]`).
/// - Byte `1`: low nibble (`b & 0x0F`) applied to accumulator (`out[1]`); high nibble is ignored.
/// - Bytes `2..33`: high nibble (`b >> 4`), then low nibble (`b & 0x0F`) for `out[2..64]`.
pub fn decode_dpcm_block(block_bytes: &[u8], out: &mut [i8; DPCM_BLOCK_SAMPLES]) -> Result<()> {
    if block_bytes.len() < DPCM_BLOCK_BYTES {
        return Err(Error::UnexpectedEof);
    }
    let mut acc = block_bytes[0] as i8;
    out[0] = acc;
    acc = acc.wrapping_add(DPCM_DELTA_TABLE[(block_bytes[1] & 0x0F) as usize]);
    out[1] = acc;

    let mut pos = 2usize;
    let mut i = 2usize;
    while i < DPCM_BLOCK_BYTES {
        let b = block_bytes[i];
        acc = acc.wrapping_add(DPCM_DELTA_TABLE[(b >> 4) as usize]);
        out[pos] = acc;
        acc = acc.wrapping_add(DPCM_DELTA_TABLE[(b & 0x0F) as usize]);
        out[pos + 1] = acc;
        pos += 2;
        i += 1;
    }
    Ok(())
}

/// Decode a Pokemon Gen 3 4-bit quadratic DPCM payload into a caller-provided signed 8-bit buffer.
///
/// Returns the number of decoded samples written (`out.len()`).
pub fn decode_dpcm_into(compressed: &[u8], out: &mut [i8]) -> Result<usize> {
    if out.is_empty() {
        return Ok(0);
    }
    let blocks_needed = out.len().div_ceil(DPCM_BLOCK_SAMPLES);
    if compressed.len() < blocks_needed * DPCM_BLOCK_BYTES {
        return Err(Error::UnexpectedEof);
    }
    let mut tmp = [0i8; DPCM_BLOCK_SAMPLES];
    let mut written = 0usize;
    for blk in 0..blocks_needed {
        let src = &compressed[blk * DPCM_BLOCK_BYTES..(blk + 1) * DPCM_BLOCK_BYTES];
        decode_dpcm_block(src, &mut tmp)?;
        let count = (out.len() - written).min(DPCM_BLOCK_SAMPLES);
        out[written..written + count].copy_from_slice(&tmp[..count]);
        written += count;
    }
    Ok(written)
}

/// Decode a Pokemon Gen 3 4-bit quadratic DPCM payload into an owned `Vec<i8>`.
#[cfg(feature = "alloc")]
pub fn decode_dpcm_vec(compressed: &[u8], sample_count: usize) -> Result<alloc::vec::Vec<i8>> {
    let mut out = alloc::vec![0i8; sample_count];
    decode_dpcm_into(compressed, &mut out)?;
    Ok(out)
}

/// Parsed 16-byte Nintendo M4A `WaveData` header.
///
/// Binary layout (little-endian):
/// - `+0x00` (`u16`): `wave_type` (0 = 8-bit signed PCM)
/// - `+0x02` (`u8`):  `stat` (reserved / compression status)
/// - `+0x03` (`u8`):  `flags` (`0x40` or `0xC0` = looped)
/// - `+0x04` (`u32`): `base_freq_q10` (base sample rate multiplied by 1024)
/// - `+0x08` (`u32`): `loop_start` (loop start offset in samples)
/// - `+0x0C` (`u32`): `size` (total sample length in bytes/samples)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaveDataHeader {
    pub wave_type: u16,
    pub stat: u8,
    pub flags: u8,
    pub base_freq_q10: u32,
    pub loop_start: u32,
    pub size: u32,
}

impl WaveDataHeader {
    /// Parse a 16-byte `WaveData` header from the beginning of `bytes`.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < WAVE_DATA_HEADER_SIZE {
            return Err(Error::UnexpectedEof);
        }
        let wave_type = u16::from_le_bytes([bytes[0], bytes[1]]);
        let stat = bytes[2];
        let flags = bytes[3];
        let base_freq_q10 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let loop_start = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let size = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);

        if size == 0 {
            return Err(Error::InvalidSample("WaveData size is zero"));
        }
        if (flags & 0xC0) != 0 && loop_start >= size {
            return Err(Error::InvalidSample("WaveData loop_start >= size"));
        }

        Ok(Self {
            wave_type,
            stat,
            flags,
            base_freq_q10,
            loop_start,
            size,
        })
    }

    /// Returns `true` if the `WaveData` loop flag (`flags & 0xC0 != 0`) is set,
    /// matching `SoundMainRAM`'s `tst r0, #0xC0` check.
    #[inline]
    #[must_use]
    pub const fn is_looped(&self) -> bool {
        (self.flags & 0xC0) != 0
    }

    /// Returns `true` if `WaveData.wave_type != 0` (Pokemon Gen 3 4-bit quadratic DPCM).
    #[inline]
    #[must_use]
    pub const fn is_compressed_dpcm(&self) -> bool {
        self.wave_type != 0
    }

    /// Base playback frequency in Hz (`base_freq_q10 / 1024`).
    #[inline]
    #[must_use]
    pub const fn base_sample_rate_hz(&self) -> u32 {
        self.base_freq_q10 >> 10
    }
}

/// Zero-allocation view into a Nintendo M4A `WaveData` sample (16-byte header + signed 8-bit PCM).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct M4aSampleView<'a> {
    pub header: WaveDataHeader,
    /// Raw PCM bytes starting immediately after the 16-byte header.
    /// May contain trailing guard bytes (at index `header.size`) used by
    /// `SoundMainRAM`'s linear interpolator when reading `sample[size - 1]` and `sample[size]`.
    pub pcm_bytes: &'a [u8],
}

impl<'a> M4aSampleView<'a> {
    /// Parse a `WaveData` structure (16-byte header + PCM or DPCM payload) without heap allocation.
    pub fn from_wavdata_bytes(bytes: &'a [u8]) -> Result<Self> {
        let header = WaveDataHeader::parse(bytes)?;
        let pcm_all = &bytes[WAVE_DATA_HEADER_SIZE..];
        let min_len = if header.is_compressed_dpcm() {
            dpcm_encoded_len(header.size as usize)
        } else {
            header.size as usize
        };
        if pcm_all.len() < min_len {
            return Err(Error::UnexpectedEof);
        }
        Ok(Self {
            header,
            pcm_bytes: pcm_all,
        })
    }

    /// Construct a view directly from an existing header and PCM/DPCM byte slice.
    pub fn from_parts(header: WaveDataHeader, pcm_bytes: &'a [u8]) -> Result<Self> {
        let min_len = if header.is_compressed_dpcm() {
            dpcm_encoded_len(header.size as usize)
        } else {
            header.size as usize
        };
        if pcm_bytes.len() < min_len {
            return Err(Error::UnexpectedEof);
        }
        Ok(Self { header, pcm_bytes })
    }

    /// Logical sample length (`header.size`).
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.header.size as usize
    }

    /// Returns `true` if the sample has zero length.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.header.size == 0
    }

    /// Returns `true` if looping is enabled.
    #[inline]
    #[must_use]
    pub const fn is_looped(&self) -> bool {
        self.header.is_looped()
    }

    /// Loop start index in samples.
    #[inline]
    #[must_use]
    pub const fn loop_start(&self) -> usize {
        self.header.loop_start as usize
    }

    /// Loop length (`size - loop_start`).
    #[inline]
    #[must_use]
    pub const fn loop_len(&self) -> usize {
        self.header.size.saturating_sub(self.header.loop_start) as usize
    }

    /// Read a signed 8-bit PCM sample at `index` (including trailing guard byte if present).
    /// Returns `0` if `index` is out of bounds of `pcm_bytes`.
    #[inline]
    #[must_use]
    pub fn sample_i8(&self, index: usize) -> i8 {
        self.pcm_bytes.get(index).copied().unwrap_or(0) as i8
    }

    /// Decode and read a single signed 8-bit sample at `index` from a Pokemon Gen 3
    /// 4-bit quadratic DPCM payload (`header.is_compressed_dpcm()`).
    ///
    /// Returns `0` if the block containing `index` is out of bounds.
    #[must_use]
    pub fn sample_i8_dpcm(&self, index: usize) -> i8 {
        let blk = index >> 6;
        let off = blk.saturating_mul(DPCM_BLOCK_BYTES);
        let Some(slice) = self.pcm_bytes.get(off..off + DPCM_BLOCK_BYTES) else {
            return 0;
        };
        let mut buf = [0i8; DPCM_BLOCK_SAMPLES];
        if decode_dpcm_block(slice, &mut buf).is_ok() {
            buf[index & 0x3F]
        } else {
            0
        }
    }

    /// Logical PCM byte slice (`&pcm_bytes[..len().min(pcm_bytes.len())]`).
    #[inline]
    #[must_use]
    pub fn logical_pcm_bytes(&self) -> &'a [u8] {
        let end = self.len().min(self.pcm_bytes.len());
        &self.pcm_bytes[..end]
    }

    /// Returns `true` if every logical sample is zero.
    #[must_use]
    pub fn is_silent(&self) -> bool {
        self.logical_pcm_bytes().iter().all(|&b| b == 0)
    }

    /// Compute `(min, max)` signed 8-bit amplitude across the logical PCM payload.
    #[must_use]
    pub fn min_max(&self) -> (i8, i8) {
        let logical = self.logical_pcm_bytes();
        if logical.is_empty() {
            return (0, 0);
        }
        let mut min_v = i8::MAX;
        let mut max_v = i8::MIN;
        for &b in logical {
            let s = b as i8;
            if s < min_v {
                min_v = s;
            }
            if s > max_v {
                max_v = s;
            }
        }
        (min_v, max_v)
    }

    /// Copy (or decode, if DPCM-compressed) the logical PCM payload into an owned [`M4aSample`].
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn to_owned_sample(&self) -> M4aSample {
        let pcm_data = if self.header.is_compressed_dpcm() {
            decode_dpcm_vec(self.pcm_bytes, self.len()).unwrap_or_default()
        } else {
            self.logical_pcm_bytes().iter().map(|&b| b as i8).collect()
        };
        M4aSample {
            header: self.header,
            pcm_data,
        }
    }
}

/// Owned M4A `WaveData` sample with signed 8-bit PCM buffer.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct M4aSample {
    pub header: WaveDataHeader,
    pub pcm_data: alloc::vec::Vec<i8>,
}

#[cfg(feature = "alloc")]
impl M4aSample {
    /// Parse and copy a `WaveData` sample into an owned [`M4aSample`].
    pub fn from_wavdata_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(M4aSampleView::from_wavdata_bytes(bytes)?.to_owned_sample())
    }

    /// Number of samples.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.pcm_data.len()
    }

    /// Returns `true` if empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pcm_data.is_empty()
    }

    /// Returns `true` if looping is enabled.
    #[inline]
    #[must_use]
    pub const fn is_looped(&self) -> bool {
        self.header.is_looped()
    }

    /// Loop start index in samples.
    #[inline]
    #[must_use]
    pub const fn loop_start(&self) -> usize {
        self.header.loop_start as usize
    }

    /// Loop length (`size - loop_start`).
    #[inline]
    #[must_use]
    pub const fn loop_len(&self) -> usize {
        self.header.size.saturating_sub(self.header.loop_start) as usize
    }

    /// Read a signed 8-bit PCM sample at `index`, returning `0` if out of bounds.
    #[inline]
    #[must_use]
    pub fn sample_i8(&self, index: usize) -> i8 {
        self.pcm_data.get(index).copied().unwrap_or(0)
    }

    /// Returns `true` if every sample is zero.
    #[must_use]
    pub fn is_silent(&self) -> bool {
        self.pcm_data.iter().all(|&s| s == 0)
    }

    /// Compute `(min, max)` signed 8-bit amplitude.
    #[must_use]
    pub fn min_max(&self) -> (i8, i8) {
        if self.pcm_data.is_empty() {
            return (0, 0);
        }
        let mut min_v = i8::MAX;
        let mut max_v = i8::MIN;
        for &s in &self.pcm_data {
            if s < min_v {
                min_v = s;
            }
            if s > max_v {
                max_v = s;
            }
        }
        (min_v, max_v)
    }
}
