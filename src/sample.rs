//! Zero-allocation `WaveData` sample views and owned M4A PCM samples.

use crate::error::{Error, Result};

/// Size of a Nintendo M4A `WaveData` header in bytes.
pub const WAVE_DATA_HEADER_SIZE: usize = 16;

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
    #[must_use]
    pub const fn is_looped(&self) -> bool {
        (self.flags & 0xC0) != 0
    }

    /// Base playback frequency in Hz (`base_freq_q10 / 1024`).
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
    /// Parse a `WaveData` structure (16-byte header + PCM payload) without heap allocation.
    pub fn from_wavdata_bytes(bytes: &'a [u8]) -> Result<Self> {
        let header = WaveDataHeader::parse(bytes)?;
        let pcm_all = &bytes[WAVE_DATA_HEADER_SIZE..];
        if pcm_all.len() < header.size as usize {
            return Err(Error::UnexpectedEof);
        }
        Ok(Self {
            header,
            pcm_bytes: pcm_all,
        })
    }

    /// Construct a view directly from an existing header and PCM byte slice.
    pub fn from_parts(header: WaveDataHeader, pcm_bytes: &'a [u8]) -> Result<Self> {
        if pcm_bytes.len() < header.size as usize {
            return Err(Error::UnexpectedEof);
        }
        Ok(Self { header, pcm_bytes })
    }

    /// Logical sample length (`header.size`).
    #[must_use]
    pub const fn len(&self) -> usize {
        self.header.size as usize
    }

    /// Returns `true` if the sample has zero length.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.header.size == 0
    }

    /// Returns `true` if looping is enabled.
    #[must_use]
    pub const fn is_looped(&self) -> bool {
        self.header.is_looped()
    }

    /// Loop start index in samples.
    #[must_use]
    pub const fn loop_start(&self) -> usize {
        self.header.loop_start as usize
    }

    /// Loop length (`size - loop_start`).
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

    /// Logical PCM byte slice (`&pcm_bytes[..len()]`).
    #[must_use]
    pub fn logical_pcm_bytes(&self) -> &'a [u8] {
        &self.pcm_bytes[..self.len()]
    }

    /// Returns `true` if every logical sample is zero.
    #[must_use]
    pub fn is_silent(&self) -> bool {
        self.logical_pcm_bytes().iter().all(|&b| b == 0)
    }

    /// Compute `(min, max)` signed 8-bit amplitude across the logical PCM payload.
    #[must_use]
    pub fn min_max(&self) -> (i8, i8) {
        let mut min_v = i8::MAX;
        let mut max_v = i8::MIN;
        for &b in self.logical_pcm_bytes() {
            let s = b as i8;
            if s < min_v {
                min_v = s;
            }
            if s > max_v {
                max_v = s;
            }
        }
        if self.is_empty() {
            (0, 0)
        } else {
            (min_v, max_v)
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
        let view = M4aSampleView::from_wavdata_bytes(bytes)?;
        let pcm_data = view
            .logical_pcm_bytes()
            .iter()
            .map(|&b| b as i8)
            .collect();
        Ok(Self {
            header: view.header,
            pcm_data,
        })
    }

    /// Number of samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pcm_data.len()
    }

    /// Returns `true` if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pcm_data.is_empty()
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
