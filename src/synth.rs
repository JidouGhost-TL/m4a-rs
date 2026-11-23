//! Dual synthesis pipelines for Nintendo MusicPlayer2000 (M4A / Sappy):
//! - [`AccuracySynth`]: Bit-exact emulation of Nintendo's ARM7TDMI `SoundMainRAM` IWRAM mixer.
//! - [`StudioSynth`]: 48,000 Hz `f32` audiophile renderer with band-limited windowed-sinc interpolation.

use crate::envelope::{M4aEnvelope, SF_LOOP, SF_START};
use crate::freq::{div_freq_for_rate, freq_step_q23};
use crate::sample::{decode_dpcm_block, M4aSampleView, DPCM_BLOCK_BYTES, DPCM_BLOCK_SAMPLES};

/// Standard M4A DirectSound DMA ring buffer size per channel (`0x630` = 1,584 bytes).
pub const PCM_DMA_BUF_SIZE: usize = 0x630;

/// Nintendo R&D1 (*Metroid Fusion* / *Zero Mission*) DirectSound DMA ring buffer stride (`0x600` bytes).
pub const RD1_DMA_BUF_SIZE: usize = 0x600;

/// Binary-verified ARM7TDMI `SoundMainRAM` mixer variants observed across the GBA M4A corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum M4aMixerVariant {
    /// Standard Nintendo M4A SDK v2 stereo Q23 linear mixer (~68% of titles:
    /// *Advance Wars*, *Golden Sun*, *Fire Emblem*, *Mother 3*, *Mario & Luigi*, etc.).
    #[default]
    Standard,
    /// Game Freak Pokemon Gen 3 M4A SDK v2 stereo mixer (*FireRed*, *LeafGreen*, *Emerald*,
    /// *Ruby*, *Sapphire*): adds `start_offset` on key-on, 4-bit quadratic DPCM
    /// (`voice_type & 0x20`, `WaveData.wave_type != 0`), and reverse playback (`voice_type & 0x10`).
    PokemonGen3,
    /// Nintendo M4A SDK v2 mono mixer (~24% of titles: *Mario Kart: Super Circuit*,
    /// *The Legend of Zelda: The Minish Cap*, *Castlevania: AoS / HoD*, *FFTA*, *Mega Man Zero*).
    /// Uses 2-sample `>> 8` reverb and `(vol_r + vol_l) >> 9` mono volume scaling.
    Mono,
    /// Early 2001 Nintendo M4A SDK v1 stereo mixer (~2.5% of titles: *EZ-Talk*, *Momotarou Matsuri*).
    /// Uses an exact-Hz modulo phase accumulator (`0..sample_rate`) and double-multiply linear
    /// interpolation `s0 + ((diff * fw * div_freq) >> 23)` with zero Q23 pitch drift.
    EarlyV1Stereo,
    /// Early 2001 Nintendo M4A SDK v1 mono mixer (*Mr. Driller 2*).
    /// Combines the exact-Hz modulo resampler with mono `>> 8` reverb and `(vol_r + vol_l) >> 9` volume.
    EarlyV1Mono,
    /// Nintendo R&D1 high-precision 32-bit stereo mixer (*Metroid Fusion*, *Metroid: Zero Mission*).
    /// Uses Q14 linear interpolation (`lsr #14`), 32-bit `i32` stereo accumulation, `<< 7` reverb
    /// pre-scaling, reverse playback (`voice_type == 0x20`), and a 9-bit signed saturating `>> 7` clamp.
    MetroidRd1,
}

/// Detect the [`M4aMixerVariant`] present in a raw GBA ROM image (`#![no_std]`, zero-allocation).
///
/// Scans for the binary-verified ARM/Thumb instruction signatures of Nintendo's `SoundMainRAM`
/// (SDK v1/v2 stereo & mono, Game Freak Pokemon Gen 3) and Nintendo R&D1's IWRAM ARM mixer.
#[must_use]
pub fn detect_mixer_variant(rom: &[u8]) -> Option<M4aMixerVariant> {
    // 1. Scan for standard M4A SoundMainRAM Thumb prologue:
    //    `ldrb r3, [r0, #5]; cmp r3, #0` (43 79 00 2B) followed within 12 bytes by
    //    `adr r1, #0/#4; bx r1` (00/01 A1 08 47).
    let mut i = 0usize;
    while i + 0x10 <= rom.len() {
        if rom[i] == 0x43 && rom[i + 1] == 0x79 && rom[i + 2] == 0x00 && rom[i + 3] == 0x2B {
            let win_end = (i + 16).min(rom.len());
            let mut has_arm_switch = false;
            let mut j = i + 4;
            while j + 4 <= win_end {
                if (rom[j] == 0x00 || rom[j] == 0x01)
                    && rom[j + 1] == 0xA1
                    && rom[j + 2] == 0x08
                    && rom[j + 3] == 0x47
                {
                    has_arm_switch = true;
                    break;
                }
                j += 2;
            }
            if has_arm_switch {
                let body_end = (i + 0x3A0).min(rom.len());
                let body = &rom[i..body_end];
                // Check reverb shift: `asr r0, r1, #8` (0xE1A00441 -> 41 04 A0 E1) = Mono
                //                     `asr r0, r1, #9` (0xE1A004C1 -> C1 04 A0 E1) = Stereo
                let rev_end = body.len().min(0x60);
                let mut is_mono = false;
                let mut k = 0usize;
                while k + 4 <= rev_end {
                    let w = u32::from_le_bytes([body[k], body[k + 1], body[k + 2], body[k + 3]]);
                    if w == 0xE1A0_0441 {
                        is_mono = true;
                        break;
                    }
                    k += 2;
                }

                // Check for Q23 `bic r?, r?, #0x3F800000` (word & 0x0FE00FFF == 0x03C005FE)
                // and Pokemon Gen 3 hook `tst r0, #0x30` (0xE3100030).
                let mut has_q23 = false;
                let mut has_pokemon_hook = false;
                let mut k = 0x100usize.min(body.len());
                while k + 4 <= body.len() {
                    let w = u32::from_le_bytes([body[k], body[k + 1], body[k + 2], body[k + 3]]);
                    if (w & 0x0FE0_0FFF) == 0x03C0_05FE {
                        has_q23 = true;
                    }
                    if w == 0xE310_0030 {
                        has_pokemon_hook = true;
                    }
                    k += 2;
                }

                return Some(if !has_q23 {
                    if is_mono {
                        M4aMixerVariant::EarlyV1Mono
                    } else {
                        M4aMixerVariant::EarlyV1Stereo
                    }
                } else if is_mono {
                    M4aMixerVariant::Mono
                } else if has_pokemon_hook {
                    M4aMixerVariant::PokemonGen3
                } else {
                    M4aMixerVariant::Standard
                });
            }
        }
        i += 2;
    }

    // 2. Scan for Nintendo R&D1 ARM mixer Q14 mask setup:
    //    `mov r7, #0; sub r7, r7, #1; lsr r7, r7, #0x12`
    const RD1_SIG: [u8; 12] = [
        0x00, 0x70, 0xA0, 0xE3, 0x01, 0x70, 0x47, 0xE2, 0x27, 0x79, 0xA0, 0xE1,
    ];
    let mut p = 0usize;
    while p + RD1_SIG.len() <= rom.len() {
        if rom[p..p + RD1_SIG.len()] == RD1_SIG {
            return Some(M4aMixerVariant::MetroidRd1);
        }
        p += 4;
    }

    None
}

/// Hardware DirectSound channel state matching Nintendo's 64-byte `SoundChannel` struct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct M4aSoundChannel {
    /// Envelope state (`status_flags`, ADSR parameters, `envelope_volume`, pseudo-echo).
    pub envelope: M4aEnvelope,
    /// Voice type flags (`0x08` = fixed 1:1 frequency mode, `0x00` = resampled Q23 linear mode).
    pub voice_type: u8,
    /// Right channel volume (`0..=255`, mixed into primary PCM buffer `+0x000`).
    pub right_volume: u8,
    /// Left channel volume (`0..=255`, mixed into secondary PCM buffer `+0x630`).
    pub left_volume: u8,
    /// Playback frequency in Hz (used in resampled mode: `step_q23 = div_freq * frequency`).
    pub frequency: u32,
    /// Initial sample start offset (when `SF_START` is set) or current sample index.
    pub position: usize,
    /// Remaining samples until sample end / loop wrap (`SoundChannel.count`).
    pub remaining: u32,
    /// Fractional sample phase in Q23 (`0..0x0080_0000`, `SoundChannel.fw`).
    pub frac_q23: u32,
}

impl M4aSoundChannel {
    /// Create a new channel ready for key-on (`SF_START`).
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub const fn new(
        voice_type: u8,
        right_volume: u8,
        left_volume: u8,
        attack: u8,
        decay: u8,
        sustain: u8,
        release: u8,
        frequency: u32,
        start_offset: usize,
    ) -> Self {
        Self {
            envelope: M4aEnvelope::new(attack, decay, sustain, release, 0, 0),
            voice_type,
            right_volume,
            left_volume,
            frequency,
            position: start_offset,
            remaining: 0,
            frac_q23: 0,
        }
    }

    /// Create a channel with pseudo-echo parameters configured.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub const fn with_echo(
        voice_type: u8,
        right_volume: u8,
        left_volume: u8,
        attack: u8,
        decay: u8,
        sustain: u8,
        release: u8,
        echo_volume: u8,
        echo_length: u8,
        frequency: u32,
        start_offset: usize,
    ) -> Self {
        Self {
            envelope: M4aEnvelope::new(attack, decay, sustain, release, echo_volume, echo_length),
            voice_type,
            right_volume,
            left_volume,
            frequency,
            position: start_offset,
            remaining: 0,
            frac_q23: 0,
        }
    }

    /// Returns `true` if this channel uses fixed 1:1 playback (`voice_type & 0x08 != 0`).
    #[inline]
    #[must_use]
    pub const fn is_fixed_pitch(&self) -> bool {
        (self.voice_type & 0x08) != 0
    }

    /// Trigger key-off (Release).
    #[inline]
    pub fn key_off(&mut self) {
        self.envelope.key_off();
    }
}

/// Bit-exact software emulation of Nintendo's ARM7TDMI `SoundMainRAM` IWRAM mixer.
#[derive(Debug, Default, Clone, Copy)]
pub struct AccuracySynth;

impl AccuracySynth {
    /// Apply Nintendo's exact `SoundMainRAM` reverb feedback pass (`0x081DC0B0..0x081DC0F8`).
    ///
    /// For each sample `i`:
    /// ```text
    /// sum = cur_r[i] + cur_l[i] + next_r[i] + next_l[i]
    /// out = (sum * reverb) >> 9
    /// if (out & 0x80) != 0 { out += 1; }
    /// cur_r[i] = out as i8;
    /// cur_l[i] = out as i8;
    /// ```
    ///
    /// When `reverb == 0`, zeroes `cur_r` and `cur_l` (matching `0x081DC104..0x081DC130`).
    pub fn reverb_pass(
        cur_r: &mut [i8],
        cur_l: &mut [i8],
        next_r: &[i8],
        next_l: &[i8],
        reverb: u8,
    ) {
        let len = cur_r.len().min(cur_l.len());
        if reverb == 0 {
            cur_r[..len].fill(0);
            cur_l[..len].fill(0);
            return;
        }
        let nlen = len.min(next_r.len()).min(next_l.len());
        if reverb <= 64 {
            // Fast path: max |sum * rev| = 512 * 64 = 32,768 fits in i16 (2x SIMD width).
            let rev16 = reverb as i16;
            for (((cr, cl), &nr), &nl) in cur_r[..nlen]
                .iter_mut()
                .zip(&mut cur_l[..nlen])
                .zip(&next_r[..nlen])
                .zip(&next_l[..nlen])
            {
                let sum = (*cr as i16) + (*cl as i16) + (nr as i16) + (nl as i16);
                let out = (sum * rev16) >> 9;
                let biased = (out + ((out >> 7) & 1)) as i8;
                *cr = biased;
                *cl = biased;
            }
        } else {
            let rev = reverb as i32;
            for (((cr, cl), &nr), &nl) in cur_r[..nlen]
                .iter_mut()
                .zip(&mut cur_l[..nlen])
                .zip(&next_r[..nlen])
                .zip(&next_l[..nlen])
            {
                let sum = (*cr as i32) + (*cl as i32) + (nr as i32) + (nl as i32);
                let out = (sum * rev) >> 9;
                // Exact ARM `tst r0, #0x80; addne r0, r0, #1` bias correction (branchless)
                let biased = (out + ((out >> 7) & 1)) as i8;
                *cr = biased;
                *cl = biased;
            }
        }
    }

    /// Tick a single [`M4aSoundChannel`] for one VBlank frame and mix it into
    /// `out_r` (primary buffer `+0x000`) and `out_l` (secondary buffer `+0x630`),
    /// matching Nintendo's `SoundMainRAM` bit-for-bit.
    ///
    /// `sample_rate` is the DirectSound output rate (e.g. `13379`).
    /// `master_volume` is in `0..=15`.
    pub fn mix_channel_frame(
        sample: &M4aSampleView<'_>,
        ch: &mut M4aSoundChannel,
        sample_rate: u32,
        master_volume: u8,
        out_r: &mut [i8],
        out_l: &mut [i8],
    ) {
        let was_start = (ch.envelope.status_flags & SF_START) != 0;
        let Some((vol_r, vol_l)) = ch.envelope.tick(
            master_volume,
            ch.right_volume,
            ch.left_volume,
            sample.is_looped(),
        ) else {
            return;
        };

        if was_start {
            ch.remaining = u32::try_from(ch.position)
                .map_or(0, |start_off| sample.header.size.saturating_sub(start_off));
            ch.frac_q23 = 0;
        }

        if ch.remaining == 0 {
            ch.envelope.status_flags = 0;
            return;
        }

        if ch.is_fixed_pitch() {
            Self::mix_fixed_inner(sample, ch, vol_r, vol_l, out_r, out_l);
        } else {
            let div_freq = div_freq_for_rate(sample_rate);
            Self::mix_resampled_inner(sample, ch, div_freq, vol_r, vol_l, out_r, out_l);
        }
    }

    /// Fixed-pitch (1:1) ARM mixing inner loop (`0x081DC278..0x081DC394`).
    ///
    /// Processes contiguous runs in bounds-check-free slices so LLVM can auto-vectorize.
    fn mix_fixed_inner(
        sample: &M4aSampleView<'_>,
        ch: &mut M4aSoundChannel,
        vol_r: u8,
        vol_l: u8,
        out_r: &mut [i8],
        out_l: &mut [i8],
    ) {
        let frames = out_r.len().min(out_l.len());
        if frames == 0 {
            return;
        }
        let out_r = &mut out_r[..frames];
        let out_l = &mut out_l[..frames];
        let pcm = sample.pcm_bytes;
        let vr = vol_r as i32;
        let vl = vol_l as i32;
        let looped = (ch.envelope.status_flags & SF_LOOP) != 0 && sample.loop_len() > 0;
        let loop_start = sample.loop_start();
        let loop_len = sample.loop_len() as u32;

        let mut pos = ch.position;
        let mut remaining = ch.remaining;
        let mut done = 0usize;

        while done < frames {
            let run = (remaining as usize).min(frames - done);
            if pos < pcm.len() {
                let in_bounds = run.min(pcm.len() - pos);
                let pcm_chunk = &pcm[pos..pos + in_bounds];
                let r_chunk = &mut out_r[done..done + in_bounds];
                let l_chunk = &mut out_l[done..done + in_bounds];
                for ((r, l), &b) in r_chunk.iter_mut().zip(l_chunk.iter_mut()).zip(pcm_chunk) {
                    let s = (b as i8) as i32;
                    // Exact ARM `mul r1, sl, r0; bic r1, r1, #0xFF0000; add r6, r1, r6, ror #8`
                    *r = r.wrapping_add(((s * vr) >> 8) as i8);
                    *l = l.wrapping_add(((s * vl) >> 8) as i8);
                }
            }
            // Any tail beyond `pcm.len()` has sample value 0 -> wrapping_add(0) is a no-op.
            pos += run;
            done += run;
            remaining -= run as u32;

            if remaining == 0 {
                if looped {
                    pos = loop_start;
                    remaining = loop_len;
                } else {
                    ch.envelope.status_flags = 0;
                    break;
                }
            }
        }

        ch.position = pos;
        ch.remaining = remaining;
    }

    /// Resampled Q23 linear-interpolation ARM mixing inner loop (`0x081DC398..0x081DC430`).
    fn mix_resampled_inner(
        sample: &M4aSampleView<'_>,
        ch: &mut M4aSoundChannel,
        div_freq: u32,
        vol_r: u8,
        vol_l: u8,
        out_r: &mut [i8],
        out_l: &mut [i8],
    ) {
        let frames = out_r.len().min(out_l.len());
        if frames == 0 {
            return;
        }
        let out_r = &mut out_r[..frames];
        let out_l = &mut out_l[..frames];
        let pcm = sample.pcm_bytes;
        let vr = vol_r as i32;
        let vl = vol_l as i32;
        let step = freq_step_q23(div_freq, ch.frequency);
        let loop_len_usize = sample.loop_len();
        let looped = (ch.envelope.status_flags & SF_LOOP) != 0
            && loop_len_usize > 0
            && loop_len_usize <= i32::MAX as usize;
        let loop_start = sample.loop_start();
        let loop_len = loop_len_usize as i32;

        let mut pos = ch.position;
        let mut count = ch.remaining as i32;
        let mut fw = ch.frac_q23;

        // Fast path: when no loop wrap, end-of-sample, or out-of-bounds PCM read can occur
        // across this entire frame, hoist all boundary checks out of the inner loop.
        if fw < 0x0080_0000 && step < 0x3F80_0000 && count > 0 {
            let total_int_steps = ((fw as u64) + (frames as u64) * (step as u64)) >> 23;
            if (count as u64) > total_int_steps {
                let end_idx = pos
                    .saturating_add(total_int_steps as usize)
                    .saturating_add(1);
                if end_idx < pcm.len() {
                    let pcm_win = &pcm[pos..=end_idx];
                    let mut rel = 0usize;
                    let mut s0 = (pcm_win[0] as i8) as i32;
                    let mut s1 = (pcm_win[1] as i8) as i32;
                    let mut diff = s1 - s0;

                    if step < 0x0080_0000 {
                        // Pitch <= 1.0x: int_step is always 0 or 1.
                        for (r, l) in out_r.iter_mut().zip(out_l.iter_mut()) {
                            let interp = s0 + (((fw as i32).wrapping_mul(diff)) >> 23);
                            *r = r.wrapping_add(((interp * vr) >> 8) as i8);
                            *l = l.wrapping_add(((interp * vl) >> 8) as i8);

                            fw += step;
                            if fw >= 0x0080_0000 {
                                fw &= 0x007F_FFFF;
                                rel += 1;
                                s0 = s1;
                                s1 = (pcm_win[rel + 1] as i8) as i32;
                                diff = s1 - s0;
                            }
                        }
                    } else {
                        // Pitch >= 1.0x: int_step is always >= 1; branchless L1 loads avoid
                        // branch mispredictions when int_step alternates (e.g. 1 and 2).
                        for (r, l) in out_r.iter_mut().zip(out_l.iter_mut()) {
                            let interp = s0 + (((fw as i32).wrapping_mul(diff)) >> 23);
                            *r = r.wrapping_add(((interp * vr) >> 8) as i8);
                            *l = l.wrapping_add(((interp * vl) >> 8) as i8);

                            fw += step;
                            let int_step = (fw >> 23) as usize;
                            fw &= 0x007F_FFFF;
                            rel += int_step;
                            s0 = (pcm_win[rel] as i8) as i32;
                            s1 = (pcm_win[rel + 1] as i8) as i32;
                            diff = s1 - s0;
                        }
                    }

                    ch.position = pos + rel;
                    ch.remaining = (count - rel as i32) as u32;
                    ch.frac_q23 = fw;
                    return;
                }
            }
        }

        // General path: handles loop wraps, sample termination, and trailing guard bytes.
        let mut s0 = sample.sample_i8(pos) as i32;
        let mut s1 = sample.sample_i8(pos.saturating_add(1)) as i32;
        let mut diff = s1 - s0;

        for (r, l) in out_r.iter_mut().zip(out_l.iter_mut()) {
            // Exact ARM:
            //   mul lr, sb, r1
            //   add lr, r0, lr, asr #23
            let interp = s0 + (((fw as i32).wrapping_mul(diff)) >> 23);

            // Exact ARM 8-bit wrapping accumulation per channel byte
            *r = r.wrapping_add(((interp * vr) >> 8) as i8);
            *l = l.wrapping_add(((interp * vl) >> 8) as i8);

            fw = fw.wrapping_add(step);
            let int_step = fw >> 23;
            if int_step != 0 {
                // Exact ARM `bic sb, sb, #0x3F800000`
                fw &= !0x3F80_0000;
                count = count.saturating_sub(int_step as i32);
                if count <= 0 {
                    if looped {
                        // Exact ARM loop wrap (`0x081DC33C..0x081DC350`)
                        let mut lr = -count;
                        loop {
                            count += loop_len;
                            if count > 0 {
                                break;
                            }
                            lr -= loop_len;
                        }
                        pos = loop_start + (lr as usize);
                        s0 = sample.sample_i8(pos) as i32;
                        s1 = sample.sample_i8(pos.saturating_add(1)) as i32;
                        diff = s1 - s0;
                    } else {
                        ch.envelope.status_flags = 0;
                        ch.position = pos;
                        ch.remaining = 0;
                        ch.frac_q23 = fw;
                        return;
                    }
                } else {
                    pos = pos.saturating_add(int_step as usize);
                    if int_step == 1 {
                        s0 = s1;
                    } else {
                        s0 = sample.sample_i8(pos) as i32;
                    }
                    s1 = sample.sample_i8(pos.saturating_add(1)) as i32;
                    diff = s1 - s0;
                }
            }
        }

        ch.position = pos;
        ch.remaining = count.max(0) as u32;
        ch.frac_q23 = fw;
    }

    /// Saturating DirectSound 8-bit DMA FIFO clamp helper for high-headroom mixing pipelines
    /// that accumulate into `i16`/`i32` before writing to `REG_FIFO_A`/`REG_FIFO_B`.
    #[inline]
    #[must_use]
    pub const fn clamp_i32_to_dma8(sample: i32) -> i8 {
        if sample < -128 {
            -128
        } else if sample > 127 {
            127
        } else {
            sample as i8
        }
    }

    /// Apply Nintendo's exact Mono `SoundMainRAM` reverb feedback pass (`M4aMixerVariant::Mono`
    /// and `M4aMixerVariant::EarlyV1Mono`, e.g. *Mario Kart: Super Circuit* `0x0805F2F4`).
    ///
    /// For each sample `i`:
    /// ```text
    /// sum = cur[i] + next[i]
    /// out = (sum * reverb) >> 8
    /// if (out & 0x80) != 0 { out += 1; }
    /// cur[i] = out as i8;
    /// ```
    pub fn reverb_pass_mono(cur: &mut [i8], next: &[i8], reverb: u8) {
        if reverb == 0 {
            cur.fill(0);
            return;
        }
        let len = cur.len().min(next.len());
        let rev = reverb as i32;
        for (c, &n) in cur[..len].iter_mut().zip(&next[..len]) {
            let sum = (*c as i32) + (n as i32);
            let out = (sum * rev) >> 8;
            let biased = (out + ((out >> 7) & 1)) as i8;
            *c = biased;
        }
    }

    /// Apply Nintendo R&D1's exact reverb pre-pass (`0x08003D68` in *Metroid Fusion* `AMTE`),
    /// initializing the 32-bit stereo mixing accumulators `acc_r` and `acc_l`.
    ///
    /// For each sample `i`:
    /// ```text
    /// sum = cur_r[i] + cur_l[i] + next_r[i] + next_l[i]
    /// out = (sum * reverb) >> 9
    /// if out < 0 { out += 1; }   // ARM `asrs r6, r6, #9; addmi r6, r6, #1`
    /// acc_r[i] = out << 7;
    /// acc_l[i] = out << 7;
    /// ```
    pub fn reverb_pass_rd1(
        cur_r: &[i8],
        cur_l: &[i8],
        next_r: &[i8],
        next_l: &[i8],
        acc_r: &mut [i32],
        acc_l: &mut [i32],
        reverb: u8,
    ) {
        let len = cur_r
            .len()
            .min(cur_l.len())
            .min(next_r.len())
            .min(next_l.len())
            .min(acc_r.len())
            .min(acc_l.len());
        if reverb == 0 {
            acc_r[..len].fill(0);
            acc_l[..len].fill(0);
            return;
        }
        let rev = reverb as i32;
        for i in 0..len {
            let sum = (cur_r[i] as i32)
                + (cur_l[i] as i32)
                + (next_r[i] as i32)
                + (next_l[i] as i32);
            let mut out = (sum * rev) >> 9;
            if out < 0 {
                out += 1;
            }
            let pre = out << 7;
            acc_r[i] = pre;
            acc_l[i] = pre;
        }
    }

    /// Finalize Nintendo R&D1's 32-bit stereo mixing accumulators into 8-bit DirectSound DMA
    /// bytes using R&D1's exact `>> 7` shift and 9-bit signed saturating clamp LUT
    /// (`0x08003E0C` in *Metroid Fusion* `AMTE`).
    pub fn finalize_rd1_dma8(acc_r: &[i32], acc_l: &[i32], out_r: &mut [i8], out_l: &mut [i8]) {
        let len = acc_r
            .len()
            .min(acc_l.len())
            .min(out_r.len())
            .min(out_l.len());
        for i in 0..len {
            let idx_r = ((acc_r[i] as u32) >> 7) & 0x1FF;
            let s9_r = if idx_r < 256 {
                idx_r as i32
            } else {
                (idx_r as i32) - 512
            };
            out_r[i] = Self::clamp_i32_to_dma8(s9_r);

            let idx_l = ((acc_l[i] as u32) >> 7) & 0x1FF;
            let s9_l = if idx_l < 256 {
                idx_l as i32
            } else {
                (idx_l as i32) - 512
            };
            out_l[i] = Self::clamp_i32_to_dma8(s9_l);
        }
    }

    /// Mix a single channel into Nintendo R&D1's 32-bit stereo mixing accumulators (`acc_r`, `acc_l`),
    /// matching *Metroid Fusion* (`AMTE` `0x08003EBC`) and *Metroid: Zero Mission* (`BMXE` `0x08004464`)
    /// bit-for-bit.
    ///
    /// - `ch.voice_type == 0x08`: fixed 1:1 pitch (`ch.position` is sample index).
    /// - `ch.voice_type == 0x20`: reverse Q14 linear interpolation (`ch.frac_q23` is Q14 phase).
    /// - `ch.voice_type == 0x00`: forward Q14 linear interpolation (`ch.frac_q23` is Q14 phase).
    /// - `step_q14`: phase increment per output sample in Q14 (`1.0x = 0x4000`).
    #[allow(clippy::too_many_arguments)]
    pub fn mix_channel_frame_rd1(
        sample: &M4aSampleView<'_>,
        ch: &mut M4aSoundChannel,
        step_q14: u32,
        vol_r: u8,
        vol_l: u8,
        acc_r: &mut [i32],
        acc_l: &mut [i32],
    ) {
        let frames = acc_r.len().min(acc_l.len());
        if frames == 0 || sample.is_empty() || ch.envelope.status_flags == 0 {
            return;
        }
        let len = sample.len();
        let looped = sample.is_looped() && sample.loop_len() > 0;
        let loop_start = sample.loop_start();
        let loop_len = sample.loop_len();
        let vr = vol_r as i32;
        let vl = vol_l as i32;

        if ch.voice_type == 0x08 {
            // Fixed 1:1 (`0x080041EC`)
            let mut pos = ch.position;
            for (ar, al) in acc_r[..frames].iter_mut().zip(&mut acc_l[..frames]) {
                let s = sample.sample_i8(pos) as i32;
                *ar = ar.wrapping_add(s.wrapping_mul(vr));
                *al = al.wrapping_add(s.wrapping_mul(vl));
                pos += 1;
                if pos == len {
                    if looped {
                        pos = loop_start;
                    } else {
                        ch.envelope.status_flags = 0;
                        break;
                    }
                }
            }
            ch.position = pos;
        } else if ch.voice_type == 0x20 {
            // Reverse Q14 linear interpolation (`0x08004318`)
            let mut phase = ch.frac_q23;
            for (ar, al) in acc_r[..frames].iter_mut().zip(&mut acc_l[..frames]) {
                let idx = (phase >> 14) as usize;
                let s0 = sample.sample_i8(idx) as i32;
                let s1 = sample.sample_i8(idx.saturating_add(1)) as i32;
                let diff = s1 - s0;
                let frac = phase & 0x3FFF;
                // Exact ARM `mul r4, r5, r4; add r3, r3, r4, lsr #14` (unsigned shift!)
                let interp = s0.wrapping_add(((frac.wrapping_mul(diff as u32)) >> 14) as i32);
                *ar = ar.wrapping_add(interp.wrapping_mul(vr));
                *al = al.wrapping_add(interp.wrapping_mul(vl));
                phase = phase.wrapping_sub(step_q14);
                if ((phase >> 14) as i32) <= 0 {
                    ch.envelope.status_flags = 0;
                    break;
                }
            }
            ch.frac_q23 = phase;
        } else {
            // Forward Q14 linear interpolation (`0x08003EF8`)
            let mut phase = ch.frac_q23;
            for (ar, al) in acc_r[..frames].iter_mut().zip(&mut acc_l[..frames]) {
                let idx = (phase >> 14) as usize;
                let s0 = sample.sample_i8(idx) as i32;
                let s1 = sample.sample_i8(idx.saturating_add(1)) as i32;
                let diff = s1 - s0;
                let frac = phase & 0x3FFF;
                // Exact ARM `mul r4, r5, r4; add r3, r3, r4, lsr #14` (unsigned shift!)
                let interp = s0.wrapping_add(((frac.wrapping_mul(diff as u32)) >> 14) as i32);
                *ar = ar.wrapping_add(interp.wrapping_mul(vr));
                *al = al.wrapping_add(interp.wrapping_mul(vl));
                phase = phase.wrapping_add(step_q14);
                let next_idx = (phase >> 14) as usize;
                if next_idx >= len {
                    if looped {
                        let wrapped = next_idx.saturating_sub(loop_len);
                        phase = (phase & 0x3FFF) | ((wrapped as u32) << 14);
                    } else {
                        ch.envelope.status_flags = 0;
                        break;
                    }
                }
            }
            ch.frac_q23 = phase;
        }
    }

    /// Tick a single [`M4aSoundChannel`] for one VBlank frame using a specific [`M4aMixerVariant`].
    ///
    /// - For [`M4aMixerVariant::Standard`]: matches standard Nintendo SDK v2 stereo `SoundMainRAM`.
    /// - For [`M4aMixerVariant::PokemonGen3`]: additionally supports `start_offset` on key-on,
    ///   4-bit quadratic DPCM (`voice_type & 0x20` / `WaveData.wave_type != 0`), and reverse
    ///   playback (`voice_type & 0x10`).
    /// - For [`M4aMixerVariant::Mono`] and [`M4aMixerVariant::EarlyV1Mono`]: mixes mono output into
    ///   `out_r` using Nintendo's exact `(vol_r + vol_l) >> 9` mono volume scaling (`out_l` is unused).
    /// - For [`M4aMixerVariant::EarlyV1Stereo`]: uses the 2001 SDK v1 exact-Hz modulo resampler.
    /// - For [`M4aMixerVariant::MetroidRd1`]: use [`Self::mix_channel_frame_rd1`] + [`Self::finalize_rd1_dma8`]
    ///   for the 32-bit accumulator pipeline (calling this method with `MetroidRd1` falls back to `Standard`).
    #[allow(clippy::too_many_arguments)]
    pub fn mix_channel_frame_variant(
        variant: M4aMixerVariant,
        sample: &M4aSampleView<'_>,
        ch: &mut M4aSoundChannel,
        sample_rate: u32,
        master_volume: u8,
        out_r: &mut [i8],
        out_l: &mut [i8],
    ) {
        let was_start = (ch.envelope.status_flags & SF_START) != 0;
        let Some((vol_r, vol_l)) = ch.envelope.tick(
            master_volume,
            ch.right_volume,
            ch.left_volume,
            sample.is_looped(),
        ) else {
            return;
        };

        if was_start {
            match variant {
                M4aMixerVariant::PokemonGen3 => {
                    ch.remaining = u32::try_from(ch.position)
                        .map_or(0, |start_off| sample.header.size.saturating_sub(start_off));
                    if (ch.voice_type & 0x10) != 0 {
                        // Reverse playback starts from `size - start_offset`
                        ch.position = sample.len().saturating_sub(ch.position);
                    }
                }
                _ => {
                    // Standard / Mono / EarlyV1 `SoundMainRAM` resets position to 0 on SF_START
                    ch.position = 0;
                    ch.remaining = sample.header.size;
                }
            }
            ch.frac_q23 = 0;
        }

        if ch.remaining == 0 {
            ch.envelope.status_flags = 0;
            return;
        }

        match variant {
            M4aMixerVariant::Standard | M4aMixerVariant::MetroidRd1 => {
                if ch.is_fixed_pitch() {
                    Self::mix_fixed_inner(sample, ch, vol_r, vol_l, out_r, out_l);
                } else {
                    let div_freq = div_freq_for_rate(sample_rate);
                    Self::mix_resampled_inner(sample, ch, div_freq, vol_r, vol_l, out_r, out_l);
                }
            }
            M4aMixerVariant::PokemonGen3 => {
                if (ch.voice_type & 0x30) != 0 {
                    let div_freq = div_freq_for_rate(sample_rate);
                    Self::mix_pokemon_gen3_ext(sample, ch, div_freq, vol_r, vol_l, out_r, out_l);
                } else if ch.is_fixed_pitch() {
                    Self::mix_fixed_inner(sample, ch, vol_r, vol_l, out_r, out_l);
                } else {
                    let div_freq = div_freq_for_rate(sample_rate);
                    Self::mix_resampled_inner(sample, ch, div_freq, vol_r, vol_l, out_r, out_l);
                }
            }
            M4aMixerVariant::Mono => {
                let scaled = (ch.envelope.envelope_volume as u32 * (master_volume as u32 + 1)) >> 4;
                let vol_mono =
                    ((scaled * (ch.right_volume as u32 + ch.left_volume as u32)) >> 9) as u8;
                ch.envelope.envelope_volume_right = vol_mono;
                if ch.is_fixed_pitch() {
                    Self::mix_fixed_mono_inner(sample, ch, vol_mono, out_r);
                } else {
                    let div_freq = div_freq_for_rate(sample_rate);
                    Self::mix_resampled_mono_inner(sample, ch, div_freq, vol_mono, out_r);
                }
            }
            M4aMixerVariant::EarlyV1Stereo => {
                if ch.is_fixed_pitch() {
                    Self::mix_fixed_inner(sample, ch, vol_r, vol_l, out_r, out_l);
                } else {
                    let div_freq = div_freq_for_rate(sample_rate);
                    Self::mix_early_v1_stereo_inner(
                        sample,
                        ch,
                        sample_rate,
                        div_freq,
                        vol_r,
                        vol_l,
                        out_r,
                        out_l,
                    );
                }
            }
            M4aMixerVariant::EarlyV1Mono => {
                let scaled = (ch.envelope.envelope_volume as u32 * (master_volume as u32 + 1)) >> 4;
                let vol_mono =
                    ((scaled * (ch.right_volume as u32 + ch.left_volume as u32)) >> 9) as u8;
                ch.envelope.envelope_volume_right = vol_mono;
                if ch.is_fixed_pitch() {
                    Self::mix_fixed_mono_inner(sample, ch, vol_mono, out_r);
                } else {
                    let div_freq = div_freq_for_rate(sample_rate);
                    Self::mix_early_v1_mono_inner(
                        sample,
                        ch,
                        sample_rate,
                        div_freq,
                        vol_mono,
                        out_r,
                    );
                }
            }
        }
    }

    /// Game Freak Pokemon Gen 3 extended mixer (`0x081DC460..0x081DC7A0` in `BPRE`):
    /// handles 4-bit quadratic DPCM (`WaveData.wave_type != 0`, forward & reverse) and
    /// reverse uncompressed PCM (`voice_type & 0x10 != 0`).
    fn mix_pokemon_gen3_ext(
        sample: &M4aSampleView<'_>,
        ch: &mut M4aSoundChannel,
        div_freq: u32,
        vol_r: u8,
        vol_l: u8,
        out_r: &mut [i8],
        out_l: &mut [i8],
    ) {
        let frames = out_r.len().min(out_l.len());
        if frames == 0 {
            return;
        }
        let out_r = &mut out_r[..frames];
        let out_l = &mut out_l[..frames];
        let vr = vol_r as i32;
        let vl = vol_l as i32;
        let step = if ch.is_fixed_pitch() {
            0x0080_0000
        } else {
            freq_step_q23(div_freq, ch.frequency)
        };
        let is_reverse = (ch.voice_type & 0x10) != 0;
        let is_dpcm = sample.header.is_compressed_dpcm();

        let mut pos = ch.position;
        let mut count = ch.remaining as i32;
        let mut fw = ch.frac_q23;

        if !is_dpcm {
            // Uncompressed reverse PCM (`0x081DC664..0x081DC6EC`)
            if !is_reverse || pos == 0 {
                return;
            }
            pos -= 1;
            let mut s0 = sample.sample_i8(pos) as i32;
            let mut s1 = if pos > 0 {
                sample.sample_i8(pos - 1) as i32
            } else {
                0
            };
            let mut diff = s1 - s0;

            for (r, l) in out_r.iter_mut().zip(out_l.iter_mut()) {
                let interp = s0 + (((fw as i32).wrapping_mul(diff)) >> 23);
                *r = r.wrapping_add(((interp * vr) >> 8) as i8);
                *l = l.wrapping_add(((interp * vl) >> 8) as i8);

                fw = fw.wrapping_add(step);
                let int_step = fw >> 23;
                if int_step != 0 {
                    fw &= !0x3F80_0000;
                    count = count.saturating_sub(int_step as i32);
                    if count <= 0 {
                        ch.envelope.status_flags = 0;
                        ch.position = pos;
                        ch.remaining = 0;
                        ch.frac_q23 = fw;
                        return;
                    }
                    pos = pos.saturating_sub(int_step as usize);
                    s0 = sample.sample_i8(pos) as i32;
                    s1 = if pos > 0 {
                        sample.sample_i8(pos - 1) as i32
                    } else {
                        0
                    };
                    diff = s1 - s0;
                }
            }
            ch.position = pos.saturating_add(1);
            ch.remaining = count.max(0) as u32;
            ch.frac_q23 = fw;
            return;
        }

        // Block-cached 4-bit quadratic DPCM fetcher matching `0x081DC71C`
        let mut cached_blk: usize = usize::MAX;
        let mut blk_buf = [0i8; DPCM_BLOCK_SAMPLES];
        let mut fetch_dpcm = |idx: usize| -> i32 {
            let blk = idx >> 6;
            if blk != cached_blk {
                cached_blk = blk;
                let off = blk.saturating_mul(DPCM_BLOCK_BYTES);
                if let Some(slice) = sample.pcm_bytes.get(off..off + DPCM_BLOCK_BYTES) {
                    let _ = decode_dpcm_block(slice, &mut blk_buf);
                } else {
                    blk_buf.fill(0);
                }
            }
            blk_buf[idx & 0x3F] as i32
        };

        if is_reverse {
            // Reverse DPCM (`0x081DC5BC..0x081DC660`)
            if pos == 0 {
                return;
            }
            pos -= 1;
            let mut s0 = fetch_dpcm(pos);
            let mut s1 = if pos > 0 {
                pos -= 1;
                fetch_dpcm(pos)
            } else {
                0
            };
            let mut diff = s1 - s0;

            for (r, l) in out_r.iter_mut().zip(out_l.iter_mut()) {
                let interp = s0 + (((fw as i32).wrapping_mul(diff)) >> 23);
                *r = r.wrapping_add(((interp * vr) >> 8) as i8);
                *l = l.wrapping_add(((interp * vl) >> 8) as i8);

                fw = fw.wrapping_add(step);
                let int_step = fw >> 23;
                if int_step != 0 {
                    fw &= !0x3F80_0000;
                    count = count.saturating_sub(int_step as i32);
                    if count <= 0 {
                        ch.envelope.status_flags = 0;
                        ch.position = pos;
                        ch.remaining = 0;
                        ch.frac_q23 = fw;
                        return;
                    }
                    if int_step == 1 {
                        s0 = s1;
                    } else {
                        pos = pos.saturating_sub((int_step - 1) as usize);
                        s0 = fetch_dpcm(pos);
                    }
                    if pos > 0 {
                        pos -= 1;
                        s1 = fetch_dpcm(pos);
                    } else {
                        s1 = 0;
                    }
                    diff = s1 - s0;
                }
            }
            ch.position = pos.saturating_add(1);
            ch.remaining = count.max(0) as u32;
            ch.frac_q23 = fw;
        } else {
            // Forward DPCM (`0x081DC4F0..0x081DC5BC`)
            let loop_len_usize = sample.loop_len();
            let looped = (ch.envelope.status_flags & SF_LOOP) != 0
                && loop_len_usize > 0
                && loop_len_usize <= i32::MAX as usize;
            let loop_start = sample.loop_start();
            let loop_len = loop_len_usize as i32;

            let mut s0 = fetch_dpcm(pos);
            pos = pos.saturating_add(1);
            let mut s1 = fetch_dpcm(pos);
            let mut diff = s1 - s0;

            for (r, l) in out_r.iter_mut().zip(out_l.iter_mut()) {
                let interp = s0 + (((fw as i32).wrapping_mul(diff)) >> 23);
                *r = r.wrapping_add(((interp * vr) >> 8) as i8);
                *l = l.wrapping_add(((interp * vl) >> 8) as i8);

                fw = fw.wrapping_add(step);
                let int_step = fw >> 23;
                if int_step != 0 {
                    fw &= !0x3F80_0000;
                    count = count.saturating_sub(int_step as i32);
                    if count <= 0 {
                        if looped {
                            let mut lr = -count;
                            loop {
                                count += loop_len;
                                if count > 0 {
                                    break;
                                }
                                lr -= loop_len;
                            }
                            pos = loop_start + (lr as usize);
                            s0 = fetch_dpcm(pos);
                            pos = pos.saturating_add(1);
                            s1 = fetch_dpcm(pos);
                            diff = s1 - s0;
                        } else {
                            ch.envelope.status_flags = 0;
                            ch.position = pos;
                            ch.remaining = 0;
                            ch.frac_q23 = fw;
                            return;
                        }
                    } else {
                        if int_step == 1 {
                            s0 = s1;
                        } else {
                            pos = pos.saturating_add((int_step - 1) as usize);
                            s0 = fetch_dpcm(pos);
                        }
                        pos = pos.saturating_add(1);
                        s1 = fetch_dpcm(pos);
                        diff = s1 - s0;
                    }
                }
            }
            ch.position = pos.saturating_sub(1);
            ch.remaining = count.max(0) as u32;
            ch.frac_q23 = fw;
        }
    }

    /// Mono fixed-pitch 1:1 ARM mixing inner loop (`M4aMixerVariant::Mono` / `EarlyV1Mono`).
    fn mix_fixed_mono_inner(
        sample: &M4aSampleView<'_>,
        ch: &mut M4aSoundChannel,
        vol_mono: u8,
        out: &mut [i8],
    ) {
        let frames = out.len();
        if frames == 0 {
            return;
        }
        let pcm = sample.pcm_bytes;
        let vm = vol_mono as i32;
        let looped = (ch.envelope.status_flags & SF_LOOP) != 0 && sample.loop_len() > 0;
        let loop_start = sample.loop_start();
        let loop_len = sample.loop_len() as u32;

        let mut pos = ch.position;
        let mut remaining = ch.remaining;
        let mut done = 0usize;

        while done < frames {
            let run = (remaining as usize).min(frames - done);
            if pos < pcm.len() {
                let in_bounds = run.min(pcm.len() - pos);
                for (o, &b) in out[done..done + in_bounds]
                    .iter_mut()
                    .zip(&pcm[pos..pos + in_bounds])
                {
                    let s = (b as i8) as i32;
                    *o = o.wrapping_add(((s * vm) >> 8) as i8);
                }
            }
            pos += run;
            done += run;
            remaining -= run as u32;

            if remaining == 0 {
                if looped {
                    pos = loop_start;
                    remaining = loop_len;
                } else {
                    ch.envelope.status_flags = 0;
                    break;
                }
            }
        }
        ch.position = pos;
        ch.remaining = remaining;
    }

    /// Mono Q23 resampled ARM mixing inner loop (`M4aMixerVariant::Mono`, e.g. `AMKE` / `BZME`).
    fn mix_resampled_mono_inner(
        sample: &M4aSampleView<'_>,
        ch: &mut M4aSoundChannel,
        div_freq: u32,
        vol_mono: u8,
        out: &mut [i8],
    ) {
        let frames = out.len();
        if frames == 0 {
            return;
        }
        let vm = vol_mono as i32;
        let step = freq_step_q23(div_freq, ch.frequency);
        let loop_len_usize = sample.loop_len();
        let looped = (ch.envelope.status_flags & SF_LOOP) != 0
            && loop_len_usize > 0
            && loop_len_usize <= i32::MAX as usize;
        let loop_start = sample.loop_start();
        let loop_len = loop_len_usize as i32;

        let mut pos = ch.position;
        let mut count = ch.remaining as i32;
        let mut fw = ch.frac_q23;
        let mut s0 = sample.sample_i8(pos) as i32;
        let mut s1 = sample.sample_i8(pos.saturating_add(1)) as i32;
        let mut diff = s1 - s0;

        for o in out.iter_mut() {
            let interp = s0 + (((fw as i32).wrapping_mul(diff)) >> 23);
            *o = o.wrapping_add(((interp * vm) >> 8) as i8);

            fw = fw.wrapping_add(step);
            let int_step = fw >> 23;
            if int_step != 0 {
                fw &= !0x3F80_0000;
                count = count.saturating_sub(int_step as i32);
                if count <= 0 {
                    if looped {
                        let mut lr = -count;
                        loop {
                            count += loop_len;
                            if count > 0 {
                                break;
                            }
                            lr -= loop_len;
                        }
                        pos = loop_start + (lr as usize);
                        s0 = sample.sample_i8(pos) as i32;
                        s1 = sample.sample_i8(pos.saturating_add(1)) as i32;
                        diff = s1 - s0;
                    } else {
                        ch.envelope.status_flags = 0;
                        ch.position = pos;
                        ch.remaining = 0;
                        ch.frac_q23 = fw;
                        return;
                    }
                } else {
                    pos = pos.saturating_add(int_step as usize);
                    s0 = if int_step == 1 {
                        s1
                    } else {
                        sample.sample_i8(pos) as i32
                    };
                    s1 = sample.sample_i8(pos.saturating_add(1)) as i32;
                    diff = s1 - s0;
                }
            }
        }
        ch.position = pos;
        ch.remaining = count.max(0) as u32;
        ch.frac_q23 = fw;
    }

    /// Early 2001 SDK v1 stereo exact-Hz modulo resampler (`A22J` `0x0801ABC4`).
    #[allow(clippy::too_many_arguments)]
    fn mix_early_v1_stereo_inner(
        sample: &M4aSampleView<'_>,
        ch: &mut M4aSoundChannel,
        sample_rate: u32,
        div_freq: u32,
        vol_r: u8,
        vol_l: u8,
        out_r: &mut [i8],
        out_l: &mut [i8],
    ) {
        let frames = out_r.len().min(out_l.len());
        if frames == 0 || sample_rate == 0 {
            return;
        }
        let vr = vol_r as i32;
        let vl = vol_l as i32;
        let df = div_freq as i32;
        let looped = (ch.envelope.status_flags & SF_LOOP) != 0 && sample.loop_len() > 0;
        let loop_start = sample.loop_start();
        let loop_len = sample.loop_len() as u32;

        let mut pos = ch.position;
        let mut remaining = ch.remaining;
        let mut fw = ch.frac_q23;

        for (r, l) in out_r[..frames].iter_mut().zip(&mut out_l[..frames]) {
            while fw >= sample_rate {
                fw -= sample_rate;
                remaining = remaining.saturating_sub(1);
                if remaining == 0 {
                    if looped {
                        pos = loop_start;
                        remaining = loop_len;
                    } else {
                        ch.envelope.status_flags = 0;
                        ch.position = pos;
                        ch.remaining = 0;
                        ch.frac_q23 = fw;
                        return;
                    }
                } else {
                    pos = pos.saturating_add(1);
                }
            }
            let s0 = sample.sample_i8(pos) as i32;
            let s1 = sample.sample_i8(pos.saturating_add(1)) as i32;
            let diff = s1 - s0;
            // Exact ARM: `mul r6, r1, r7; mul r1, r6, ip; add r6, r0, r1, asr #23`
            let interp = s0 + (diff.wrapping_mul(fw as i32).wrapping_mul(df) >> 23);
            *r = r.wrapping_add(((interp * vr) >> 8) as i8);
            *l = l.wrapping_add(((interp * vl) >> 8) as i8);
            fw = fw.wrapping_add(ch.frequency);
        }

        ch.position = pos;
        ch.remaining = remaining;
        ch.frac_q23 = fw;
    }

    /// Early 2001 SDK v1 mono exact-Hz modulo resampler (`AD2J` `0x08012766`).
    fn mix_early_v1_mono_inner(
        sample: &M4aSampleView<'_>,
        ch: &mut M4aSoundChannel,
        sample_rate: u32,
        div_freq: u32,
        vol_mono: u8,
        out: &mut [i8],
    ) {
        if out.is_empty() || sample_rate == 0 {
            return;
        }
        let vm = vol_mono as i32;
        let df = div_freq as i32;
        let looped = (ch.envelope.status_flags & SF_LOOP) != 0 && sample.loop_len() > 0;
        let loop_start = sample.loop_start();
        let loop_len = sample.loop_len() as u32;

        let mut pos = ch.position;
        let mut remaining = ch.remaining;
        let mut fw = ch.frac_q23;

        for o in out.iter_mut() {
            while fw >= sample_rate {
                fw -= sample_rate;
                remaining = remaining.saturating_sub(1);
                if remaining == 0 {
                    if looped {
                        pos = loop_start;
                        remaining = loop_len;
                    } else {
                        ch.envelope.status_flags = 0;
                        ch.position = pos;
                        ch.remaining = 0;
                        ch.frac_q23 = fw;
                        return;
                    }
                } else {
                    pos = pos.saturating_add(1);
                }
            }
            let s0 = sample.sample_i8(pos) as i32;
            let s1 = sample.sample_i8(pos.saturating_add(1)) as i32;
            let diff = s1 - s0;
            let interp = s0 + (diff.wrapping_mul(fw as i32).wrapping_mul(df) >> 23);
            *o = o.wrapping_add(((interp * vm) >> 8) as i8);
            fw = fw.wrapping_add(ch.frequency);
        }

        ch.position = pos;
        ch.remaining = remaining;
        ch.frac_q23 = fw;
    }
}

/// Number of fractional polyphase steps in [`LANCZOS4_LUT`].
const LANCZOS_PHASES: usize = 256;

/// Compile-time precomputed, row-normalized 8-tap Lanczos-4 polyphase filter bank
/// (`257 * 8 * 4 = 8,224` bytes, L1-cache resident, `#![no_std]` zero-init).
const LANCZOS4_LUT: [[f32; 8]; LANCZOS_PHASES + 1] = build_lanczos4_lut();

/// Build the normalized 8-tap Lanczos-4 polyphase lookup table at compile time.
const fn build_lanczos4_lut() -> [[f32; 8]; LANCZOS_PHASES + 1] {
    let mut table = [[0.0f32; 8]; LANCZOS_PHASES + 1];
    let mut p = 0usize;
    while p <= LANCZOS_PHASES {
        let frac = (p as f64) / (LANCZOS_PHASES as f64);
        let mut weights = [0.0f64; 8];
        let mut sum = 0.0f64;
        let mut idx = 0usize;
        while idx < 8 {
            let k = (idx as i32) - 3; // taps -3..=+4
            let w = lanczos_kernel(frac - (k as f64), 4.0);
            weights[idx] = w;
            sum += w;
            idx += 1;
        }
        let inv_sum = if sum > 1e-12 { 1.0 / sum } else { 0.0 };
        let mut j = 0usize;
        while j < 8 {
            table[p][j] = (weights[j] * inv_sum) as f32;
            j += 1;
        }
        p += 1;
    }
    table
}

/// High-fidelity 48,000 Hz `f32` software synthesizer with windowed-sinc band-limited
/// interpolation and constant-power stereo panning.
#[derive(Debug, Default, Clone, Copy)]
pub struct StudioSynth;

impl StudioSynth {
    /// Standard studio output sample rate (48,000 Hz).
    pub const SAMPLE_RATE: f64 = 48_000.0;

    /// Render a single M4A sample into stereo `f32` buffers using a precomputed 256-phase
    /// 8-tap Lanczos windowed-sinc polyphase filter and true equal-power stereo panning.
    ///
    /// - `playback_hz`: Effective sample playback frequency in Hz (from `midi_key_to_freq`).
    /// - `gain`: Linear gain (`0.0..=1.0`).
    /// - `pan`: Stereo pan (`0.0` = hard left, `0.5` = center, `1.0` = hard right).
    /// - `phase`: Current floating-point sample position (updated in place).
    /// - Returns `false` when a non-looped sample finishes or inputs are invalid.
    #[allow(clippy::too_many_arguments)]
    pub fn mix_voice_f32(
        sample: &M4aSampleView<'_>,
        phase: &mut f64,
        playback_hz: f64,
        output_rate_hz: f64,
        gain: f32,
        pan: f32,
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> bool {
        let len = sample.len().min(sample.pcm_bytes.len());
        if len == 0
            || !output_rate_hz.is_finite()
            || !playback_hz.is_finite()
            || output_rate_hz <= 0.0
            || playback_hz <= 0.0
            || !phase.is_finite()
        {
            return false;
        }
        if *phase < 0.0 {
            *phase = 0.0;
        }
        let frames = out_l.len().min(out_r.len());
        if frames == 0 {
            return true;
        }
        let out_l = &mut out_l[..frames];
        let out_r = &mut out_r[..frames];
        let pcm = sample.pcm_bytes;

        let step = playback_hz / output_rate_hz;
        let pan_clamped = pan.clamp(0.0, 1.0) as f64;
        let angle = pan_clamped * core::f64::consts::FRAC_PI_2;
        // Fold `1 / 128` signed-8-bit normalization into stereo gains.
        let scale = (gain as f64) * (1.0 / 128.0);
        let gain_l = (scale * cos_f64(angle)) as f32;
        let gain_r = (scale * sin_f64(angle)) as f32;

        let loop_start_usize = sample.loop_start();
        let loop_len_usize = sample.loop_len();
        let looped = sample.is_looped() && loop_len_usize > 0 && loop_start_usize < len;
        let loop_start = loop_start_usize as f64;
        let loop_len = loop_len_usize as f64;
        let len_f = len as f64;
        let safe_max_base = len.saturating_sub(5);
        let mut ph = *phase;
        let mut idx = 0usize;

        while idx < frames {
            if ph >= len_f {
                if looped {
                    ph -= loop_len;
                    if ph >= len_f {
                        ph = loop_start + ((ph - loop_start) % loop_len);
                    }
                    if ph < loop_start {
                        ph = loop_start;
                    }
                } else {
                    *phase = ph;
                    return false;
                }
            }

            // Fast interior run: Q32.32 fixed-point phase with zero boundary checks or f64 casts.
            let base = ph as usize;
            if base >= 3 && base <= safe_max_base && step < (1u64 << 31) as f64 {
                let safe_limit = (safe_max_base as f64) + 0.999_999;
                let max_interior = (((safe_limit - ph) / step) as usize) + 1;
                let run = max_interior.min(frames - idx);
                if run >= 4 {
                    const Q32_SCALE: f64 = 4_294_967_296.0;
                    const INV_Q24: f32 = 1.0 / 16_777_216.0;
                    let mut ph_q32 = (ph * Q32_SCALE) as u64;
                    let step_q32 = (step * Q32_SCALE) as u64;
                    let last_base = ((ph_q32.saturating_add(step_q32.saturating_mul((run - 1) as u64))) >> 32) as usize;
                    if last_base <= safe_max_base {
                        for (ol, or) in out_l[idx..idx + run]
                            .iter_mut()
                            .zip(&mut out_r[idx..idx + run])
                        {
                            let b = (ph_q32 >> 32) as usize;
                            let p = ((ph_q32 >> 24) & 0xFF) as usize;
                            let t = ((ph_q32 & 0x00FF_FFFF) as f32) * INV_Q24;
                            let w0 = &LANCZOS4_LUT[p];
                            let w1 = &LANCZOS4_LUT[p + 1];
                            let win = &pcm[b - 3..b + 5];
                            let s0 = (win[0] as i8) as f32;
                            let s1 = (win[1] as i8) as f32;
                            let s2 = (win[2] as i8) as f32;
                            let s3 = (win[3] as i8) as f32;
                            let s4 = (win[4] as i8) as f32;
                            let s5 = (win[5] as i8) as f32;
                            let s6 = (win[6] as i8) as f32;
                            let s7 = (win[7] as i8) as f32;
                            let a0 = s0 * w0[0]
                                + s1 * w0[1]
                                + s2 * w0[2]
                                + s3 * w0[3]
                                + s4 * w0[4]
                                + s5 * w0[5]
                                + s6 * w0[6]
                                + s7 * w0[7];
                            let a1 = s0 * w1[0]
                                + s1 * w1[1]
                                + s2 * w1[2]
                                + s3 * w1[3]
                                + s4 * w1[4]
                                + s5 * w1[5]
                                + s6 * w1[6]
                                + s7 * w1[7];
                            let val = a0 + t * (a1 - a0);
                            *ol += val * gain_l;
                            *or += val * gain_r;
                            ph_q32 += step_q32;
                        }
                        ph += (run as f64) * step;
                        idx += run;
                        continue;
                    }
                }
            }

            // Boundary / loop-edge single-sample step
            let frac = ph - (base as f64);
            let idx_f = frac * (LANCZOS_PHASES as f64);
            let p = (idx_f as usize).min(LANCZOS_PHASES - 1);
            let t = (idx_f - (p as f64)) as f32;
            let w0 = &LANCZOS4_LUT[p];
            let w1 = &LANCZOS4_LUT[p + 1];

            let mut s = [0.0f32; 8];
            let base_i = base as isize;
            for (slot, k) in s.iter_mut().zip(-3isize..=4) {
                let sample_idx = base_i + k;
                if sample_idx >= 0 {
                    let u = sample_idx as usize;
                    if u < len {
                        *slot = (pcm[u] as i8) as f32;
                    } else if looped {
                        let wrapped =
                            loop_start_usize + ((u - loop_start_usize) % loop_len_usize);
                        *slot = (pcm[wrapped] as i8) as f32;
                    }
                }
            }

            let a0 = s[0] * w0[0]
                + s[1] * w0[1]
                + s[2] * w0[2]
                + s[3] * w0[3]
                + s[4] * w0[4]
                + s[5] * w0[5]
                + s[6] * w0[6]
                + s[7] * w0[7];
            let a1 = s[0] * w1[0]
                + s[1] * w1[1]
                + s[2] * w1[2]
                + s[3] * w1[3]
                + s[4] * w1[4]
                + s[5] * w1[5]
                + s[6] * w1[6]
                + s[7] * w1[7];
            let val = a0 + t * (a1 - a0);

            out_l[idx] += val * gain_l;
            out_r[idx] += val * gain_r;
            ph += step;
            idx += 1;
        }

        *phase = ph;
        true
    }
}

/// `#![no_std]` Lanczos windowed-sinc kernel `sinc(x) * sinc(x / a)`.
const fn lanczos_kernel(x: f64, a: f64) -> f64 {
    let ax = if x < 0.0 { -x } else { x };
    if ax < 1e-12 {
        return 1.0;
    }
    if ax >= a {
        return 0.0;
    }
    let pi_x = core::f64::consts::PI * x;
    let pi_x_a = pi_x / a;
    (sin_f64(pi_x) / pi_x) * (sin_f64(pi_x_a) / pi_x_a)
}

/// Pure `#![no_std]` `sin(x)` using range reduction + 11th-order Taylor polynomial.
const fn sin_f64(mut x: f64) -> f64 {
    const TWO_PI: f64 = core::f64::consts::TAU;
    const PI: f64 = core::f64::consts::PI;
    const HALF_PI: f64 = core::f64::consts::FRAC_PI_2;

    // Reduce to [-PI, PI]
    let k = (x / TWO_PI) as i64;
    x -= (k as f64) * TWO_PI;
    if x > PI {
        x -= TWO_PI;
    } else if x < -PI {
        x += TWO_PI;
    }
    // Fold to [-PI/2, PI/2]
    if x > HALF_PI {
        x = PI - x;
    } else if x < -HALF_PI {
        x = -PI - x;
    }
    let x2 = x * x;
    x * (1.0
        + x2 * (-1.0 / 6.0
            + x2 * (1.0 / 120.0
                + x2 * (-1.0 / 5_040.0 + x2 * (1.0 / 362_880.0 - x2 / 39_916_800.0)))))
}

/// Pure `#![no_std]` `cos(x)` via `sin(x + PI/2)`.
const fn cos_f64(x: f64) -> f64 {
    sin_f64(x + core::f64::consts::FRAC_PI_2)
}
