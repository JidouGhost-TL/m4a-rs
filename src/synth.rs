//! Dual synthesis pipelines for Nintendo MusicPlayer2000 (M4A / Sappy):
//! - [`AccuracySynth`]: Bit-exact emulation of Nintendo's ARM7TDMI `SoundMainRAM` IWRAM mixer.
//! - [`StudioSynth`]: 48,000 Hz `f32` audiophile renderer with band-limited windowed-sinc interpolation.

use crate::envelope::{M4aEnvelope, SF_LOOP, SF_START};
use crate::freq::{div_freq_for_rate, freq_step_q23};
use crate::sample::M4aSampleView;

/// Standard M4A DirectSound DMA ring buffer size per channel (`0x630` = 1,584 bytes).
pub const PCM_DMA_BUF_SIZE: usize = 0x630;

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
