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
        let rev = reverb as i32;
        let nlen = len.min(next_r.len()).min(next_l.len());
        for i in 0..nlen {
            let sum = (cur_r[i] as i32)
                + (cur_l[i] as i32)
                + (next_r[i] as i32)
                + (next_l[i] as i32);
            let mut out = (sum * rev) >> 9;
            // Exact ARM `tst r0, #0x80; addne r0, r0, #1` bias correction
            if (out & 0x80) != 0 {
                out += 1;
            }
            cur_r[i] = out as i8;
            cur_l[i] = out as i8;
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
            let start_off = ch.position;
            ch.remaining = sample.header.size.saturating_sub(start_off as u32);
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
    fn mix_fixed_inner(
        sample: &M4aSampleView<'_>,
        ch: &mut M4aSoundChannel,
        vol_r: u8,
        vol_l: u8,
        out_r: &mut [i8],
        out_l: &mut [i8],
    ) {
        let frames = out_r.len().min(out_l.len());
        let vr = vol_r as i32;
        let vl = vol_l as i32;
        let looped = (ch.envelope.status_flags & SF_LOOP) != 0 && sample.loop_len() > 0;
        let loop_start = sample.loop_start();
        let loop_len = sample.loop_len() as u32;

        for i in 0..frames {
            let s = sample.sample_i8(ch.position) as i32;
            ch.position += 1;

            // Exact ARM `mul r1, sl, r0; bic r1, r1, #0xFF0000; add r6, r1, r6, ror #8`
            out_r[i] = out_r[i].wrapping_add(((s * vr) >> 8) as i8);
            out_l[i] = out_l[i].wrapping_add(((s * vl) >> 8) as i8);

            ch.remaining -= 1;
            if ch.remaining == 0 {
                if looped {
                    ch.position = loop_start;
                    ch.remaining = loop_len;
                } else {
                    ch.envelope.status_flags = 0;
                    break;
                }
            }
        }
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
        let vr = vol_r as i32;
        let vl = vol_l as i32;
        let step = freq_step_q23(div_freq, ch.frequency);
        let looped = (ch.envelope.status_flags & SF_LOOP) != 0 && sample.loop_len() > 0;
        let loop_start = sample.loop_start();
        let loop_len = sample.loop_len() as i32;

        let mut pos = ch.position;
        let mut count = ch.remaining as i32;
        let mut fw = ch.frac_q23;

        let mut s0 = sample.sample_i8(pos) as i32;
        let mut s1 = sample.sample_i8(pos + 1) as i32;
        let mut diff = s1 - s0;

        for i in 0..frames {
            // Exact ARM:
            //   mul lr, sb, r1
            //   add lr, r0, lr, asr #23
            let interp = s0 + (((fw as i32).wrapping_mul(diff)) >> 23);

            // Exact ARM 8-bit wrapping accumulation per channel byte
            out_r[i] = out_r[i].wrapping_add(((interp * vr) >> 8) as i8);
            out_l[i] = out_l[i].wrapping_add(((interp * vl) >> 8) as i8);

            fw = fw.wrapping_add(step);
            let int_step = fw >> 23;
            if int_step != 0 {
                // Exact ARM `bic sb, sb, #0x3F800000`
                fw &= !0x3F80_0000;
                count -= int_step as i32;
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
                        s1 = sample.sample_i8(pos + 1) as i32;
                        diff = s1 - s0;
                    } else {
                        ch.envelope.status_flags = 0;
                        ch.position = pos;
                        ch.remaining = 0;
                        ch.frac_q23 = fw;
                        return;
                    }
                } else {
                    pos += int_step as usize;
                    if int_step == 1 {
                        s0 += diff;
                    } else {
                        s0 = sample.sample_i8(pos) as i32;
                    }
                    s1 = sample.sample_i8(pos + 1) as i32;
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
    pub fn clamp_i32_to_dma8(sample: i32) -> i8 {
        sample.clamp(-128, 127) as i8
    }
}

/// High-fidelity 48,000 Hz `f32` software synthesizer with windowed-sinc band-limited
/// interpolation and constant-power stereo panning.
pub struct StudioSynth;

impl StudioSynth {
    /// Standard studio output sample rate (48,000 Hz).
    pub const SAMPLE_RATE: f64 = 48_000.0;

    /// Render a single M4A sample into stereo `f32` buffers using an 8-tap Lanczos
    /// windowed-sinc interpolator and true equal-power stereo panning.
    ///
    /// - `playback_hz`: Effective sample playback frequency in Hz (from `midi_key_to_freq`).
    /// - `gain`: Linear gain (`0.0..=1.0`).
    /// - `pan`: Stereo pan (`0.0` = hard left, `0.5` = center, `1.0` = hard right).
    /// - `phase`: Current floating-point sample position (updated in place).
    /// - Returns `false` when a non-looped sample finishes.
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
        let len = sample.len();
        if len == 0 || output_rate_hz <= 0.0 || playback_hz <= 0.0 {
            return false;
        }
        let frames = out_l.len().min(out_r.len());
        let step = playback_hz / output_rate_hz;
        let pan_clamped = pan.clamp(0.0, 1.0) as f64;
        let angle = pan_clamped * (core::f64::consts::FRAC_PI_2);
        let gain_l = (gain as f64 * cos_f64(angle)) as f32;
        let gain_r = (gain as f64 * sin_f64(angle)) as f32;

        let looped = sample.is_looped() && sample.loop_len() > 0;
        let loop_start = sample.loop_start() as f64;
        let loop_len = sample.loop_len() as f64;
        let len_f = len as f64;

        const A: i32 = 4; // 8-tap Lanczos kernel (-3..=+4)

        for i in 0..frames {
            if *phase >= len_f {
                if looped {
                    while *phase >= len_f {
                        *phase -= loop_len;
                    }
                    if *phase < loop_start {
                        *phase = loop_start;
                    }
                } else {
                    return false;
                }
            }

            let base = *phase as i64;
            let frac = *phase - (base as f64);

            let mut acc = 0.0f64;
            let mut weight_sum = 0.0f64;
            for k in (1 - A)..=A {
                let idx = base + k as i64;
                let s_val = if idx < 0 {
                    0.0
                } else if (idx as usize) < len {
                    (sample.sample_i8(idx as usize) as f64) / 128.0
                } else if looped {
                    let wrapped = sample.loop_start()
                        + (((idx as usize) - sample.loop_start()) % sample.loop_len());
                    (sample.sample_i8(wrapped) as f64) / 128.0
                } else {
                    0.0
                };
                let x = frac - (k as f64);
                let w = lanczos_kernel(x, A as f64);
                acc += s_val * w;
                weight_sum += w;
            }

            let val = if weight_sum > 1e-9 {
                (acc / weight_sum) as f32
            } else {
                0.0
            };

            out_l[i] += val * gain_l;
            out_r[i] += val * gain_r;
            *phase += step;
        }

        true
    }
}

/// `#![no_std]` Lanczos windowed-sinc kernel `sinc(x) * sinc(x / a)`.
fn lanczos_kernel(x: f64, a: f64) -> f64 {
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

/// Pure `#![no_std]` `sin(x)` using range reduction + 9th-order minimax/Taylor polynomial.
fn sin_f64(mut x: f64) -> f64 {
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
fn cos_f64(x: f64) -> f64 {
    sin_f64(x + core::f64::consts::FRAC_PI_2)
}
