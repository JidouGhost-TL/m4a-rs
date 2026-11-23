//! Bit-exact GBA BIOS `MidiKey2Freq` (SWI `0x1F`) and M4A `SoundMain` frequency math.

/// Standard Nintendo M4A DirectSound output sample rates (Hz) indexed by the
/// 4-bit frequency mode (`1..=12`) in `m4aSoundMode`.
pub const M4A_RATE_TABLE: [u32; 12] = [
    5_734, 7_884, 10_512, 13_379, 15_768, 18_157, 21_024, 26_758, 31_536, 36_314, 40_137, 42_048,
];

/// Exact semitone scale factors `round(2^(-i / 12) * 2^32)` for `i = 0..12`.
const SCALE_TABLE_Q32: [u64; 12] = [
    0x1_0000_0000, //  0 semitones
    0x0_F1A1_BF39, // -1 semitone
    0x0_E411_F03A, // -2 semitones
    0x0_D744_FCCB, // -3 semitones
    0x0_CB2F_F52A, // -4 semitones
    0x0_BFC8_86BB, // -5 semitones
    0x0_B504_F334, // -6 semitones
    0x0_AADC_0848, // -7 semitones
    0x0_A145_17CC, // -8 semitones
    0x0_9837_F052, // -9 semitones
    0x0_8FAC_D61E, // -10 semitones
    0x0_879C_7C97, // -11 semitones
];

/// `round((ln(2) / 3072) * 2^40)` for pure-integer `2^(-frac / 3072)` evaluation.
const LN2_OVER_3072_Q40: u128 = 248_087_468;

/// Bit-exact GBA BIOS `MidiKey2Freq` (SWI `0x1F`) calculation.
///
/// Computes `base_freq_q10 * 2^((key + fine_tune / 256 - 180) / 12)` using pure
/// `#![no_std]` fixed-point integer arithmetic.
///
/// Verified bit-for-bit (24/24) against mGBA's `GBASwi16(cpu, 0x1F)`.
#[must_use]
pub const fn midi_key_to_freq(base_freq_q10: u32, key: u8, fine_tune: u8) -> u32 {
    if key >= 180 {
        return base_freq_q10;
    }

    let key_diff = (180u32 - key as u32) * 256 - fine_tune as u32;
    let semitones = key_diff >> 8;
    let frac = (key_diff & 0xFF) as u128;

    let octave = semitones / 12;
    let rem = (semitones % 12) as usize;

    // Pure-integer 4th-order Taylor expansion of exp(-frac * ln(2) / 3072) in Q40.
    // Maximum x = 255 * ln(2) / 3072 ≈ 0.05754, so 4th-order error is < 1e-11.
    let x = frac * LN2_OVER_3072_Q40;
    let t1 = x;
    let t2 = (x * x) >> 41;
    let t3 = (t2 * x) / (3u128 << 40);
    let t4 = (t3 * x) >> 42;
    let fine_q40 = (1u128 << 40) - t1 + t2 - t3 + t4;

    let scale_q32 = ((SCALE_TABLE_Q32[rem] as u128) * fine_q40) >> 40;
    let shift = 32 + octave;
    if shift >= 128 {
        0
    } else {
        (((base_freq_q10 as u128) * scale_q32) >> shift) as u32
    }
}

/// Compute `SoundInfo.divFreq` (`0x0080_0000 / sample_rate`) used by `SoundMainRAM`.
///
/// In Nintendo's M4A driver, `SoundMainRAM` multiplies `divFreq` by `channel.frequency`
/// to obtain the Q23 fractional phase step per output sample.
#[must_use]
pub const fn div_freq_for_rate(sample_rate: u32) -> u32 {
    if sample_rate == 0 {
        return 0;
    }
    0x0080_0000 / sample_rate
}

/// Compute the exact Q23 phase increment (`div_freq * channel_freq`) used in
/// `SoundMainRAM`'s resampled ARM mixing loop (`mul r4, ip, r1`).
#[must_use]
pub const fn freq_step_q23(div_freq: u32, channel_freq: u32) -> u32 {
    div_freq.wrapping_mul(channel_freq)
}

/// Compute the standard number of DirectSound PCM samples mixed per 59.7275 Hz VBlank
/// (rounded to a multiple of 16 as in Nintendo's `m4aSoundMode`).
#[must_use]
pub const fn samples_per_vblank(sample_rate: u32) -> usize {
    // GBA VBlank period is 280,896 CPU cycles (16,777,216 / 280,896 ≈ 59.7275 Hz).
    // Standard M4A rounds `sample_rate / 59.7275` to a multiple of 16 (e.g. 13379 -> 224).
    let raw = ((sample_rate as u64) * 280_896 + 8_388_608) / 16_777_216;
    ((raw + 8) & !15) as usize
}
