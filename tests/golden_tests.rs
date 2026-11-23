//! Byte-Exact mGBA ARM7TDMI Golden Integration Tests for Nintendo MusicPlayer2000 (m4a-rs)
//!
//! Verifies:
//! - **Tier 1**: Bit-exact `midi_key_to_freq` against GBA BIOS SWI `0x1F` (`MidiKey2Freq`)
//!   executed in `libmgba` across 24 diverse `(base_freq, key, fine_tune)` tuples.
//! - **Tier 2**: Bit-exact 60 Hz ADSR + pseudo-echo envelope progression (`status_flags`,
//!   `envelope_volume`, `envelope_volume_right`, `envelope_volume_left`) against
//!   *Pokemon FireRed* (`BPRE`) `SoundMainRAM` (`0x081DC0A4`) across 40 frames.
//! - **Tier 3**: Strict byte-for-byte `AccuracySynth` DirectSound DMA FIFO verification
//!   against real ARM7TDMI `SoundMainRAM` execution in *Pokemon FireRed* (`BPRE`, `0x081DC0A4`)
//!   and *Advance Wars* (`AWRE`, `0x08077634`) across 6 scenarios (2,688 ARM mixer bytes):
//!   1. Fixed 1:1 pitch stereo mix
//!   2. Resampled Q23 linear interpolation stereo mix
//!   3. Looped sample wrap across loop boundary
//!   4. Multi-voice (3-channel) stereo accumulation
//!   5. Multi-frame reverb feedback (`reverb = 64`) with negative-bias correction
//!   6. Cross-title verification on *Advance Wars* (`AWRE`)

#[cfg(feature = "alloc")]
use m4a::{decode_dpcm_vec, M4aSample};
use m4a::{
    decode_dpcm_block, decode_dpcm_into, detect_mixer_variant, div_freq_for_rate, midi_key_to_freq,
    samples_per_vblank, AccuracySynth, EnvelopeStage, M4aEnvelope, M4aEvent, M4aMixerVariant,
    M4aSampleView, M4aSoundChannel, StudioSynth, TrackDecoder, WaveDataHeader, DPCM_BLOCK_BYTES,
    DPCM_BLOCK_SAMPLES, M4A_RATE_TABLE,
};

const MANIFEST_JSON: &str = include_str!("golden/manifest.json");

const BPRE_SAMPLE_084B0634: &[u8] = include_bytes!("golden/bpre_sample_084b0634.wavdata");
const BPRE_SAMPLE_08502768: &[u8] = include_bytes!("golden/bpre_sample_08502768.wavdata");
const AWRE_SAMPLE_0814BE54: &[u8] = include_bytes!("golden/awre_sample_0814be54.wavdata");

const MIDI_KEY_TO_FREQ_GOLDEN: &[u8] = include_bytes!("golden/midi_key_to_freq.bin");
const BPRE_ADSR_TRACE_GOLDEN: &[u8] = include_bytes!("golden/bpre_adsr_trace.bin");

const BPRE_MIX_FIXED_1TO1: &[u8] = include_bytes!("golden/bpre_mix_fixed_1to1.bin");
const BPRE_MIX_RESAMPLED: &[u8] = include_bytes!("golden/bpre_mix_resampled.bin");
const BPRE_MIX_LOOPED: &[u8] = include_bytes!("golden/bpre_mix_looped.bin");
const BPRE_MIX_MULTIVOICE: &[u8] = include_bytes!("golden/bpre_mix_multivoice.bin");
const BPRE_MIX_REVERB: &[u8] = include_bytes!("golden/bpre_mix_reverb.bin");
const AWRE_MIX_RESAMPLED: &[u8] = include_bytes!("golden/awre_mix_resampled.bin");

// Clean-room synthetic WaveData + multi-variant libmgba ARM7TDMI golden fixtures
const SYNTH_PCM_512: &[u8] = include_bytes!("golden/synth_pcm_512.wavdata");
const SYNTH_DPCM_384: &[u8] = include_bytes!("golden/synth_dpcm_384.wavdata");
const AGSE_STD_RESAMPLED: &[u8] = include_bytes!("golden/agse_std_resampled.bin");
const A3UJ_STD_RESAMPLED: &[u8] = include_bytes!("golden/a3uj_std_resampled.bin");
const BPRE_DPCM_FORWARD: &[u8] = include_bytes!("golden/bpre_dpcm_forward.bin");
const BPRE_REVERSE_PCM: &[u8] = include_bytes!("golden/bpre_reverse_pcm.bin");
const BPRE_REVERSE_DPCM: &[u8] = include_bytes!("golden/bpre_reverse_dpcm.bin");
const AMKE_MONO_RESAMPLED: &[u8] = include_bytes!("golden/amke_mono_resampled.bin");
const AMKE_MONO_REVERB: &[u8] = include_bytes!("golden/amke_mono_reverb.bin");
const A22J_EARLY_V1_STEREO: &[u8] = include_bytes!("golden/a22j_early_v1_stereo.bin");
const AD2J_EARLY_V1_MONO: &[u8] = include_bytes!("golden/ad2j_early_v1_mono.bin");
const AMTE_RD1_Q14: &[u8] = include_bytes!("golden/amte_rd1_q14.bin");
const AMTE_RD1_FIXED: &[u8] = include_bytes!("golden/amte_rd1_fixed.bin");
const AMTE_RD1_REVERB_CLAMP: &[u8] = include_bytes!("golden/amte_rd1_reverb_clamp.bin");
const AMTE_RD1_REVERSE: &[u8] = include_bytes!("golden/amte_rd1_reverse.bin");

const FRAMES: usize = 224;

#[test]
fn test_manifest_integrity() {
    assert!(!MANIFEST_JSON.is_empty());
    for expected in [
        "bpre_sample_084b0634.wavdata",
        "bpre_sample_08502768.wavdata",
        "awre_sample_0814be54.wavdata",
        "midi_key_to_freq.bin",
        "bpre_adsr_trace.bin",
        "bpre_mix_fixed_1to1.bin",
        "bpre_mix_resampled.bin",
        "bpre_mix_looped.bin",
        "bpre_mix_multivoice.bin",
        "bpre_mix_reverb.bin",
        "awre_mix_resampled.bin",
    ] {
        assert!(
            MANIFEST_JSON.contains(expected),
            "manifest missing {expected}"
        );
    }
}

// ============================================================================
// Tier 1: Bit-exact GBA BIOS SWI 0x1F (MidiKey2Freq) & Sample Parsing
// ============================================================================

#[test]
fn test_tier1_bios_midi_key_to_freq_bit_exact() {
    const CASES: [(u32, u8, u8); 24] = [
        (13_700_096, 60, 0),
        (13_700_096, 69, 0),
        (13_700_096, 72, 0),
        (13_700_096, 48, 0),
        (13_700_096, 36, 0),
        (13_700_096, 84, 0),
        (13_700_096, 60, 128),
        (13_700_096, 60, 255),
        (13_700_096, 61, 64),
        (13_700_096, 59, 192),
        (13_700_096, 0, 0),
        (13_700_096, 127, 0),
        (13_700_096, 127, 255),
        (13_700_096, 179, 0),
        (13_700_096, 180, 0),
        (6_850_048, 60, 0),
        (6_850_048, 72, 128),
        (11_289_600, 64, 37),
        (16_777_216, 60, 0),
        (16_777_216, 67, 99),
        (14_109_696, 55, 200),
        (14_109_696, 90, 15),
        (8_388_608, 45, 170),
        (22_050_000, 76, 42),
    ];

    assert_eq!(MIDI_KEY_TO_FREQ_GOLDEN.len(), CASES.len() * 4);
    for (i, &(base_freq, key, fine_tune)) in CASES.iter().enumerate() {
        let expected = u32::from_le_bytes([
            MIDI_KEY_TO_FREQ_GOLDEN[i * 4],
            MIDI_KEY_TO_FREQ_GOLDEN[i * 4 + 1],
            MIDI_KEY_TO_FREQ_GOLDEN[i * 4 + 2],
            MIDI_KEY_TO_FREQ_GOLDEN[i * 4 + 3],
        ]);
        let actual = midi_key_to_freq(base_freq, key, fine_tune);
        assert_eq!(
            actual, expected,
            "MidiKey2Freq mismatch at case {i} (base={base_freq}, key={key}, fp={fine_tune})"
        );
    }
}

#[test]
fn test_tier1_sample_view_parsing() {
    let s1 = M4aSampleView::from_wavdata_bytes(BPRE_SAMPLE_084B0634).unwrap();
    assert!(!s1.is_looped());
    assert_eq!(s1.len(), 1222);
    assert_eq!(s1.header.base_sample_rate_hz(), 13_379);
    assert!(!s1.is_silent());

    let s2 = M4aSampleView::from_wavdata_bytes(BPRE_SAMPLE_08502768).unwrap();
    assert!(s2.is_looped());
    assert_eq!(s2.loop_start(), 213);
    assert_eq!(s2.len(), 1978);
    assert_eq!(s2.loop_len(), 1978 - 213);

    let s3 = M4aSampleView::from_wavdata_bytes(AWRE_SAMPLE_0814BE54).unwrap();
    assert!(s3.is_looped());
    assert_eq!(s3.loop_start(), 41);
    assert_eq!(s3.len(), 1744);

    assert_eq!(div_freq_for_rate(13_379), 626);
    assert_eq!(samples_per_vblank(13_379), 224);
    assert_eq!(M4A_RATE_TABLE[3], 13_379);
}

#[cfg(feature = "alloc")]
#[test]
fn test_tier1_owned_sample_matches_view() {
    let view = M4aSampleView::from_wavdata_bytes(BPRE_SAMPLE_084B0634).unwrap();
    let owned = M4aSample::from_wavdata_bytes(BPRE_SAMPLE_084B0634).unwrap();
    assert_eq!(owned.len(), view.len());
    assert_eq!(owned.min_max(), view.min_max());
    for i in 0..view.len() {
        assert_eq!(owned.pcm_data[i], view.sample_i8(i), "mismatch at {i}");
    }
}

// ============================================================================
// Tier 2: Bit-exact 60 Hz ADSR + Pseudo-Echo Envelope Progression
// ============================================================================

#[test]
fn test_tier2_bpre_adsr_envelope_bit_exact() {
    assert_eq!(BPRE_ADSR_TRACE_GOLDEN.len(), 40 * 4);
    let mut env = M4aEnvelope::new(64, 200, 120, 180, 24, 4);

    for tick in 0..40usize {
        if tick == 20 {
            env.key_off();
        }
        let _ = env.tick(15, 200, 100, true);

        let exp_status = BPRE_ADSR_TRACE_GOLDEN[tick * 4];
        let exp_env = BPRE_ADSR_TRACE_GOLDEN[tick * 4 + 1];
        let exp_r = BPRE_ADSR_TRACE_GOLDEN[tick * 4 + 2];
        let exp_l = BPRE_ADSR_TRACE_GOLDEN[tick * 4 + 3];

        assert_eq!(
            env.status_flags, exp_status,
            "ADSR status_flags mismatch at tick {tick}"
        );
        assert_eq!(
            env.envelope_volume, exp_env,
            "ADSR envelope_volume mismatch at tick {tick}"
        );
        assert_eq!(
            env.envelope_volume_right, exp_r,
            "ADSR envelope_volume_right mismatch at tick {tick}"
        );
        assert_eq!(
            env.envelope_volume_left, exp_l,
            "ADSR envelope_volume_left mismatch at tick {tick}"
        );

        match tick {
            0..=2 => assert_eq!(env.stage(), EnvelopeStage::Attack),
            3..=6 => assert_eq!(env.stage(), EnvelopeStage::Decay),
            7..=19 => assert_eq!(env.stage(), EnvelopeStage::Sustain),
            20..=23 => assert_eq!(env.stage(), EnvelopeStage::Release),
            24..=27 => assert_eq!(env.stage(), EnvelopeStage::PseudoEcho),
            _ => assert_eq!(env.stage(), EnvelopeStage::Off),
        }
    }
}

// ============================================================================
// Tier 3: Strict Byte-for-Byte mGBA ARM7TDMI SoundMainRAM Mixer Verification
// ============================================================================

fn assert_stereo_golden(name: &str, out_r: &[i8], out_l: &[i8], golden: &[u8]) {
    assert_eq!(golden.len(), FRAMES * 2);
    for i in 0..FRAMES {
        assert_eq!(
            out_r[i], golden[i] as i8,
            "{name}: Right channel mismatch at frame {i}"
        );
        assert_eq!(
            out_l[i],
            golden[FRAMES + i] as i8,
            "{name}: Left channel mismatch at frame {i}"
        );
    }
}

#[test]
fn test_tier3_bpre_arm_golden_fixed_1to1() {
    let sample = M4aSampleView::from_wavdata_bytes(BPRE_SAMPLE_084B0634).unwrap();
    let mut ch = M4aSoundChannel::new(0x08, 220, 110, 255, 255, 255, 0, 13_379, 0);
    let mut out_r = [0i8; FRAMES];
    let mut out_l = [0i8; FRAMES];

    AccuracySynth::mix_channel_frame(&sample, &mut ch, 13_379, 15, &mut out_r, &mut out_l);
    assert_stereo_golden("BPRE fixed 1:1", &out_r, &out_l, BPRE_MIX_FIXED_1TO1);
}

#[test]
fn test_tier3_bpre_arm_golden_resampled_q23() {
    let sample = M4aSampleView::from_wavdata_bytes(BPRE_SAMPLE_084B0634).unwrap();
    let mut ch = M4aSoundChannel::new(0x00, 200, 140, 255, 255, 255, 0, 18_500, 0);
    let mut out_r = [0i8; FRAMES];
    let mut out_l = [0i8; FRAMES];

    AccuracySynth::mix_channel_frame(&sample, &mut ch, 13_379, 15, &mut out_r, &mut out_l);
    assert_stereo_golden("BPRE resampled Q23", &out_r, &out_l, BPRE_MIX_RESAMPLED);
}

#[test]
fn test_tier3_bpre_arm_golden_looped_wrap() {
    let sample = M4aSampleView::from_wavdata_bytes(BPRE_SAMPLE_08502768).unwrap();
    let mut ch = M4aSoundChannel::new(0x00, 192, 192, 255, 255, 255, 0, 16_000, 1918);
    let mut out_r = [0i8; FRAMES];
    let mut out_l = [0i8; FRAMES];

    AccuracySynth::mix_channel_frame(&sample, &mut ch, 13_379, 15, &mut out_r, &mut out_l);
    assert_stereo_golden("BPRE looped wrap", &out_r, &out_l, BPRE_MIX_LOOPED);
}

#[test]
fn test_tier3_bpre_arm_golden_multivoice_stereo() {
    let s0 = M4aSampleView::from_wavdata_bytes(BPRE_SAMPLE_084B0634).unwrap();
    let s2 = M4aSampleView::from_wavdata_bytes(BPRE_SAMPLE_08502768).unwrap();

    let mut ch0 = M4aSoundChannel::new(0x08, 128, 64, 255, 255, 255, 0, 13_379, 0);
    let mut ch1 = M4aSoundChannel::new(0x00, 64, 128, 255, 255, 255, 0, 16_500, 100);
    let mut ch2 = M4aSoundChannel::new(0x00, 100, 100, 255, 255, 255, 0, 9_800, 200);

    let mut out_r = [0i8; FRAMES];
    let mut out_l = [0i8; FRAMES];

    AccuracySynth::mix_channel_frame(&s0, &mut ch0, 13_379, 15, &mut out_r, &mut out_l);
    AccuracySynth::mix_channel_frame(&s0, &mut ch1, 13_379, 15, &mut out_r, &mut out_l);
    AccuracySynth::mix_channel_frame(&s2, &mut ch2, 13_379, 15, &mut out_r, &mut out_l);

    assert_stereo_golden("BPRE multivoice", &out_r, &out_l, BPRE_MIX_MULTIVOICE);
}

#[test]
fn test_tier3_bpre_arm_golden_reverb_feedback() {
    let sample = M4aSampleView::from_wavdata_bytes(BPRE_SAMPLE_084B0634).unwrap();
    let mut ch = M4aSoundChannel::new(0x08, 180, 90, 255, 255, 255, 0, 13_379, 0);

    let mut slot0_r = [0i8; FRAMES];
    let mut slot0_l = [0i8; FRAMES];
    let mut slot1_r = [0i8; FRAMES];
    let mut slot1_l = [0i8; FRAMES];

    // Tick 0: slot 0, next = slot 1 (dma_counter=3)
    AccuracySynth::reverb_pass(&mut slot0_r, &mut slot0_l, &slot1_r, &slot1_l, 64);
    AccuracySynth::mix_channel_frame(&sample, &mut ch, 13_379, 15, &mut slot0_r, &mut slot0_l);

    // Tick 1: slot 1, next = slot 0 (dma_counter=2 wraps to base)
    AccuracySynth::reverb_pass(&mut slot1_r, &mut slot1_l, &slot0_r, &slot0_l, 64);
    AccuracySynth::mix_channel_frame(&sample, &mut ch, 13_379, 15, &mut slot1_r, &mut slot1_l);

    // Tick 2: slot 0, next = slot 1 (dma_counter=3) -> feeds back BOTH slot 0 & slot 1!
    AccuracySynth::reverb_pass(&mut slot0_r, &mut slot0_l, &slot1_r, &slot1_l, 64);
    AccuracySynth::mix_channel_frame(&sample, &mut ch, 13_379, 15, &mut slot0_r, &mut slot0_l);

    assert_stereo_golden("BPRE reverb feedback", &slot0_r, &slot0_l, BPRE_MIX_REVERB);
}

#[test]
fn test_tier3_awre_arm_golden_resampled_q23() {
    let sample = M4aSampleView::from_wavdata_bytes(AWRE_SAMPLE_0814BE54).unwrap();
    let mut ch = M4aSoundChannel::new(0x00, 210, 130, 255, 255, 255, 0, 15_200, 0);
    let mut out_r = [0i8; FRAMES];
    let mut out_l = [0i8; FRAMES];

    AccuracySynth::mix_channel_frame(&sample, &mut ch, 13_379, 15, &mut out_r, &mut out_l);
    assert_stereo_golden("AWRE resampled Q23", &out_r, &out_l, AWRE_MIX_RESAMPLED);
}

// ============================================================================
// Tier 4: Multi-Variant Corpus Byte-Exact ARM7TDMI Verification
// (Standard Cross-Studio, Pokemon Gen 3 DPCM/Reverse, Mono SDK v2,
//  Early SDK v1 Stereo/Mono, Nintendo R&D1 Metroid Fusion / Zero Mission)
// ============================================================================

fn assert_mono_golden(name: &str, out: &[i8], golden: &[u8]) {
    assert_eq!(golden.len(), FRAMES);
    for i in 0..FRAMES {
        assert_eq!(
            out[i], golden[i] as i8,
            "{name}: Mono channel mismatch at frame {i}"
        );
    }
}

#[test]
fn test_tier4_cross_studio_standard_agse_and_a3uj() {
    let sample = M4aSampleView::from_wavdata_bytes(SYNTH_PCM_512).unwrap();
    let mut ch = M4aSoundChannel::new(0x00, 200, 140, 255, 255, 255, 0, 17_500, 0);
    let mut out_r = [0i8; FRAMES];
    let mut out_l = [0i8; FRAMES];

    AccuracySynth::mix_channel_frame_variant(
        M4aMixerVariant::Standard,
        &sample,
        &mut ch,
        13_379,
        15,
        &mut out_r,
        &mut out_l,
    );
    assert_stereo_golden("AGSE (Golden Sun) Standard Q23", &out_r, &out_l, AGSE_STD_RESAMPLED);
    assert_stereo_golden("A3UJ (Mother 3) Standard Q23", &out_r, &out_l, A3UJ_STD_RESAMPLED);
}

#[test]
fn test_tier4_pokemon_gen3_dpcm_forward() {
    let sample = M4aSampleView::from_wavdata_bytes(SYNTH_DPCM_384).unwrap();
    assert!(sample.header.is_compressed_dpcm());
    assert_eq!(sample.len(), 384);

    // Also verify block & slice DPCM decoders agree with sample_i8_dpcm
    let mut blk0 = [0i8; DPCM_BLOCK_SAMPLES];
    decode_dpcm_block(&sample.pcm_bytes[..DPCM_BLOCK_BYTES], &mut blk0).unwrap();
    let mut all = [0i8; 384];
    assert_eq!(decode_dpcm_into(sample.pcm_bytes, &mut all).unwrap(), 384);
    for (i, &v) in blk0.iter().enumerate() {
        assert_eq!(v, all[i]);
        assert_eq!(v, sample.sample_i8_dpcm(i));
    }
    #[cfg(feature = "alloc")]
    {
        let owned = decode_dpcm_vec(sample.pcm_bytes, 384).unwrap();
        assert_eq!(owned.as_slice(), &all[..]);
        let owned_sample = sample.to_owned_sample();
        assert_eq!(owned_sample.pcm_data.as_slice(), &all[..]);
    }

    let mut ch = M4aSoundChannel::new(0x20, 210, 130, 255, 255, 255, 0, 15_500, 0);
    let mut out_r = [0i8; FRAMES];
    let mut out_l = [0i8; FRAMES];
    AccuracySynth::mix_channel_frame_variant(
        M4aMixerVariant::PokemonGen3,
        &sample,
        &mut ch,
        13_379,
        15,
        &mut out_r,
        &mut out_l,
    );
    assert_stereo_golden("BPRE 4-bit DPCM forward", &out_r, &out_l, BPRE_DPCM_FORWARD);
}

#[test]
fn test_tier4_pokemon_gen3_reverse_pcm() {
    let sample = M4aSampleView::from_wavdata_bytes(SYNTH_PCM_512).unwrap();
    let mut ch = M4aSoundChannel::new(0x10, 190, 150, 255, 255, 255, 0, 16_200, 0);
    let mut out_r = [0i8; FRAMES];
    let mut out_l = [0i8; FRAMES];

    AccuracySynth::mix_channel_frame_variant(
        M4aMixerVariant::PokemonGen3,
        &sample,
        &mut ch,
        13_379,
        15,
        &mut out_r,
        &mut out_l,
    );
    assert_stereo_golden("BPRE reverse PCM", &out_r, &out_l, BPRE_REVERSE_PCM);
}

#[test]
fn test_tier4_pokemon_gen3_reverse_dpcm() {
    let sample = M4aSampleView::from_wavdata_bytes(SYNTH_DPCM_384).unwrap();
    let mut ch = M4aSoundChannel::new(0x30, 180, 160, 255, 255, 255, 0, 14_800, 0);
    let mut out_r = [0i8; FRAMES];
    let mut out_l = [0i8; FRAMES];

    AccuracySynth::mix_channel_frame_variant(
        M4aMixerVariant::PokemonGen3,
        &sample,
        &mut ch,
        13_379,
        15,
        &mut out_r,
        &mut out_l,
    );
    assert_stereo_golden("BPRE reverse 4-bit DPCM", &out_r, &out_l, BPRE_REVERSE_DPCM);
}

#[test]
fn test_tier4_mono_sdk_v2_amke_resampled_and_reverb() {
    let sample = M4aSampleView::from_wavdata_bytes(SYNTH_PCM_512).unwrap();
    let mut ch = M4aSoundChannel::new(0x00, 200, 140, 255, 255, 255, 0, 17_500, 0);
    let mut out = [0i8; FRAMES];
    let mut unused = [0i8; FRAMES];

    AccuracySynth::mix_channel_frame_variant(
        M4aMixerVariant::Mono,
        &sample,
        &mut ch,
        13_379,
        15,
        &mut out,
        &mut unused,
    );
    assert_mono_golden("AMKE/BZME Mono Q23", &out, AMKE_MONO_RESAMPLED);

    // 3-tick mono reverb feedback (reverb = 64)
    let mut ch_rev = M4aSoundChannel::new(0x08, 180, 100, 255, 255, 255, 0, 13_379, 0);
    let mut slot0 = [0i8; FRAMES];
    let mut slot1 = [0i8; FRAMES];
    AccuracySynth::reverb_pass_mono(&mut slot0, &slot1, 64);
    AccuracySynth::mix_channel_frame_variant(
        M4aMixerVariant::Mono,
        &sample,
        &mut ch_rev,
        13_379,
        15,
        &mut slot0,
        &mut unused,
    );
    AccuracySynth::reverb_pass_mono(&mut slot1, &slot0, 64);
    AccuracySynth::mix_channel_frame_variant(
        M4aMixerVariant::Mono,
        &sample,
        &mut ch_rev,
        13_379,
        15,
        &mut slot1,
        &mut unused,
    );
    AccuracySynth::reverb_pass_mono(&mut slot0, &slot1, 64);
    AccuracySynth::mix_channel_frame_variant(
        M4aMixerVariant::Mono,
        &sample,
        &mut ch_rev,
        13_379,
        15,
        &mut slot0,
        &mut unused,
    );
    assert_mono_golden("AMKE Mono reverb 3-tick", &slot0, AMKE_MONO_REVERB);
}

#[test]
fn test_tier4_early_v1_stereo_and_mono() {
    let sample = M4aSampleView::from_wavdata_bytes(SYNTH_PCM_512).unwrap();

    // Early 2001 SDK v1 Stereo (A22J)
    let mut ch_s = M4aSoundChannel::new(0x00, 200, 140, 255, 255, 255, 0, 17_500, 0);
    let mut out_r = [0i8; FRAMES];
    let mut out_l = [0i8; FRAMES];
    AccuracySynth::mix_channel_frame_variant(
        M4aMixerVariant::EarlyV1Stereo,
        &sample,
        &mut ch_s,
        13_379,
        15,
        &mut out_r,
        &mut out_l,
    );
    assert_stereo_golden("A22J Early SDK v1 Stereo", &out_r, &out_l, A22J_EARLY_V1_STEREO);

    // Early 2001 SDK v1 Mono (AD2J Mr. Driller 2)
    let mut ch_m = M4aSoundChannel::new(0x00, 200, 140, 255, 255, 255, 0, 17_500, 0);
    let mut out_m = [0i8; FRAMES];
    let mut unused = [0i8; FRAMES];
    AccuracySynth::mix_channel_frame_variant(
        M4aMixerVariant::EarlyV1Mono,
        &sample,
        &mut ch_m,
        13_379,
        15,
        &mut out_m,
        &mut unused,
    );
    assert_mono_golden("AD2J Early SDK v1 Mono", &out_m, AD2J_EARLY_V1_MONO);
}

#[test]
fn test_tier4_metroid_rd1_q14_fixed_reverse_and_reverb_clamp() {
    let sample = M4aSampleView::from_wavdata_bytes(SYNTH_PCM_512).unwrap();

    // 1. R&D1 Q14 forward resampled
    let mut ch = M4aSoundChannel::new(0x00, 110, 75, 255, 255, 255, 0, 13_379, 0);
    let mut acc_r = [0i32; FRAMES];
    let mut acc_l = [0i32; FRAMES];
    let mut out_r = [0i8; FRAMES];
    let mut out_l = [0i8; FRAMES];
    AccuracySynth::mix_channel_frame_rd1(&sample, &mut ch, 21_428, 110, 75, &mut acc_r, &mut acc_l);
    AccuracySynth::finalize_rd1_dma8(&acc_r, &acc_l, &mut out_r, &mut out_l);
    assert_stereo_golden("AMTE/BMXE R&D1 Q14", &out_r, &out_l, AMTE_RD1_Q14);

    // 2. R&D1 Fixed 1:1
    let mut ch_fix = M4aSoundChannel::new(0x08, 120, 60, 255, 255, 255, 0, 13_379, 0);
    acc_r.fill(0);
    acc_l.fill(0);
    AccuracySynth::mix_channel_frame_rd1(&sample, &mut ch_fix, 0, 120, 60, &mut acc_r, &mut acc_l);
    AccuracySynth::finalize_rd1_dma8(&acc_r, &acc_l, &mut out_r, &mut out_l);
    assert_stereo_golden("AMTE/BMXE R&D1 Fixed 1:1", &out_r, &out_l, AMTE_RD1_FIXED);

    // 3. R&D1 Reverse Q14 (`voice_type = 0x20`)
    let mut ch_rev = M4aSoundChannel::new(0x20, 95, 85, 255, 255, 255, 0, 13_379, 0);
    ch_rev.frac_q23 = 510 << 14;
    acc_r.fill(0);
    acc_l.fill(0);
    AccuracySynth::mix_channel_frame_rd1(&sample, &mut ch_rev, 19_000, 95, 85, &mut acc_r, &mut acc_l);
    AccuracySynth::finalize_rd1_dma8(&acc_r, &acc_l, &mut out_r, &mut out_l);
    assert_stereo_golden("AMTE/BMXE R&D1 Reverse Q14", &out_r, &out_l, AMTE_RD1_REVERSE);

    // 4. R&D1 3-tick reverb feedback + 2-voice 32-bit accumulation + 9-bit saturating clamp
    let mut slot0_r = [0i8; FRAMES];
    let mut slot0_l = [0i8; FRAMES];
    let mut slot1_r = [0i8; FRAMES];
    let mut slot1_l = [0i8; FRAMES];
    let mut v0 = M4aSoundChannel::new(0x08, 120, 90, 255, 255, 255, 0, 13_379, 0);
    let mut v1 = M4aSoundChannel::new(0x00, 100, 120, 255, 255, 255, 0, 13_379, 0);
    v1.frac_q23 = 32 << 14;

    for tick in 0..3usize {
        if tick % 2 == 0 {
            AccuracySynth::reverb_pass_rd1(
                &slot0_r, &slot0_l, &slot1_r, &slot1_l, &mut acc_r, &mut acc_l, 64,
            );
            AccuracySynth::mix_channel_frame_rd1(&sample, &mut v0, 0, 120, 90, &mut acc_r, &mut acc_l);
            AccuracySynth::mix_channel_frame_rd1(&sample, &mut v1, 18_500, 100, 120, &mut acc_r, &mut acc_l);
            AccuracySynth::finalize_rd1_dma8(&acc_r, &acc_l, &mut slot0_r, &mut slot0_l);
        } else {
            AccuracySynth::reverb_pass_rd1(
                &slot1_r, &slot1_l, &slot0_r, &slot0_l, &mut acc_r, &mut acc_l, 64,
            );
            AccuracySynth::mix_channel_frame_rd1(&sample, &mut v0, 0, 120, 90, &mut acc_r, &mut acc_l);
            AccuracySynth::mix_channel_frame_rd1(&sample, &mut v1, 18_500, 100, 120, &mut acc_r, &mut acc_l);
            AccuracySynth::finalize_rd1_dma8(&acc_r, &acc_l, &mut slot1_r, &mut slot1_l);
        }
    }
    assert_stereo_golden(
        "AMTE/BMXE R&D1 Reverb + Clamp",
        &slot0_r,
        &slot0_l,
        AMTE_RD1_REVERB_CLAMP,
    );
}

#[test]
fn test_detect_mixer_variant_signatures() {
    // Synthetic ROM slices containing each variant's exact ARM/Thumb signature
    let mut std_rom = [0u8; 0x200];
    std_rom[0..12].copy_from_slice(&[0x43, 0x79, 0x00, 0x2B, 0x2C, 0xD0, 0x01, 0xA1, 0x08, 0x47, 0x00, 0x00]);
    std_rom[0x3C..0x40].copy_from_slice(&0xE1A0_04C1u32.to_le_bytes()); // asr r0, r1, #9 (Stereo)
    std_rom[0x140..0x144].copy_from_slice(&0xE3CE_E5FEu32.to_le_bytes()); // bic lr, lr, #0x3F800000
    assert_eq!(detect_mixer_variant(&std_rom), Some(M4aMixerVariant::Standard));

    let mut pkmn_rom = std_rom;
    pkmn_rom[0x120..0x124].copy_from_slice(&0xE310_0030u32.to_le_bytes()); // tst r0, #0x30
    assert_eq!(detect_mixer_variant(&pkmn_rom), Some(M4aMixerVariant::PokemonGen3));

    let mut mono_rom = std_rom;
    mono_rom[0x2C..0x30].copy_from_slice(&0xE1A0_0441u32.to_le_bytes()); // asr r0, r1, #8 (Mono)
    assert_eq!(detect_mixer_variant(&mono_rom), Some(M4aMixerVariant::Mono));

    let mut v1_stereo = [0u8; 0x200];
    v1_stereo[0..12].copy_from_slice(&[0x43, 0x79, 0x00, 0x2B, 0x2C, 0xD0, 0x01, 0xA1, 0x08, 0x47, 0x00, 0x00]);
    v1_stereo[0x3C..0x40].copy_from_slice(&0xE1A0_04C1u32.to_le_bytes());
    assert_eq!(detect_mixer_variant(&v1_stereo), Some(M4aMixerVariant::EarlyV1Stereo));

    let mut v1_mono = v1_stereo;
    v1_mono[0x2C..0x30].copy_from_slice(&0xE1A0_0441u32.to_le_bytes());
    assert_eq!(detect_mixer_variant(&v1_mono), Some(M4aMixerVariant::EarlyV1Mono));

    let mut rd1_rom = [0u8; 0x40];
    rd1_rom[0x10..0x1C].copy_from_slice(&[
        0x00, 0x70, 0xA0, 0xE3, 0x01, 0x70, 0x47, 0xE2, 0x27, 0x79, 0xA0, 0xE1,
    ]);
    assert_eq!(detect_mixer_variant(&rd1_rom), Some(M4aMixerVariant::MetroidRd1));

    assert_eq!(detect_mixer_variant(&[0u8; 64]), None);
}

// ============================================================================
// StudioSynth & Track Bytecode Sanity Checks
// ============================================================================

#[test]
fn test_studio_synth_48khz_band_limited() {
    let sample = M4aSampleView::from_wavdata_bytes(BPRE_SAMPLE_08502768).unwrap();
    let mut out_l = [0.0f32; 256];
    let mut out_r = [0.0f32; 256];
    let mut phase = 0.0f64;

    let active = StudioSynth::mix_voice_f32(
        &sample,
        &mut phase,
        13_379.0,
        StudioSynth::SAMPLE_RATE,
        0.8,
        0.25,
        &mut out_l,
        &mut out_r,
    );
    assert!(active);
    let energy_l: f32 = out_l.iter().map(|x| x * x).sum();
    let energy_r: f32 = out_r.iter().map(|x| x * x).sum();
    assert!(energy_l > energy_r * 2.0, "pan=0.25 should favor left channel");

    // Verify DC preservation of the precomputed normalized Lanczos-4 polyphase LUT
    let dc_pcm = [64u8; 64];
    let dc_hdr = WaveDataHeader {
        wave_type: 0,
        stat: 0,
        flags: 0x40,
        base_freq_q10: 13_700_096,
        loop_start: 0,
        size: 64,
    };
    let dc_sample = M4aSampleView::from_parts(dc_hdr, &dc_pcm).unwrap();
    let mut dc_l = [0.0f32; 64];
    let mut dc_r = [0.0f32; 64];
    let mut dc_phase = 10.125f64;
    assert!(StudioSynth::mix_voice_f32(
        &dc_sample,
        &mut dc_phase,
        13_379.0,
        StudioSynth::SAMPLE_RATE,
        1.0,
        0.0, // hard left: gain_l = 1.0, gain_r = 0.0
        &mut dc_l,
        &mut dc_r,
    ));
    for &v in &dc_l {
        assert!((v - 0.5).abs() < 1e-4, "expected DC 0.5, got {v}");
    }
    for &v in &dc_r {
        assert!(v.abs() < 1e-6, "expected hard-left right=0.0, got {v}");
    }

    // Verify NaN phase is rejected gracefully without panic
    let mut nan_phase = f64::NAN;
    assert!(!StudioSynth::mix_voice_f32(
        &dc_sample,
        &mut nan_phase,
        13_379.0,
        StudioSynth::SAMPLE_RATE,
        1.0,
        0.5,
        &mut dc_l,
        &mut dc_r,
    ));
}

#[test]
fn test_track_decoder_basic() {
    // Tempo 75 (150 BPM), Voice 4, Note len=24 key=60 vel=100, Wait 24, Fine
    let stream: [u8; 10] = [0xBB, 75, 0xBD, 4, 0xE7, 60, 100, 0x98, 0xB1, 0x00];
    let mut dec = TrackDecoder::new(&stream);
    assert_eq!(dec.next_event().unwrap(), Some(M4aEvent::Tempo(75)));
    assert_eq!(dec.next_event().unwrap(), Some(M4aEvent::Voice(4)));
    assert_eq!(
        dec.next_event().unwrap(),
        Some(M4aEvent::Note {
            length: 24,
            key: 60,
            velocity: 100
        })
    );
    assert_eq!(dec.next_event().unwrap(), Some(M4aEvent::Wait(24)));
    assert_eq!(dec.next_event().unwrap(), Some(M4aEvent::Fine));
}
