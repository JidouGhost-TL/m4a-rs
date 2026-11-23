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

use m4a::sequence::TrackDecoder;
#[cfg(feature = "alloc")]
use m4a::M4aSample;
use m4a::{
    div_freq_for_rate, midi_key_to_freq, samples_per_vblank, AccuracySynth, EnvelopeStage,
    M4aEnvelope, M4aEvent, M4aSampleView, M4aSoundChannel, StudioSynth, M4A_RATE_TABLE,
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
