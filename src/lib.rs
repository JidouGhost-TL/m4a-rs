//! # m4a-rs
//!
//! Pure Rust, zero-dependency parser, software synthesizer, and byte-exact
//! mGBA ARM golden-tested emulator for the **Nintendo MusicPlayer2000 (M4A / Sappy)**
//! sound engine on the Game Boy Advance.
//!
//! MusicPlayer2000 is Nintendo's official GBA audio SDK (`m4a`), powering
//! hundreds of commercial titles including *Pokemon FireRed/LeafGreen/Emerald*,
//! *Advance Wars*, *Castlevania: Harmony of Dissonance*, *Metroid Fusion*,
//! *Mother 3*, and *Golden Sun*.
//!
//! ## Features
//! - **Zero external dependencies** (`[dependencies]` is empty).
//! - **`#![no_std]` compatible** with `default = ["std"]`, `std = ["alloc"]`, `alloc = []`.
//! - **`#![deny(unsafe_code)]`**: 100% safe Rust.
//! - **Dual synthesis pipelines**:
//!   - [`AccuracySynth`]: Bit-exact emulation of Nintendo's ARM7TDMI `SoundMainRAM`
//!     IWRAM mixer (Q23 linear interpolation, 60 Hz ADSR + pseudo-echo, reverb feedback,
//!     and DirectSound DMA FIFO output).
//!   - [`StudioSynth`]: 48,000 Hz `f32` high-fidelity renderer with band-limited
//!     windowed-sinc interpolation and true stereo panning.
//!
//! ## Quickstart
//!
//! ```rust
//! use m4a::{AccuracySynth, M4aSampleView, M4aSoundChannel, midi_key_to_freq};
//!
//! let wavdata: &[u8] = &[
//!     0x00, 0x00, 0x00, 0x00, // type=0, stat=0, flags=0 (one-shot)
//!     0x00, 0x0C, 0xD1, 0x00, // base_freq_q10 = 13,700,096 (13,379 Hz)
//!     0x00, 0x00, 0x00, 0x00, // loop_start = 0
//!     0x08, 0x00, 0x00, 0x00, // size = 8
//!     0x00, 0x20, 0x40, 0x20, 0x00, 0xE0, 0xC0, 0xE0,
//! ];
//! let sample = M4aSampleView::from_wavdata_bytes(wavdata).unwrap();
//! let freq = midi_key_to_freq(sample.header.base_freq_q10, 64, 0);
//! let mut ch = M4aSoundChannel::new(0x00, 200, 140, 255, 255, 255, 0, freq, 0);
//! let mut out_r = [0i8; 64];
//! let mut out_l = [0i8; 64];
//! AccuracySynth::mix_channel_frame(&sample, &mut ch, 13_379, 15, &mut out_r, &mut out_l);
//! assert!(out_r.iter().any(|&s| s != 0));
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_code)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod envelope;
pub mod error;
pub mod freq;
pub mod sample;
pub mod sequence;
pub mod synth;

pub use envelope::{
    EnvelopeStage, M4aEnvelope, SF_ENV_ATTACK, SF_ENV_DECAY, SF_ENV_MASK, SF_ENV_OFF,
    SF_ENV_SUSTAIN, SF_IEC, SF_LOOP, SF_START, SF_STOP,
};
pub use error::{Error, Result};
pub use freq::{
    div_freq_for_rate, freq_step_q23, midi_key_to_freq, samples_per_vblank, M4A_RATE_TABLE,
};
#[cfg(feature = "alloc")]
pub use sample::{decode_dpcm_vec, M4aSample};
pub use sample::{
    decode_dpcm_block, decode_dpcm_into, dpcm_encoded_len, M4aSampleView, WaveDataHeader,
    DPCM_BLOCK_BYTES, DPCM_BLOCK_SAMPLES, DPCM_DELTA_TABLE, WAVE_DATA_HEADER_SIZE,
};
pub use sequence::{
    M4aChannel, M4aEvent, M4aVoice, SongHeaderView, TrackDecoder, VoiceKind, CLOCK_TABLE,
    VOICE_ENTRY_SIZE,
};
pub use synth::{
    detect_mixer_variant, AccuracySynth, M4aMixerVariant, M4aSoundChannel, StudioSynth,
    PCM_DMA_BUF_SIZE, RD1_DMA_BUF_SIZE,
};
