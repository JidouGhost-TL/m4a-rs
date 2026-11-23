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
