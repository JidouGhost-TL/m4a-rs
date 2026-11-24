# m4a-rs 🎵

**Zero-dependency, pure Rust parser, software synthesizer, and byte-exact mGBA ARM golden-tested emulator for the Nintendo MusicPlayer2000 (M4A / Sappy) Sound Engine (Game Boy Advance).**

[![Crates.io](https://img.shields.io/crates/v/m4a.svg)](https://crates.io/crates/m4a)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)
[![no_std](https://img.shields.io/badge/no__std-compatible-success.svg)](#features)

> **Looking for a ready-to-use CLI?** You can scan GBA ROMs, export MIDI + WAV samples, and compile SoundFont 2.04 (`.sf2`) banks directly from the command line using our **[gbakit](https://github.com/JidouGhost-TL/gbakit)** toolkit (`gbakit audio scan`, `gbakit audio export`, `gbakit audio soundfont`).

**MusicPlayer2000 (`m4a`)** is Nintendo's official Game Boy Advance audio SDK, powering hundreds of classic GBA titles including *Pokemon FireRed / LeafGreen / Emerald*, *Advance Wars*, *Castlevania: Harmony of Dissonance*, *Metroid Fusion*, *Mother 3*, and *Golden Sun*.

`m4a-rs` provides a clean-room, reverse-engineered implementation designed for audio preservation, DAWs, PC source ports, and emulators.

---

## Features

- **Zero External Dependencies**: Pure Rust with no dependencies in `[dependencies]`.
- **`#![no_std]` Compatible**: Ready for embedded targets and WASM runtimes (`default = ["std"]`, `std = ["alloc"]`, `alloc = []`).
- **100% Safe Rust**: `#![deny(unsafe_code)]` across the entire crate.
- **Dual Synthesis Engines**:
  - **`AccuracySynth`**: Bit-exact emulation of Nintendo's ARM7TDMI `SoundMainRAM` IWRAM mixer — exact Q23 linear interpolation, 60 Hz ADSR + pseudo-echo envelopes, reverb feedback with negative-bias correction, and DirectSound DMA FIFO output.
  - **`StudioSynth`**: Renders at 48,000 Hz, 32-bit floating-point precision with 8-tap Lanczos windowed-sinc interpolation and equal-power stereo panning.
- **Byte-Exact mGBA ARM Golden Tested**: Every frequency lookup, ADSR envelope tick, loop wrap, reverb pass, and mixer output byte is verified with strict `assert_eq!` against real ARM7TDMI execution in `libmgba`.

---

## Quickstart

### Zero-Allocation Mixing (`#![no_std]`)

```rust
use m4a::{AccuracySynth, M4aSampleView, M4aSoundChannel, midi_key_to_freq};

// Wrap a raw 16-byte WaveData header + signed 8-bit PCM bytes without heap allocation
let wavdata: &[u8] = &[
    0x00, 0x00, 0x00, 0x00, // type=0, stat=0, flags=0 (non-looped)
    0x00, 0x0C, 0xD1, 0x00, // base_freq_q10 = 13,700,096 (13,379 Hz)
    0x00, 0x00, 0x00, 0x00, // loop_start = 0
    0x08, 0x00, 0x00, 0x00, // size = 8
    0x00, 0x20, 0x40, 0x20, 0x00, 0xE0, 0xC0, 0xE0, // signed 8-bit PCM
];
let sample = M4aSampleView::from_wavdata_bytes(wavdata).unwrap();

// Compute exact playback frequency for MIDI key 64 (E4) via GBA BIOS SWI 0x1F math
let freq = midi_key_to_freq(sample.header.base_freq_q10, 64, 0);

// Resampled stereo channel (right=200, left=140, instant attack, full sustain)
let mut ch = M4aSoundChannel::new(0x00, 200, 140, 255, 255, 255, 0, freq, 0);
let mut out_r = [0i8; 224];
let mut out_l = [0i8; 224];

AccuracySynth::mix_channel_frame(&sample, &mut ch, 13_379, 15, &mut out_r, &mut out_l);
```

---

## Golden Test Methodology

To guarantee 100% mathematical accuracy without guesswork, `m4a-rs` is verified against real ARM7TDMI execution inside headless `libmgba`:

1. **Tier 1 — GBA BIOS SWI `0x1F` (`MidiKey2Freq`)**: 24/24 bit-exact 32-bit frequency outputs across keys `0..=180`, fine-tunes `0..=255`, and diverse base sample rates.
2. **Tier 2 — 60 Hz ADSR & Pseudo-Echo Envelope**: 40/40 exact ticks (`status_flags`, `envelope_volume`, `envelope_volume_right`, `envelope_volume_left`) verified against *Pokemon FireRed* (`BPRE`) `SoundMainRAM` (`0x081DC0A4`) through Attack → Decay → Sustain → Release → Pseudo-Echo → Off.
3. **Tier 3 — Standard `SoundMainRAM` DirectSound Mixer**: 2,688/2,688 byte-exact DirectSound FIFO samples verified against *Pokemon FireRed* (`BPRE`, `0x081DC0A4`) and *Advance Wars* (`AWRE`, `0x08077634`) across fixed 1:1 pitch, Q23 linear resampling, looped sample wrap, 3-voice stereo mixing, and multi-frame reverb feedback.
4. **Tier 4 — Multi-Variant Corpus Golden Verification**: 4,480/4,480 additional byte-exact ARM7TDMI samples spanning every distinct M4A mixer variant in the GBA library (`7,424 / 7,424` total byte-exact golden assertions):
   - **`M4aMixerVariant::Standard`** (~68% of titles): cross-verified on *Golden Sun* (`AGSE`) and *Mother 3* (`A3UJ`).
   - **`M4aMixerVariant::PokemonGen3`**: *Pokemon FireRed/LeafGreen/Emerald/Ruby/Sapphire* 4-bit quadratic DPCM (`WaveData.wave_type != 0`, 64-sample / 33-byte blocks) and reverse PCM/DPCM playback (`voice_type & 0x30`).
   - **`M4aMixerVariant::Mono`** (~24% of titles): *Mario Kart: Super Circuit* (`AMKE`) & *The Legend of Zelda: The Minish Cap* (`BZME`) mono `>> 8` reverb and `(vol_r + vol_l) >> 9` mono mixing.
   - **`M4aMixerVariant::EarlyV1Stereo` & `EarlyV1Mono`** (2001 launch SDK v1): *EZ-Talk* (`A22J`) and *Mr. Driller 2* (`AD2J`) exact-Hz modulo resampler with zero Q23 pitch drift.
   - **`M4aMixerVariant::MetroidRd1`**: *Metroid Fusion* (`AMTE`) & *Metroid: Zero Mission* (`BMXE`) Nintendo R&D1 32-bit stereo accumulator, Q14 linear interpolation (`lsr #14`), reverse playback, `<< 7` reverb, and 9-bit saturating `>> 7` clamp LUT.

---

## Benchmarks

```sh
cargo bench --bench bench_audio
```

| Benchmark | Throughput |
| --- | ---: |
| `AccuracySynth` fixed 1:1 (224f) | ~3.63 GSamples/s |
| `AccuracySynth` resampled Q23 (224f) | ~664 MSamples/s |
| `AccuracySynth` 3-voice stereo (224f) | ~332 MSamples/s |
| `AccuracySynth` reverb pass (224f) | ~6.71 GSamples/s |
| `StudioSynth` Lanczos-4 48 kHz (800f) | ~187 MSamples/s |

---

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))
