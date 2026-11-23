//! Core Nintendo M4A sequence, track bytecode, voicegroup, and channel types.

use crate::error::{Error, Result};

/// Size of a single 12-byte M4A voicegroup entry.
pub const VOICE_ENTRY_SIZE: usize = 12;

/// Standard 49-entry M4A tick/clock table used for waits (`0x80..=0xB0`)
/// and note lengths (`0xD0..=0xFF`, indices `1..=48`).
pub const CLOCK_TABLE: [u8; 49] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28,
    30, 32, 36, 40, 42, 44, 48, 52, 54, 56, 60, 64, 66, 68, 72, 76, 78, 80, 84, 88, 90, 92, 96,
];

/// Instrument / voice type byte in an M4A voicegroup entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceKind {
    DirectSound,
    DirectSoundFixed,
    DirectSoundAlt,
    Square1,
    Square2,
    ProgrammableWave,
    Noise,
    Keysplit,
    Drumkit,
    Unknown(u8),
}

impl VoiceKind {
    /// Decode a raw M4A voice type byte.
    #[must_use]
    pub const fn from_raw(raw: u8) -> Self {
        match raw {
            0x00 => Self::DirectSound,
            0x08 => Self::DirectSoundFixed,
            0x10 => Self::DirectSoundAlt,
            0x01 | 0x09 => Self::Square1,
            0x02 | 0x0A => Self::Square2,
            0x03 | 0x0B => Self::ProgrammableWave,
            0x04 | 0x0C => Self::Noise,
            0x40 => Self::Keysplit,
            0x80 => Self::Drumkit,
            other => Self::Unknown(other),
        }
    }

    /// Convert back to canonical raw type byte.
    #[must_use]
    pub const fn to_raw(self) -> u8 {
        match self {
            Self::DirectSound => 0x00,
            Self::DirectSoundFixed => 0x08,
            Self::DirectSoundAlt => 0x10,
            Self::Square1 => 0x01,
            Self::Square2 => 0x02,
            Self::ProgrammableWave => 0x03,
            Self::Noise => 0x04,
            Self::Keysplit => 0x40,
            Self::Drumkit => 0x80,
            Self::Unknown(raw) => raw,
        }
    }

    /// Returns `true` if this voice uses 8-bit DirectSound PCM samples.
    #[must_use]
    pub const fn is_directsound(self) -> bool {
        matches!(
            self,
            Self::DirectSound | Self::DirectSoundFixed | Self::DirectSoundAlt
        )
    }
}

/// A 12-byte M4A voicegroup / instrument entry (`ToneData`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct M4aVoice {
    pub kind: VoiceKind,
    pub base_key: u8,
    pub length: u8,
    pub pan_or_sweep: u8,
    pub data_ptr: u32,
    pub attack: u8,
    pub decay: u8,
    pub sustain: u8,
    pub release: u8,
}

impl M4aVoice {
    /// Parse a 12-byte M4A voice entry from `bytes`.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < VOICE_ENTRY_SIZE {
            return Err(Error::UnexpectedEof);
        }
        Ok(Self {
            kind: VoiceKind::from_raw(bytes[0]),
            base_key: bytes[1],
            length: bytes[2],
            pan_or_sweep: bytes[3],
            data_ptr: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            attack: bytes[8],
            decay: bytes[9],
            sustain: bytes[10],
            release: bytes[11],
        })
    }
}

/// Zero-allocation view of an M4A song header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SongHeaderView<'a> {
    pub track_count: u8,
    pub block_count: u8,
    pub priority: u8,
    pub reverb: u8,
    pub voicegroup_ptr: u32,
    track_ptr_bytes: &'a [u8],
}

impl<'a> SongHeaderView<'a> {
    /// Parse an M4A song header from `bytes`.
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(Error::UnexpectedEof);
        }
        let track_count = bytes[0];
        if track_count > 16 {
            return Err(Error::InvalidHeader("track_count exceeds 16"));
        }
        let needed = 8 + (track_count as usize) * 4;
        if bytes.len() < needed {
            return Err(Error::UnexpectedEof);
        }
        Ok(Self {
            track_count,
            block_count: bytes[1],
            priority: bytes[2],
            reverb: bytes[3],
            voicegroup_ptr: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            track_ptr_bytes: &bytes[8..needed],
        })
    }

    /// Return the 32-bit GBA ROM bus pointer for track `index`.
    #[must_use]
    pub fn track_ptr(&self, index: usize) -> Option<u32> {
        if index >= self.track_count as usize {
            return None;
        }
        let off = index * 4;
        Some(u32::from_le_bytes([
            self.track_ptr_bytes[off],
            self.track_ptr_bytes[off + 1],
            self.track_ptr_bytes[off + 2],
            self.track_ptr_bytes[off + 3],
        ]))
    }
}

/// Track sequencer channel state (`MusicPlayerTrack`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct M4aChannel {
    pub active: bool,
    pub voice: u8,
    pub volume: u8,
    pub pan: u8,
    pub bend: i8,
    pub bend_range: u8,
    pub key_shift: i8,
    pub fine_tune: i8,
    pub priority: u8,
    pub tempo_bpm: u16,
    pub mod_depth: u8,
    pub mod_type: u8,
    pub lfo_speed: u8,
    pub lfo_delay: u8,
}

impl Default for M4aChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl M4aChannel {
    /// Initialize a track channel with M4A defaults (volume=100, center pan=64, 150 BPM).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            active: true,
            voice: 0,
            volume: 100,
            pan: 64,
            bend: 0,
            bend_range: 2,
            key_shift: 0,
            fine_tune: 0,
            priority: 0,
            tempo_bpm: 150,
            mod_depth: 0,
            mod_type: 0,
            lfo_speed: 22,
            lfo_delay: 0,
        }
    }
}

/// Decoded M4A track bytecode event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum M4aEvent {
    /// `0x80..=0xB0`: Wait for the decoded number of ticks.
    Wait(u8),
    /// `0xB1`: End of track.
    Fine,
    /// `0xB2`: Unconditional jump to target ROM bus address.
    Goto(u32),
    /// `0xB3`: Call pattern subroutine at target ROM bus address.
    Patt(u32),
    /// `0xB4`: End of pattern subroutine.
    Pend,
    /// `0xB5`: Repeat loop (`count` iterations jumping to `target`).
    Rept { count: u8, target: u32 },
    /// `0xB9`: Memory access command.
    MemAcc { op: u8, address: u8, data: u8 },
    /// `0xBA`: Track priority.
    Prio(u8),
    /// `0xBB`: Song tempo (`BPM = value * 2`).
    Tempo(u8),
    /// `0xBC`: Signed key shift (transpose in semitones).
    Keysh(i8),
    /// `0xBD`: Instrument / voice selection.
    Voice(u8),
    /// `0xBE`: Track volume (`0..=127`).
    Vol(u8),
    /// `0xBF`: Track pan (`0..=127`, center 64).
    Pan(u8),
    /// `0xC0`: Pitch bend (`0..=127`, center 64).
    Bend(u8),
    /// `0xC1`: Pitch bend range (semitones).
    Bendr(u8),
    /// `0xC2`: LFO speed.
    Lfos(u8),
    /// `0xC3`: LFO delay.
    Lfodl(u8),
    /// `0xC4`: Modulation depth.
    Mod(u8),
    /// `0xC5`: Modulation type.
    Modt(u8),
    /// `0xC8`: Fine tune (`0..=127`, center 64).
    Tune(u8),
    /// `0xCD`: Extended sound command (`sub_cmd`, first argument byte).
    Xcmd { sub_cmd: u8, arg0: u8 },
    /// `0xCE`: End of tied note (optional key byte).
    Eot(Option<u8>),
    /// `0xCF`: Tied note on (`key`, `velocity`).
    Tie { key: u8, velocity: u8 },
    /// `0xD0..=0xFF`: Note event with decoded tick `length`, `key`, and `velocity`.
    Note { length: u8, key: u8, velocity: u8 },
}

/// Zero-allocation streaming decoder for M4A track bytecode.
#[derive(Debug, Clone, Copy)]
pub struct TrackDecoder<'a> {
    bytes: &'a [u8],
    pos: usize,
    running_cmd: Option<u8>,
    last_key: u8,
    last_velocity: u8,
}

impl<'a> TrackDecoder<'a> {
    /// Create a new decoder at the start of `bytes`.
    #[must_use]
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
            running_cmd: None,
            last_key: 60,
            last_velocity: 100,
        }
    }

    /// Current byte offset within the track slice.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.pos
    }

    fn read_u8(&mut self) -> Result<u8> {
        if self.pos >= self.bytes.len() {
            return Err(Error::UnexpectedEof);
        }
        let b = self.bytes[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_u32_le(&mut self) -> Result<u32> {
        if self.pos + 4 > self.bytes.len() {
            return Err(Error::UnexpectedEof);
        }
        let v = u32::from_le_bytes([
            self.bytes[self.pos],
            self.bytes[self.pos + 1],
            self.bytes[self.pos + 2],
            self.bytes[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    /// Decode the next [`M4aEvent`], returning `Ok(None)` at end of slice.
    pub fn next_event(&mut self) -> Result<Option<M4aEvent>> {
        if self.pos >= self.bytes.len() {
            return Ok(None);
        }
        let byte = self.read_u8()?;
        if byte < 0x80 {
            let cmd = self
                .running_cmd
                .ok_or(Error::InvalidTrack("running status before any command"))?;
            return self.decode_running(cmd, byte).map(Some);
        }
        let ev = match byte {
            0x80..=0xB0 => M4aEvent::Wait(CLOCK_TABLE[(byte - 0x80) as usize]),
            0xB1 => M4aEvent::Fine,
            0xB2 => M4aEvent::Goto(self.read_u32_le()?),
            0xB3 => M4aEvent::Patt(self.read_u32_le()?),
            0xB4 => M4aEvent::Pend,
            0xB5 => {
                let count = self.read_u8()?;
                let target = self.read_u32_le()?;
                M4aEvent::Rept { count, target }
            }
            0xB9 => {
                let op = self.read_u8()?;
                let address = self.read_u8()?;
                let data = self.read_u8()?;
                M4aEvent::MemAcc { op, address, data }
            }
            0xBA..=0xC5 | 0xC8 => {
                self.running_cmd = Some(byte);
                let param = self.read_u8()?;
                Self::decode_single_param(byte, param)?
            }
            0xCD => {
                self.running_cmd = Some(byte);
                let sub_cmd = self.read_u8()?;
                let arg0 = if self.pos < self.bytes.len() && self.bytes[self.pos] < 0x80 {
                    self.read_u8()?
                } else {
                    0
                };
                if sub_cmd == 0x08 {
                    // Skip remaining 2 args of 3-byte extended command if present
                    for _ in 0..2 {
                        if self.pos < self.bytes.len() && self.bytes[self.pos] < 0x80 {
                            let _ = self.read_u8()?;
                        }
                    }
                }
                M4aEvent::Xcmd { sub_cmd, arg0 }
            }
            0xCE => {
                self.running_cmd = Some(byte);
                let key = if self.pos < self.bytes.len() && self.bytes[self.pos] < 0x80 {
                    let k = self.read_u8()?;
                    self.last_key = k;
                    Some(k)
                } else {
                    None
                };
                M4aEvent::Eot(key)
            }
            0xCF => {
                self.running_cmd = Some(byte);
                if self.pos < self.bytes.len() && self.bytes[self.pos] < 0x80 {
                    self.last_key = self.read_u8()?;
                    if self.pos < self.bytes.len() && self.bytes[self.pos] < 0x80 {
                        self.last_velocity = self.read_u8()?;
                    }
                }
                M4aEvent::Tie {
                    key: self.last_key,
                    velocity: self.last_velocity,
                }
            }
            0xD0..=0xFF => {
                self.running_cmd = Some(byte);
                let mut length = CLOCK_TABLE[(byte - 0xCF) as usize];
                if self.pos < self.bytes.len() && self.bytes[self.pos] < 0x80 {
                    self.last_key = self.read_u8()?;
                    if self.pos < self.bytes.len() && self.bytes[self.pos] < 0x80 {
                        self.last_velocity = self.read_u8()?;
                        if self.pos < self.bytes.len() && self.bytes[self.pos] <= 3 {
                            length = length.saturating_add(self.read_u8()?);
                        }
                    }
                }
                M4aEvent::Note {
                    length,
                    key: self.last_key,
                    velocity: self.last_velocity,
                }
            }
            other => return Err(Error::InvalidOpcode(other)),
        };
        Ok(Some(ev))
    }

    fn decode_single_param(cmd: u8, param: u8) -> Result<M4aEvent> {
        Ok(match cmd {
            0xBA => M4aEvent::Prio(param),
            0xBB => M4aEvent::Tempo(param),
            0xBC => M4aEvent::Keysh(param as i8),
            0xBD => M4aEvent::Voice(param),
            0xBE => M4aEvent::Vol(param),
            0xBF => M4aEvent::Pan(param),
            0xC0 => M4aEvent::Bend(param),
            0xC1 => M4aEvent::Bendr(param),
            0xC2 => M4aEvent::Lfos(param),
            0xC3 => M4aEvent::Lfodl(param),
            0xC4 => M4aEvent::Mod(param),
            0xC5 => M4aEvent::Modt(param),
            0xC8 => M4aEvent::Tune(param),
            other => return Err(Error::InvalidOpcode(other)),
        })
    }

    fn decode_running(&mut self, cmd: u8, first_byte: u8) -> Result<M4aEvent> {
        match cmd {
            0xBA..=0xC5 | 0xC8 => Self::decode_single_param(cmd, first_byte),
            0xCE => {
                self.last_key = first_byte;
                Ok(M4aEvent::Eot(Some(first_byte)))
            }
            0xCF => {
                self.last_key = first_byte;
                if self.pos < self.bytes.len() && self.bytes[self.pos] < 0x80 {
                    self.last_velocity = self.read_u8()?;
                }
                Ok(M4aEvent::Tie {
                    key: self.last_key,
                    velocity: self.last_velocity,
                })
            }
            0xD0..=0xFF => {
                let mut length = CLOCK_TABLE[(cmd - 0xCF) as usize];
                self.last_key = first_byte;
                if self.pos < self.bytes.len() && self.bytes[self.pos] < 0x80 {
                    self.last_velocity = self.read_u8()?;
                    if self.pos < self.bytes.len() && self.bytes[self.pos] <= 3 {
                        length = length.saturating_add(self.read_u8()?);
                    }
                }
                Ok(M4aEvent::Note {
                    length,
                    key: self.last_key,
                    velocity: self.last_velocity,
                })
            }
            other => Err(Error::InvalidOpcode(other)),
        }
    }
}
