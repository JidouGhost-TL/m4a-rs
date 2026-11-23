//! Bit-exact Nintendo M4A ADSR envelope state machine (60 Hz frame ticks).
//!
//! Matches the Thumb ADSR routine inside `SoundMainRAM` (`0x081DC15C..0x081DC21C`
//! in *Pokemon FireRed* `BPRE`).

/// Channel status flag: note start / key-on (`0x80`).
pub const SF_START: u8 = 0x80;
/// Channel status flag: note stop / key-off / release (`0x40`).
pub const SF_STOP: u8 = 0x40;
/// Channel status flag: sample looping enabled (`0x10`).
pub const SF_LOOP: u8 = 0x10;
/// Channel status flag: pseudo-echo active (`0x04`).
pub const SF_IEC: u8 = 0x04;
/// Channel status flag mask for ADSR stage (`0x03`).
pub const SF_ENV_MASK: u8 = 0x03;
/// ADSR stage: Attack (`3`).
pub const SF_ENV_ATTACK: u8 = 0x03;
/// ADSR stage: Decay (`2`).
pub const SF_ENV_DECAY: u8 = 0x02;
/// ADSR stage: Sustain (`1`).
pub const SF_ENV_SUSTAIN: u8 = 0x01;
/// ADSR stage: Off (`0`).
pub const SF_ENV_OFF: u8 = 0x00;

/// High-level envelope stage derived from `status_flags`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvelopeStage {
    Off,
    Attack,
    Decay,
    Sustain,
    Release,
    PseudoEcho,
}

/// Bit-exact M4A ADSR + pseudo-echo envelope state for a single DirectSound channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct M4aEnvelope {
    pub status_flags: u8,
    pub attack: u8,
    pub decay: u8,
    pub sustain: u8,
    pub release: u8,
    pub envelope_volume: u8,
    pub envelope_volume_right: u8,
    pub envelope_volume_left: u8,
    pub pseudo_echo_volume: u8,
    pub pseudo_echo_length: u8,
}

impl M4aEnvelope {
    /// Create a new envelope configured for key-on (`SF_START`).
    #[must_use]
    pub const fn new(
        attack: u8,
        decay: u8,
        sustain: u8,
        release: u8,
        pseudo_echo_volume: u8,
        pseudo_echo_length: u8,
    ) -> Self {
        Self {
            status_flags: SF_START,
            attack,
            decay,
            sustain,
            release,
            envelope_volume: 0,
            envelope_volume_right: 0,
            envelope_volume_left: 0,
            pseudo_echo_volume,
            pseudo_echo_length,
        }
    }

    /// Trigger key-off (sets `SF_STOP`, entering Release on the next tick).
    #[inline]
    pub fn key_off(&mut self) {
        self.status_flags |= SF_STOP;
    }

    /// Returns `true` if the channel is active (`status_flags & 0xC7 != 0`).
    #[inline]
    #[must_use]
    pub const fn is_active(&self) -> bool {
        (self.status_flags & 0xC7) != 0
    }

    /// Decode the current [`EnvelopeStage`] from `status_flags`.
    #[must_use]
    pub const fn stage(&self) -> EnvelopeStage {
        if !self.is_active() {
            EnvelopeStage::Off
        } else if (self.status_flags & SF_IEC) != 0 {
            EnvelopeStage::PseudoEcho
        } else if (self.status_flags & SF_STOP) != 0 {
            EnvelopeStage::Release
        } else {
            match self.status_flags & SF_ENV_MASK {
                SF_ENV_ATTACK => EnvelopeStage::Attack,
                SF_ENV_DECAY => EnvelopeStage::Decay,
                SF_ENV_SUSTAIN => EnvelopeStage::Sustain,
                _ => EnvelopeStage::Off,
            }
        }
    }

    /// Advance the envelope by one 60 Hz frame tick, matching Nintendo's `SoundMainRAM`.
    ///
    /// Returns `Some((envelope_volume_right, envelope_volume_left))` if the channel
    /// remains active for mixing on this frame, or `None` if the channel stopped.
    pub fn tick(
        &mut self,
        master_volume: u8,
        right_volume: u8,
        left_volume: u8,
        sample_looped: bool,
    ) -> Option<(u8, u8)> {
        let mut status = self.status_flags;
        if (status & 0xC7) == 0 {
            return None;
        }

        let mut env: u32;
        if (status & SF_START) != 0 {
            if (status & SF_STOP) != 0 {
                self.status_flags = 0;
                return None;
            }
            status = SF_ENV_ATTACK;
            if sample_looped {
                status |= SF_LOOP;
            }
            self.status_flags = status;
            // Start with env = 0 and immediately run the first Attack step.
            env = self.attack as u32;
            if env >= 0xFF {
                env = 0xFF;
                status -= 1; // Attack (3) -> Decay (2)
                self.status_flags = status;
            }
        } else {
            env = self.envelope_volume as u32;
            if (status & SF_IEC) != 0 {
                let next_len = self.pseudo_echo_length.wrapping_sub(1);
                self.pseudo_echo_length = next_len;
                if next_len == 0 {
                    self.status_flags = 0;
                    return None;
                }
            } else if (status & SF_STOP) != 0 {
                env = (env * self.release as u32) >> 8;
                if env <= self.pseudo_echo_volume as u32 {
                    env = self.pseudo_echo_volume as u32;
                    if env == 0 {
                        self.status_flags = 0;
                        return None;
                    }
                    status |= SF_IEC;
                    self.status_flags = status;
                }
            } else {
                match status & SF_ENV_MASK {
                    SF_ENV_DECAY => {
                        env = (env * self.decay as u32) >> 8;
                        if env <= self.sustain as u32 {
                            env = self.sustain as u32;
                            if env == 0 {
                                // Sustain == 0 enters pseudo-echo check (0x081DC1C4)
                                env = self.pseudo_echo_volume as u32;
                                if env == 0 {
                                    self.status_flags = 0;
                                    return None;
                                }
                                status |= SF_IEC;
                                self.status_flags = status;
                            } else {
                                status -= 1; // Decay (2) -> Sustain (1)
                                self.status_flags = status;
                            }
                        }
                    }
                    SF_ENV_ATTACK => {
                        env += self.attack as u32;
                        if env >= 0xFF {
                            env = 0xFF;
                            status -= 1; // Attack (3) -> Decay (2)
                            self.status_flags = status;
                        }
                    }
                    _ => {}
                }
            }
        }

        self.envelope_volume = env as u8;
        let scaled = (env * (master_volume as u32 + 1)) >> 4;
        let vol_r = ((scaled * right_volume as u32) >> 8) as u8;
        let vol_l = ((scaled * left_volume as u32) >> 8) as u8;
        self.envelope_volume_right = vol_r;
        self.envelope_volume_left = vol_l;
        Some((vol_r, vol_l))
    }
}
