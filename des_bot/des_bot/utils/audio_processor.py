import numpy as np

def bytes_needed_for_resample(n_frames_out, sr_in, sr_out, channels=1, sample_width_bytes=2):
    """
    Calculate number of bytes needed from input buffer to produce n_frames_out at fs_out.

    Args:
        n_frames_out (int): Number of output frames at fs_out
        sr_in (int): Input sample rate (Hz)
        sr_out (int): Output sample rate (Hz)
        channels (int): Number of audio channels (default 1)
        sample_width_bytes (int): Bytes per sample (int16=2)

    Returns:
        int: Number of bytes to pull from input buffer, always even
    """
    n_frames_in = n_frames_out * sr_in / sr_out
    n_frames_in = int(n_frames_out * sr_in / sr_out)
    if n_frames_in % 2 != 0:
        n_frames_in += 1
    return int(n_frames_in * channels * sample_width_bytes)

def resample_linear(audio: np.ndarray, orig_sr: int = 16000, target_sr: int = 24000) -> np.ndarray:
    """
    Resample audio using linear interpolation.

    Parameters:
        audio (np.ndarray): 1D array of float32 audio samples.
        orig_sr (int): Original sample rate (default 16k).
        target_sr (int): Target sample rate (default 24k).

    Returns:
        np.ndarray: Resampled audio as float32.
    """
    # Original and new lengths
    orig_len = len(audio)
    duration = orig_len / orig_sr
    new_len = int(np.round(duration * target_sr))

    # Time indices
    orig_time = np.linspace(0, duration, orig_len, endpoint=False)
    new_time = np.linspace(0, duration, new_len, endpoint=False)

    # Linear interpolation
    resampled = np.interp(new_time, orig_time, audio).astype(np.float32)

    return resampled

          
class OutputAudioProcessorInt16:
    '''Post process audio before sending to callback.'''
    def __init__(self, data: bytes, samplerate=24000):
        self.samplerate = samplerate
        # store as float32 normalized [-1, 1]
        self.samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    def bit_crush(self, bits=8):
        step_count = 2 ** bits
        self.samples = np.round(self.samples * step_count) / step_count
        return self
    
    def pwm(self, freq=200):
        '''shts and giggles only'''
        t = np.arange(len(self.samples)) / self.samplerate
        carrier = np.sin(2 * np.pi * freq * t)
        self.samples = np.where(self.samples > carrier, 1.0, -1.0)
        return self

    def ring_mod(self, freq=80.0):
        t = np.arange(len(self.samples)) / self.samplerate
        modulator = np.sin(2 * np.pi * freq * t)
        self.samples *= modulator
        return self

    def hard_clip(self, threshold=0.3):
        self.samples = np.clip(self.samples, -threshold, threshold)
        self.samples /= threshold  # normalize
        return self

    def sample_rate_reduction(self, factor=4):
        reduced = self.samples[::factor]
        expanded = np.repeat(reduced, factor)[:len(self.samples)]
        self.samples = expanded
        return self

    def comb_filter(self, delay=120, feedback=0.7):
        out = np.copy(self.samples)
        delay = min(delay, len(out)-1)  # safety
        for i in range(delay, len(out)):
            out[i] += feedback * out[i - delay]
        self.samples = np.clip(out, -1.0, 1.0)
        return self
    
    def xor_glitch(self, mask=0x0F):
        pcm16 = (self.samples * 32768).astype(np.int16)
        pcm16 = pcm16 ^ mask
        self.samples = pcm16.astype(np.float32) / 32768.0
        return self
    
    def tremolo(self, freq=8.0, depth=0.5):
        t = np.arange(len(self.samples)) / self.samplerate
        modulator = 1.0 - depth + depth * np.sin(2 * np.pi * freq * t)
        self.samples *= modulator
        return self
    
    def formant_shift(self, shift=1.5):
        # naive: resample then back to same pitch
        indices = np.arange(0, len(self.samples), shift)
        indices = indices[indices < len(self.samples)].astype(int)
        shifted = self.samples[indices]
        self.samples = np.interp(
            np.linspace(0, len(shifted), len(self.samples)),
            np.arange(len(shifted)),
            shifted
        )
        return self
    
    def square_tremolo(self, freq=40.0):
        t = np.arange(len(self.samples)) / self.samplerate
        modulator = np.sign(np.sin(2 * np.pi * freq * t))
        self.samples *= modulator
        return self
    
    def stutter(self, grain_size=200):
        grains = []
        for i in range(0, len(self.samples), grain_size):
            g = self.samples[i:i+grain_size]
            if np.random.rand() > 0.5:  # randomly repeat or skip
                grains.append(np.tile(g, 2))
            else:
                grains.append(g)
        self.samples = np.concatenate(grains)[:len(self.samples)]
        return self
        
    def echo(self, delay_ms=250, decay=0.4):
        delay_samples = int(self.samplerate * delay_ms / 1000)
        out = np.copy(self.samples)
        for i in range(delay_samples, len(out)):
            out[i] += decay * out[i - delay_samples]
        self.samples = np.clip(out, -1.0, 1.0)
        return self
    
    
    def chorus(self, delay_ms=20, depth=0.003, rate=0.25):
        delay_samples = int(self.samplerate * delay_ms / 1000)
        t = np.arange(len(self.samples)) / self.samplerate
        lfo = (np.sin(2 * np.pi * rate * t) * depth) * self.samplerate
        delayed = np.zeros_like(self.samples)
        for i in range(delay_samples, len(self.samples)):
            idx = int(i - delay_samples + lfo[i])
            if 0 <= idx < len(self.samples):
                delayed[i] = self.samples[idx]
        self.samples = (self.samples + delayed) * 0.5
        return self
    
    ## more subtle stuff
    def flatten_dynamics(self, strength=0.7):
        rms = np.sqrt(np.mean(self.samples**2)) + 1e-6
        self.samples = (1 - strength) * self.samples + strength * (self.samples / rms * 0.1)
        return self
    
    def tilt_eq(self, tilt=0.2):
        """
        tilt > 0 brightens (more treble), tilt < 0 darkens (more bass).
        """
        n = np.arange(len(self.samples))
        curve = 1.0 + tilt * (n / len(n) - 0.5)
        self.samples *= curve
        return self
    
    def detune(self, cents=10, mix=0.3):
        factor = 2 ** (cents / 1200.0)
        indices = np.arange(0, len(self.samples), factor)
        indices = indices[indices < len(self.samples)].astype(int)
        shifted = self.samples[indices]
        shifted = np.interp(np.linspace(0, len(shifted), len(self.samples)),
                            np.arange(len(shifted)), shifted)
        self.samples = (1 - mix) * self.samples + mix * shifted
        return self

    def downsample(self, num_frames = None, target_rate=16000):
        if self.samplerate == target_rate:
            return self
        else:
            # print(f"Start resample: {time.perf_counter()}")
            audio = self.samples
            sr_orig = self.samplerate
            orig_len = len(audio)

            # Determine number of frames if not provided
            if num_frames is None:
                duration = orig_len / sr_orig
                num_frames = int(round(duration * target_rate))

            #------------------------------------------------------
            # Interpolation indices
            x_old = np.arange(orig_len)
            x_new = np.linspace(0, orig_len - 1, num_frames)

            # Resample
            resampled = np.interp(x_new, x_old, audio)

            # # Calculate indices in the original audio
            # indices = np.linspace(0, orig_len - 1, num_frames)

            # # Linear interpolation
            # left_idx = np.floor(indices).astype(int)
            # right_idx = np.ceil(indices).astype(int)
            # right_idx = np.clip(right_idx, 0, orig_len - 1)
            # alpha = indices - left_idx
            # resampled = (1 - alpha) * audio[left_idx] + alpha * audio[right_idx]
            #---------------------------------------------------

            # Update state
            self.samples = resampled.astype(np.float32)
            self.samplerate = target_rate
            return self 

    def process(self) -> bytes:
        # Convert back to PCM16
        processed = np.clip(self.samples * 32768.0, -32768, 32767).astype(np.int16)
        return processed.tobytes()