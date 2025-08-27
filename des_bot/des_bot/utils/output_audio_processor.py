import numpy as np
          
class AudioProcessorInt16:
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


    def process(self) -> bytes:
        # Convert back to PCM16
        processed = np.clip(self.samples * 32768.0, -32768, 32767).astype(np.int16)
        return processed.tobytes()