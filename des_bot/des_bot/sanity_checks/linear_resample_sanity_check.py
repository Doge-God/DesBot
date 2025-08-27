import numpy as np
import sounddevice as sd, time

def resample_16k_to_24k(audio: np.ndarray) -> np.ndarray:
    """
    Resample from 16 kHz to 24 kHz using linear interpolation (3/2 ratio).
    NumPy-only, efficient for this specific case.
    """
    assert audio.ndim == 1, "Only 1D audio arrays supported"
    
    N = len(audio)

    # Step 1: Upsample by 3 (insert 2 interpolated samples)
    up_len = N * 3
    upsampled = np.empty(up_len, dtype=np.float32)

    # Original samples go to positions 0,3,6,...
    upsampled[0::3] = audio

    # Linear interpolation for in-between samples
    upsampled[1::3] = (2/3) * audio[:-1] + (1/3) * audio[1:]
    upsampled[2::3] = (1/3) * audio[:-1] + (2/3) * audio[1:]

    # Handle last two positions (repeat last sample to keep length consistent)
    upsampled[-2:] = audio[-1]

    # Step 2: Downsample by 2 (take every 2nd sample)
    resampled = upsampled[::2]

    return resampled

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

# Example usage
if __name__ == "__main__":
    # Fake sinewave at 440 Hz
    t = np.linspace(0, 1, 16000, endpoint=False)
    sine_16k = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    t0 = time.perf_counter()
    sine_24k = resample_linear(sine_16k, 16000, 24000)
    t1 = time.perf_counter()

    print(f"Original length: {len(sine_16k)} → Resampled length: {len(sine_24k)} in {t1-t0} s")