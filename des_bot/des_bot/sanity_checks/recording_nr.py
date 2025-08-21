import sounddevice as sd
import numpy as np
import noisereduce as nr
import soundfile as sf

# Parameters
device_id = 1
duration = 3  # seconds
chunk_size = 1920
samplerate = 16000  # adjust for your device
output_file = "cleaned_stream.wav"

# Reference noise file (mono, 32float WAV)
noise_file = "/home/fz/Documents/DesBot/des_bot/des_bot/sanity_checks/sample_noise.wav"

# Load noise reference
import soundfile as sf
noise_data, noise_sr = sf.read(noise_file, dtype='float32')
if noise_sr != samplerate:
    raise ValueError(f"Noise file sample rate ({noise_sr}) does not match target ({samplerate})")
if noise_data.ndim > 1:
    noise_data = noise_data[:, 0]

# Prepare output file
out_file = sf.SoundFile(output_file, mode='w', samplerate=samplerate,
                        channels=1, subtype='FLOAT')

# Callback for streaming processing
def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    # Flatten chunk
    chunk = indata[:, 0]
    # Reduce noise using reference noise profile
    cleaned_chunk = nr.reduce_noise(y=chunk, y_noise=noise_data, sr=samplerate, stationary=True)
    # Write processed chunk to file
    out_file.write(cleaned_chunk)

print("Recording and reducing noise in real-time...")
with sd.InputStream(
                    channels=1,
                    samplerate=samplerate,
                    blocksize=chunk_size,
                    dtype='float32',
                    callback=callback):
    sd.sleep(int(duration * 1000))

out_file.close()
print(f"Cleaned audio saved as {output_file}")
