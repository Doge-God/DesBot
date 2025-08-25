from openai import OpenAI
import sounddevice as sd, time
import numpy as np
import requests
import wave
import asyncio

# client = OpenAI(
#     base_url="http://localhost:8880/v1", api_key="not-needed"
# )

# with client.audio.speech.with_streaming_response.create(
#     model="kokoro",
#     voice="af_bella", #single or multiple voicepack combo
#     input=" My house was at the very tip of the egg, only fifty yards from the Sound, and squeezed between two huge places that rented for twelve or fifteen thousand a season.",
#     speed=1.2,
#     response_format="pcm",
#   ) as response:
#     # response.stream_to_file("test_tts.wav")
#     # wav_file = wave.open("test_audio.wav", 'wb')
#     # wav_file.setnchannels(1)
#     # wav_file.setsampwidth(2)
#     # wav_file.setframerate(24000)
#     with sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16) as stream:
#         stream.start()

#         duration = 0.2
#         t = np.arange(int(duration*24000)) / 24000
#         signal = np.sin(2* np.pi * 440 * t)


#         signal_int16 = (signal * 32767).astype(np.int16)

#         stream.write(signal_int16)
#         for chunk in response.iter_bytes(1024):
#             if chunk:
#                 # wav_file.writeframes(chunk)
#                 audio_array = np.frombuffer(chunk, dtype=np.int16)
#                 stream.write(audio_array)


input = "I lived at West Egg, the—well, the less fashionable of the two, though this is a most superficial tag to express the bizarre and not a little sinister contrast between them."

response = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "input": input,
        "voice": "af_bella",
        "response_format": "pcm",
        "normalization_options": {
            "normalize": False,
        }
    },
    stream=True
)
            
t0 = time.perf_counter()
with sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16) as stream:
    t1 = time.perf_counter()
    print(f"Opened stream in: {(t1-t0)*1000:.2f} ms")
    duration = 0.2
    t = np.arange(int(duration*24000)) / 24000
    signal = np.sin(2* np.pi * 440 * t)
    signal_int16 = (signal * 32767).astype(np.int16)
    stream.write(signal_int16)

    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
           
            stream.start()

            audio_array = np.frombuffer(chunk, dtype=np.int16)
            stream.write(audio_array)


