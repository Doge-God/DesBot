import sounddevice as sd, time
import numpy as np
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

# response = requests.post(
#     "http://192.168.137.1:8880/v1/audio/speech",
#     json={
#         "input": input,
#         "voice": "bf_v0emma",
#         "response_format": "pcm",
#         "speed": 1.1
#         # "normalization_options": {
#         #     "normalize": False,
#         # },

#     },
#     stream=True
# )
            
# t0 = time.perf_counter()
# with sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16) as stream:
#     t1 = time.perf_counter()
#     print(f"Opened stream in: {(t1-t0)*1000:.2f} ms")
#     # duration = 0.2
#     # t = np.arange(int(duration*24000)) / 24000
#     # signal = np.sin(2* np.pi * 440 * t)
#     # signal_int16 = (signal * 32767).astype(np.int16)
#     # stream.write(signal_int16)

#     for chunk in response.iter_content(chunk_size=1024):
#         if chunk:
           
#             stream.start()

#             audio_array = np.frombuffer(chunk, dtype=np.int16)
#             stream.write(audio_array)

from typing import NamedTuple
from numpy.typing import NDArray

class SentencePiece(NamedTuple):
    text:str
    audio:NDArray
    is_audio_complete: bool


import aiohttp

class TtsStreamSanity:
    def __init__(self):
        self.tts_session = aiohttp.ClientSession()
        self.speaker_audio_queue = asyncio.Queue()

        def speaker_audio_callback(outdata, frames, time, status):
            """Sounddevice callback function."""
            if status:
                print(status)
            try:
                # Get data from the asyncio queue (blocking in this thread, but put_nowait is used)
                data = self.speaker_audio_queue.get_nowait()
                if data is None: # End of stream
                    raise sd.CallbackStop
                outdata[:] = data.reshape(-1, 1) # Reshape for single channel
            except asyncio.QueueEmpty:
                outdata.fill(0) # Fill with zeros if no data is available
            except sd.CallbackStop:
                raise
        self.speaker_stream = sd.OutputStream(samplerate=24000, channels=1, callback=speaker_audio_callback)
        self.speaker_audio_send_task = asyncio.create_task(self.send_audio_from_sentence_piece_tts_coroutine())


        self.test_queue:asyncio.Queue[SentencePieceTts] = asyncio.Queue()
        self.test_queue.put_nowait(SentencePieceTts("I ate an apple today." ,self.tts_session))
        self.test_queue.put_nowait(SentencePieceTts("I ate an apple today." ,self.tts_session))

    async def send_audio_from_sentence_piece_tts_coroutine(self):
        while True:
            sentence_piece_tts = await self.test_queue.get()
            while not sentence_piece_tts.is_complete_audio:
                audio_data = await sentence_piece_tts.generated_audio.get()
                self.speaker_audio_queue.put_nowait(audio_data)

        
class SentencePieceTts:
    def __init__(self, text: str, session: aiohttp.ClientSession):
        self.text = text
        self.session = session
        self.generated_audio:asyncio.Queue[np.typing.NDArray] = asyncio.Queue()
        self.is_complete_audio = False
        self.fetch_coroutine = self._fetch()
        asyncio.create_task(self.fetch_coroutine)

    async def _fetch(self):
        print(f"V Start fetch: [{self.text}]")
        payload = {
            "input": self.text,
            "voice": "af_bella",
            "response_format": "pcm",
            "speed": 1,
            "stream": True,
        }
        leftover = b""
        async with self.session.post("http://localhost:8880/v1/audio/speech",json=payload) as resp:
            async for chunk in resp.content.iter_chunked(1024): #iter_chunks(1024):
                if not chunk:
                    continue
                data = leftover + chunk
                # Ensure even number of bytes
                if len(data) % 2 != 0:
                    leftover, data = data[-1:], data[:-1]  # keep last byte as leftover
                else:
                    leftover = b""

                if data:
                    np_audio = np.frombuffer(data, dtype=np.int16)
                    self.generated_audio.put_nowait(np_audio)
        self.is_complete_audio = True
        print(f"^ Done fetch: [{self.text}]")

    async def __aiter__(self):
        if not self.is_complete_audio:
            yield await self.generated_audio.get()
            

# ---- Global playback system ----
global_audio_queue = asyncio.Queue() 

def audio_callback(outdata, frames, time, status):
    if status:
        print("Audio callback status:", status)
    try:
        data = global_audio_queue.get_nowait()
    except Exception:
        outdata[:] = np.zeros((frames, 1), dtype=np.float32)
    else:
        if len(data) < frames:
            outdata[:len(data), 0] = data
            outdata[len(data):, 0] = 0
        else:
            outdata[:, 0] = data[:frames]
            if len(data) > frames:
                # Push remainder back
                global_audio_queue.put_nowait(data[frames:])


async def feed_tts_to_queue(tts: SentencePieceTts):
    async for chunk in tts:
        global_audio_queue.put_nowait(chunk)


async def main():
    async with aiohttp.ClientSession() as session:
        # Create two TTS tasks
        tts1 = SentencePieceTts("Hello, this is the first sentence.", session)
        tts2 = SentencePieceTts("And this is the second sentence.", session)

        # Start audio stream
        stream = sd.OutputStream(
            samplerate=24000,  # adjust to your API’s sample rate
            channels=1,
            dtype="int16",
            callback=audio_callback,
            blocksize=1024,
        )
        stream.start()

        # Feed both TTS into the global queue sequentially
        await feed_tts_to_queue(tts1)
        await feed_tts_to_queue(tts2)

        # wait a bit for last audio to finish playing
        await asyncio.sleep(2)
        stream.stop()
        stream.close()

if __name__ == "__main__":
    asyncio.run(main())

                    

