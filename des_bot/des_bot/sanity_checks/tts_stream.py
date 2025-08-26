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

from typing import NamedTuple, Optional
from numpy.typing import NDArray

class SentencePiece(NamedTuple):
    text:str
    audio:NDArray
    is_audio_complete: bool


import aiohttp
            

        
class SentencePieceTts:
    def __init__(self, text: str, session: aiohttp.ClientSession, loop:asyncio.AbstractEventLoop):
        self.text = text
        self.session = session
        self.audio_buffer = b""
        self.is_complete_audio_fetched = False
        self.is_all_audio_consumed = asyncio.Event()
        self.loop = loop

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

        async with self.session.post("http://192.168.137.1:8880/v1/audio/speech",json=payload) as resp:
            async for chunk in resp.content.iter_chunked(1024): #iter_chunks(1024):
                # np_audio = np.frombuffer(chunk, dtype=np.int16)#.astype(np.float32)
                # print(np_audio.shape)
                # self.generated_audio.put_nowait(np_audio)
                if not chunk:
                    continue
                self.audio_buffer += chunk
        
        self.is_complete_audio_fetched = True
        print(f"^ Done fetch: [{self.text}]. Got [{len(self.audio_buffer)}] bytes")

    def force_get_samples(self, samples_required:int):
        '''Get x samples, indicate if all audio generated is emptied with this read.'''
        if len(self.audio_buffer) < samples_required:
            # Not enough samples, return what we have and pad with zeros
            result = self.audio_buffer
            padding = b'\x00' * (samples_required - len(self.audio_buffer))
            self.audio_buffer = b''

            print(f"Outputting bytes:  [{len(result+padding)}] -------------------- PADDED")

            if self.is_complete_audio_fetched:
                self.loop.call_soon_threadsafe(
                    self.is_all_audio_consumed.set()
                )
                
                print(f"# CONSUMED ALL AUDIO: [{self.text}] <<<<<<<<<<<<<<<<<<<<<<<<<<")

            return result + padding
        else:
            # Enough samples, return requested amount and keep the rest
            result = self.audio_buffer[:samples_required]
            self.audio_buffer = self.audio_buffer[samples_required:]
            print(f"Outputing bytes: [{len(result)}]")
            return result
            

# ---- Global playback system ----

class Mock():
    def __init__(self):
        self.sentence_piece_tts_queue:asyncio.Queue[SentencePieceTts] = asyncio.Queue()
        self.speaker_output_stream = None
        self.current_sentence_piece_tts:Optional[SentencePieceTts] = None
        self.advance_sentence_piece_tts_task = self.advance_sentence_piece_tts()

        self.tts_session = aiohttp.ClientSession()

        self.is_state_correct = True

        self.spoken = []

        asyncio.create_task(self.advance_sentence_piece_tts_task)
        self.make_audio_stream()

        self.sentence_piece_tts_queue.put_nowait(SentencePieceTts("This is the first sentence.", self.tts_session, asyncio.get_running_loop()))
        self.sentence_piece_tts_queue.put_nowait(SentencePieceTts("This is the second sentence.", self.tts_session, asyncio.get_running_loop()))
        self.sentence_piece_tts_queue.put_nowait(SentencePieceTts("This is the third sentence.", self.tts_session, asyncio.get_running_loop()))


    async def advance_sentence_piece_tts(self):
        while self.is_state_correct:
            if (not self.current_sentence_piece_tts or 
            self.current_sentence_piece_tts.is_all_audio_consumed.is_set() ):
                self.current_sentence_piece_tts = await self.sentence_piece_tts_queue.get()
                self.spoken.append(self.current_sentence_piece_tts.text)
                print(f"NEW sentence tts obj: [{self.current_sentence_piece_tts.text}]")
                await self.current_sentence_piece_tts.is_all_audio_consumed.wait()
                print("Recieved. CONTINUE----------------------------------------------------------------")

    def make_audio_stream(self, start_now=True):
        def audio_callback(outdata, frames, time, status):
            '''ASSUME int16 audio: each frame = 16bit (2bytes)'''
            if status:
                print("Audio callback status:", status)
            if not self.current_sentence_piece_tts or self.current_sentence_piece_tts.is_all_audio_consumed.is_set():
                outdata[:] = b'\x00' * frames*2 #16bit frames: 
                return

            buffered_data = self.current_sentence_piece_tts.force_get_samples(frames*2)
          
            outdata[:] = buffered_data

        self.speaker_output_stream = sd.RawOutputStream(
            samplerate=24000,  # adjust to your API’s sample rate
            channels=1,
            dtype="int16",
            callback=audio_callback
        )
        if start_now:
            self.speaker_output_stream.start()





async def main():
    Mock()
    await asyncio.sleep(10)

asyncio.run(main())

                    

