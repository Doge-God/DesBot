
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
        int: Number of bytes to pull from input buffer
    """
    n_frames_in = int(n_frames_out * sr_in / sr_out)
    if n_frames_in % 2 != 0:
        n_frames_in += 1
    
    return int(n_frames_in * channels * sample_width_bytes)
        
class SoundProcessor:
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
        '''Decimate down sampler'''
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

            # Calculate indices in the original audio
            indices = np.linspace(0, orig_len - 1, num_frames)

            # Linear interpolation
            left_idx = np.floor(indices).astype(int)
            right_idx = np.ceil(indices).astype(int)
            right_idx = np.clip(right_idx, 0, orig_len - 1)
            alpha = indices - left_idx
            resampled = (1 - alpha) * audio[left_idx] + alpha * audio[right_idx]

            # Update state
            self.samples = resampled.astype(np.float32)
            self.samplerate = target_rate
            # print(f"Stop resample: {time.perf_counter()}")
            return self 
    

    def process(self) -> bytes:
        # Convert back to PCM16
        processed = np.clip(self.samples * 32768.0, -32768, 32767).astype(np.int16)
        return processed.tobytes()
        
class SentencePieceTts: #192.168.137.1
    def __init__(self, text: str, session: aiohttp.ClientSession, loop:asyncio.AbstractEventLoop, init_wait=0.0,base_url="http://192.168.137.1:8880"):
        self.text = text
        self.session = session
        self.audio_buffer = b""
        self.is_complete_audio_fetched = False
        self.is_all_audio_consumed = asyncio.Event()
        self.loop = loop
        self.init_wait = init_wait
        self.base_url = base_url

        self.fetch_coroutine = self._fetch()
        asyncio.create_task(self.fetch_coroutine)

    async def _fetch(self):
        payload = {
            "input": self.text,
            "voice": "bf_v0isabella", #bf_v0isabella
            "response_format": "pcm",
            "speed": 1.1,
            "stream": True,
            "language_code":"a",
            "normalization_options": {
                "normalize": False,
            }
        }
        if self.init_wait:
            await asyncio.sleep(self.init_wait)

        print(f"V Start fetch: [{self.text}]")
        #http://192.168.137.1:8880
        async with self.session.post(self.base_url+"/v1/audio/speech",json=payload) as resp:
            async for chunk in resp.content.iter_chunked(1024): #iter_chunks(1024):
                # np_audio = np.frombuffer(chunk, dtype=np.int16)#.astype(np.float32)
                # print(np_audio.shape)
                # self.generated_audio.put_nowait(np_audio)
                if not chunk:
                    continue
                self.audio_buffer += chunk

        # trim off 300ms of end silence.
        self.audio_buffer = self.audio_buffer[:-14400] # 24000 frames/s * 0.3s * 2(byte per frame)
        self.is_complete_audio_fetched = True
        print(f"^ Done fetch: [{self.text}]. Got [{len(self.audio_buffer)}] bytes")

    def force_get_bytes(self, samples_required:int):
        '''Get x samples, indicate if all audio generated is emptied with this read.'''
        if len(self.audio_buffer) < samples_required:
            # Not enough samples, return what we have and pad with zeros
            result = self.audio_buffer
            padding = b'\x00' * (samples_required - len(self.audio_buffer))
            self.audio_buffer = b''

            if len(result) > 0:
                print(f"Give partially filled bytes: {time.perf_counter()}")
            # print(f"Outputting bytes:  [{len(result+padding)}] -------------------- PADDED")

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
        
            print(f"SentencePieceTts yield audio at [{time.perf_counter()}]: [{len(result)}] bytes")
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

        
        print(f"Started requesting: {time.perf_counter()}")
        self.sentence_piece_tts_queue.put_nowait(SentencePieceTts("This is the first sentence.", self.tts_session, asyncio.get_running_loop()))
        self.sentence_piece_tts_queue.put_nowait(SentencePieceTts("This is the second sentence.", self.tts_session, asyncio.get_running_loop(),init_wait=1))
        self.sentence_piece_tts_queue.put_nowait(SentencePieceTts("This is the third sentence.", self.tts_session, asyncio.get_running_loop(),init_wait=2))
        self.sentence_piece_tts_queue.put_nowait(SentencePieceTts("This is the fourth sentence.", self.tts_session, asyncio.get_running_loop(),init_wait=3))
        self.sentence_piece_tts_queue.put_nowait(SentencePieceTts("This is the fifth sentence.", self.tts_session, asyncio.get_running_loop(),init_wait=4))
        self.sentence_piece_tts_queue.put_nowait(SentencePieceTts("It was a matter of chance that I should have rented a house in one of the strangest communities in North America. It was on that slender riotous island which extends itself due east of New York and where there are, among other natural curiosities, two unusual formations of land. Twenty miles from the city a pair of enormous eggs, identical in contour and separated only by a courtesy bay, jut out into the most domesticated body of salt water in the Western Hemisphere, the great wet barnyard of Long Island Sound. They are not perfect ovals—like the egg in the Columbus story they are both crushed flat at the contact end—but their physical resemblance must be a source of perpetual confusion to the gulls that fly overhead. To the wingless a more arresting phenomenon is their dissimilarity in every particular except shape and size.", self.tts_session, asyncio.get_running_loop(),init_wait=5))
        



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
        def audio_callback(outdata, frames, _ , status):
            '''ASSUME int16 audio: each frame = 16bit (2bytes)'''
            if status:
                print("Audio callback status:", status)
            if not self.current_sentence_piece_tts or self.current_sentence_piece_tts.is_all_audio_consumed.is_set():
                outdata[:] = b'\x00' * frames*2 #16bit frames: 
                return

            buffered_data = self.current_sentence_piece_tts.force_get_bytes(bytes_needed_for_resample(frames, 24000,16000))
            # print(f"Start post process: {time.perf_counter()}")
            processed = (
                SoundProcessor(buffered_data)
                # .ring_mod(30)
                # .comb_filter(delay=75, feedback=0.4)
                # .square_tremolo(10)
                .downsample(num_frames=frames, target_rate=16000)
                .process()
            )
            # print(f"End post process: {time.perf_counter()}")
            outdata[:] = processed

        self.speaker_output_stream = sd.RawOutputStream(
            samplerate=16000,  
            channels=1,
            dtype="int16",
            callback=audio_callback,
            device=0
        )
        if start_now:
            self.speaker_output_stream.start()





async def main():
    Mock()
    await asyncio.sleep(20)

asyncio.run(main())

                    

