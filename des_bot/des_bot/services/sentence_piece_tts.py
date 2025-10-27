import aiohttp
import asyncio
import sounddevice as time

class SentencePiecePoisonPill:
    pass

class SentencePieceTts:
    def __init__(self, text: str, session: aiohttp.ClientSession, loop:asyncio.AbstractEventLoop, init_wait=0.0,base_url="http://192.168.137.1:8880"):
        self.text = text
        self.session = session
        self.audio_buffer = b""
        self.is_complete_audio_fetched = False
        self.is_all_audio_consumed = asyncio.Event()
        self.loop = loop
        self.init_wait = init_wait
        self.base_url = base_url

        self.fetch_task = asyncio.run_coroutine_threadsafe(self._fetch(), self.loop)

    async def _fetch(self):
        try:
            payload = {
                "input": self.text,
                "voice": "bf_v0isabella",
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
            async with self.session.post(self.base_url+"/v1/audio/speech",json=payload) as resp:
                async for chunk in resp.content.iter_chunked(1024): #iter_chunks(1024):
                    # np_audio = np.frombuffer(chunk, dtype=np.int16)#.astype(np.float32)
                    # print(np_audio.shape)
                    # self.generated_audio.put_nowait(np_audio)
                    if not chunk:
                        continue
                    self.audio_buffer += chunk

            # trim off 300ms of end silence.
            # self.audio_buffer = self.audio_buffer[:-4800] # 24000 frames/s * 0.1s * 2(byte per frame)
            self.is_complete_audio_fetched = True
            print(f"^ Done fetch: [{self.text[:20]}..]. Got [{len(self.audio_buffer)}] bytes")
        except asyncio.CancelledError:
            self.is_complete_audio_fetched = True
            print(f"^ Cancelled fetch: [{self.text[:20]}..].")
        finally:
            pass

    def force_get_bytes(self, samples_required:int):
        '''Get x samples, indicate if all audio generated is emptied with this read.'''
        if len(self.audio_buffer) < samples_required:
            # Not enough samples, return what we have and pad with zeros
            result = self.audio_buffer
            padding = b'\x00' * (samples_required - len(self.audio_buffer))
            self.audio_buffer = b''

            # print(f"Outputting bytes:  [{len(result+padding)}] -------------------- PADDED")

            if self.is_complete_audio_fetched:
      
                self.loop.call_soon_threadsafe(
                    self.is_all_audio_consumed.set
                )
    
                # print(f"# CONSUMED ALL AUDIO: [{self.text[:20]}..]")

            return result + padding
        else:
            # Enough samples, return requested amount and keep the rest
            result = self.audio_buffer[:samples_required]
            self.audio_buffer = self.audio_buffer[samples_required:]
        
            # print(f"Outputing bytes: [{len(result)}]")
            return result
