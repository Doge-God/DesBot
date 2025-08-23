import threading
import librosa
import msgpack
import websockets
import numpy as np
import sounddevice as sd
import asyncio
from typing import TYPE_CHECKING, AsyncIterator
from ..utils.stt_types import SttMessage, SttMessageAdapter, SttStepMessage, SttReadyMessage, SttWordMessage, SttEndWordMessage, SttErrorMessage, SttMarkerMessage
from ..utils.exponential_moving_avg import ExponentialMovingAverage

if TYPE_CHECKING:
    from rclpy.node import Clock
    from rclpy.impl.rcutils_logger import RcutilsLogger

class SttSessionKyutai:
    '''Maintain websocket connection.
    Iterate over for messages recieved from the server.'''
    def __init__(self, node_clock:"Clock", node_logger:"RcutilsLogger", delay_sensitivity_idx=2, delay_sec=0.5, frame_time_sec=1920/24000):
        self.websocket = None
        self.node_clock = node_clock
        self.node_logger = node_logger

        # model related settings
        self.delay_sec = delay_sec  
        self.frame_time_sec = frame_time_sec # 80ms for 24kHz sample rate
        self.delay_sensitivity_idx = delay_sensitivity_idx


        self.first_audio_sent_timestamp = None # NODE TIME IN NANOSEC

        self.current_time_sec = -delay_sec # detected current time lag behind real time
        '''Time for confirmed text stream'''

        self.pause_predictor = ExponentialMovingAverage(
            attack_time=0.01, release_time=0.01, initial_value=1.0
        )

        # status flags
        self.shutdown_complete = asyncio.Event() #false default
        self.is_ready = asyncio.Event()

    async def start_up(self, url, api_key='public_key'):
        '''Remember api/asr-streaming part'''
        headers = {"kyutai-api-key": api_key}
        self.websocket = await websockets.connect(url, additional_headers=headers)
    
    async def shutdown(self):
        if self.shutdown_complete.is_set():
            return
        if not self.websocket:
            raise RuntimeError("Attempting to shutdown without an active websocket connection.")
        
        await self.websocket.close()
        await self.shutdown_complete.wait()

    async def send_audio(self, audio_data):
        '''Process frame from audio stream. Assume 1920 chuck
        size (80ms), 24kHz sample rate, mono channel.'''
        formatted = {"type": "Audio", "pcm": [float(x) for x in audio_data]}
        
        if self.first_audio_sent_timestamp is None:
            self.first_audio_sent_timestamp = self.node_clock.now().nanoseconds

        to_send = msgpack.packb(
            formatted, use_single_float=True
        )
        if self.websocket:
            await self.websocket.send(to_send)
        else:
            self.node_logger.warning("Trying to send audio without an active websocket")
        
    async def __aiter__(self) -> AsyncIterator[SttWordMessage]:
        if not self.websocket:
            raise RuntimeError("WebSocket connection not established. Call start_up first.")
        
        steps_to_wait = 12 # skip ~1s for pause prediction to warm up

        try: 
            async for message in self.websocket:
                unpacked = msgpack.unpackb(message, raw=False)
                stt_message = SttMessageAdapter.validate_python(unpacked)

                match stt_message:
                    case SttWordMessage():
                        self.node_logger.info(f"word: {stt_message.text}")
                        yield stt_message
                    case SttStepMessage():
                        self.current_time_sec += self.frame_time_sec
                        if steps_to_wait > 0:
                            steps_to_wait -= 1
                        else:
                            self.pause_predictor.update(dt=self.frame_time_sec, new_value=stt_message.prs[self.delay_sensitivity_idx])
                        
                    case SttReadyMessage():
                        self.is_ready.set()
                        self.node_logger.info("STT is ready.")
                    case SttEndWordMessage():
                        continue
                    case SttMarkerMessage():
                        continue
                    case SttErrorMessage():
                        self.node_logger.error(f"STT Error: {stt_message.message}")
                    case _:
                        self.node_logger.warning(f"Received unexpected message type: {stt_message.type}")
                
        
        except websockets.ConnectionClosed:
            self.node_logger.info("WebSocket connection closed.")
        
        finally:
            self.shutdown_complete.set()

    
        
