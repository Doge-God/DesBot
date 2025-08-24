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

        self.current_time_sec = -delay_sec # detected current time lag behind real time
        '''Time for confirmed text stream'''

        self.pause_predictor = ExponentialMovingAverage(
            attack_time=0.01, release_time=0.01, initial_value=1.0
        )

        # status flags
        self.shutdown_complete = asyncio.Event() #false default

    async def start_up(self, url, api_key='public_key'):
        '''Remember api/asr-streaming part'''
        headers = {"kyutai-api-key": api_key}
        self.websocket = await websockets.connect(url, additional_headers=headers)

        try: 
            message_bytes = await self.websocket.recv()
            message_dict = msgpack.unpackb(message_bytes)
            message = SttMessageAdapter.validate_python(message_dict)

            if isinstance(message, SttReadyMessage):
                self.node_logger.info("STT session ready.")
                return
            else:
                raise RuntimeError("Unexpected return from STT session recived on startup.")
        
        except Exception as e:
            self.node_logger.error(f"Failed starting up STT session.")
            await self.websocket.close()
            self.websocket = None
            raise e
        
    
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
        # self.node_logger.info("sending audio")
        
        formatted = {"type": "Audio", "pcm": [float(x) for x in audio_data]}

        to_send = msgpack.packb(
            formatted, use_single_float=True, #use_bin_type = True
        )
        if self.websocket:
            await self.websocket.send(to_send)
        else:
            self.node_logger.warning("Trying to send audio without an active websocket")
        
    async def __aiter__(self) -> AsyncIterator[SttWordMessage]:
        self.node_logger.info(f"stt listen to server itr called")
        if not self.websocket:
            raise RuntimeError("WebSocket connection not established. Call start_up first.")
        
        steps_to_wait = 12 # skip ~1s for pause prediction to warm up

        try: 
            async for message_bytes in self.websocket:
                # self.node_logger.debug(f"server msg")
                unpacked = msgpack.unpackb(message_bytes, raw=False) #, raw=False
                stt_message = SttMessageAdapter.validate_python(unpacked)

                match stt_message:
                    case SttWordMessage():
                        self.node_logger.info(f"word: {stt_message.text}")
                        yield stt_message
                    case SttStepMessage():
                        # self.node_logger.info(str(stt_message.prs[2]))
                        self.current_time_sec += self.frame_time_sec
                        if steps_to_wait > 0:
                            steps_to_wait -= 1
                        else:
                            self.pause_predictor.update(dt=self.frame_time_sec, new_value=stt_message.prs[self.delay_sensitivity_idx])
                        
                    case SttReadyMessage():
                        continue
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

    
        
