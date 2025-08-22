import asyncio
import librosa
import rclpy
from rclpy.node import Node
from enum import Enum
from .baml_client import b, partial_types, types
from .baml_client.types import Message
from .baml_client.stream_types import ReplyTool
import numpy as np
from dotenv import load_dotenv
import sounddevice as sd

from .services.stt_session import SttSessionKyutai

load_dotenv()

class ConversationState(Enum):
    NO_CONVERSATION = 0
    LISTENING = 1
    SPEAKING = 2
    THINKING = 3
    SHUTTING_DOWN = 4

class ConversationManagerNode(Node):
    def __init__(self):
        super().__init__('conversation_manager')
        self.state = ConversationState.NO_CONVERSATION
        self.get_logger().info('ConversationManagerNode has been started.')
        self.SAMPLE_RATE = 16000
        self.TARGET_RATE = 24000

        
        # self.get_logger().info('Creating a minimal chat agent...')
        # chat = [
        #     Message( role="assistant",content="Hello! How can I assist you today?"),
        #     Message( role ="user", content="Hmm, what can you do? Give me a detailed list."),
        # ]
        # stream = b.stream.MinimalChatAgent(chat)
        # for partial in stream:
        #     if isinstance(partial,ReplyTool):
        #         # print(f"Streamed: {partial}")
        #         if partial.response:
        #             print(partial.response)
        # final = stream.get_final_response()

        self.stt_session = SttSessionKyutai(
            node_clock=self.get_clock(),
            node_logger=self.get_logger(),
        )

        asyncio.run(self.stt_session.start_up(
            url="ws://192.168.0.1:8080/api/asr-streaming",
        ))
        asyncio.run(self.stt_session.is_ready.wait())

    def _audio_streaming_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._stream_and_send_audio())

    async def _stream_and_send_audio(self):
        blocksize = 1920
        # Use a blocking stream to read audio in chunks
        with sd.InputStream(
            channels=1,
            samplerate=self.SAMPLE_RATE,
            dtype='float32',
            blocksize=blocksize
        ) as stream:
            while rclpy.ok():
                indata, _ = stream.read(blocksize)
                mono_audio = indata[:, 0]  # float32
                # Resample to 24kHz with librosa
                resampled = librosa.resample(
                    mono_audio, orig_sr=self.SAMPLE_RATE, target_sr=self.TARGET_RATE
                ).astype(np.float32)
                await self.stt_session.send_audio(resampled)

                



def main(args=None):
    rclpy.init(args=args)
    node = ConversationManagerNode()
    timer_period = 2.0  # seconds
    node.create_timer(timer_period, node.timer_callback)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()