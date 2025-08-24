import asyncio
import librosa
import rclpy
from rclpy.node import Node
from enum import Enum
import numpy as np
from dotenv import load_dotenv
import sounddevice as sd
from typing import List, Optional
import math

from .services.stt_session import SttSessionKyutai
# from .baml_client import b, partial_types, types
from .baml_client.async_client import b
from .baml_client.types import Message
from .baml_client.stream_types import ReplyTool

# sanity check stuff for llm
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
load_dotenv()

class ConversationState(Enum):
    NO_CONVERSATION = 0
    STARTING_UP = 1
    USER_TURN = 2
    ROBOT_TURN = 3
    SHUTTING_DOWN = 4

class ChatHistory:
    def __init__(self):
        pass

class ConversationManagerNode(Node):
    def __init__(self):
        super().__init__('conversation_manager')
        self.get_logger().info('ConversationManagerNode has been started.')
        self.SAMPLE_RATE = 16000
        self.TARGET_RATE = 24000

        
        # tasks & websocket session managment
        self.robot_response_task = None
        self.sentence_piece_tts_task_queue = asyncio.Queue()

        self.mic_stream_task = None
        self.stt_session = None
        self.stt_stream_task = None
        '''Listents to STT server returns. e.g. transcribed words and audio events.'''
        self.tts_task = None
        '''Loop through queued tasks for speaking sentence pieces'''

        
        self.mic_audio_queue = asyncio.Queue()

        # conversation state, flag and buffers
        self.state = ConversationState.NO_CONVERSATION
        self.message_history:List[Message] = []
        self.stt_word_buffer = []
        self.robot_generated_buffer = []
        self.end_of_flush_time = None
        '''Internal time for stt session: time for when text stream is guaranteed to be complete up to flush point.'''

        asyncio.run(self.run_conversation())


    def change_state(new_state:ConversationState):
        pass

    
    async def speak_sentence_piece(self, input:str):
        self.robot_generated_buffer.append(input)
        self.get_logger().info(f"Sim start saying: {input}")
        await asyncio.sleep(2)
        self.get_logger().info(f"Sim finished saying: {input}")


    def should_transition_user_to_robot(self):
        '''Is user turn, detected pause, and have some transcript'''
        if not self.stt_session:
            return False
        if (self.state == ConversationState.USER_TURN 
            and self.stt_session.pause_predictor.value > 0.6
            and len(self.stt_word_buffer)>=1):
            return True
        else:
            return False


    async def handle_stt_triggered_state_change(self):
        '''Handle conversation state transition. Run every tick (~per 80ms; 1920/24k; 12.5Hz) of mic data sent.'''
        
        # is not in the process of flushing audio
        if self.end_of_flush_time is None:
            if self.should_transition_user_to_robot():
                self.get_logger().info("Detected user end speech: begin flushing.")

                # flush stt process on server with blank audio to immediate get last bits of transcription back
                num_frames = int(math.ceil(0.5 / (0.08))) + 1 # some extra for safety
                blank_audio = np.zeros(1920, dtype=np.float32)
                self.end_of_flush_time = self.stt_session.current_time_sec + self.stt_session.delay_sec
                for _ in range (num_frames):
                    await self.stt_session.send_audio(blank_audio)
        
        # audio flushing in process
        else:
            # we are sure transcription is complete at time of silence detection
            if self.stt_session.current_time_sec > self.end_of_flush_time:

                new_user_message = Message(role='user', content= " ".join(self.stt_word_buffer))
                self.message_history.append(new_user_message)
                self.get_logger().info(f"User turn finished: Done flushing. Message added: {str(new_user_message)}")

                # reset buffer
                self.stt_word_buffer = []
                self.end_of_flush_time = None

                self.state = ConversationState.ROBOT_TURN
                self.robot_response_task = self.generate_response()
                asyncio.create_task(self.robot_response_task)
                
                

    async def generate_response(self):
        """Get response from LLM and run tts."""
        stream = b.stream.MinimalChatAgent(self.message_history)
        

        async for partial in stream:
            if isinstance(partial,ReplyTool):
                if partial.response:
                    print(partial.response)

        ## TEMP ######################################################################
        final = await stream.get_final_response()
        if isinstance(final, ReplyTool):
            print("REPLY: " + final.response)
        self.message_history.append(Message(role='assistant', content=final.response))

        await self.speaker_audio_queue
        self.get_logger().info("Robot generation finished. Yield to USER_TURN")
        self.state = ConversationState.USER_TURN
            

    async def run_conversation(self, robot_invoke_reason=None):
        self.get_logger().info("Started conversation.")
        self.state = ConversationState.STARTING_UP

        self.stt_session =  SttSessionKyutai(
            node_clock=self.get_clock(),
            node_logger=self.get_logger(),
        )

        await self.stt_session.start_up(
            url="ws://192.168.137.1:8080/api/asr-streaming",
            api_key="public_token"
        )

        if not robot_invoke_reason:
            self.state = ConversationState.USER_TURN

        self.stt_stream_task = self.run_stt_stream()
        self.mic_stream_task = self.run_mic_stream()
       
        asyncio.create_task(self.stt_stream_task)
        asyncio.create_task(self.mic_stream_task)
        

        await asyncio.gather(
            self.stt_stream_task,
            self.mic_stream_task,
        )

        self.get_logger().info("Conversation finished cleanly.")

    async def run_mic_stream(self, loop:Optional[asyncio.AbstractEventLoop]=None):
        self.get_logger().info("Begin mic stream, sending data")
        # Start the audio stream
        
        if not loop:
            this_loop = asyncio.get_running_loop()
        else:
            this_loop = loop

        def audio_callback(data:np.ndarray, frames, time, status):
            this_loop.call_soon_threadsafe(
                # get copy of mono channel data
                self.mic_audio_queue.put_nowait, data[:,0].astype(np.float32).copy()
            )

        with sd.InputStream(
            samplerate=self.SAMPLE_RATE, 
            channels=1, 
            blocksize=1920,
            dtype='float32',
            callback=audio_callback):
        

            while self.state != ConversationState.SHUTTING_DOWN:
                audio_data = await self.mic_audio_queue.get()
                # Resample to 24kHz with librosa
                resampled = librosa.resample(
                    audio_data, orig_sr=self.SAMPLE_RATE, target_sr=self.TARGET_RATE
                )  #.astype(np.float32)

                # dont send audio when flushing
                if not self.end_of_flush_time:
                    await self.stt_session.send_audio(resampled)
                
                await self.handle_stt_triggered_state_change()

        self.get_logger().info("Mic stream closed cleanly.")

    async def run_tts_tasks(self):
        while self.state == ConversationState.ROBOT_TURN:
            task = self.sentence_piece_tts_task_queue.get()
            await task

    async def run_stt_stream(self):
        '''closes by calling shutdown on stt_session instance'''
        self.get_logger().info("Begin stt stream, listing for transcribed words")
        if not self.stt_session:
            raise RuntimeError("Trying to listen to stt response without instantiating object first.")
        
        async for message in self.stt_session:
            # first utterance in user turn, reset predictor value
            if len(self.stt_word_buffer) == 0:
                self.stt_session.pause_predictor.value = 0

            self.stt_word_buffer.append(message.text)

        self.get_logger().info("STT task finished.")


                
               
    
    # async def 

    


def main(args=None):
    rclpy.init(args=args)
    node = ConversationManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()