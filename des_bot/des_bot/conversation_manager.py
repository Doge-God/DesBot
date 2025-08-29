import asyncio
import aiohttp
import librosa
import rclpy
from rclpy.node import Node
from enum import Enum, Flag, auto
import numpy as np
from dotenv import load_dotenv
import sounddevice as sd, time
from typing import List, Optional, TYPE_CHECKING
import math


from .services.stt_session import SttSessionKyutai
from .utils.llm_stream_parser import SemanticDeltaParser
from .utils.audio_processor import OutputAudioProcessorInt16, bytes_needed_for_resample, resample_linear

from .baml_client.async_client import b
from .baml_client.types import Message
from .baml_client.stream_types import ReplyTool

from .services.sentence_piece_tts import SentencePieceTts

load_dotenv()

class ConversationState(Flag):
    NO_CONVERSATION = auto()
    STARTING_UP = auto()
    USER_TURN = auto()
    ROBOT_RECIEVED = auto()
    ROBOT_USING_TOOLS = auto()
    ROBOT_SPEAKING = auto()
    SHUTTING_DOWN = auto()
    #-----------------------------------------------
    ROBOT_TURN = ROBOT_RECIEVED | ROBOT_SPEAKING | ROBOT_USING_TOOLS

class StateChangeCases(Enum):
    INTERRUPT_ROBOT = auto()

class ChatHistory:
    def __init__(self):
        pass

class ConversationManagerNode(Node):
    def __init__(self):
        super().__init__('conversation_manager')
        self.get_logger().info('ConversationManagerNode has been started.')
        self.SAMPLE_RATE = 16000
        self.TARGET_RATE = 24000
        self.MIN_USER_MESSAGE_GAP_SEC = 2.0

        
        # tasks ---------------------------------------------------------------------------
        self.robot_response_task = None
        self.mic_stream_task = None
        self.stt_stream_task = None
        '''Listents to STT server returns. e.g. transcribed words and audio events.'''

        # queues & stream/connection management structures
        self.stt_session:Optional[SttSessionKyutai] = None # this one has its own shutdown ---------------------------
        '''Manager for websocket connection and parses input from server'''
        self.tts_session = None
        self.speaker_output_stream = None
        self.sentence_piece_tts_queue:Optional[asyncio.Queue["SentencePieceTts"]] = None
        self.current_sentence_piece_tts:Optional["SentencePieceTts"] = None
        self.current_sentence_piece_tts_lock = asyncio.Lock()

        # conversation state, flag and buffers
        self.state = ConversationState.NO_CONVERSATION
        self.message_history:List[Message] = []
        self.stt_word_buffer = []
        self.robot_spoken_buffer = []
        self.is_llm_generation_complete = False
        self.end_of_flush_time = None
        '''Internal time for stt session: time for when text stream is guaranteed to be complete up to flush point.'''
        self.last_user_word_heard_stamp = None

        asyncio.run(self.run_conversation())


    def change_state(new_state:ConversationState):
        pass
    


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
    
    def should_interrupt_robot(self):
        if (self.state in ConversationState.ROBOT_TURN and self.stt_word_buffer):
            return True
        else:
            return False


    async def handle_stt_triggered_state_change(self):
        '''Handle conversation state transition. Polled every tick (~per 80ms; 1920/24k; 12.5Hz) of mic data sent.'''
        
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
            
            elif self.should_interrupt_robot():
                t0 = time.perf_counter()
                print(f"============ INTERRUPTED ROBOT ============")
                self.robot_spoken_buffer.append("[INTERRUPTED]")
                self.robot_response_task.cancel()
                t1 = time.perf_counter()

                print(f"Interrupted task in {[t1-t0]} ms")
                self.state = ConversationState.USER_TURN
        
        # audio flushing in process
        else:
            # we are sure transcription is complete at time of silence detection
            if self.stt_session.current_time_sec > self.end_of_flush_time:

                new_user_message = Message(role='user', content= " ".join(self.stt_word_buffer))
                self.message_history.append(new_user_message)
                self.get_logger().info(f"User turn finished: Done flushing. Message added: {str(new_user_message)}")

                # reset buffer & flush indicator flag
                self.stt_word_buffer = []
                self.end_of_flush_time = None

                # try getting response
                self.state = ConversationState.ROBOT_TURN
                self.robot_response_task = asyncio.create_task(self.generate_response())
                
    async def advance_sentence_piece_tts(self):
        '''Loop to grab sentence piece tts from queue for playback.'''
        try: # for cancellation
            while self.state in ConversationState.ROBOT_TURN:
                # grab from queue
                self.current_sentence_piece_tts = await self.sentence_piece_tts_queue.get()
                # add to tracker & log
                self.robot_spoken_buffer.append(self.current_sentence_piece_tts.text)
                print(f"vv NEW sentence tts obj: [{self.current_sentence_piece_tts.text}]--------------------")
                #wait for current sentence piece tts to finish
                await self.current_sentence_piece_tts.is_all_audio_consumed.wait()
                # before trying to wait for next sentence piece tts to be in queue, check if current
                # one is the last sentence piece generated by llm
                if self.sentence_piece_tts_queue.qsize() == 0:
                    if self.is_llm_generation_complete:
                        break
                print(f"^^ Finished sentence tts obj: [{self.current_sentence_piece_tts.text}] CONTINUE-------")
        finally:
            self.current_sentence_piece_tts = None
            print(f"SPOKEN ENTIRE LLM RESPONSE.")            

    async def generate_response(self):
        """Get response from LLM and run tts."""
        try:
            # prepare llm related actions
            stream = b.stream.MinimalChatAgent(self.message_history)
            parser = SemanticDeltaParser()
            self.is_llm_generation_complete = False

            # prepare tts related actions
            min_tts_request_gap_sec = 1.0
            next_allowed_tts_request_stamp = self.get_clock().now().nanoseconds
            advance_sentence_piece_tts_task = asyncio.create_task(self.advance_sentence_piece_tts())
            
            # process streamed llm response
            async for partial in stream:
                if isinstance(partial,ReplyTool):
                    if partial.response:
                        new_pieces, unclosed_piece = parser.parse_new_input(partial.response)
                        for new_piece in new_pieces:

                            now_stamp = self.get_clock().now().nanoseconds
                            # Request NOT allowed immediately: less than specified sec apart from previous request: 
                            #   needed to ensure fastkoko not drowning and degrade first audio time performance
                            if now_stamp <= next_allowed_tts_request_stamp:
                                delta_sec = (next_allowed_tts_request_stamp-now_stamp) / 1_000_000_000
                                self.sentence_piece_tts_queue.put_nowait(
                                    SentencePieceTts(new_piece, self.tts_session, asyncio.get_running_loop(), delta_sec)
                                )
                            # Request allowed, start fetching immediatly
                            else:
                                self.sentence_piece_tts_queue.put_nowait(
                                    SentencePieceTts(new_piece, self.tts_session, asyncio.get_running_loop() )
                                )
                            
                            next_allowed_tts_request_stamp += min_tts_request_gap_sec * 1_000_000_000
                            
            if unclosed_piece:
                # unlikely to happen. Also when it gets here other tts request are likely complete anyway.
                self.sentence_piece_tts_queue.put_nowait(
                    SentencePieceTts(unclosed_piece, self.tts_session, asyncio.get_running_loop() )
                )
        
            self.is_llm_generation_complete = True
            await asyncio.gather(advance_sentence_piece_tts_task)

        except asyncio.CancelledError:
        # when generation task is cancelled: cancel tts obj cycling task, empty sentence_piece_tts_queue
            advance_sentence_piece_tts_task.cancel()
            try:
                while True:
                    sentence_piece_tts = self.sentence_piece_tts_queue.get_nowait()
                    sentence_piece_tts.fetch_task.cancel()
                    sentence_piece_tts.audio_buffer = b""
        
            except asyncio.QueueEmpty:
                pass
  
            raise

        finally:
            self.get_logger().info("Robot generation finished. Yield to USER_TURN")
            self.message_history.append(Message(role='assistant', content=" ".join(self.robot_spoken_buffer)))
            self.robot_spoken_buffer = []
            self.state = ConversationState.USER_TURN
    
    async def run_converstion_startup(self):
        '''Start connections, set up queues etc. NOT setting state flag.'''
        self.get_logger().info("Starting conversation...")
        t0 = self.get_clock().now().nanoseconds

        # initialize streams, tasks, queues etc etc.
        self.mic_audio_queue = asyncio.Queue()
        self.sentence_piece_tts_queue = asyncio.Queue()

        ## START TTS AUDIO STREAM ------------------------------------------------
        # Contiously try grabbing audio from current sentence piece tts.
        # Fill with silence in any other case
        def audio_callback(outdata, frames, time, status):
            '''ASSUME int16 audio: each frame = 16bit (2bytes)'''
            if status:
                print("Audio callback status:", status)
            current_tts = self.current_sentence_piece_tts
            if not current_tts or current_tts.is_all_audio_consumed.is_set():
                outdata[:] = b'\x00' * frames*2 #16bit frames: 
                return
            try:
                buffered_data = current_tts.force_get_bytes(bytes_needed_for_resample(frames, sr_in=24000, sr_out=16000))
            except:
                outdata[:] = b'\x00' * frames*2 #16bit frames: 
                return
            processed = (
                OutputAudioProcessorInt16(buffered_data)
                .downsample(num_frames=frames, target_rate=16000)
                .process()
            )
            outdata[:] = processed


        self.speaker_output_stream = sd.RawOutputStream(
            samplerate=16000,  
            channels=1,
            dtype="int16",
            callback=audio_callback,
            device=0
        )
        self.speaker_output_stream.start()
        #-------------------------------------------------------------------------

        self.message_history:List[Message] = []
        self.stt_word_buffer = []
        self.robot_spoken_buffer = []
        self.end_of_flush_time = None
        
        # networking sessions ---------------------------------
        self.stt_session =  SttSessionKyutai(
            node_clock=self.get_clock(),
            node_logger=self.get_logger(),
        )
        await self.stt_session.start_up(
            url="ws://192.168.137.1:8080/api/asr-streaming",
            api_key="public_token"
        )
        self.tts_session = aiohttp.ClientSession()

        t1 = self.get_clock().now().nanoseconds
        self.get_logger().info(f"Started Conversation. Readied in: {(t1-t0)/1000000:.2f} ms")

    def conversation_clean_up(self):
        '''Reset all flags, queues etc.'''
        if self.speaker_output_stream:
            self.speaker_output_stream.close()
        else:
            self.get_logger().warning("Cleaning up conversation: trying to close non-existing speaker output stream.")

        # tasks ---------------------------------------------------------------------------
        self.robot_response_task = None
        self.mic_stream_task = None
        self.stt_stream_task = None
        '''Listents to STT server returns. e.g. transcribed words and audio events.'''
        self.advance_sentence_piece_tts_task = None
        '''Loop through queued tasks for speaking sentence pieces'''

        # queues & stream/connection management structures
        self.stt_session:Optional[SttSessionKyutai] = None # this one has its own shutdown ---------------------------
        '''Manager for websocket connection and parses input from server'''
        self.tts_session = None
        self.speaker_output_stream = None
        self.sentence_piece_tts_queue:Optional[asyncio.Queue["SentencePieceTts"]] = None
        self.current_sentence_piece_tts:Optional["SentencePieceTts"] = None

        # conversation state, flag and buffers
        self.state = ConversationState.NO_CONVERSATION
        self.message_history:List[Message] = []
        self.stt_word_buffer = []
        self.robot_spoken_buffer = []
        self.is_llm_generation_complete = False
        self.end_of_flush_time = None

    async def run_conversation(self, robot_invoke_reason=None):

        self.state = ConversationState.STARTING_UP

        await self.run_converstion_startup()

        # run "Main" loops
        if not robot_invoke_reason:
            self.state = ConversationState.USER_TURN

        self.stt_stream_task = asyncio.create_task(self.run_stt_stream())
        self.mic_stream_task = asyncio.create_task(self.run_mic_stream())
        
        await asyncio.gather(
            self.stt_stream_task,
            self.mic_stream_task,
        )

        # Clean up
        self.conversation_clean_up()

        self.get_logger().info("Conversation finished cleanly.")

    

    async def run_mic_stream(self, loop:Optional[asyncio.AbstractEventLoop]=None):
        '''Close by setting conversation state to shutting down'''
        self.get_logger().info("Begin mic stream, sending data")
        # Start the audio stream
        
        if not loop:
            this_loop = asyncio.get_running_loop()
        else:
            this_loop = loop

        def audio_callback(data:np.ndarray, frames, time, status):
            # get copy of mono channel data, drop all values < 0.2
            data_copy = data[:,0].astype(np.float32).copy()
            data_copy[np.abs(data_copy) < 0.01] = 0
            this_loop.call_soon_threadsafe(
                
                
                self.mic_audio_queue.put_nowait, data_copy
            )

        with sd.InputStream(
                samplerate=16000, 
                channels=1, 
                blocksize=1920,
                dtype='float32',
                callback=audio_callback,
                device=0
            ):

            # 1920 samples at 16000 hz ~ 120ms
            while self.state != ConversationState.SHUTTING_DOWN:
                audio_data = await self.mic_audio_queue.get()
                # Resample to 24kHz NOT with librosa thing takes 2s
                resampled = resample_linear(audio_data, orig_sr=16000, target_sr=24000)

                # dont send audio when flushing
                if not self.end_of_flush_time:
                    await self.stt_session.send_audio(resampled)
                
                await self.handle_stt_triggered_state_change()

        self.get_logger().info("Mic stream closed cleanly.")
         

    async def run_stt_stream(self):
        '''closes by calling shutdown on stt_session instance'''
        self.get_logger().info("Begin stt stream, listing for transcribed words")
        if not self.stt_session:
            raise RuntimeError("Trying to listen to stt response without instantiating object first.")
        
        # breaks gracefully when shutdown func called on stt_session
        async for message in self.stt_session:
            # first utterance in user turn, reset predictor value
            if len(self.stt_word_buffer) == 0:
                self.stt_session.pause_predictor.value = 0

            self.stt_word_buffer.append(message.text)
            self.last_user_word_heard_stamp = self.get_clock().now().nanoseconds

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