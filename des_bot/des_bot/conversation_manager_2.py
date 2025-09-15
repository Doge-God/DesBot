import asyncio
import aiohttp
import librosa
import rclpy
from rclpy.node import Node
from statemachine import State, StateMachine
from std_msgs.msg import String
from enum import Enum, Flag, auto
import numpy as np
from dotenv import load_dotenv
import sounddevice as sd, time
from typing import List, Optional, TYPE_CHECKING, get_type_hints
import math
import threading


from .services.stt_session import SttSessionKyutai
from .utils.llm_stream_parser import SemanticDeltaParser
from .utils.audio_processor import OutputAudioProcessorInt16, bytes_needed_for_resample, resample_linear

from .baml_client.async_client import b
from .baml_client.types import Message
from .baml_client.stream_types import ReplyTool, StopTool

from .services.sentence_piece_tts import SentencePieceTts, SentencePiecePoisonPill

from des_bot_interfaces.srv import RunConversation

load_dotenv()

class ConversationState(Flag):
    IDLE = auto()

    USER_TURN = auto()
    
    ROBOT_PRE = auto()
    '''Processing conversation: can choose tool OR respond/farewell without tool'''
    ROBOT_TOOL_CALLING = auto()
    '''Calling tool'''
    ROBOT_AFT = auto()
    '''Can only generate response/farewell given tool result. (prevent tool loops)'''

    #-----------------------------------------------
    ROBOT_TURN = ROBOT_PRE | ROBOT_TOOL_CALLING | ROBOT_AFT
  

class ConversationManagerNode2(Node):
    def __init__(self):
        super().__init__('conversation_manager_node')
        
        # handle event loop in seperate thread: don't block ROS spin --------------------------------
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._start_asyncio_loop, daemon=True)
        self.state_publisher = self.create_publisher(String, 'conversation_state', 10)
        self._loop_thread.start()

        # param TODO might be good to use ros param --------------------------------------------------
        self.SAMPLE_RATE = 16000
        self.TARGET_RATE = 24000
        self.MIN_USER_MESSAGE_GAP_SEC = 2
        self.MIN_TTS_REQUEST_GAP_SEC = 1.0

        # Self STATE --------------------------------------------------------------------------------
        self.state = ConversationState.IDLE
        
        # tasks --------------------------------------------------------------------------------------
        # self.conversation_task = None # <========= HIGHEST LEVEL TASK IN NODE
        self.robot_task = None
        self.mic_stream_task = None
        self.stt_stream_task = None
        '''Listents to STT server returns. e.g. transcribed words and audio events.'''
        self.user_idle_timer_task = None
        '''Timer task to detect user silence, to end conversation'''

        # queues & stream/connection management structures--------------------------------------------
        self.stt_session:Optional[SttSessionKyutai] = None # this one has its own shutdown 
        '''Manager for websocket connection and parses input from server'''
        self.tts_session = None
        self.speaker_output_stream = None
        self.sentence_piece_tts_queue:Optional[asyncio.Queue["SentencePieceTts"]] = None
        self.current_sentence_piece_tts:Optional["SentencePieceTts"|"SentencePiecePoisonPill"] = None
        self.mic_audio_queue = asyncio.Queue()

         # conversation state, flag and buffers---------------------------------------------------------
        self.message_history:List[Message] = []
        self.stt_word_buffer = []
        self.robot_spoken_buffer = []
        self.end_of_flush_time = None
        '''Internal time for stt session: time for when text stream is guaranteed to be complete up to flush point.'''
        self.next_allowed_tts_request_stamp = None
        self.last_interaction_stamp = None
        '''Last time either user word heard / robot spoke something. Used for auto sleep.'''

        self.srv = self.create_service(
            RunConversation,
            'run_conversation',
            self.run_conversation_callback
        )

    ################################################################################################
    ############## Service Callback & Explicit State Transition ############################################
    def run_conversation_callback(self, _ , response):
        if self.state != ConversationState.IDLE:
            self.get_logger().warn("Conversation already ongoing, rejecting new request.")
            response.is_successful = False
            return response
        
        response.is_successful = True
        self.get_logger().info("## STARTING NEW CONVERSATION ##")
        asyncio.run(self.handle_USER_START_CONVERSATION())
        return response
    
    async def tick_stt_events(self):
        '''Check if state transition should happen based on stt status values. Polled every tick (~per 80ms; 1920/24k; 12.5Hz) of mic data sent.'''
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
                await self.handle_INTERRUPT_ROBOT()
        
        # audio flushing in process
        else:
            # we are sure transcription is complete, at time of silence detection
            if self.stt_session.current_time_sec > self.end_of_flush_time:
                await self.handle_USER_DONE_SPEAKING()
                
 
    # UTIL ---------------------------------------------------------------------------------------------------------
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
        '''In robot turn BUT there are words in stt buffer'''
        if (self.state in ConversationState.ROBOT_TURN and self.stt_word_buffer):
            return True
        else:
            return False

    
    ################################################################################################
    ############## State transition handler ############################################
    async def handle_USER_START_CONVERSATION(self):
        await self.prepare_conversation_startup()

        self.stt_stream_task = asyncio.run_coroutine_threadsafe(self.run_stt_stream(), self._loop)
        self.mic_stream_task = asyncio.run_coroutine_threadsafe(self.run_mic_stream(self._loop), self._loop)

        self.state = ConversationState.USER_TURN
        self.get_logger().info("FINISHED HANDLE START CONV STATE CHANGE")

    def handle_INTERRUPT_ROBOT(self):
        self.get_logger().info(">> INTERRUPT: User interrupted robot's turn.")
        pass

    def handle_USER_DONE_SPEAKING(self):
        new_user_message = Message(role='user', content= " ".join(self.stt_word_buffer))
        self.message_history.append(new_user_message)
        self.get_logger().info(f">> User turn finished: Done flushing. Message added: [{new_user_message.content[:20]}...]")
        # reset buffer & flush indicator flag
        self.stt_word_buffer.clear()
        self.end_of_flush_time = None
         # try getting response
        self.state = ConversationState.ROBOT_PRE
        self.robot_task = asyncio.run_coroutine_threadsafe(self.run_robot_pre(), self._loop)
        self.state_publisher.publish(self.create_std_str_msg("ROBOT_RECIEVED"))
    
    def handle_ROBOT_FINISHED(self):
        new_robot_message = Message(role='assistant',content=" ".join(self.robot_spoken_buffer))
        self.message_history.append(new_robot_message)
        self.robot_spoken_buffer.clear()
        self.get_logger().info(f">> Robot turn finished. Enter USER_TURN Message added: [{new_robot_message.content[:20]}...]")
       
        self.state = ConversationState.USER_TURN
        self.state_publisher.publish(self.create_std_str_msg("USER_TURN"))

        # TODO start user idle timer task

    def handle_ROBOT_REQUEST_TOOL(self):
        pass

    def handle_ROBOT_FAREWELL(self):
        pass
    
    # UTIL --------------------------------------------------------------------------------------------------
    async def prepare_conversation_startup(self):
        '''Start connections, set up queues etc. NOT setting state flag.'''
        if self.state != ConversationState.IDLE:
            self.get_logger().warn("Tried starting conversation with one in progress already. Ignored.")
            return

        self.get_logger().info("Readying conversation prerequisites...")
        t0 = self.get_clock().now().nanoseconds
        
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
        ## START TTS AUDIO STREAM ------------------------------------------------
        # Contiously try grabbing audio from current sentence piece tts.
        # Fill with silence in any other case
        def speaker_output_stream_callback(outdata, frames, time, status):
            '''ASSUME int16 audio: each frame = 16bit (2bytes)'''
            if status:
                print("Audio callback status:", status)

            current_tts = self.current_sentence_piece_tts
            if not current_tts or current_tts.is_all_audio_consumed.is_set():
                outdata[:] = b'\x00' * frames*2 #16bit frames: 
                return
            try:
                buffered_data = current_tts.force_get_bytes(bytes_needed_for_resample(frames, sr_in=24000, sr_out=16000))
            except RuntimeError as e:
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
            callback=speaker_output_stream_callback,
            device=0
        )
        self.speaker_output_stream.start()
        #-------------------------------------------------------------------------
        self.sentence_piece_tts_queue = asyncio.Queue()
        self.current_sentence_piece_tts = None
        self.mic_audio_queue = asyncio.Queue()


        self.message_history:List[Message] = []
        self.stt_word_buffer = []
        self.robot_spoken_buffer = []
        self.end_of_flush_time = None
        self.next_allowed_tts_request_stamp = None
        self.last_interaction_stamp = self.get_clock().now().nanoseconds
        self.last_user_word_heard_stamp = None

        t1 = self.get_clock().now().nanoseconds
        self.get_logger().info(f"Conversation ready. Spend: {(t1-t0)/1e6:.2f} ms")

    ################################################################################################
    ############# ROBOT TASKS ###############################################################  
    async def run_robot_pre(self):
        '''Generate response/farewell/tool call from llm. If response, queue tts, play them, add to robot spoken buffer'''
        try:
            # prepare llm related actions
            stream = b.stream.PrelimAgent(messages=self.message_history, activities="None")
            parser = SemanticDeltaParser()
            self.is_llm_generation_complete = False

            # prepare tts related actions
            advance_sentence_piece_tts_task = asyncio.run_coroutine_threadsafe(self.advance_sentence_piece_tts() ,self._loop)
            self.next_allowed_tts_request_stamp = self.get_clock().now().nanoseconds

            # process streamed llm response
            async for partial in stream:
                if isinstance(partial,ReplyTool):
                    if partial.response:
                        new_pieces, _ = parser.parse_new_input(partial.response) # IGNORE any final piece from llm not closed by separators 
                        self.queue_sentence_pieces_to_speak(new_pieces)
            

            # llm generation complete, add poinson pill to tts queue to end tts advance loop
            self.sentence_piece_tts_queue.put_nowait(SentencePiecePoisonPill())
            await asyncio.gather(advance_sentence_piece_tts_task)

            final =  await stream.get_final_response()
            # Transition to next state based on final tool------------------------------------
            # use run to block main event loop until state transition complete
            if isinstance(final, ReplyTool):
                asyncio.run(self.handle_ROBOT_FINISHED())
            elif isinstance(final, StopTool):
                asyncio.run(self.handle_ROBOT_FAREWELL())
            else:
                asyncio.run(self.handle_ROBOT_REQUEST_TOOL(final))

        except asyncio.CancelledError:
        # when generation task is cancelled: cancel tts obj cycling task, empty sentence_piece_tts_queue
            advance_sentence_piece_tts_task.cancel()
            try:
                while True:
                    sentence_piece_tts = self.sentence_piece_tts_queue.get_nowait()
                    if isinstance(sentence_piece_tts, SentencePieceTts):
                        sentence_piece_tts.fetch_task.cancel()
                        sentence_piece_tts.audio_buffer = b""
            except asyncio.QueueEmpty:
                pass
            # DOES NOT ADVANCE STATE!!!!!!!!
            raise






    
    # -- Util ------------------------------------------------------------------------------------
    def pop_last_message_of_role(self, role:str):
        '''Get last messge from converation of given role & remove from history.'''
        for i in range(len(self.message_history)-1, -1, -1):
            if self.message_history[i].role == role:
                return self.message_history.pop(i)
        return None

    ################################################################################################
    ############# TTS Related Tools & tasks ###############################################################       
    async def advance_sentence_piece_tts(self):
        '''Loop to grab sentence piece tts objects from queue for playback. Add to robot spoken buffer'''
        try: # for cancellation
            while self.state in ConversationState.ROBOT_TURN:
                # grab from queue
                new_current_sentence_piece_tts = await self.sentence_piece_tts_queue.get()

                if isinstance(new_current_sentence_piece_tts, SentencePiecePoisonPill):
                    print("^^ Received poison pill, ending tts advance loop.")
                    break

                # swap out spent one 
                self.current_sentence_piece_tts = new_current_sentence_piece_tts

                # add to tracker & log
                self.robot_spoken_buffer.append(self.current_sentence_piece_tts.text)
                print(f"vv NEW sentence tts obj: [{self.current_sentence_piece_tts.text}]--------------------")

                # update state
                #TODO publish stuff for visual report: start speaking

                #wait for current sentence piece tts to finish
                await self.current_sentence_piece_tts.is_all_audio_consumed.wait()


        except asyncio.CancelledError:
            self.current_sentence_piece_tts = None
               
        finally:
            self.current_sentence_piece_tts = None
            print(f"SPOKEN ENTIRE LLM RESPONSE.")            

    def queue_sentence_pieces_to_speak(self, pieces:List[str]):
        '''Make SentencePieceTts objects from strings and queue them for processing, respects self.next_allowed_tts_request_stamp (should be reset per generation task.)'''   
        for new_piece in pieces:              
            now_stamp = self.get_clock().now().nanoseconds
            # Request NOT allowed immediately: less than specified sec apart from previous request: needed to ensure fastkoko not drowning and degrade first audio time performance
            if now_stamp <= self.next_allowed_tts_request_stamp:
                delta_sec = (self.next_allowed_tts_request_stamp-now_stamp) / 1_000_000_000
                self.sentence_piece_tts_queue.put_nowait(
                    SentencePieceTts(new_piece, self.tts_session, asyncio.get_running_loop(), delta_sec)
                )
            # Request allowed, start fetching immediatly
            else:
                self.sentence_piece_tts_queue.put_nowait(
                    SentencePieceTts(new_piece, self.tts_session, asyncio.get_running_loop() )
                )
            self.next_allowed_tts_request_stamp += self.MIN_TTS_REQUEST_GAP_SEC * 1_000_000_000
    

    ################################################################################################
    ############# MIC & STT Tools & tasks ###############################################################  

    async def run_mic_stream(self, loop:Optional[asyncio.AbstractEventLoop]=None):
        try:
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
                while True:
                    audio_data = await self.mic_audio_queue.get()
                    # Resample to 24kHz NOT with librosa thing takes 2s
                    resampled = resample_linear(audio_data, orig_sr=16000, target_sr=24000)

                    # dont send audio when flushing
                    try:
                        if not self.end_of_flush_time:
                            await self.stt_session.send_audio(resampled)
                    except Exception as e:
                        self.get_logger().warning("Caught when sending audio: "+str(e))
                    
                    await self.tick_stt_events()

        except asyncio.CancelledError:
            self.get_logger().info("Mic stream closed cleanly.")
            raise
   
    async def run_stt_stream(self):
        '''closes by calling shutdown on stt_session instance'''
        self.get_logger().info("Begin stt stream, listing for transcribed words")
        if not self.stt_session:
            raise RuntimeError("Trying to listen to stt response without instantiating object first.")
        
        # breaks gracefully when shutdown func called on stt_session
        async for message in self.stt_session:
            print()
            # first utterance in user turn, reset predictor value
            if len(self.stt_word_buffer) == 0:
                self.stt_session.pause_predictor.value = 0

            self.stt_word_buffer.append(message.text)
            self.last_user_word_heard_stamp = self.get_clock().now().nanoseconds

        self.get_logger().info("STT task finished.")



    ################################################################################################
    ############## Asyncio loop control util ############################################
    def _start_asyncio_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def destroy_node(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join()
        super().destroy_node()

    ################################################################################################
    ############# Util util, actual simply isolated shorthand stuff ############################################
    def create_std_str_msg(self, message:str):
        msg = String()
        msg.data = message
        return msg
    
    



def main(args=None):
    rclpy.init(args=args)
    node = ConversationManagerNode2()


    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()
   


if __name__ == '__main__':
    main()