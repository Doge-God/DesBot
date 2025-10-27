import asyncio
import os
import aiohttp
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import ListenV2SocketClientResponse
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

from .utils.llm_stream_parser import SemanticDeltaParser
from .utils.audio_processor import OutputAudioProcessorInt16, bytes_needed_for_resample, resample_linear

from .baml_client.async_client import b
from .baml_client.types import Message, ReplyTool, StopTool, BookActivityTool, SuggestActivityTool
from .baml_client import stream_types

from .services.sentence_piece_tts import SentencePieceTts, SentencePiecePoisonPill
from .utils.conversation_state_types import ConversationState

from des_bot_interfaces.srv import StartConversation, EndConversation

load_dotenv()

class ConversationManagerDG():
    def __init__(self, node:Node):
    
        self.node = node
        # handle event loop in seperate thread: don't block ROS spin --------------------------------
        # self._loop = asyncio.new_event_loop()
        # self._loop_thread = threading.Thread(target=self._start_asyncio_loop, daemon=True)
        self.state_publisher = self.node.create_publisher(String, 'conversation_state', 10)
        self.stt_pulse_publisher = self.node.create_publisher(String, 'stt_pulse', 10)
        # self._loop_thread.start()

        # param TODO might be good to use ros param --------------------------------------------------
        self.SAMPLE_RATE = 16000
        self.TARGET_RATE = 24000
        self.MIN_USER_MESSAGE_GAP_SEC = 1
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
        self.is_stt_ready = asyncio.Event()
        '''Last time either user word heard / robot spoke something. Used for auto sleep.'''

        self.srv = self.node.create_service(
            StartConversation,
            'start_conversation',
            self.run_conversation_callback
        )

        self.srv = self.node.create_service(
            EndConversation,
            "end_conversation",
            self.end_conversation_callback
        )

    ################################################################################################
    ############## Service Callback & Explicit State Transition ############################################
    def run_conversation_callback(self, _ , response):
        if self.state != ConversationState.IDLE:
            self.node.get_logger().warn("Conversation already ongoing, rejecting new request.")
            response.is_successful = False
            return response
        
        response.is_successful = True
        self.node.get_logger().info("## STARTING NEW CONVERSATION ##")
        asyncio.gather(self.handle_USER_START_CONVERSATION())
        return response
    
    def end_conversation_callback(self, _ , response):
        if self.state == ConversationState.IDLE:
            self.node.get_logger().warn("Idle, rejecting end conversation request.")
            response.is_successful = False
            return response
        
        self.node.get_logger().info("## USER END CONVERSATION ##")
        self.handle_USER_END_CONVERSATION()
        response.is_successful = True
        return response
    
    # async def tick_stt_events(self):
    #     '''Check if state transition should happen based on stt status values. Polled every tick (~per 80ms; 1920/24k; 12.5Hz) of mic data sent.'''
    #     # is not in the process of flushing audio
    #     if self.end_of_flush_time is None:
    #         if self.should_transition_user_to_robot():
    #             self.node.get_logger().info("Detected user end speech: begin flushing.")
    #             # flush stt process on server with blank audio to immediate get last bits of transcription back
    #             num_frames = int(math.ceil(0.5 / (0.08))) + 1 # some extra for safety
    #             blank_audio = np.zeros(1920, dtype=np.float32)
    #             self.end_of_flush_time = self.stt_session.current_time_sec + self.stt_session.delay_sec
    #             for _ in range (num_frames):
    #                 await self.stt_session.send_audio(blank_audio)
            
    #         elif self.should_interrupt_robot():
    #             self.handle_INTERRUPT_ROBOT()
        
    #     # audio flushing in process
    #     else:
    #         # we are sure transcription is complete, at time of silence detection
    #         if self.stt_session.current_time_sec > self.end_of_flush_time:
    #             self.handle_USER_DONE_SPEAKING()
            

    
    ################################################################################################
    ############## State transition handler ############################################
    async def handle_USER_START_CONVERSATION(self):

        # start stt stream first:   need to recieve stream ready from server
        self.stt_stream_task = asyncio.create_task(self.run_stt_stream())

        await self.prepare_conversation_startup()

      
        self.mic_stream_task = asyncio.create_task(self.run_mic_stream())

        self.state = ConversationState.USER_TURN
        self.state_publisher.publish(self.create_std_str_msg(ConversationState.USER_TURN.name))
        self.node.get_logger().info("FINISHED HANDLE START CONV STATE CHANGE")

    def handle_INTERRUPT_ROBOT(self):
        self.node.get_logger().info("## INTERRUPT: User interrupted robot's turn.")
        #stop current task whatever it is
        self.robot_task.cancel()
        # check for case: [user speech] [very short accidental pause] [robot speech start] [user speecch]
        # then ignore robot task
        if (self.node.get_clock().now().nanoseconds - self.last_user_done_speaking_stamp) < self.MIN_USER_MESSAGE_GAP_SEC * 1e9:
            self.robot_spoken_buffer.clear()
            last_user_msg = self.pop_last_message_of_role('user')
            if last_user_msg:
                self.stt_word_buffer.insert(0, last_user_msg.content)
            
        # normal case: simply interrupting robot
        # add interrupt label and add to chat history
        else:        
            self.robot_spoken_buffer.append("[INTERRUPTED]")
            new_robot_message = Message(role='assistant',content=" ".join(self.robot_spoken_buffer))
            self.message_history.append(new_robot_message)
            self.robot_spoken_buffer.clear()

        self.state = ConversationState.USER_TURN
        self.state_publisher.publish(self.create_std_str_msg(ConversationState.USER_TURN.name))

    def handle_USER_DONE_SPEAKING(self):
        self.node.get_logger().info("## HANDLE USER DONE SPEAKING")
        new_user_message = Message(role='user', content= " ".join(self.stt_word_buffer))
        self.message_history.append(new_user_message)
        self.node.get_logger().info(f">> User turn finished: Done flushing. Message added: [{new_user_message.content[:20]}...]")
        # reset buffer & flush indicator flag
        self.stt_word_buffer.clear()
        self.end_of_flush_time = None
         # try getting response
        self.state = ConversationState.ROBOT_PRE
        self.robot_task = asyncio.create_task(self.run_robot_pre()) # <===== TASK STARTED HERE
        # keep time stamp of last user done speaking time
        self.last_user_done_speaking_stamp = self.node.get_clock().now().nanoseconds
        # inform other nodes and log and stuff
        self.state_publisher.publish(self.create_std_str_msg(ConversationState.ROBOT_PRE.name))
        self.node.get_logger().info("## DONE: HANDLE USER DONE SPEAKING")
    
    def handle_ROBOT_FINISHED(self):
        self.node.get_logger().info("## HANDLE ROBOT FINISHED")
        new_robot_message = Message(role='assistant',content=" ".join(self.robot_spoken_buffer))
        self.message_history.append(new_robot_message)
        self.robot_spoken_buffer.clear()
        self.node.get_logger().info(f">> Robot turn finished. Enter USER_TURN Message added: [{new_robot_message.content[:20]}...]")
       
        self.state = ConversationState.USER_TURN
        self.state_publisher.publish(self.create_std_str_msg(ConversationState.USER_TURN.name))

        # TODO start user idle timer task

    def handle_ROBOT_REQUEST_TOOL(self, tool_call:BookActivityTool | SuggestActivityTool):
        self.node.get_logger().info("## HANDLE ROBOT TOOLCALLING")
        self.state = ConversationState.ROBOT_TOOL_CALLING
        self.state_publisher.publish(self.create_std_str_msg(ConversationState.ROBOT_TOOL_CALLING.name))
        self.robot_task = asyncio.create_task(self.run_robot_tool_call(tool_call))

    def handle_ROBOT_GENERATE_RESPONSE(self):
        self.node.get_logger().info("## HANDLE ROBOT GENERATE RESPONSE")
        self.state = ConversationState.ROBOT_AFT
        self.state_publisher.publish(self.create_std_str_msg(ConversationState.ROBOT_AFT.name))
        self.robot_task = asyncio.create_task(self.run_robot_aft())

    def handle_ROBOT_FAREWELL(self):
        self.node.get_logger().info("## HANDLE ROBOT FAREWELL")
        self.state_publisher.publish(self.create_std_str_msg(ConversationState.ROBOT_COUNTDOWN.name))
        self.state = ConversationState.ROBOT_COUNTDOWN
        self.robot_task = asyncio.create_task(self.run_robot_countdown())

    def handle_ROBOT_END_CONVERSATION(self):
        self.node.get_logger().info("## HANDLE ROBOT END CONVERSATION")
        self.state_publisher.publish(self.create_std_str_msg(ConversationState.CONVERSATION_SHUTDOWN.name))
        self.state = ConversationState.CONVERSATION_SHUTDOWN
        asyncio.create_task(self.run_conversation_shutdown())
    
    def handle_USER_END_CONVERSATION(self):
        self.node.get_logger().info("## HANDLE ROBOT END CONVERSATION")
        self.state_publisher.publish(self.create_std_str_msg(ConversationState.CONVERSATION_SHUTDOWN.name))
        self.state = ConversationState.CONVERSATION_SHUTDOWN
        asyncio.create_task(self.run_conversation_shutdown())

    def handle_RESET_TO_IDLE(self):
        self.node.get_logger().info("## HANDLE RESET TO IDLE")
        self.state_publisher.publish(self.create_std_str_msg(ConversationState.IDLE.name))
        self.state = ConversationState.IDLE

    # UTIL --------------------------------------------------------------------------------------------------
    async def prepare_conversation_startup(self):
        '''Start connections, set up queues etc. NOT setting state flag.'''
        if self.state != ConversationState.IDLE:
            self.node.get_logger().warn("Tried starting conversation with one in progress already. Ignored.")
            return

        self.node.get_logger().info("Readying conversation prerequisites...")
        t0 = self.node.get_clock().now().nanoseconds

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
                outdata[:] = b'\x00' * frames*4 #16bit frames: 
                return
            try:
                buffered_data = current_tts.force_get_bytes(bytes_needed_for_resample(frames, sr_in=24000, sr_out=16000))
            except RuntimeError as e:
                outdata[:] = b'\x00' * frames*4#16bit frames: 
                return
            processed = (
                OutputAudioProcessorInt16(buffered_data)
                .downsample(num_frames=frames, target_rate=16000)
                # .ring_mod(freq=40)
                .process()
            )
            outdata[:] = processed

        self.speaker_output_stream = sd.RawOutputStream(
            channels=1,
            samplerate=16000,  
            dtype="int16",
            callback=speaker_output_stream_callback,
            # device = 1
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
        self.last_interaction_stamp = self.node.get_clock().now().nanoseconds
        self.last_user_done_speaking_stamp = None

        t1 = self.node.get_clock().now().nanoseconds

        # self.node.get_logger().info("WAITING FOR STT READY...")
        # # wait for stt to be ready
        # await self.is_stt_ready.wait()

        self.node.get_logger().info(f"Conversation ready. Spend: {(t1-t0)/1e6:.2f} ms")

    ################################################################################################
    ############# ROBOT TASKS ###############################################################  
    async def run_robot_pre(self):
        '''Generate response/farewell/tool call from llm. If response, queue tts, play them, add to robot spoken buffer'''
        try:
            # prepare llm related actions
            stream = b.stream.PrelimAgent(messages=self.message_history, activities="None")
            parser = SemanticDeltaParser()

            # prepare tts related actions
            advance_sentence_piece_tts_task = asyncio.create_task(self.advance_sentence_piece_tts())
            self.next_allowed_tts_request_stamp = self.node.get_clock().now().nanoseconds

            # process streamed llm response
            async for partial in stream:
                if isinstance(partial,stream_types.ReplyTool):
                    if partial.response:
                        new_pieces, _ = parser.parse_new_input(partial.response) # IGNORE any final piece from llm not closed by separators 
                        self.queue_sentence_pieces_to_speak(new_pieces)
            

            # llm generation complete, add poinson pill to tts queue to end tts advance loop
            # await generation to be spoken
            self.sentence_piece_tts_queue.put_nowait(SentencePiecePoisonPill())
            await asyncio.gather(advance_sentence_piece_tts_task)

            final =  await stream.get_final_response()
            # Transition to next state based on final tool------------------------------------
            if isinstance(final, ReplyTool):
                self.handle_ROBOT_FINISHED()
            elif isinstance(final, StopTool):
                self.handle_ROBOT_FAREWELL()
            else:
                self.handle_ROBOT_REQUEST_TOOL(final)
            
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
    
    async def run_robot_tool_call(self, tool_call:BookActivityTool | SuggestActivityTool):
        
        if isinstance(tool_call, BookActivityTool):
            self.node.get_logger().info(f"## MOCK ROBOT TOOL CALL: BookActivityTool with params: {tool_call.activity}")
            await asyncio.sleep(0.5)
            new_tool_result_msg = Message(role='user', content=f"Tool call result: Successfully booked activity [{tool_call.activity}] for user.")
            self.message_history.append(new_tool_result_msg)

        elif isinstance(tool_call, SuggestActivityTool):
            self.node.get_logger().info(f"## MOCK ROBOT TOOL CALL: SuggestActivityTool with params: {tool_call.description}")
            await asyncio.sleep(0.5)
            new_tool_result_msg = Message(role='user', content=f"Tool call result: Successfully submitted activity suggestion [{tool_call.description}] to Oceania.")
            self.message_history.append(new_tool_result_msg)

        self.handle_ROBOT_GENERATE_RESPONSE()

    async def run_robot_aft(self):
        '''Generate response/farewell call from llm. If response, queue tts, play them, add to robot spoken buffer'''
        try:
            # prepare llm related actions
            stream = b.stream.RespondAgent(messages=self.message_history, activities="None")
            parser = SemanticDeltaParser()

            # prepare tts related actions
            advance_sentence_piece_tts_task = asyncio.create_task(self.advance_sentence_piece_tts())
            self.next_allowed_tts_request_stamp = self.node.get_clock().now().nanoseconds

            # process streamed llm response
            async for partial in stream:
                if isinstance(partial,stream_types.ReplyTool):
                    if partial.response:
                        new_pieces, _ = parser.parse_new_input(partial.response) # IGNORE any final piece from llm not closed by separators 
                        self.queue_sentence_pieces_to_speak(new_pieces)
            

            # llm generation complete, add poinson pill to tts queue to end tts advance loop
            # await generation to be spoken
            self.sentence_piece_tts_queue.put_nowait(SentencePiecePoisonPill())
            await asyncio.gather(advance_sentence_piece_tts_task)

            final =  await stream.get_final_response()
            # Transition to next state based on final tool------------------------------------
            if isinstance(final, ReplyTool):
                self.handle_ROBOT_FINISHED()
            elif isinstance(final, StopTool):
                self.handle_ROBOT_FAREWELL()
          
            
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
    
    async def run_robot_countdown(self):
        '''Say farewell message and start countdown to end conversation'''
        try:
            # prepare llm related actions
            stream = b.stream.FarewellAgent(messages=self.message_history)
            parser = SemanticDeltaParser()

            # prepare tts related actions
            advance_sentence_piece_tts_task = asyncio.create_task(self.advance_sentence_piece_tts())
            self.next_allowed_tts_request_stamp = self.node.get_clock().now().nanoseconds

            # process streamed llm response
            async for partial in stream:
                new_pieces, _ = parser.parse_new_input(partial) # IGNORE any final piece from llm not closed by separators 
                self.queue_sentence_pieces_to_speak(new_pieces)
            

            # llm generation complete, add poinson pill to tts queue to end tts advance loop
            # await generation to be spoken
            self.sentence_piece_tts_queue.put_nowait(SentencePiecePoisonPill())
            await asyncio.gather(advance_sentence_piece_tts_task)

            ### count down to transition:
            for i in range (5,0,-1):
                # TODO display on screen somehow
                self.node.get_logger().info(f"## COUNTDOWN END CONV: {i}")
                await asyncio.sleep(1)
            
            self.handle_ROBOT_END_CONVERSATION()
                

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
    
    async def run_conversation_shutdown(self):
        try:
            self.speaker_output_stream.abort()
            self.mic_stream_task.cancel()
            self.stt_stream_task.cancel()
            await self.tts_session.close()
        except Exception as e:
            self.node.get_logger().warning("Caught error when shutting down: "+str(e))
        
        self.handle_RESET_TO_IDLE()

    
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
            now_stamp = self.node.get_clock().now().nanoseconds
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
            self.node.get_logger().info("Begin mic stream, sending data")
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
                    self.mic_audio_queue.put_nowait, data_copy.tobytes()
                )

            input_stream = sd.InputStream(
                    samplerate=16000, 
                    # channels=1, 
                    blocksize=1280, # 80ms per block at 16kHz
                    dtype='float32',
                    callback=audio_callback,
                    device=1
                )
            input_stream.start()

        except asyncio.CancelledError:
            input_stream.stop()
            self.node.get_logger().info("Mic stream closed cleanly.")
            raise
   
    async def run_stt_stream(self):
        '''connect to deepgram flux stt service, send mic audio from mic_audio_queue, handle returned messages/events'''
        
        # if not self.stt_session:
        #     raise RuntimeError("Trying to listen to stt response without instantiating object first.")
        
        # # breaks gracefully when shutdown func called on stt_session
        # async for message in self.stt_session:
            
        #     # first utterance in user turn, reset predictor value
        #     if len(self.stt_word_buffer) == 0:
        #         self.stt_session.pause_predictor.value = 0

        #     self.stt_word_buffer.append(message.text)

        # self.node.get_logger().info("STT task finished.")

     
        try:
            self.node.get_logger().info("create stt client")
            client = AsyncDeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))
            listening_task = None
            self.node.get_logger().info("start setting up stt stream before try")
        # Connect to Flux with auto-detection for streaming audio
        # SDK automatically connects to: wss://api.deepgram.com/v2/listen?model=flux-general-en&encoding=linear16&sample_rate=16000
            self.node.get_logger().info("start setting up stt stream")
            async with client.listen.v2.connect(
                model="flux-general-en",
                eot_threshold="0.7",
                eot_timeout_ms="6000",

                encoding="linear32",
                sample_rate="16000"
            ) as connection:
                self.node.get_logger().info("connection setup")
                # Define message handler function

                def on_message(message: ListenV2SocketClientResponse) -> None:

                    msg_type = getattr(message, "type", "Unknown")
                    # self.node.get_logger().info(str(message))

                    # Show transcription results
                    # if hasattr(message, 'event') and message.event == "Update" and message.transcript:
                    #     self.stt_pulse_publisher.publish(self.create_std_str_msg("bru"))

                    if hasattr(message, 'event') and message.event == "StartOfTurn":
                        if self.state == ConversationState.ROBOT_TURN:
                            self.handle_INTERRUPT_ROBOT()
                    
                    elif hasattr(message, 'event') and message.event == "EndOfTurn":
                        self.node.get_logger().info(f"End turn msg: {str(message)}")
                        self.stt_word_buffer.append(message.transcript)
                        self.handle_USER_DONE_SPEAKING()

                    elif msg_type == "Connected":
                        self.is_stt_ready.set()
                        print(f"✅ Connected to Deepgram Flux - Ready for audio!")


                # Attach the message handler to the connection & start listening to event
                connection.on(EventType.MESSAGE, on_message)
                connection.on(EventType.OPEN, lambda _: self.node.get_logger().info("stt connection open"))
                connection.on(EventType.CLOSE, lambda _: self.node.get_logger().info("Connection closed"))
                connection.on(EventType.ERROR, lambda error: self.node.get_logger().info(f"Caught: {error}"))

                listening_task = asyncio.create_task(connection.start_listening())
        
                while True:
                    # Get audio data from mic audio queue
                    audio_data = await self.mic_audio_queue.get()
                    # Send audio data to Deepgram
                    await connection._send(audio_data)
        
        except asyncio.CancelledError:
            connection.send_control({"type": "CloseStream"})
            if listening_task:
                listening_task.cancel()
        
        except Exception as e:
               self.node.get_logger().info(str(e))
               self.node.get_logger().info(os.getenv("DEEPGRAM_API_KEY"))


        # finally:
   
        #     self.is_stt_ready.clear()



    ################################################################################################
    ############## Asyncio loop control util ############################################
    # def _start_asyncio_loop(self):
    #     asyncio.set_event_loop(self._loop)
    #     self._loop.run_forever()

    # def destroy_node(self):
    #     self._loop.call_soon_threadsafe(self._loop.stop)
    #     self._loop_thread.join()
    #     super().destroy_node()

    ################################################################################################
    ############# Util util, actual simply isolated shorthand stuff #################################
    def create_std_str_msg(self, message:str):
        msg = String()
        msg.data = message
        return msg
    
    
rclpy.init()
node = Node("conversation_manager_dg")
conversation_manager = ConversationManagerDG(node)


async def ros_loop():
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0)
        await asyncio.sleep(1e-4)  

def main():
    print("node started")
    asyncio.run(ros_loop())
    

if __name__ == '__main__':
    main()