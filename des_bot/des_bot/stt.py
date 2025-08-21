#!/usr/bin/env python3
import asyncio
import threading
import librosa
import msgpack
import websockets
import numpy as np
import sounddevice as sd
import rclpy
from rclpy.node import Node
from des_bot_interfaces.srv import ControlSttState


class ControlSttStateService(Node):
    def __init__(self):
        super().__init__('control_stt_state_server')

        self.SAMPLE_RATE = 16000
        self.TARGET_RATE = 24000

        # The VAD has several prediction heads, each of which tries to determine whether there
        # has been a pause of a given length. The lengths are 0.5, 1.0, 2.0, and 3.0 seconds.
        # Lower indices predict pauses more aggressively. In Unmute, we use 2.0 seconds = index 2.
        self.PAUSE_PREDICTION_HEAD_INDEX = 2

        self.is_ready = False

        # Create the service
        self.srv = self.create_service(
            ControlSttState,
            'control_stt_state',
            self.service_callback
        )

        self.streaming_thread = None
        self.stream_task = None
        self.send_task = None
        self.receive_task = None
        self.ready_event = threading.Event()
        self.recognized_utterances = []

        self.get_logger().info('ControlSttState service is ready.')


    def service_callback(self, request, response):
        """
        request: ControlSttState.Request
        response: ControlSttState.Response
        """
        self.get_logger().info(f'Received request: targetState={request.target_state}')

        if request.target_state:
            self.streaming_thread = threading.Thread(
                target=lambda: asyncio.run(self.stream_audio(
                    url="ws://192.168.137.1:8080/api/asr-streaming",
                    api_key="public_token",
                    show_vad=True
                )),
                daemon=True
            ).start()

            try:
                self.ready_event.wait(timeout=3)
                response.is_ready = True
            except TimeoutError:
                response.is_ready = False
                self.ready_event.clear()
            
        else:
            if self.send_task:
                self.send_task.cancel()
            if self.receive_task:
                self.receive_task.cancel()
            self.ready_event.clear()
            response.is_ready = True

        # Now wait for ready event
      
        return response


    async def receive_messages(self, websocket, show_vad: bool = False):
        """Receive and process messages from the WebSocket server."""
        try:
            speech_started = False
            async for message in websocket:
                data = msgpack.unpackb(message, raw=False)
                # print(data)

                # The Step message only gets sent if the model has semantic VAD available
                if data["type"] == "Step" and show_vad:
                    pause_prediction = data["prs"][self.PAUSE_PREDICTION_HEAD_INDEX]
                    if pause_prediction > 0.7 and speech_started:
                        print("| ", end="", flush=True)
                        speech_started = False

                elif data["type"] == "Word":
                    print(data["text"], end=" ", flush=True)
                    speech_started = True

                elif data["type"] == "Ready":
                    print("HEARD READY EVENT")
                    self.ready_event.set()


        except websockets.ConnectionClosed:
            print("Connection closed while receiving messages.")


    async def send_messages(self, websocket, audio_queue):
        """Send audio data from microphone to WebSocket server."""
        try:
            # Start by draining the queue to avoid lags
            while not audio_queue.empty():
                await audio_queue.get()

            print("Starting the transcription")

            while True:
                audio_data = await audio_queue.get()
                chunk = {"type": "Audio", "pcm": [float(x) for x in audio_data]}
                msg = msgpack.packb(chunk, use_single_float=True) #use_bin_type=True
                await websocket.send(msg)

        except websockets.ConnectionClosed:
            print("Connection closed while sending messages.")

    async def stream_audio(self, url: str, api_key: str, show_vad: bool):

        self.is_ready = False
        """Stream audio data to a WebSocket server (resampled to 24 kHz with librosa)."""
        print("Starting microphone recording...")
        print("Press Ctrl+C to stop recording")
        audio_queue = asyncio.Queue()

        loop = asyncio.get_event_loop()

        def audio_callback(indata, frames, time, status):
            mono_audio = indata[:, 0]  # float32
            # Resample to 24kHz with librosa
            resampled = librosa.resample(
                mono_audio, orig_sr=self.SAMPLE_RATE, target_sr=self.TARGET_RATE
            ).astype(np.float32)
            loop.call_soon_threadsafe(audio_queue.put_nowait, resampled.copy())

        with sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=audio_callback,
            blocksize=1920,  # 80ms
        ):
            headers = {"kyutai-api-key": api_key}
            async with websockets.connect(url, additional_headers=headers) as websocket:
                self.send_task = asyncio.create_task(self.send_messages(websocket, audio_queue))
                self.receive_task = asyncio.create_task(
                    self.receive_messages(websocket, show_vad=show_vad)
                )
                await asyncio.gather(self.send_task, self.receive_task)



def main(args=None):
    rclpy.init(args=args)
    node = ControlSttStateService()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
