# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "msgpack",
#     "numpy",
#     "sounddevice",
#     "websockets",
# ]
# ///
import argparse
import asyncio
import signal

import msgpack
import numpy as np
import sounddevice as sd
import websockets
import librosa

SAMPLE_RATE = 16000
TARGET_RATE = 24000

# The VAD has several prediction heads, each of which tries to determine whether there
# has been a pause of a given length. The lengths are 0.5, 1.0, 2.0, and 3.0 seconds.
# Lower indices predict pauses more aggressively. In Unmute, we use 2.0 seconds = index 2.
PAUSE_PREDICTION_HEAD_INDEX = 2


async def receive_messages(websocket, show_vad: bool = False):
    """Receive and process messages from the WebSocket server."""
    try:
        speech_started = False
        async for message in websocket:
            data = msgpack.unpackb(message, raw=False)

            # The Step message only gets sent if the model has semantic VAD available
            if data["type"] == "Step" and show_vad:
                pause_prediction = data["prs"][PAUSE_PREDICTION_HEAD_INDEX]
                if pause_prediction > 0.8 and speech_started:
                    print("| ", end="", flush=True)
                    speech_started = False

            elif data["type"] == "Word":
                print(data["text"], end=" ", flush=True)
                speech_started = True
    except websockets.ConnectionClosed:
        print("Connection closed while receiving messages.")


async def send_messages(websocket, audio_queue):
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

async def stream_audio(url: str, api_key: str, show_vad: bool):
    """Stream audio data to a WebSocket server (resampled to 24 kHz with librosa)."""
    print("Starting microphone recording...")
    print("Press Ctrl+C to stop recording")
    audio_queue = asyncio.Queue()

    loop = asyncio.get_event_loop()

    def audio_callback(indata, frames, time, status):
        mono_audio = indata[:, 0]  # float32
        # Resample to 24kHz with librosa
        resampled = librosa.resample(
            mono_audio, orig_sr=SAMPLE_RATE, target_sr=TARGET_RATE
        ).astype(np.float32)
        loop.call_soon_threadsafe(audio_queue.put_nowait, resampled.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=1920,  # 80ms
    ):
        headers = {"kyutai-api-key": api_key}
        async with websockets.connect(url, additional_headers=headers) as websocket:
            send_task = asyncio.create_task(send_messages(websocket, audio_queue))
            receive_task = asyncio.create_task(
                receive_messages(websocket, show_vad=show_vad)
            )
            await asyncio.gather(send_task, receive_task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time microphone transcription")
    parser.add_argument(
        "--url",
        help="The URL of the server to which to send the audio",
        default="ws://192.168.137.1:8080",
    )
    parser.add_argument("--api-key", default="public_token")
    parser.add_argument(
        "--list-devices", action="store_true", help="List available audio devices"
    )
    parser.add_argument(
        "--device", type=int, help="Input device ID (use --list-devices to see options)"
    )
    parser.add_argument(
        "--show-vad",
        action="store_true",
        help="Visualize the predictions of the semantic voice activity detector with a '|' symbol",
    )

    args = parser.parse_args()

    def handle_sigint(signum, frame):
        print("Interrupted by user")  # Don't complain about KeyboardInterrupt
        exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    if args.list_devices:
        print("Available audio devices:")
        print(sd.query_devices())
        exit(0)

    if args.device is not None:
        sd.default.device[0] = args.device  # Set input device

    url = f"{args.url}/api/asr-streaming"
    asyncio.run(stream_audio(url, args.api_key, args.show_vad))