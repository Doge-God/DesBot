import sherpa_onnx
import os
import pyaudio
import numpy as np
import sounddevice as sd

def main():
    print("> Creating recognizer.")
    # recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    #     tokens=os.path.expanduser("~/Documents/sherpa-onnx/sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt"),
    #     encoder=os.path.expanduser("~/Documents/sherpa-onnx/sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-128.onnx"),
    #     decoder=os.path.expanduser("~/Documents/sherpa-onnx/sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-128.onnx"),
    #     joiner=os.path.expanduser("~/Documents/sherpa-onnx/sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-128.onnx"),
    #     num_threads=2,
    #     sample_rate=16000,
    #     feature_dim=80,
    #     decoding_method="modified_beam_search", #greedy_search
    #     max_active_paths=4, #used for modified_beam_search
    #     provider="cpu",
    #     hotwords_file="",
    #     hotwords_score=1.5,
    #     blank_penalty=0.0,
    #     hr_dict_dir="",
    #     hr_rule_fsts="",
    #     hr_lexicon=""
    # )

    # recognizer = sherpa_onnx.OnlineRecognizer.from_nemo_ctc(
    #     tokens=os.path.expanduser("~/Documents/sherpa-onnx/sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8/tokens.txt"),
    #     model=os.path.expanduser("~/Documents/sherpa-onnx/sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8/model.int8.onnx"),
    # )
    # recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
    #     tokens=os.path.expanduser("~/Documents/sherpa-onnx/sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt"),
    #     encoder=os.path.expanduser("~/Documents/sherpa-onnx/sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.onnx"),
    #     decoder=os.path.expanduser("~/Documents/sherpa-onnx/sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.onnx"),
    # )

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=os.path.expanduser("~/Documents/sherpa-onnx/sherpa-onnx-lstm-en-2023-02-17/tokens.txt"),
        encoder=os.path.expanduser("~/Documents/sherpa-onnx/sherpa-onnx-lstm-en-2023-02-17/encoder-epoch-99-avg-1.onnx"),
        decoder=os.path.expanduser("~/Documents/sherpa-onnx/sherpa-onnx-lstm-en-2023-02-17/decoder-epoch-99-avg-1.onnx"),
        joiner=os.path.expanduser("~/Documents/sherpa-onnx/sherpa-onnx-lstm-en-2023-02-17/joiner-epoch-99-avg-1.onnx"),
    )
    
    print("> Recognizer ready.")



    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
        return

    print(devices)
    default_input_device_idx = 0# sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    print("Started! Please speak")

    # The model is using 16 kHz, we use 48 kHz here to demonstrate that
    # sherpa-onnx will do resampling inside.
    sample_rate = 48000
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms
    last_result = ""
    stream = recognizer.create_stream()
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            stream.accept_waveform(sample_rate, samples)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            result = recognizer.get_result(stream)
            if last_result != result:
                last_result = result
                print("\r{}".format(result), end="", flush=True)
            


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")