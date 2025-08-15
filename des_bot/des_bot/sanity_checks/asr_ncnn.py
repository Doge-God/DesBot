import sherpa_ncnn
import os
import pyaudio
import numpy as np
import sounddevice as sd

def main():
    print("> Creating recognizer.")


    # recognizer = sherpa_ncnn.Recognizer(
    #     tokens=os.path.expanduser("~/Documents/sherpa-ncnn/sherpa-ncnn-conv-emformer-transducer-2022-12-04/tokens.txt"),
    #     encoder_param=os.path.expanduser("~/Documents/sherpa-ncnn/sherpa-ncnn-conv-emformer-transducer-2022-12-04/encoder_jit_trace-pnnx.ncnn.param"),
    #     encoder_bin=os.path.expanduser("~/Documents/sherpa-ncnn/sherpa-ncnn-conv-emformer-transducer-2022-12-04/encoder_jit_trace-pnnx.ncnn.bin"),
    #     decoder_param=os.path.expanduser("~/Documents/sherpa-ncnn/sherpa-ncnn-conv-emformer-transducer-2022-12-04/decoder_jit_trace-pnnx.ncnn.param"),
    #     decoder_bin=os.path.expanduser("~/Documents/sherpa-ncnn/sherpa-ncnn-conv-emformer-transducer-2022-12-04/decoder_jit_trace-pnnx.ncnn.bin"),
    #     joiner_param=os.path.expanduser("~/Documents/sherpa-ncnn/sherpa-ncnn-conv-emformer-transducer-2022-12-04/joiner_jit_trace-pnnx.ncnn.param"),
    #     joiner_bin=os.path.expanduser("~/Documents/sherpa-ncnn/sherpa-ncnn-conv-emformer-transducer-2022-12-04/joiner_jit_trace-pnnx.ncnn.bin"),
    #     num_threads=4,
    #     hotwords_file="",
    #     hotwords_score=1.5,
    # )

    recognizer = sherpa_ncnn.Recognizer(
        tokens=os.path.expanduser("~/Documents/sherpa-ncnn/sherpa-ncnn-2022-09-05/tokens.txt"),
        encoder_param=os.path.expanduser("~/Documents/sherpa-ncnn/sherpa-ncnn-2022-09-05/encoder_jit_trace-pnnx.ncnn.param"),
        encoder_bin=os.path.expanduser("~/Documents/sherpa-ncnn/sherpa-ncnn-2022-09-05/encoder_jit_trace-pnnx.ncnn.bin"),
        decoder_param=os.path.expanduser("~/Documents/sherpa-ncnn/sherpa-ncnn-2022-09-05/decoder_jit_trace-pnnx.ncnn.param"),
        decoder_bin=os.path.expanduser("~/Documents/sherpa-ncnn/sherpa-ncnn-2022-09-05/decoder_jit_trace-pnnx.ncnn.bin"),
        joiner_param=os.path.expanduser("~/Documents/sherpa-ncnn/sherpa-ncnn-2022-09-05/joiner_jit_trace-pnnx.ncnn.param"),
        joiner_bin=os.path.expanduser("~/Documents/sherpa-ncnn/sherpa-ncnn-2022-09-05/joiner_jit_trace-pnnx.ncnn.bin"),
        num_threads=4,
        hotwords_file="",
        hotwords_score=1.5,
    )

    print("> Recognizer ready.")
    
    sample_rate = 16000
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms
    last_result = ""
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            recognizer.accept_waveform(sample_rate, samples)
            result = recognizer.text
            if last_result != result:
                last_result = result
                print("\r{}".format(result), end="", flush=True)


if __name__ == "__main__":
    devices = sd.query_devices()
    print(devices)
    default_input_device_idx = 0#sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
