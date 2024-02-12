import socket
import time

import numpy as np
import pyaudio
import torch
import whisperx
from transformers.models.pop2piano.feature_extraction_pop2piano import librosa

# Audio stream configuration
CHUNK = 960000
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 48000
TARGET_RATE = 16000
PCM_16BIT_MAX = 32768.0

# Networking configuration
HOST = "localhost"
PORT = 65432

audio_model = whisperx.load_model(
    "tiny",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="int8",
    language="fr",
)


def stereo_to_mono(data):
    audio_array = np.frombuffer(data, dtype=np.int16)
    mono_array = audio_array.reshape((-1, 2))[:, 0].astype(np.float32)

    return mono_array / PCM_16BIT_MAX


def play_stream():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=TARGET_RATE, output=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        print(f"Connected by {addr}")
        full_data = b""

        while True:
            data = b""
            while len(data) < CHUNK:
                packet = conn.recv(CHUNK - len(data))
                if not packet:
                    break
                data += packet

            if not data:
                break

            full_data += data
            audio_array = stereo_to_mono(full_data).astype(np.float32)
            audio_np_16k = librosa.resample(
                audio_array, orig_sr=RATE, target_sr=TARGET_RATE
            )

            stream.write(audio_np_16k.tobytes())
            trs = audio_model.transcribe(audio_np_16k, batch_size=8)
            print(trs["segments"])

            time.sleep(3)

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    play_stream()
