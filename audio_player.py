import socket
import time

import numpy as np
import requests
import pyaudio
import torch
import whisperx
from transformers.models.pop2piano.feature_extraction_pop2piano import librosa

# Audio stream configuration
CHUNK = 960000
FORMAT = pyaudio.paFloat32
TARGET_CHANNELS = 1
CHANNELS = 1
RATE = 48000
TARGET_RATE = 16000
PCM_16BIT_MAX = 32768.0

# Networking configuration
HOST = "localhost"
PORT = 5000

audio_model = whisperx.load_model(
    "tiny",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="int8",
    language="fr",
)


def audio_bytes_to_np(data: bytes, to_mono: bool = True):
    audio_array = np.frombuffer(data, dtype=np.int16)

    if to_mono:
        audio_array = audio_array.reshape((-1, 2))[:, 0]

    return audio_array.astype(np.float32) / PCM_16BIT_MAX


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
            audio_array = audio_bytes_to_np(full_data).astype(np.float32)
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

def fetch_audio_metadata(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    return response.json()

def main(audio_metadata_url, audio_stream_url):
    # Fetch audio metadata
    metadata = fetch_audio_metadata(audio_metadata_url)

    rate = metadata['sample_rate']
    chunk_size = metadata['chunk_size']

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=TARGET_CHANNELS,
                    rate=TARGET_RATE,
                    output=True)

    # Stream and play the audio
    with requests.get(audio_stream_url, stream=True) as r:
        r.raise_for_status()
        full_data = b""
        data = b""
        for chunk in r.iter_content(chunk_size=chunk_size):
            # If stereo, convert to mono by selecting one channel
            data += chunk

            if len(data) < chunk_size:
                continue

            full_data += data

            full_data_np = audio_bytes_to_np(full_data, to_mono=metadata['channels'] == 2)
            full_data_np = librosa.resample(full_data_np, orig_sr=rate, target_sr=TARGET_RATE)

            # Play audio chunk
            trs = audio_model.transcribe(full_data_np, batch_size=8)
            print(trs["segments"])

            stream.write(full_data_np.tobytes())
            time.sleep(2)
            data = b""

    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    # play_stream()
    file_name = "devenir-riche.wav"
    audio_metadata_url = f"http://{HOST}:{PORT}/audio/metadata/{file_name}"
    audio_stream_url = f"http://{HOST}:{PORT}/stream/audio?file_name={file_name}"
    main(audio_metadata_url, audio_stream_url)
