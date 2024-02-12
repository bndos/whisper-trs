import time

import numpy as np
import pyaudio
import requests
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


def fetch_audio_metadata(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def main(audio_metadata_url, audio_stream_url):
    # Fetch audio metadata
    metadata = fetch_audio_metadata(audio_metadata_url)

    rate = metadata["sample_rate"]
    chunk_size = metadata["chunk_size"]

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT, channels=TARGET_CHANNELS, rate=TARGET_RATE, output=True
    )

    with requests.get(audio_stream_url, stream=True) as r:
        r.raise_for_status()
        full_data = b""
        data = b""
        for chunk in r.iter_content(chunk_size=chunk_size):
            data += chunk

            if len(data) < chunk_size:
                continue

            full_data += data

            data_np = audio_bytes_to_np(data, to_mono=metadata["channels"] == 2)
            data_np = librosa.resample(
                data_np, orig_sr=rate, target_sr=TARGET_RATE
            )

            full_data_np = audio_bytes_to_np(
                full_data, to_mono=metadata["channels"] == 2
            )
            full_data_np = librosa.resample(
                full_data_np, orig_sr=rate, target_sr=TARGET_RATE
            )

            trs = audio_model.transcribe(full_data_np, batch_size=8)
            print(trs["segments"])

            stream.write(data_np.tobytes())
            time.sleep(2)
            data = b""

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    file_name = "devenir-riche.wav"
    audio_metadata_url = f"http://{HOST}:{PORT}/audio/metadata/{file_name}"
    audio_stream_url = f"http://{HOST}:{PORT}/stream/audio?file_name={file_name}"
    main(audio_metadata_url, audio_stream_url)
