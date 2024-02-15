
from pprint import pprint

import librosa
import numpy as np
import pyaudio
import requests
import torch
import whisperx

from audio_chunk_buffer import AudioChunkBuffer
from transcription_buffer import TranscriptionBuffer

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

# Model configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float32" if torch.cuda.is_available() else "int8"
LANGUAGE = "fr"

audio_model = whisperx.load_model(
    "small",
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    language=LANGUAGE,
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

    channels = metadata["channels"]
    rate = metadata["sample_rate"]
    chunk_size = metadata["chunk_size"]

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT, channels=TARGET_CHANNELS, rate=TARGET_RATE, output=True
    )

    with requests.get(audio_stream_url, stream=True) as r:
        r.raise_for_status()
        data_window = AudioChunkBuffer(maxlen=3)
        transcripts_buffer = TranscriptionBuffer(window_duration=15)
        data = b""
        time_padding = 0
        for chunk in r.iter_content(chunk_size=chunk_size):
            data += chunk

            if len(data) < chunk_size:
                continue

            if len(data_window) == 3:
                time_padding += 5

            data_window.add_chunk(data)
            window_data = data_window.get_audio_data()

            data_np = audio_bytes_to_np(data, to_mono=channels == 2)
            data_np = librosa.resample(data_np, orig_sr=rate, target_sr=TARGET_RATE)

            full_data_np = audio_bytes_to_np(window_data, to_mono=channels == 2)
            full_data_np = librosa.resample(
                full_data_np, orig_sr=rate, target_sr=TARGET_RATE
            )

            trs = audio_model.transcribe(full_data_np, batch_size=8)
            model_a, metadata = whisperx.load_align_model(
                language_code=trs["language"], device=DEVICE
            )
            trs_aligned = whisperx.align(
                trs["segments"],
                model_a,
                metadata,
                full_data_np,
                device=DEVICE,
                return_char_alignments=False,
            )

            texts = []

            stream.write(full_data_np.tobytes())

            for segment in trs_aligned["segments"]:
                words = []
                last_end = segment["start"]
                for i, word in enumerate(segment["words"]):
                    start = last_end
                    end = None
                    if "start" in word:
                        start = word["start"]
                    if "end" in word:
                        end = word["end"]
                    else:
                        for j in range(i + 1, len(segment["words"])):
                            if "start" in segment["words"][j]:
                                end = segment["words"][j]["start"]
                                break
                            elif "end" in segment["words"][j]:
                                end = segment["words"][j]["end"]
                                break
                    if end is None:
                        end = segment["end"]
                    last_end = end
                    words.append(
                        {
                            "word": word["word"],
                            "start": start + time_padding,
                            "end": end + time_padding,
                        }
                    )
                texts.append(
                    {
                        "text": segment["text"],
                        "start": segment["start"] + time_padding,
                        "end": segment["end"] + time_padding,
                        "words": words,
                    }
                )

            transcripts_buffer.add_transcription(texts)

            pprint(transcripts_buffer.get_continuous_transcription())

            data = b""

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    file_name = "devenir-riche.wav"
    audio_metadata_url = f"http://{HOST}:{PORT}/audio/metadata/{file_name}"
    audio_stream_url = f"http://{HOST}:{PORT}/stream/audio?file_name={file_name}"
    main(audio_metadata_url, audio_stream_url)
