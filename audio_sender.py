# streamer.py
import socket
import wave

import wave

def get_audio_file_properties(file_path):
    with wave.open(file_path, 'rb') as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()  # In bytes
        num_frames = wf.getnframes()
        duration = num_frames / sample_rate

        return {
            "channels": channels,
            "sample_rate": sample_rate,
            "sample_width": sample_width,
            "num_frames": num_frames,
            "duration": duration
        }

def calculate_chunk_size(sample_rate, channels, sample_width, duration_seconds):
    bytes_per_frame = channels * sample_width
    frames_per_second = sample_rate
    frames_for_duration = frames_per_second * duration_seconds
    bytes_for_duration = frames_for_duration * bytes_per_frame
    return bytes_for_duration

audio_file_path = 'audio_files/devenir-riche.wav'
properties = get_audio_file_properties(audio_file_path)
duration_seconds = 5
chunk_size = calculate_chunk_size(
    properties["sample_rate"],
    properties["channels"],
    properties["sample_width"],
    duration_seconds
)

def calculate_chunk_frames(sample_rate, duration_seconds):
    frames_for_duration = sample_rate * duration_seconds
    return frames_for_duration

chunk_frames = calculate_chunk_frames(
    properties["sample_rate"],
    duration_seconds
)

# Networking configuration
HOST = 'localhost'
PORT = 65432

def stream_audio(file_path):
    with wave.open(file_path, 'rb') as wf:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            data = wf.readframes(chunk_frames)
            while data != b'':
                s.sendall(data)
                data = wf.readframes(chunk_frames)

if __name__ == "__main__":
    stream_audio(audio_file_path)
