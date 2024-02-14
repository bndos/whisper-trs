from flask import Flask, Response, request, jsonify
import os
import socket
import wave

import wave


def get_audio_file_properties(file_path):
    with wave.open(file_path, "rb") as wf:
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
            "duration": duration,
        }


def calculate_chunk_size(sample_rate, channels, sample_width, duration_seconds):
    bytes_per_frame = channels * sample_width
    frames_per_second = sample_rate
    frames_for_duration = frames_per_second * duration_seconds
    bytes_for_duration = frames_for_duration * bytes_per_frame
    return bytes_for_duration


audio_files_path = "audio_files/"
# properties = get_audio_file_properties(audio_file_path)
# duration_seconds = 5
# chunk_size = calculate_chunk_size(
#     properties["sample_rate"],
#     properties["channels"],
#     properties["sample_width"],
#     duration_seconds,
# )


def calculate_chunk_frames(sample_rate, duration_seconds):
    frames_for_duration = sample_rate * duration_seconds
    return frames_for_duration


# chunk_frames = calculate_chunk_frames(properties["sample_rate"], duration_seconds)

# Networking configuration
HOST = "localhost"
PORT = 65432


def stream_audio(file_path, chunk_frames=1024):
    with wave.open(file_path, "rb") as wf:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            data = wf.readframes(chunk_frames)
            while data != b"":
                s.sendall(data)
                data = wf.readframes(chunk_frames)


app = Flask(__name__)

@app.route("/audio/metadata/<string:file_name>", methods=["GET"])
def audio_metadata(file_name: str):
    file_path = os.path.join(audio_files_path, file_name)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found."}), 404

    try:
        properties = get_audio_file_properties(file_path)
        chunk_size = calculate_chunk_size(
            properties["sample_rate"],
            properties["channels"],
            properties["sample_width"],
            5,
        )
        properties["chunk_size"] = chunk_size
        return jsonify(properties)
    except wave.Error as e:
        return jsonify({"error": str(e)}), 400

@app.route("/stream/audio")
def stream_audio_route():
    file_name = request.args.get("file_name")
    if not file_name:
        return jsonify({"error": "File name is required."}), 400

    file_path = os.path.join(audio_files_path, file_name)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found."}), 404

    def generate_audio_stream(file_path: str, chunk_frames: int = 1024):
        with wave.open(file_path, "rb") as wf:
            data = wf.readframes(chunk_frames)
            while data:
                yield data
                data = wf.readframes(chunk_frames)

    properties = get_audio_file_properties(file_path)
    chunk_frames = calculate_chunk_frames(properties["sample_rate"], 5)

    return Response(generate_audio_stream(file_path, chunk_frames), mimetype="audio/wav")

if __name__ == "__main__":
    app.run(debug=True)
