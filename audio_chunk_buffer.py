"""
This file contains the implementation of the AudioChunkBuffer class.
The AudioChunkBuffer class is used to store and merge a queue of audio chunks
with a maxlen of n chunks. The AudioChunkBuffer class is used to store and merge
the chunks together so that we keep a window of n audio chunks in memory at
all times, pushing the oldest chunk out of the buffer when the buffer is full.
"""

from collections import deque

class AudioChunkBuffer:
    """
    The AudioChunkBuffer class.
    Stores, merges and returns audio chunks
    in a buffer.

    Args:
        maxlen (int): The maximum length of the buffer.
    """
    def __init__(self, maxlen: int = 3):
        self.buffer = deque(maxlen=maxlen)

    def add_chunk(self, chunk: bytes):
        """
        Adds a chunk to the buffer.

        Args:
            chunk (bytes): The chunk to add to the buffer.
        """
        self.buffer.append(chunk)

    def get_audio_data(self) -> bytes:
        """
        Merges all the chunks in the buffer and returns the merged
        audio data.

        Returns:
            bytes: The merged audio data.
        """
        return b"".join(self.buffer)

    def clear(self):
        """
        Clears the buffer.
        """
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
