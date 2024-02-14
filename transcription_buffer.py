"""
This file contains the implementation of the TranscriptionBuffer class.
The TranscriptionBuffer class is used to store and merge overlapping transcriptions
without overlapping them. We need a bufferized solution because Whisperx does not
support real-time transcription or incremental transcription.
"""

from math import isclose
from copy import deepcopy
from typing import Any, Dict, List


class TranscriptionBuffer:
    """
    The TranscriptionBuffer class.
    Stores, merges and returns transcriptions.
    """

    def __init__(self, window_duration: float = 15.0, chunk_duration: float = 5.0):
        """
        Initializes the TranscriptionBuffer object.

        Args:
            buffer_duration (float): The duration of the buffer in seconds.
        """
        self.window_duration = window_duration
        self.chunk_duration = chunk_duration
        self.buffer_duration = 0.0
        self.equality_treshold = 0.2
        self.buffer = []

    def get_closest_segment(
        self, query_segment: Dict[str, Any], candidate_segments: List[Dict[str, Any]]
    ) -> List[tuple[int, Dict[str, Any], float]]:
        """
        Returns the segments sorted by distance to the segment.

        Args:
            query_segment (Dict[str, Any]): The segment to find in the segments.
            candidate_segments (List[Dict[str, Any]]): The segments to search in.

        Returns:
            List[tuple[int, Dict[str, Any], float]]: The segments sorted by distance to the segment.
        """

        if not candidate_segments or len(candidate_segments) == 0:
            raise ValueError("No candidate segments provided")

        candidate_segments_sorted = []

        for i, candidate_segment in enumerate(candidate_segments):
            distance = abs(candidate_segment["start"] - query_segment["start"])
            candidate_segments_sorted.append((i, candidate_segment, distance))

        candidate_segments_sorted = sorted(
            candidate_segments_sorted, key=lambda x: x[2]
        )

        return candidate_segments_sorted

    def get_segment_by_words(
        self,
        query_segment: Dict[str, Any],
        candidate_segments: List[tuple[int, Dict[str, Any], float]],
    ) -> tuple[int, Dict[str, Any]]:
        """
        Finds the best matching segment in the candidate_segments.
        We iterate through the words of the query_segment and search for the same word with similar
        timestamps in the candidate_segments in the candidate_segments.
        We fall back to an identical word match if we don't find a similar timestamp match.
        If that fails, we fallback to the first segment in the candidate_segments.

        Args:
            query_segment (Dict[str, Any]): The segment to find in the searched_segment.
            candidate_segments (List[Dict[str, Any]]): The segments to search in.
        Returns:
            tuple[int | None, Dict[str, Any] | None]: The index of the matching segment and the matching segment.
        """
        if not candidate_segments or len(candidate_segments) == 0:
            raise ValueError("No candidate segments provided")

        matching_words = []
        segment_index = 0
        fallback_matching_words = []
        fallback_segment_index = 0

        for c_i, candidate_segment, _ in candidate_segments:
            query_words = query_segment["words"]
            searched_words = candidate_segment["words"]

            for q_i in range(len(query_words)):
                query_word = query_words[q_i]
                for i, searched_word in enumerate(searched_words):
                    q_word = query_word["word"].lower().strip(".,!?")
                    s_word = searched_word["word"].lower().strip(".,!?")

                    if q_word == s_word:
                        if len(fallback_matching_words) == 0:
                            fallback_matching_words = (
                                query_words[:q_i] + searched_words[i:]
                            )
                            fallback_segment_index = c_i

                        if isclose(
                            searched_word["start"],
                            query_word["start"],
                            abs_tol=self.equality_treshold,
                        ):
                            matching_words = query_words[:q_i] + searched_words[i:]
                            segment_index = c_i
                            break
                if len(matching_words) > 0:
                    break

        if len(matching_words) == 0:
            matching_words = fallback_matching_words
            segment_index = fallback_segment_index

        segment = {
            "text": " ".join([word["word"] for word in matching_words]),
            "start": matching_words[0]["start"],
            "end": matching_words[-1]["end"],
            "words": matching_words,
        }

        return segment_index, segment

    def add_transcription(self, segments: List[Dict[str, Any]]):
        """
        Adds a transcription to the buffer. The total duration of the segments
        should be around self.window_duration unless the cumulative audio received
        is less than self.window_duration.

        If the cumulative audio received is less than self.window_duration, the
        segments will just replace the current buffer.

        Args:
            segments (List[Dict[str, Any]]): The segments of the transcription.
        """

        if self.buffer_duration >= self.window_duration or isclose(
            self.buffer_duration,
            self.window_duration,
            abs_tol=self.equality_treshold,
        ):
            last_segment = self.buffer[-1]
            closest_segments = self.get_closest_segment(last_segment, segments)

            matching_segment_index, matching_segment = self.get_segment_by_words(
                last_segment, closest_segments
            )
            self.buffer[-1] = matching_segment
            if matching_segment_index + 1 < len(segments):
                self.buffer.extend(segments[matching_segment_index + 1 :])

        else:
            self.buffer = deepcopy(segments)

        self.buffer_duration += self.chunk_duration

    def get_continuous_transcription(self) -> str:
        """
        Returns the continuous transcription.

        Returns:
            str: The continuous transcription.
        """
        trs_text = " ".join([segment["text"] for segment in self.buffer])
        return trs_text
