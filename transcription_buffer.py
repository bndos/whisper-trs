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

    def __init__(self, window_duration: float = 15):
        """
        Initializes the TranscriptionBuffer object.

        Args:
            buffer_duration (float): The duration of the buffer in seconds.
        """
        self.window_duration = window_duration
        self.buffer_duration = 0.0
        self.equality_treshold = 0.2
        self.buffer = []

    def get_closest_segment(
        self, segment: Dict[str, Any], segments: List[Dict[str, Any]]
    ) -> tuple[int, Dict[str, Any] | None]:
        """
        Returns the closest segment to the given segment.

        Args:
            segment (Dict[str, Any]): The segment to find the closest segment to.

        Returns:
            tuple[int, Dict[str, Any]]: The index of the closest segment and the closest segment.
        """

        if not segments or len(segments) == 0:
            return 0, None

        close_segment = None
        close_segment_index = 0

        fallback_segment = segments[0]
        fallback_segment_distance = float("inf")
        fallback_segment_index = 0

        for i, candidate_segment in enumerate(segments):
            distance = abs(candidate_segment["start"] - segment["start"])

            if (
                distance < fallback_segment_distance
                and candidate_segment["start"] < segment["start"]
            ):
                fallback_segment = candidate_segment
                fallback_segment_index = i
                fallback_segment_distance = distance
            if isclose(
                candidate_segment["start"],
                segment["start"],
                abs_tol=self.equality_treshold,
            ):
                close_segment = candidate_segment
                close_segment_index = i
                break

        if close_segment:
            return close_segment_index, close_segment

        # print("-------------------")
        # print("-------------------")
        # print("-------------------")
        # print(f"Closest segment: {fallback_segment}")
        # print(f"Query segment: {segment}")
        # print(f"Segments: {segments}")

        return fallback_segment_index, fallback_segment

    def get_segment_by_words(
        self, query_segment: Dict[str, Any], searched_segment: Dict[str, Any]
    ) -> Dict[str, Any] | None:
        """
        Finds the query_segment in the searched_segment by words.
        We use the first word of the query_segment to find matches in the searched_segment.
        We then verify which word has a similar start time and end time to the query_segment's word.

        Args:
            query_segment (Dict[str, Any]): The segment to find in the searched_segment.
            searched_segment (Dict[str, Any]): The segment to search in.
        Returns:
            List[Dict[str, Any]]: The reconstructed segment with the matching words.
        """
        query_words = query_segment["words"]
        searched_words = searched_segment["words"]

        if not query_words or not searched_words:
            return None

        matching_words = []
        # iterate over query words at 4 places
        # first place is 0, second place is 1/4, third place is 2/4, fourth place is 3/4
        for q_i in range(0, len(query_words), len(query_words) // 4):
            query_word = query_words[q_i]
            for i, searched_word in enumerate(searched_words):
                q_word = query_word["word"].lower().strip(".,!?")
                s_word = searched_word["word"].lower().strip(".,!?")
                if q_word == s_word:
                    # print("-------------------")
                    # print("-------------------")
                    # print("-------------------")
                    # print(searched_word["start"], query_word["start"])
                    # print(query_segment["text"])
                    # print(searched_segment["words"][i:])
                    # print(
                    #     isclose(
                    #         searched_word["start"],
                    #         query_word["start"],
                    #         abs_tol=self.equality_treshold,
                    #     )
                    # )
                    if isclose(
                        searched_word["start"],
                        query_word["start"],
                        abs_tol=self.equality_treshold,
                    ):
                        matching_words = searched_segment["words"][i:]
                        break
        if len(matching_words) == 0:
            return None

        segment = {
            "text": " ".join([word["word"] for word in matching_words]),
            "start": matching_words[0]["start"],
            "end": matching_words[-1]["end"],
            "words": matching_words,
        }

        return segment

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
            # print(f"Buffer before stabilization step: {self.buffer}")
            last_segment = self.buffer[-1]
            closest_segment_index, closest_segment = self.get_closest_segment(
                last_segment, segments
            )
            if closest_segment and isclose(
                closest_segment["start"],
                last_segment["start"],
                abs_tol=self.equality_treshold,
            ):
                self.buffer[-1] = closest_segment
                if closest_segment_index + 1 < len(segments):
                    self.buffer.extend(segments[closest_segment_index + 1 :])
            else:
                # we couldn't find a segment that overlaps with the last segment
                # so we need to search by word to find the overlapping segment
                # print(self.buffer[-1])
                # print(segments)
                # raise NotImplementedError(
                #     "Overlapping segments not found for buffer duration"
                # )
                matching_segment = self.get_segment_by_words(
                    last_segment, closest_segment
                )
                if matching_segment:
                    self.buffer[-1] = matching_segment
                    if closest_segment_index + 1 < len(segments):
                        self.buffer.extend(segments[closest_segment_index + 1 :])
                else:
                    raise NotImplementedError(
                        "Overlapping segments not found for buffer duration"
                    )

            # print(f"Buffer stabilization step: {self.buffer}")
        else:
            self.buffer = deepcopy(segments)
            # print(f"Buffer before window duration: {self.buffer}")

        self.buffer_duration = self.buffer[-1]["end"]
        # print(self.buffer_duration)
        # print(self.buffer_duration >= self.window_duration)
        # print(
        #     isclose(
        #         self.buffer_duration,
        #         self.window_duration,
        #         abs_tol=self.equality_treshold,
        #     )
        # )

    def get_continuous_transcription(self) -> str:
        """
        Returns the continuous transcription.

        Returns:
            str: The continuous transcription.
        """
        trs_text = " ".join([segment["text"] for segment in self.buffer])
        return trs_text
