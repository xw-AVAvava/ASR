from __future__ import annotations

import unittest

from src.asr_mvp.schemas import Segment
from src.asr_mvp.text_processing import merge_short_segments, summarize_voice_metadata
from src.asr_mvp.transcription import _parse_sensevoice_chunks, _segments_from_funasr_result


class SenseVoiceMetadataTests(unittest.TestCase):
    def test_parse_multiple_tagged_chunks(self) -> None:
        raw = (
            "<|zh|><|HAPPY|><|Laughter|><|withitn|>大家好。"
            "<|en|><|NEUTRAL|><|Speech|><|withitn|>Hello everyone."
        )

        chunks = _parse_sensevoice_chunks(raw)

        self.assertEqual(chunks[0], ("大家好。", "ZH", "HAPPY", ["LAUGHTER"]))
        self.assertEqual(chunks[1], ("Hello everyone.", "EN", "NEUTRAL", []))

    def test_funasr_result_keeps_metadata(self) -> None:
        result = [
            {
                "text": (
                    "<|zh|><|NEUTRAL|><|Speech|><|withitn|>第一句话。"
                    "<|zh|><|ANGRY|><|Cough|><|withitn|>第二句话。"
                )
            }
        ]

        segments = _segments_from_funasr_result(result, 10.0)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].emotion, "NEUTRAL")
        self.assertEqual(segments[1].emotion, "ANGRY")
        self.assertEqual(segments[1].events, ["COUGH"])
        self.assertAlmostEqual(segments[-1].end, 10.0)

    def test_merge_preserves_emotion_boundary(self) -> None:
        segments = [
            Segment(0.0, 1.0, "你好", emotion="NEUTRAL"),
            Segment(1.0, 2.0, "注意", emotion="ANGRY"),
        ]

        merged = merge_short_segments(segments, gap_seconds=1.0, max_chars=90)

        self.assertEqual(len(merged), 2)

    def test_speaker_emotion_summary(self) -> None:
        segments = [
            Segment(0.0, 1.0, "a", speaker="SPEAKER_00", emotion="HAPPY"),
            Segment(1.0, 2.0, "b", speaker="SPEAKER_00", emotion="HAPPY", events=["LAUGHTER"]),
            Segment(2.0, 3.0, "c", speaker="SPEAKER_01", emotion="NEUTRAL"),
        ]

        summary = summarize_voice_metadata(segments)

        self.assertEqual(summary["dominant_emotion_by_speaker"]["SPEAKER_00"], "HAPPY")
        self.assertEqual(summary["speaker_events"]["SPEAKER_00"]["LAUGHTER"], 1)


if __name__ == "__main__":
    unittest.main()
