from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import time
from pathlib import Path
from typing import Iterable

from .audio_io import inspect_audio, load_text
from .diarization import assign_speakers
from .outputs import write_markdown_outputs
from .schemas import PipelineConfig, Segment
from .text_processing import (
    accuracy_from_error_rate,
    character_error_rate,
    load_replacement_rules,
    merge_short_segments,
    polish_segments,
    remove_repeated_segments,
    summarize_segments,
    word_error_rate,
)
from .transcription import transcribe_audio


def run_pipeline(config: PipelineConfig) -> dict:
    start = time.time()
    print("[Pipeline] Inspecting audio...", flush=True)
    audio_info = inspect_audio(config.audio)
    print("[Pipeline] Running transcription...", flush=True)
    segments, asr_engine_used = transcribe_audio(config, audio_info)
    if not segments:
        raise RuntimeError("No transcript segments were produced.")
    raw_segments = [Segment(seg.start, seg.end, seg.text, seg.speaker) for seg in segments]
    raw_segment_count = len(segments)
    postprocess_steps = []
    if config.polish_text:
        replacements = load_replacement_rules(config.replacement_file)
        segments = polish_segments(segments, replacements)
        step = "generic text cleanup"
        if replacements:
            step += f" + {len(replacements)} custom replacement rules"
        postprocess_steps.append(step)

    speaker_aware_diarizers = {"cluster", "trained-model", "pyannote"}
    speaker_aware_merge = config.diarizer in speaker_aware_diarizers
    diarizer_used = ""
    if speaker_aware_merge:
        print("[Pipeline] Assigning speaker labels...", flush=True)
        segments, diarizer_used = assign_speakers(config, segments)

    if config.merge_gap_seconds > 0 and config.max_merged_chars > 0:
        before_merge = len(segments)
        segments = merge_short_segments(
            segments,
            config.merge_gap_seconds,
            config.max_merged_chars,
            require_same_speaker=speaker_aware_merge,
        )
        postprocess_steps.append(
            f"short-segment merge {before_merge}->{len(segments)} "
            f"(gap<={config.merge_gap_seconds}s, max_chars<={config.max_merged_chars}, "
            f"same_speaker={speaker_aware_merge})"
        )

    if config.remove_repeated_text:
        before_dedup = len(segments)
        segments, removed_repeats = remove_repeated_segments(
            segments,
            similarity_threshold=config.repeat_similarity_threshold,
            window=config.repeat_window,
        )
        if removed_repeats:
            postprocess_steps.append(
                f"repeat removal {before_dedup}->{len(segments)} "
                f"(removed={removed_repeats}, threshold={config.repeat_similarity_threshold}, "
                f"window={config.repeat_window})"
            )

    if not speaker_aware_merge:
        print("[Pipeline] Assigning speaker labels...", flush=True)
        segments, diarizer_used = assign_speakers(config, segments)

    print("[Pipeline] Generating summary and metrics...", flush=True)
    summary = summarize_segments(segments)

    raw_hypothesis = " ".join(seg.text for seg in raw_segments)
    display_hypothesis = " ".join(seg.text for seg in segments)
    reference = load_text(config.reference_file) if config.reference_file else ""
    raw_wer = word_error_rate(reference, raw_hypothesis) if reference else None
    raw_cer = character_error_rate(reference, raw_hypothesis) if reference else None
    display_wer = word_error_rate(reference, display_hypothesis) if reference else None
    display_cer = character_error_rate(reference, display_hypothesis) if reference else None
    raw_accuracy = accuracy_from_error_rate(raw_cer)
    display_accuracy = accuracy_from_error_rate(display_cer)

    metadata = {
        "asr_engine_used": asr_engine_used,
        "diarizer_used": diarizer_used,
        "runtime_seconds": round(time.time() - start, 3),
        "wer": None if raw_wer is None else round(raw_wer, 4),
        "cer": None if raw_cer is None else round(raw_cer, 4),
        "raw_wer": None if raw_wer is None else round(raw_wer, 4),
        "raw_cer": None if raw_cer is None else round(raw_cer, 4),
        "display_wer": None if display_wer is None else round(display_wer, 4),
        "display_cer": None if display_cer is None else round(display_cer, 4),
        "raw_accuracy": None if raw_accuracy is None else round(raw_accuracy, 4),
        "display_accuracy": None if display_accuracy is None else round(display_accuracy, 4),
        "raw_segment_count": raw_segment_count,
        "postprocess": "; ".join(postprocess_steps) if postprocess_steps else "none",
    }
    print("[Pipeline] Writing output files...", flush=True)
    write_markdown_outputs(config, audio_info, segments, summary, metadata, raw_segments)
    print("[Pipeline] Done.", flush=True)
    return {
        "audio_info": asdict(audio_info),
        "segments": [asdict(seg) for seg in segments],
        "summary": summary,
        "metadata": metadata,
        "output": str(config.output),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Meeting ASR Assistant MVP pipeline")
    parser.add_argument("--audio", required=True, type=Path, help="Path to an audio file")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--engine", choices=["auto", "demo", "faster-whisper", "openai-whisper", "funasr"], default="auto")
    parser.add_argument("--model", default="tiny", help="ASR model size, for example tiny/base/small")
    parser.add_argument("--compute-type", default="int8", help="faster-whisper compute type, for example int8/float32")
    parser.add_argument("--cpu-threads", type=int, default=1, help="CPU threads used by faster-whisper")
    parser.add_argument(
        "--force-cpu-isa",
        choices=["GENERIC", "SSE4_1", "AVX", "AVX2", "AVX512"],
        default=None,
        help="Optional CTranslate2 CPU instruction set override",
    )
    parser.add_argument("--asr-prompt", default=None, help="Optional custom ASR prompt/context")
    parser.add_argument(
        "--use-default-asr-prompt",
        action="store_true",
        help="Use a generic Chinese meeting prompt when language is zh",
    )
    parser.add_argument(
        "--replacement-file",
        type=Path,
        default=None,
        help="Optional text or JSON file with ASR correction rules",
    )
    parser.add_argument("--no-polish-text", action="store_true", help="Disable generic text cleanup")
    parser.add_argument("--no-remove-repeated-text", action="store_true", help="Disable repeated segment removal")
    parser.add_argument(
        "--repeat-similarity-threshold",
        type=float,
        default=0.82,
        help="Similarity threshold for repeated segment removal",
    )
    parser.add_argument("--repeat-window", type=int, default=6, help="Number of recent segments used when detecting repeated text")
    parser.add_argument("--merge-gap-seconds", type=float, default=1.0, help="Merge nearby short ASR segments")
    parser.add_argument("--max-merged-chars", type=int, default=90, help="Maximum characters after segment merging")
    parser.add_argument("--language", default=None, help="Optional language code such as en or zh")
    parser.add_argument("--speakers", type=int, default=2, help="Number of speakers for the baseline diarizer")
    parser.add_argument("--transcript-file", type=Path, default=None, help="Optional transcript text for demo mode")
    parser.add_argument("--diarizer", choices=["turns", "cluster", "pyannote", "trained-model"], default="turns")
    parser.add_argument("--speaker-model", type=Path, default=None, help="Optional trained audio classifier .pkl")
    parser.add_argument("--reference-file", type=Path, default=None, help="Optional reference transcript for WER")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PipelineConfig(
        audio=args.audio,
        output=args.output,
        engine=args.engine,
        model=args.model,
        compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
        force_cpu_isa=args.force_cpu_isa,
        asr_prompt=args.asr_prompt,
        use_default_asr_prompt=args.use_default_asr_prompt,
        replacement_file=args.replacement_file,
        polish_text=not args.no_polish_text,
        remove_repeated_text=not args.no_remove_repeated_text,
        repeat_similarity_threshold=args.repeat_similarity_threshold,
        repeat_window=args.repeat_window,
        merge_gap_seconds=args.merge_gap_seconds,
        max_merged_chars=args.max_merged_chars,
        language=args.language,
        speakers=args.speakers,
        transcript_file=args.transcript_file,
        diarizer=args.diarizer,
        speaker_model=args.speaker_model,
        reference_file=args.reference_file,
    )
    result = run_pipeline(config)
    print(json.dumps({"output": result["output"], "metadata": result["metadata"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
