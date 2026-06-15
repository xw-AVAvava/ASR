from __future__ import annotations

import os
from pathlib import Path
import re
import site
import sys

from .audio_io import load_text
from .schemas import AudioInfo, PipelineConfig, Segment


def ensure_user_site_packages() -> None:
    try:
        user_site = site.getusersitepackages()
        if user_site and os.path.isdir(user_site) and user_site not in sys.path:
            sys.path.append(user_site)
    except Exception:
        pass


ensure_user_site_packages()


DEFAULT_ZH_ASR_PROMPT = (
    "以下是一段中文多人对话或会议录音，请按原话转写为简体中文，"
    "尽量保留口语表达、关键信息和自然断句。"
)

def split_sentences(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]

def parse_timestamped_lines(text: str, duration: float | None) -> list[Segment]:
    entries: list[tuple[float, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^\[(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\]\s*(.+)$", line)
        if not match:
            match = re.match(r"^\[(\d{1,2}):(\d{2})\]\s*(.+)$", line)
            if match:
                minutes, seconds, body = match.groups()
                entries.append((int(minutes) * 60 + int(seconds), body.strip()))
            continue
        hours, minutes, seconds, body = match.groups()
        start = (int(hours or 0) * 3600) + int(minutes) * 60 + int(seconds)
        entries.append((float(start), body.strip()))

    if not entries:
        return []

    segments = []
    for i, (start, body) in enumerate(entries):
        if i + 1 < len(entries):
            end = entries[i + 1][0]
        elif duration and duration > start:
            end = duration
        else:
            end = start + max(4.0, min(20.0, len(body) / 6.0))
        segments.append(Segment(start=start, end=max(end, start + 0.5), text=body))
    return segments

def strip_speaker_prefix(text: str) -> tuple[str | None, str]:
    match = re.match(r"^\s*([A-Za-z][A-Za-z0-9 _-]{0,24})\s*:\s*(.+)$", text)
    if not match:
        return None, text.strip()
    raw_label, body = match.groups()
    normalized = raw_label.strip().upper().replace(" ", "_")
    return normalized, body.strip()

def segments_from_transcript(text: str, duration: float | None) -> list[Segment]:
    timestamped = parse_timestamped_lines(text, duration)
    if timestamped:
        return timestamped

    sentences = split_sentences(text)
    if not sentences:
        return []
    total = duration if duration and duration > 1 else max(4.0 * len(sentences), 8.0)
    step = total / len(sentences)
    segments = []
    speaker_map: dict[str, str] = {}
    for i, sentence in enumerate(sentences):
        prefix, body = strip_speaker_prefix(sentence)
        speaker = "SPEAKER_00"
        if prefix:
            if prefix not in speaker_map:
                speaker_map[prefix] = f"SPEAKER_{len(speaker_map):02d}"
            speaker = speaker_map[prefix]
        segments.append(Segment(start=i * step, end=(i + 1) * step, text=body, speaker=speaker))
    return segments

def demo_transcript() -> str:
    return (
        "Professor: Today we are testing a small meeting assistant for automatic speech recognition.\n"
        "Student: The goal is to create timestamps, speaker labels, a summary, and keywords.\n"
        "Professor: If time is limited, the simple baseline still demonstrates the full workflow.\n"
        "Student: We should explain the limitations and show how pyannote audio can improve diarization."
    )

def configure_asr_cpu_runtime(force_cpu_isa: str | None) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    if force_cpu_isa:
        os.environ["CT2_FORCE_CPU_ISA"] = force_cpu_isa

def build_asr_prompt(config: PipelineConfig) -> str | None:
    if config.asr_prompt is not None:
        return config.asr_prompt
    if config.use_default_asr_prompt and config.language and config.language.lower().startswith("zh"):
        return DEFAULT_ZH_ASR_PROMPT
    return None

def transcribe_with_faster_whisper(config: PipelineConfig) -> list[Segment]:
    configure_asr_cpu_runtime(config.force_cpu_isa)
    from faster_whisper import WhisperModel  # type: ignore

    model_ref = resolve_faster_whisper_model(config.model)
    print(f"[ASR] Loading faster-whisper model: {model_ref}", flush=True)
    print(
        "[ASR] CPU settings: "
        f"compute_type={config.compute_type}, "
        f"cpu_threads={config.cpu_threads}, "
        f"CT2_FORCE_CPU_ISA={config.force_cpu_isa or 'auto'}",
        flush=True,
    )
    model = WhisperModel(
        str(model_ref),
        device="cpu",
        compute_type=config.compute_type,
        cpu_threads=max(1, config.cpu_threads),
        num_workers=1,
    )
    print(f"[ASR] Transcribing audio: {config.audio}", flush=True)
    segments_iter, _ = model.transcribe(
        str(config.audio),
        language=config.language,
        task="transcribe",
        vad_filter=True,
        initial_prompt=build_asr_prompt(config),
    )
    segments = []
    for i, seg in enumerate(segments_iter, 1):
        text = seg.text.strip()
        if text:
            segments.append(Segment(seg.start, seg.end, text))
        if i % 10 == 0:
            print(f"[ASR] Processed {i} ASR segments...", flush=True)
    print(f"[ASR] Finished transcription with {len(segments)} text segments.", flush=True)
    return segments

def resolve_faster_whisper_model(model: str) -> str | Path:
    model_path = Path(model)
    if model_path.exists():
        return model_path

    cache_name = f"models--Systran--faster-whisper-{model}"
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / cache_name / "snapshots"
    if not cache_root.exists():
        return model

    candidates = []
    for snapshot in cache_root.iterdir():
        if not snapshot.is_dir():
            continue
        required = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]
        if all((snapshot / name).exists() for name in required):
            candidates.append(snapshot)

    if not candidates:
        return model

    return max(candidates, key=lambda path: path.stat().st_mtime)

def transcribe_with_openai_whisper(config: PipelineConfig) -> list[Segment]:
    import whisper  # type: ignore

    print(f"[ASR] Loading openai-whisper model: {config.model}", flush=True)
    model = whisper.load_model(config.model)
    print(f"[ASR] Transcribing audio: {config.audio}", flush=True)
    result = model.transcribe(
        str(config.audio),
        language=config.language,
        verbose=False,
        initial_prompt=build_asr_prompt(config),
    )
    segments = []
    for seg in result.get("segments", []):
        text = str(seg.get("text", "")).strip()
        if text:
            segments.append(Segment(float(seg["start"]), float(seg["end"]), text))
    return segments

def transcribe_audio(config: PipelineConfig, audio_info: AudioInfo) -> tuple[list[Segment], str]:
    if config.engine == "demo":
        text = load_text(config.transcript_file) or demo_transcript()
        engine_name = "provided transcript" if config.transcript_file else "demo transcript"
        return segments_from_transcript(text, audio_info.duration_seconds), engine_name

    if config.transcript_file is not None:
        text = load_text(config.transcript_file)
        return segments_from_transcript(text, audio_info.duration_seconds), "provided transcript"

    if config.engine in {"auto", "faster-whisper"}:
        try:
            return transcribe_with_faster_whisper(config), "faster-whisper"
        except Exception as exc:
            if config.engine == "faster-whisper":
                raise RuntimeError(f"faster-whisper failed: {exc}") from exc

    if config.engine in {"auto", "openai-whisper"}:
        try:
            return transcribe_with_openai_whisper(config), "openai-whisper"
        except Exception as exc:
            if config.engine == "openai-whisper":
                raise RuntimeError(f"openai-whisper failed: {exc}") from exc

    text = demo_transcript()
    return segments_from_transcript(text, audio_info.duration_seconds), "demo fallback"
