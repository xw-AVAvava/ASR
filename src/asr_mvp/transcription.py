from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import site
import sys

from .audio_io import load_text
from .schemas import AudioInfo, PipelineConfig, Segment
from .text_processing import repair_mojibake


def ensure_user_site_packages() -> None:
    try:
        user_site = site.getusersitepackages()
        if user_site and os.path.isdir(user_site) and user_site not in sys.path:
            sys.path.append(user_site)
    except Exception:
        pass


def ensure_venv_scripts_on_path() -> None:
    ffmpeg_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    current_path = os.environ.get("PATH", "")
    path_parts = [part for part in current_path.split(os.pathsep) if part]
    known_paths = {part.lower() for part in path_parts}
    scripts_dir = Path(sys.executable).resolve().parent
    candidate_dirs = [scripts_dir]

    try:
        import imageio_ffmpeg  # type: ignore

        packaged_ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
        scripts_ffmpeg = scripts_dir / ffmpeg_name
        if not scripts_ffmpeg.exists() and packaged_ffmpeg.exists():
            shutil.copy2(packaged_ffmpeg, scripts_ffmpeg)
        candidate_dirs.append(packaged_ffmpeg.parent)
    except Exception:
        pass

    for candidate_dir in candidate_dirs:
        if not (candidate_dir / ffmpeg_name).exists():
            continue
        candidate_text = str(candidate_dir)
        if candidate_text.lower() not in known_paths:
            os.environ["PATH"] = candidate_text + os.pathsep + os.environ.get("PATH", "")
            known_paths.add(candidate_text.lower())


ensure_user_site_packages()
ensure_venv_scripts_on_path()


DEFAULT_ZH_ASR_PROMPT = (
    "以下是一段中文多人对话或会议录音，请按原话转写为简体中文，"
    "尽量保留口语表达、关键信息和自然断句。"
)

def has_meaningful_text(text: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", text))

def split_sentences(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return [line for line in lines if has_meaningful_text(line)]
    parts = re.split(r"(?<=[。！？!?；;])\s*", text.strip())
    if len(parts) <= 1:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())

    sentences = []
    for part in parts:
        part = part.strip()
        if not has_meaningful_text(part):
            continue
        sentences.extend(split_long_sentence(part))
    return sentences

def split_long_sentence(text: str, max_chars: int = 70) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current = ""
    clauses = re.split(r"(?<=[，,、])\s*", text)
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        if current and len(current) + len(clause) > max_chars:
            chunks.append(current)
            current = clause
        else:
            current += clause
    if current:
        chunks.append(current)

    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
            continue
        final_chunks.extend(chunk[i : i + max_chars] for i in range(0, len(chunk), max_chars))
    return final_chunks

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
                body = body.strip()
                if has_meaningful_text(body):
                    entries.append((int(minutes) * 60 + int(seconds), body))
            continue
        hours, minutes, seconds, body = match.groups()
        body = body.strip()
        if not has_meaningful_text(body):
            continue
        start = (int(hours or 0) * 3600) + int(minutes) * 60 + int(seconds)
        entries.append((float(start), body))

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

def _seconds_from_millis(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number > 1000:
        return number / 1000.0
    return number

SENSEVOICE_LANGUAGES = {
    "zh": "ZH",
    "en": "EN",
    "yue": "YUE",
    "ja": "JA",
    "ko": "KO",
}
SENSEVOICE_EMOTIONS = {
    "HAPPY",
    "SAD",
    "ANGRY",
    "NEUTRAL",
    "FEARFUL",
    "DISGUSTED",
    "SURPRISED",
}
SENSEVOICE_EVENTS = {
    "BGM",
    "SPEECH",
    "APPLAUSE",
    "LAUGHTER",
    "CRY",
    "SNEEZE",
    "BREATH",
    "COUGH",
}


def _parse_sensevoice_chunks(text: object) -> list[tuple[str, str | None, str | None, list[str]]]:
    raw = repair_mojibake(str(text).strip())
    if not raw:
        return []

    parts = re.split(r"((?:<\|[^|]+?\|>)+)", raw)
    chunks: list[tuple[str, str | None, str | None, list[str]]] = []
    for index in range(1, len(parts), 2):
        tag_block = parts[index]
        body = parts[index + 1] if index + 1 < len(parts) else ""
        body = re.sub(r"<\|[^|]+?\|>", "", body).strip()
        if not has_meaningful_text(body):
            continue

        language = None
        emotion = None
        events: list[str] = []
        for raw_tag in re.findall(r"<\|([^|]+?)\|>", tag_block):
            tag = raw_tag.strip()
            lower_tag = tag.lower()
            upper_tag = tag.upper()
            if lower_tag in SENSEVOICE_LANGUAGES:
                language = SENSEVOICE_LANGUAGES[lower_tag]
            elif upper_tag in SENSEVOICE_EMOTIONS:
                emotion = upper_tag
            elif upper_tag in SENSEVOICE_EVENTS and upper_tag != "SPEECH":
                events.append(upper_tag)
        chunks.append((body, language, emotion, events))

    if chunks:
        return chunks

    cleaned = re.sub(r"<\|[^|]+?\|>", "", raw).strip()
    return [(cleaned, None, None, [])] if has_meaningful_text(cleaned) else []


def _clean_funasr_text(text: object) -> str:
    return " ".join(chunk[0] for chunk in _parse_sensevoice_chunks(text)).strip()


def _segments_from_sensevoice_chunks(
    chunks: list[tuple[str, str | None, str | None, list[str]]],
    duration: float | None,
) -> list[Segment]:
    if not chunks:
        return []

    weights = [max(1, len(text)) for text, _, _, _ in chunks]
    total_duration = duration if duration and duration > 0 else max(4.0, sum(weights) / 5.0)
    total_weight = sum(weights)
    offset = 0.0
    segments: list[Segment] = []

    for index, ((text, language, emotion, events), weight) in enumerate(zip(chunks, weights)):
        if index == len(chunks) - 1:
            chunk_end = total_duration
        else:
            chunk_end = offset + total_duration * weight / total_weight
        local_segments = segments_from_transcript(text, max(0.5, chunk_end - offset))
        for seg in local_segments:
            segments.append(
                Segment(
                    start=offset + seg.start,
                    end=offset + seg.end,
                    text=seg.text,
                    speaker=seg.speaker,
                    language=language,
                    emotion=emotion,
                    events=list(events),
                )
            )
        offset = chunk_end
    return segments

def _segments_from_funasr_result(result: object, duration: float | None) -> list[Segment]:
    records = result if isinstance(result, list) else [result]
    segments: list[Segment] = []

    for record in records:
        if not isinstance(record, dict):
            continue

        sentence_info = record.get("sentence_info")
        if isinstance(sentence_info, list):
            for item in sentence_info:
                if not isinstance(item, dict):
                    continue
                start = _seconds_from_millis(item.get("start"), 0.0)
                raw_text = item.get("text", "")
                cleaned_text = _clean_funasr_text(raw_text)
                end = _seconds_from_millis(item.get("end"), start + max(1.0, len(cleaned_text) / 6.0))
                item_segments = _segments_from_sensevoice_chunks(
                    _parse_sensevoice_chunks(raw_text),
                    max(0.5, end - start),
                )
                for seg in item_segments:
                    seg.start += start
                    seg.end += start
                segments.extend(item_segments)

        text = _clean_funasr_text(record.get("text", ""))
        if has_meaningful_text(text) and not sentence_info:
            total = duration if duration and duration > 1 else max(4.0, len(text) / 5.0)
            segments.extend(
                _segments_from_sensevoice_chunks(
                    _parse_sensevoice_chunks(record.get("text", "")),
                    total,
                )
            )

    return segments

def resolve_funasr_model(model: str) -> str:
    model_path = Path(model)
    if model_path.exists():
        return str(model_path)

    local_candidates = {
        "iic/SenseVoiceSmall": [Path("D:/asr_models/SenseVoiceSmall")],
        "SenseVoiceSmall": [Path("D:/asr_models/SenseVoiceSmall")],
        "paraformer-zh": [Path("D:/asr_models/paraformer-zh")],
        "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch": [
            Path("D:/asr_models/paraformer-zh")
        ],
    }
    for candidate in local_candidates.get(model, []):
        if candidate.exists():
            return str(candidate)
    return model

def transcribe_with_funasr(config: PipelineConfig, audio_info: AudioInfo) -> list[Segment]:
    configure_asr_cpu_runtime(config.force_cpu_isa)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    from funasr import AutoModel  # type: ignore

    try:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    model_name = resolve_funasr_model(config.model or "iic/SenseVoiceSmall")
    print(f"[ASR] Loading FunASR model: {model_name}", flush=True)
    print(f"[ASR] Running FunASR on {device}.", flush=True)

    is_sensevoice = "sensevoice" in model_name.lower()
    kwargs: dict[str, object] = {
        "model": model_name,
        "device": device,
        "disable_update": True,
        "vad_model": "fsmn-vad",
        "vad_kwargs": {"max_single_segment_time": 30000},
    }
    if is_sensevoice:
        kwargs["trust_remote_code"] = True
    else:
        kwargs.update(
            {
                "punc_model": "ct-punc",
            }
        )

    model = AutoModel(**kwargs)
    print(f"[ASR] Transcribing audio with FunASR: {config.audio}", flush=True)
    generate_kwargs: dict[str, object] = {
        "input": str(config.audio),
        "batch_size_s": 60,
        "hotword": build_asr_prompt(config) or "",
    }
    if is_sensevoice:
        generate_kwargs.update(
            {
                "language": config.language or "auto",
                "use_itn": True,
                "merge_vad": True,
                "merge_length_s": 15,
            }
        )
    result = model.generate(**generate_kwargs)
    segments = _segments_from_funasr_result(result, audio_info.duration_seconds)
    print(f"[ASR] Finished FunASR transcription with {len(segments)} text segments.", flush=True)
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

    if config.engine in {"auto", "funasr"}:
        try:
            model_name = resolve_funasr_model(config.model or "iic/SenseVoiceSmall")
            return transcribe_with_funasr(config, audio_info), f"FunASR ({model_name})"
        except Exception as exc:
            if config.engine == "funasr":
                raise RuntimeError(f"FunASR failed: {exc}") from exc

    text = demo_transcript()
    return segments_from_transcript(text, audio_info.duration_seconds), "demo fallback"
