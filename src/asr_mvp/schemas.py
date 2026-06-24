from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = "SPEAKER_00"
    language: str | None = None
    emotion: str | None = None
    events: list[str] = field(default_factory=list)

@dataclass
class AudioInfo:
    path: str
    exists: bool
    duration_seconds: float | None
    sample_rate: int | None
    channels: int | None
    size_bytes: int | None
    warning: str | None = None

@dataclass
class PipelineConfig:
    audio: Path
    output: Path
    engine: str = "auto"
    model: str = "tiny"
    compute_type: str = "int8"
    cpu_threads: int = 1
    force_cpu_isa: str | None = None
    asr_prompt: str | None = None
    use_default_asr_prompt: bool = False
    replacement_file: Path | None = None
    polish_text: bool = True
    remove_repeated_text: bool = True
    repeat_similarity_threshold: float = 0.82
    repeat_window: int = 6
    merge_gap_seconds: float = 1.0
    max_merged_chars: int = 90
    language: str | None = None
    speakers: int = 2
    transcript_file: Path | None = None
    diarizer: str = "turns"
    speaker_model: Path | None = None
    reference_file: Path | None = None
    use_llm_correct: bool = False
    llm_model: str = "qwen2.5:1.5b"
    llm_base_url: str = "http://localhost:11434"
