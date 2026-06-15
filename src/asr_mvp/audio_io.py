from __future__ import annotations

from pathlib import Path
import wave

from .schemas import AudioInfo


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    whole = int(seconds)
    ms = int((seconds - whole) * 1000)
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def inspect_audio(path: Path) -> AudioInfo:
    if not path.exists():
        return AudioInfo(str(path), False, None, None, None, None, "Audio file does not exist.")

    size = path.stat().st_size
    if size < 1024:
        return AudioInfo(str(path), True, None, None, None, size, "Audio file is very small; it may be empty or damaged.")

    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wav:
                frames = wav.getnframes()
                sample_rate = wav.getframerate()
                channels = wav.getnchannels()
                duration = frames / float(sample_rate) if sample_rate else None
            return AudioInfo(str(path), True, duration, sample_rate, channels, size)
        except wave.Error as exc:
            return AudioInfo(str(path), True, None, None, None, size, f"Could not parse wav metadata: {exc}")

    return AudioInfo(str(path), True, None, None, None, size, "Duration unavailable for this file type without ffmpeg.")

def load_text(path: Path | None) -> str:
    if path is None:
        return ""
    if not path.exists():
        raise FileNotFoundError(f"Transcript/reference file not found: {path}")
    return path.read_text(encoding="utf-8").strip()
