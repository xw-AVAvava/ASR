from __future__ import annotations

import json
from typing import Callable

import requests

from .schemas import Segment

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "qwen2.5:1.5b"
CHECK_TIMEOUT = 30
CORRECT_TIMEOUT = 60
PULL_TIMEOUT = 600
BATCH_TIMEOUT = 120

# Prompt for concat-then-align strategy:
# Send all segments as one continuous text, get corrected text back,
# then use SequenceMatcher to map corrections to individual segments.
BATCH_CORRECTION_PROMPT = """你是中文错别字校对器。输入是一段ASR语音转写文本。文本被 ### 分隔成多个短句。

你的唯一任务：逐句检查，只把错别字替换为正确的字。其余一概不动。

绝对禁止：
- 改变语序
- 增、删、改任何字（除了替换错别字）
- 改写、润色、总结
- 删除或合并 ### 分隔符

示例——
原文：大家好，欢迎来到今天的会议。###然后我们要讨论项目。
校对：大家好，欢迎来到今天的会议。###然后我们要讨论项目。

原文：
{text}

校对："""

SEGMENT_SEPARATOR = "\n###\n"

# Max chars per batch to keep model output reliable
BATCH_MAX_CHARS = 3000


def _get(path: str, base_url: str, timeout: int = CHECK_TIMEOUT) -> dict:
    url = f"{base_url.rstrip('/')}{path}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, body: dict, base_url: str, timeout: int = CORRECT_TIMEOUT) -> dict:
    url = f"{base_url.rstrip('/')}{path}"
    resp = requests.post(url, json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _post_stream(path: str, body: dict, base_url: str, timeout: int = PULL_TIMEOUT):
    url = f"{base_url.rstrip('/')}{path}"
    resp = requests.post(url, json=body, timeout=timeout, stream=True)
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def check_ollama_status(
    base_url: str = OLLAMA_DEFAULT_BASE_URL,
) -> tuple[bool, str, list[str]]:
    """Check Ollama connectivity and list installed models.

    Returns:
        (connected, status_message, installed_model_names)
    """
    try:
        data = _get("/api/tags", base_url)
        models = [m["name"] for m in data.get("models", [])]
        if not models:
            return True, "已连接（无已安装模型）", []
        names = ", ".join(models[:5])
        suffix = "..." if len(models) > 5 else ""
        return True, f"已连接，{len(models)} 个模型: {names}{suffix}", models
    except requests.exceptions.ConnectionError:
        return False, "无法连接 — 请确认 Ollama 正在运行", []
    except requests.exceptions.Timeout:
        return False, "连接超时 — 请确认 Ollama 正在运行", []
    except Exception as exc:
        return False, f"连接异常: {exc}", []


def ensure_model_available(
    model: str,
    base_url: str = OLLAMA_DEFAULT_BASE_URL,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> bool:
    """Ensure the specified Ollama model is pulled locally.

    If the model is not found, triggers /api/pull with streaming progress.
    Calls progress_callback(status, completed, total) on each progress event.
    Returns True if the model is ready, False otherwise.
    """
    # check if model already exists
    try:
        data = _get("/api/tags", base_url)
        installed = [m["name"] for m in data.get("models", [])]
        # match exact name or name:tag
        if model in installed:
            return True
    except Exception:
        return False

    # model not found — pull it
    print(f"[LLM] Model '{model}' not found locally. Pulling from Ollama registry...", flush=True)
    if progress_callback:
        progress_callback("downloading", 0, 0)

    try:
        last_completed = 0
        for event in _post_stream("/api/pull", {"name": model, "stream": True}, base_url):
            if "total" in event and "completed" in event:
                total = int(event["total"])
                completed = int(event["completed"])
                if completed > last_completed:
                    last_completed = completed
                    pct = round(completed / max(total, 1) * 100, 1)
                    print(f"\r[LLM] Pulling {model}: {pct}% ({_fmt_bytes(completed)}/{_fmt_bytes(total)})", end="", flush=True)
                    if progress_callback:
                        progress_callback("downloading", completed, total)
            elif "status" in event:
                status = event["status"]
                if status.startswith("pulling"):
                    digest = event.get("digest", "")[:12]
                    total = int(event.get("total", 0))
                    completed = int(event.get("completed", 0))
                    if total > 0:
                        pct = round(completed / total * 100, 1)
                        print(f"\r[LLM] Pulling {model}: {pct}% ({_fmt_bytes(completed)}/{_fmt_bytes(total)})", end="", flush=True)
                    if progress_callback:
                        progress_callback("downloading", completed, total)
                elif status == "verifying sha256 digest":
                    print(f"\r[LLM] Verifying {model}...", end="", flush=True)
                elif status == "writing manifest":
                    print(f"\r[LLM] Writing manifest...", end="", flush=True)
                elif status == "success":
                    print(f"\r[LLM] {model} pulled successfully!           ", flush=True)
                    if progress_callback:
                        progress_callback("done", 1, 1)
                    return True
    except Exception as exc:
        print(f"\n[LLM] Failed to pull model '{model}': {exc}", flush=True)
        if progress_callback:
            progress_callback("error", 0, 0)
        return False

    # fallback: verify model is now in list
    try:
        data = _get("/api/tags", base_url)
        installed = [m["name"] for m in data.get("models", [])]
        if model in installed:
            return True
    except Exception:
        pass
    return False


SINGLE_CORRECTION_PROMPT = """你是中文错别字校对器。请逐字检查以下ASR识别短句，只替换错别字。

示例——
原文：大家号，欢饮来到今天的会以。
校对：大家好，欢迎来到今天的会议。

示例——
原文：下周一前踢交暴告。
校对：下周一前提交报告。

现在校对：
原文：{text}
校对："""


def _correct_single(
    text: str,
    model: str = OLLAMA_DEFAULT_MODEL,
    base_url: str = OLLAMA_DEFAULT_BASE_URL,
    timeout: int = CORRECT_TIMEOUT,
) -> str:
    """Correct a single segment (used as fallback when batching fails)."""
    if not text.strip():
        return text
    prompt = SINGLE_CORRECTION_PROMPT.format(text=text)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 256},
    }
    try:
        data = _post("/api/chat", payload, base_url, timeout)
        result = data.get("message", {}).get("content", "").strip()
        # Safety: reject if length changed too much
        if result and 0.5 < len(result) / max(len(text), 1) < 2.0:
            return result
    except Exception:
        pass
    return text


def _correct_batch(
    text: str,
    model: str = OLLAMA_DEFAULT_MODEL,
    base_url: str = OLLAMA_DEFAULT_BASE_URL,
    timeout: int = BATCH_TIMEOUT,
) -> str:
    """Send batch text to Ollama, return corrected version."""
    prompt = BATCH_CORRECTION_PROMPT.format(text=text)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 4096},
    }
    data = _post("/api/chat", payload, base_url, timeout)
    return data.get("message", {}).get("content", "").strip()


def _split_corrected_output(raw: str, expected_count: int) -> list[str]:
    """Split model output by ### separators.

    Returns list of corrected segment texts.  If the split count doesn't
    match expected, returns empty list to signal fallback.
    """
    import re

    # Try various split patterns the model might use
    parts = re.split(r"\n?###\n?", raw.strip())

    # Filter empty parts and strip whitespace
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) == expected_count:
        return parts

    # Maybe the model used different separators
    if len(parts) != expected_count:
        print(f"[LLM] Separator split gave {len(parts)} parts, expected {expected_count}.", flush=True)
        return []

    return parts


def correct_segments_with_llm(
    segments: list[Segment],
    model: str = OLLAMA_DEFAULT_MODEL,
    base_url: str = OLLAMA_DEFAULT_BASE_URL,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[list[Segment], int]:
    """Correct each segment individually via Ollama.

    Per-segment correction is chosen over batching because:
    - The 1.5B model does not reliably follow output formatting instructions
    - Per-segment ensures each output maps exactly to one input segment
    - Each call is fast (~3s) — total time scales linearly with segment count

    Returns:
        (corrected_segments, number_of_corrected_segments)
    """
    if not ensure_model_available(model, base_url, progress_callback):
        print(f"[LLM] Model '{model}' is not available. Skipping LLM correction.", flush=True)
        return segments, 0

    total = len(segments)
    print(f"[LLM] Correcting {total} segments with '{model}' (~3s per segment)...", flush=True)

    n_corrected = 0
    result: list[Segment] = []

    for i, seg in enumerate(segments):
        new_text = _correct_single(seg.text, model, base_url)

        if new_text != seg.text:
            n_corrected += 1
            if n_corrected <= 3:
                print(f"[LLM]  [{i+1}/{total}] '{seg.text[:50]}' → '{new_text[:50]}'", flush=True)

        result.append(Segment(
            seg.start, seg.end, new_text, seg.speaker,
            language=seg.language, emotion=seg.emotion, events=list(seg.events),
        ))

        if progress_callback and total > 0:
            progress_callback("correcting", i + 1, total)

    print(f"[LLM] Done: {n_corrected}/{total} segments corrected.", flush=True)
    return result, n_corrected


def _fmt_bytes(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}GB"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}MB"
    if n >= 1_000:
        return f"{n / 1_000:.0f}KB"
    return f"{n}B"
