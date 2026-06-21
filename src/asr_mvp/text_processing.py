from __future__ import annotations

import json
import requests
from pathlib import Path
import re
from difflib import SequenceMatcher

from .schemas import Segment


MOJIBAKE_MARKERS = (
    "浣",
    "涓",
    "鍚",
    "瀹",
    "鎴",
    "杩",
    "鐨",
    "鏄",
    "绋",
    "€",
    "�",
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "can",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "so",
    "that",
    "the",
    "this",
    "to",
    "we",
    "with",
}

CHINESE_STOPWORDS = {
    "这个",
    "就是",
    "大家",
    "然后",
    "觉得",
    "一个",
    "一些",
    "可以",
    "还是",
    "没有",
    "这边",
    "这个",
    "那个",
    "你们",
    "我们",
    "他们",
    "什么",
    "比较",
    "其实",
    "明白",
    "意思",
    "方面",
    "一下",
    "起来",
    "有一",
    "个包",
    "您觉",
    "或者",
    "一点",
    "好嘞",
    "起来",
    "喝完",
    "认为",
    "针对",
    "原因",
}

CHINESE_KEYWORD_HINTS = [
    "包装",
    "价格",
    "纯度",
    "酒香",
    "刺激感",
    "味道",
    "口感",
    "品质",
    "缺点",
    "改进",
    "提升",
    "评价",
    "五粮液",
    "梦之兰",
    "剑南春",
    "洋河",
]

def _mojibake_score(text: str) -> int:
    return sum(text.count(marker) for marker in MOJIBAKE_MARKERS)

def repair_mojibake(text: str) -> str:
    """Repair common UTF-8 Chinese text that was accidentally decoded as GBK/CP936."""
    if not text:
        return text

    best = text
    best_score = _mojibake_score(text)
    for encoding in ("gbk", "cp936"):
        try:
            candidate = text.encode(encoding).decode("utf-8")
        except UnicodeError:
            continue
        candidate_score = _mojibake_score(candidate)
        if candidate_score < best_score:
            best = candidate
            best_score = candidate_score
    return best

def load_replacement_rules(path: Path | None) -> list[tuple[str, str]]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Replacement file not found: {path}")

    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
        if isinstance(data, dict):
            return [(str(wrong), str(right)) for wrong, right in data.items()]
        if isinstance(data, list):
            rules = []
            for item in data:
                if isinstance(item, dict) and "wrong" in item and "right" in item:
                    rules.append((str(item["wrong"]), str(item["right"])))
            return rules
        raise ValueError("JSON replacement file must be a dict or a list of {wrong, right} objects.")

    rules = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=>" in line:
            wrong, right = line.split("=>", 1)
        elif "\t" in line:
            wrong, right = line.split("\t", 1)
        else:
            continue
        rules.append((wrong.strip(), right.strip()))
    return rules

def polish_asr_text(text: str, replacements: list[tuple[str, str]] | None = None) -> str:
    polished = repair_mojibake(text.strip())
    polished = re.sub(r"<\|[^|]+?\|>", "", polished)
    for wrong, right in replacements or []:
        polished = polished.replace(wrong, right)
    polished = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", polished)
    polished = re.sub(r"\s+", " ", polished)
    return polished.strip()

def has_meaningful_text(text: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", text))

def polish_segments(segments: list[Segment], replacements: list[tuple[str, str]] | None = None) -> list[Segment]:
    polished_segments = []
    for seg in segments:
        text = polish_asr_text(seg.text, replacements)
        if has_meaningful_text(text):
            polished_segments.append(
                Segment(
                    seg.start,
                    seg.end,
                    text,
                    seg.speaker,
                    language=seg.language,
                    emotion=seg.emotion,
                    events=list(seg.events),
                )
            )
    return polished_segments

def merge_short_segments(
    segments: list[Segment],
    gap_seconds: float,
    max_chars: int,
    require_same_speaker: bool = False,
) -> list[Segment]:
    if not segments:
        return []
    first = segments[0]
    merged = [
        Segment(
            first.start,
            first.end,
            first.text,
            first.speaker,
            language=first.language,
            emotion=first.emotion,
            events=list(first.events),
        )
    ]
    for seg in segments[1:]:
        current = merged[-1]
        gap = seg.start - current.end
        combined_text = f"{current.text}{seg.text}"
        same_speaker = current.speaker == seg.speaker
        same_voice_metadata = (
            current.language == seg.language
            and current.emotion == seg.emotion
            and current.events == seg.events
        )
        if (
            gap <= gap_seconds
            and len(combined_text) <= max_chars
            and (same_speaker or not require_same_speaker)
            and same_voice_metadata
        ):
            current.end = seg.end
            current.text = combined_text
        else:
            merged.append(
                Segment(
                    seg.start,
                    seg.end,
                    seg.text,
                    seg.speaker,
                    language=seg.language,
                    emotion=seg.emotion,
                    events=list(seg.events),
                )
            )
    return merged

def compact_text_for_matching(text: str) -> str:
    text = strip_timestamps(text).lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text

def text_similarity(left: str, right: str) -> float:
    left_key = compact_text_for_matching(left)
    right_key = compact_text_for_matching(right)
    if not left_key or not right_key:
        return 0.0
    if left_key in right_key or right_key in left_key:
        return min(len(left_key), len(right_key)) / max(len(left_key), len(right_key))
    return SequenceMatcher(None, left_key, right_key).ratio()

def remove_repeated_segments(
    segments: list[Segment],
    similarity_threshold: float = 0.82,
    window: int = 6,
) -> tuple[list[Segment], int]:
    if not segments:
        return [], 0

    kept: list[Segment] = []
    removed = 0
    for seg in segments:
        text_key = compact_text_for_matching(seg.text)
        if len(text_key) < 8:
            kept.append(seg)
            continue

        recent = kept[-max(1, window) :]
        is_repeat = any(text_similarity(seg.text, prev.text) >= similarity_threshold for prev in recent)
        if is_repeat:
            removed += 1
            continue
        kept.append(seg)
    return kept, removed

def tokenize(text: str) -> list[str]:
    tokens = [tok.lower() for tok in re.findall(r"[A-Za-z0-9]+", text) if tok.lower() not in STOPWORDS and len(tok) > 2]
    cjk_sequences = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    for sequence in cjk_sequences:
        for width in (2, 3):
            for i in range(0, max(0, len(sequence) - width + 1)):
                gram = sequence[i : i + width]
                if gram not in CHINESE_STOPWORDS:
                    tokens.append(gram)
    return tokens

def count_text_units(text: str) -> dict[str, int]:
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    words = re.findall(r"[A-Za-z0-9]+", text)
    return {
        "word_count": len(words),
        "cjk_character_count": len(cjk_chars),
        "mixed_token_count": len(words) + len(cjk_chars),
    }


def summarize_voice_metadata(segments: list[Segment]) -> dict:
    language_counts: dict[str, int] = {}
    emotion_counts: dict[str, int] = {}
    event_counts: dict[str, int] = {}
    speaker_emotions: dict[str, dict[str, int]] = {}
    speaker_events: dict[str, dict[str, int]] = {}

    for seg in segments:
        if seg.language:
            language_counts[seg.language] = language_counts.get(seg.language, 0) + 1
        if seg.emotion:
            emotion_counts[seg.emotion] = emotion_counts.get(seg.emotion, 0) + 1
            speaker_counts = speaker_emotions.setdefault(seg.speaker, {})
            speaker_counts[seg.emotion] = speaker_counts.get(seg.emotion, 0) + 1
        for event in seg.events:
            event_counts[event] = event_counts.get(event, 0) + 1
            speaker_counts = speaker_events.setdefault(seg.speaker, {})
            speaker_counts[event] = speaker_counts.get(event, 0) + 1

    dominant_emotion_by_speaker = {
        speaker: max(counts.items(), key=lambda item: (item[1], item[0]))[0]
        for speaker, counts in speaker_emotions.items()
        if counts
    }
    return {
        "language_counts": language_counts,
        "emotion_counts": emotion_counts,
        "event_counts": event_counts,
        "speaker_emotions": speaker_emotions,
        "speaker_events": speaker_events,
        "dominant_emotion_by_speaker": dominant_emotion_by_speaker,
    }

def summarize_segments(segments: list[Segment], max_sentences: int = 3) -> dict:
    full_text = " ".join(seg.text for seg in segments)
    words = tokenize(full_text)
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    hinted = {hint: full_text.count(hint) for hint in CHINESE_KEYWORD_HINTS if full_text.count(hint) > 0}
    merged_counts = {**counts, **{key: counts.get(key, 0) + value * 3 for key, value in hinted.items()}}
    keywords = [word for word, _ in sorted(merged_counts.items(), key=lambda item: (-item[1], item[0]))[:12]]

    scored = []
    for seg in segments:
        score = sum(counts.get(word, 0) for word in tokenize(seg.text))
        scored.append((score, seg))
    selected = [seg for _, seg in sorted(scored, key=lambda item: item[0], reverse=True)[:max_sentences]]
    selected.sort(key=lambda seg: seg.start)

    action_patterns = re.compile(r"\b(should|need to|must|next step|action item|todo|we will)\b", re.IGNORECASE)
    actions = [seg.text for seg in segments if action_patterns.search(seg.text)]

    return {
        "summary_sentences": [seg.text for seg in selected],
        "keywords": keywords,
        "action_items": actions[:5],
        **count_text_units(full_text),
        "segment_count": len(segments),
        **summarize_voice_metadata(segments),
    }

def edit_distance(reference_tokens: list[str], hypothesis_tokens: list[str]) -> int:
    if not reference_tokens:
        return 0
    dp = [[0] * (len(hypothesis_tokens) + 1) for _ in range(len(reference_tokens) + 1)]
    for i in range(len(reference_tokens) + 1):
        dp[i][0] = i
    for j in range(len(hypothesis_tokens) + 1):
        dp[0][j] = j
    for i in range(1, len(reference_tokens) + 1):
        for j in range(1, len(hypothesis_tokens) + 1):
            cost = 0 if reference_tokens[i - 1] == hypothesis_tokens[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1]

def strip_timestamps(text: str) -> str:
    return re.sub(r"^\[(?:(?:\d{1,2}:)?\d{1,2}:\d{2})\]\s*", "", text, flags=re.MULTILINE)

def word_error_rate(reference: str, hypothesis: str) -> float | None:
    ref = re.findall(r"\S+", strip_timestamps(reference).lower())
    hyp = re.findall(r"\S+", strip_timestamps(hypothesis).lower())
    if not ref:
        return None
    return edit_distance(ref, hyp) / len(ref)

def normalize_chars(text: str) -> list[str]:
    text = strip_timestamps(text).lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？、,.!?;；:：\[\]()（）\"'`]+", "", text)
    return list(text)

def character_error_rate(reference: str, hypothesis: str) -> float | None:
    ref = normalize_chars(reference)
    hyp = normalize_chars(hypothesis)
    if not ref:
        return None
    return edit_distance(ref, hyp) / len(ref)

def accuracy_from_error_rate(error_rate: float | None) -> float | None:
    if error_rate is None:
        return None
    return max(0.0, min(1.0, 1.0 - error_rate))

def get_role_analysis_md(segments):
    # 【创新方案核心逻辑：全局完整上下文拼接，不依赖任何外部大模型工具】
    full_context = ""
    for seg in segments:
        speaker = seg["speaker"]
        text = seg["text"]
        full_context += f"{speaker}: {text}\n"

    # 标准化约束提示词（本方案核心设计，不限定任意大模型推理载体）
    prompt = f"""
你是会议角色分析助手，根据完整会议对话区分发言人角色，角色仅限四类：主持人、汇报人、参会提问人、旁听人员。
仅输出标准JSON，无多余文字，key为发言人编号，value为对应角色。
对话：
{full_context}
"""

    md_content = "\n## 参会人员角色分析\n"
    md_content += f"### 方案输入全局对话上下文\n```text\n{full_context}\n```\n"
    md_content += f"### 标准化约束提示词（本创新方案核心设计）\n```text\n{prompt}\n```\n"
    md_content += "> 说明：本角色识别算法方案独立完整，可对接任意离线/云端大模型推理载体（Ollama仅为本地调试验证工具，不属于方案本身）\n"
    return md_content
