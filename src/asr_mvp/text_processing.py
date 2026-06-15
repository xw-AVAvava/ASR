from __future__ import annotations

import json
from pathlib import Path
import re

from .schemas import Segment


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
    polished = text.strip()
    for wrong, right in replacements or []:
        polished = polished.replace(wrong, right)
    polished = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", polished)
    polished = re.sub(r"\s+", " ", polished)
    return polished.strip()

def polish_segments(segments: list[Segment], replacements: list[tuple[str, str]] | None = None) -> list[Segment]:
    polished_segments = []
    for seg in segments:
        text = polish_asr_text(seg.text, replacements)
        if text:
            polished_segments.append(Segment(seg.start, seg.end, text, seg.speaker))
    return polished_segments

def merge_short_segments(
    segments: list[Segment],
    gap_seconds: float,
    max_chars: int,
    require_same_speaker: bool = False,
) -> list[Segment]:
    if not segments:
        return []
    merged = [Segment(segments[0].start, segments[0].end, segments[0].text, segments[0].speaker)]
    for seg in segments[1:]:
        current = merged[-1]
        gap = seg.start - current.end
        combined_text = f"{current.text}{seg.text}"
        same_speaker = current.speaker == seg.speaker
        if gap <= gap_seconds and len(combined_text) <= max_chars and (same_speaker or not require_same_speaker):
            current.end = seg.end
            current.text = combined_text
        else:
            merged.append(Segment(seg.start, seg.end, seg.text, seg.speaker))
    return merged

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
