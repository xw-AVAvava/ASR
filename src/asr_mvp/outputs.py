from __future__ import annotations

from dataclasses import asdict
import json

from .audio_io import format_timestamp
from .schemas import AudioInfo, PipelineConfig, Segment


def format_voice_labels(seg: Segment) -> str:
    labels = [label for label in (seg.language, seg.emotion) if label]
    labels.extend(seg.events)
    return " ".join(f"[{label}]" for label in labels)


def format_count_map(counts: dict[str, int]) -> str:
    if not counts:
        return "无"
    return ", ".join(f"{label}={count}" for label, count in sorted(counts.items()))


def write_markdown_outputs(
    config: PipelineConfig,
    audio_info: AudioInfo,
    segments: list[Segment],
    summary: dict,
    metadata: dict,
    raw_segments: list[Segment] | None = None,
) -> None:
    config.output.mkdir(parents=True, exist_ok=True)
    speaker_durations: dict[str, float] = {}
    for seg in segments:
        speaker_durations[seg.speaker] = speaker_durations.get(seg.speaker, 0.0) + max(0.0, seg.end - seg.start)
    speaker_duration_lines = [
        f"- {speaker}: `{round(duration, 2)}` 秒"
        for speaker, duration in sorted(speaker_durations.items())
    ]

    if raw_segments is not None:
        raw_lines = ["# 原始 ASR 转写文本", ""]
        for seg in raw_segments:
            voice_labels = format_voice_labels(seg)
            label_prefix = f" {voice_labels}" if voice_labels else ""
            raw_lines.append(
                f"- `{format_timestamp(seg.start)} -> {format_timestamp(seg.end)}`{label_prefix}: {seg.text}"
            )
        (config.output / "raw_transcript.md").write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    transcript_lines = ["# 清理后的带说话人标签转写文本", ""]
    for seg in segments:
        voice_labels = format_voice_labels(seg)
        label_suffix = f" {voice_labels}" if voice_labels else ""
        transcript_lines.append(
            f"- `{format_timestamp(seg.start)} -> {format_timestamp(seg.end)}` "
            f"**{seg.speaker}**{label_suffix}: {seg.text}"
        )
    (config.output / "transcript.md").write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")

    speaker_emotion_lines = [
        f"- **{speaker}**: 主导情绪 `{summary['dominant_emotion_by_speaker'].get(speaker, 'N/A')}`；"
        f"{format_count_map(counts)}"
        for speaker, counts in sorted(summary["speaker_emotions"].items())
    ]
    speaker_event_lines = [
        f"- **{speaker}**: {format_count_map(counts)}"
        for speaker, counts in sorted(summary["speaker_events"].items())
    ]
    emotion_timeline = []
    previous_emotion: dict[str, str] = {}
    for seg in sorted(segments, key=lambda item: item.start):
        if not seg.emotion or previous_emotion.get(seg.speaker) == seg.emotion:
            continue
        previous_emotion[seg.speaker] = seg.emotion
        preview = seg.text if len(seg.text) <= 70 else seg.text[:67] + "..."
        emotion_timeline.append(
            f"- `{format_timestamp(seg.start)}` **{seg.speaker}** -> `{seg.emotion}`: {preview}"
        )

    summary_lines = [
        "# 摘要",
        "",
        "## 自动摘要",
        "",
        *[f"- {sentence}" for sentence in summary["summary_sentences"]],
        "",
        "## 关键词",
        "",
        ", ".join(summary["keywords"]) if summary["keywords"] else "没有提取到关键词。",
        "",
        "## 行动项",
        "",
        *([f"- {item}" for item in summary["action_items"]] or ["没有发现明确行动项。"]),
        "",
        "## 说话人情绪",
        "",
        *(speaker_emotion_lines or ["没有检测到情绪标签。"]),
        "",
        "## 音频事件",
        "",
        f"- 总体事件: {format_count_map(summary['event_counts'])}",
        *(speaker_event_lines or ["- 没有检测到说话人相关音频事件。"]),
        "",
        "## 情绪转折时间线",
        "",
        *(emotion_timeline or ["没有检测到情绪转折。"]),
        "",
    ]
    (config.output / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    report_lines = [
        "# 实验报告",
        "",
        "## 输入信息",
        "",
        f"- 音频路径: `{audio_info.path}`",
        f"- 文件存在: `{audio_info.exists}`",
        f"- 音频时长（秒）: `{audio_info.duration_seconds}`",
        f"- 采样率: `{audio_info.sample_rate}`",
        f"- 声道数: `{audio_info.channels}`",
        f"- 文件大小（bytes）: `{audio_info.size_bytes}`",
        f"- 警告信息: `{audio_info.warning}`",
        "",
        "## 处理流程",
        "",
        f"- 使用的转写来源: `{metadata['asr_engine_used']}`",
        f"- 使用的说话人标注方式: `{metadata['diarizer_used']}`",
        f"- 请求模型: `{config.model}`",
        f"- 请求语言: `{config.language}`",
        f"- 运行时间（秒）: `{metadata['runtime_seconds']}`",
        f"- 原始 ASR 片段数: `{metadata.get('raw_segment_count')}`",
        f"- 后处理方式: `{metadata.get('postprocess')}`",
        "",
        "## 结果",
        "",
        f"- 文本片段数: `{summary['segment_count']}`",
        f"- 英文/数字词数: `{summary['word_count']}`",
        f"- 中文字符数: `{summary['cjk_character_count']}`",
        f"- 混合 token 数: `{summary['mixed_token_count']}`",
        f"- 说话人标签数: `{len(set(seg.speaker for seg in segments))}`",
        f"- 语言标签统计: `{format_count_map(summary['language_counts'])}`",
        f"- 情绪标签统计: `{format_count_map(summary['emotion_counts'])}`",
        f"- 音频事件统计: `{format_count_map(summary['event_counts'])}`",
        f"- 原始 ASR WER: `{metadata.get('raw_wer')}`",
        f"- 原始 ASR CER: `{metadata.get('raw_cer')}`",
        f"- 展示文本 WER: `{metadata.get('display_wer')}`",
        f"- 展示文本 CER: `{metadata.get('display_cer')}`",
        f"- 原始 ASR 字符准确率: `{metadata.get('raw_accuracy')}`",
        f"- 展示文本字符准确率: `{metadata.get('display_accuracy')}`",
        "",
        "## 说话人统计",
        "",
        *speaker_duration_lines,
        "",
        "## 说话人情绪统计",
        "",
        *(speaker_emotion_lines or ["- 没有检测到情绪标签。"]),
        "",
        "## 说话人音频事件统计",
        "",
        *(speaker_event_lines or ["- 没有检测到音频事件。"]),
        "",
        "## 局限性",
        "",
        "- 默认说话人标签是一个基线方法，不是完整的说话人日志分离模型。",
        "- 真实重叠语音仍然需要 pyannote.audio 或说话人嵌入聚类。",
        "- 真实 ASR 需要安装 Whisper 或 faster-whisper，并下载模型权重。",
        "- 可选自定义词表纠错属于规则后处理，不能替代更大的 ASR 模型或人工校对。",
        "",
        "## 后续改进",
        "",
        "- 接入带 `HF_TOKEN` 的 pyannote.audio 说话人分离。",
        "- 增加噪声音频实验，并比较 WER/CER。",
        "- 比较 `tiny`、`base`、`small` 等 ASR 模型的速度和质量。",
    ]
    (config.output / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    json_segments = [asdict(seg) for seg in segments]
    (config.output / "segments.json").write_text(json.dumps(json_segments, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics = {
        "duration_seconds": audio_info.duration_seconds,
        "segment_count": summary["segment_count"],
        "word_count": summary["word_count"],
        "cjk_character_count": summary["cjk_character_count"],
        "mixed_token_count": summary["mixed_token_count"],
        "speaker_count": len(set(seg.speaker for seg in segments)),
        "wer": metadata.get("wer"),
        "cer": metadata.get("cer"),
        "raw_wer": metadata.get("raw_wer"),
        "raw_cer": metadata.get("raw_cer"),
        "display_wer": metadata.get("display_wer"),
        "display_cer": metadata.get("display_cer"),
        "raw_accuracy": metadata.get("raw_accuracy"),
        "display_accuracy": metadata.get("display_accuracy"),
        "runtime_seconds": metadata["runtime_seconds"],
        "asr_engine_used": metadata["asr_engine_used"],
        "diarizer_used": metadata["diarizer_used"],
        "raw_segment_count": metadata.get("raw_segment_count"),
        "postprocess": metadata.get("postprocess"),
        "speaker_durations_seconds": {key: round(value, 3) for key, value in sorted(speaker_durations.items())},
        "language_counts": summary["language_counts"],
        "emotion_counts": summary["emotion_counts"],
        "event_counts": summary["event_counts"],
        "speaker_emotions": summary["speaker_emotions"],
        "speaker_events": summary["speaker_events"],
        "dominant_emotion_by_speaker": summary["dominant_emotion_by_speaker"],
    }
    (config.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
