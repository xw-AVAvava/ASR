from __future__ import annotations

from dataclasses import asdict
import json

from .audio_io import format_timestamp
from .schemas import AudioInfo, PipelineConfig, Segment


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
            raw_lines.append(f"- `{format_timestamp(seg.start)} -> {format_timestamp(seg.end)}`: {seg.text}")
        (config.output / "raw_transcript.md").write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    transcript_lines = ["# 清理后的带说话人标签转写文本", ""]
    for seg in segments:
        transcript_lines.append(f"- `{format_timestamp(seg.start)} -> {format_timestamp(seg.end)}` **{seg.speaker}**: {seg.text}")
    (config.output / "transcript.md").write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")

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
        f"- 原始 ASR WER: `{metadata.get('raw_wer')}`",
        f"- 原始 ASR CER: `{metadata.get('raw_cer')}`",
        f"- 展示文本 WER: `{metadata.get('display_wer')}`",
        f"- 展示文本 CER: `{metadata.get('display_cer')}`",
        "",
        "## 说话人统计",
        "",
        *speaker_duration_lines,
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
        "runtime_seconds": metadata["runtime_seconds"],
        "asr_engine_used": metadata["asr_engine_used"],
        "diarizer_used": metadata["diarizer_used"],
        "raw_segment_count": metadata.get("raw_segment_count"),
        "postprocess": metadata.get("postprocess"),
        "speaker_durations_seconds": {key: round(value, 3) for key, value in sorted(speaker_durations.items())},
    }
    (config.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
