# 项目报告草稿

## 项目标题

会议音频智能助手：面向多人对话的 ASR、匿名说话人聚类与摘要生成系统

## 项目动机

课堂、会议和访谈录音通常信息量很大，人工回听成本高。用户往往需要快速知道：音频中说了什么、什么时候说的、不同说话人大致出现在哪些片段、有哪些关键词和结论。本项目希望把原始多人音频整理成结构化文本，方便检索、总结和复盘。

## 系统概览

系统流程如下：

```text
音频输入 -> ASR/人工转写 -> 文本清理 -> 匿名说话人聚类 -> 摘要与关键词 -> 指标评估 -> 报告输出
```

系统会生成：

- `raw_transcript.md`：原始 ASR 输出文本
- `transcript.md`：清理后的带时间戳和 speaker 标签的文本
- `summary.md`：自动摘要、关键词和行动项
- `report.md`：音频元数据、流程说明、指标和局限性
- `segments.json`：机器可读的片段数据
- `metrics.json`：CER/WER、speaker 数量、运行时间等指标

## 项目架构

项目已经拆成模块化结构：

```text
src/asr_mvp/
  schemas.py          数据结构
  audio_io.py         音频检查和文本读取
  transcription.py    ASR 与人工 transcript 解析
  diarization.py      说话人标签与聚类
  text_processing.py  文本清理、摘要、WER/CER
  outputs.py          Markdown/JSON 输出
  pipeline.py         主流程编排
```

这种设计让 ASR、说话人标注、摘要、评估和输出相互解耦，方便替换模型或增加实验。

## 方法

### 1. 自动语音识别

项目支持两种转写模式：

- 人工 transcript 模式：使用已有文本，稳定验证完整流程。
- 真实 ASR 模式：使用 `faster-whisper` 对音频进行自动转写。

在当前实验中，我们使用 `faster-whisper base` 模型对约 9 分钟中文多人对话进行转写，并使用人工文本计算 CER/WER。

### 2. 匿名说话人标注

项目不识别真实姓名，而是进行匿名 speaker diarization：

```text
SPEAKER_00
SPEAKER_01
SPEAKER_02
...
```

目前支持两种方法：

- `turns`：轮流标签基线。
- `cluster`：从每个 ASR 时间片中切出音频，提取频谱统计特征和 MFCC-like 特征，经过标准化后使用 KMeans 聚类。

`cluster` 方法体现了课程中的特征提取、标准化和无监督学习思想。它比简单轮流标签更接近真实 diarization，但仍不是商业级模型。

### 3. 文本后处理与摘要

系统对 ASR 文本进行通用清理，例如去掉中文字符之间多余空格，并把过短片段合并成更适合阅读的段落。

摘要采用抽取式方法：根据关键词频率给片段打分，选出代表性片段，同时提取关键词和行动项。

### 4. 评估指标

项目主要关注中文 CER，因为中文没有天然空格，WER 对中文不够稳定。系统同时保留：

- `raw_cer` / `raw_wer`：原始 ASR 输出指标
- `display_cer` / `display_wer`：清理合并后的展示文本指标

## 真实多人对话样本

项目使用团队提供的真实多人对话音频和人工文本：

```text
project/多人对话.wav
project/多人对话文本.txt
```

音频元数据：

- 时长：约 539.9 秒
- 采样率：16 kHz
- 声道数：1

当前较好的真实 ASR 实验结果：

```text
模型: faster-whisper base
运行时间: 约 88 秒
原始 ASR CER: 0.2863
原始 ASR 片段数: 287
后处理片段数: 32
```

这说明 `base` 模型在该音频上比 `tiny` 更准确，但仍会受到多人对话、口语表达、重叠说话和音频质量影响。

## 局限性

- 当前 speaker diarization 是匿名聚类，不能识别真实姓名。
- 如果要识别具体身份，需要每个说话人的参考声音样本或人工标注。
- MFCC + KMeans 对重叠语音和短句片段仍然有限。
- 中文 WER 不够稳定，报告中应重点解释 CER。
- `faster-whisper` 依赖本地 Python/CPU 环境，部署时需要额外环境检查。

## 后续工作

- 在更多自备多人音频上测试 ASR 和 speaker clustering 稳定性。
- 引入更强说话人嵌入模型，例如 SpeechBrain 或 pyannote。
- 增加 VAD、降噪、音量归一化等预处理。
- 比较 `tiny`、`base`、`small` 等 ASR 模型的速度和准确率。
- 如果有真实 speaker 标签，可计算 diarization error rate 或聚类纯度。

## 总结

本项目完成了一个端到端多人音频处理流程：从音频输入，到真实 ASR、匿名说话人聚类、文本后处理、摘要生成、CER/WER 评估和 GUI 展示。项目没有使用老师提供的范例项目作为成果数据，核心实验基于团队提供的多人音频和文本。整体上，它既能运行，也能展示机器学习课程中的特征提取、无监督聚类、模型评估和工程化组织思想。
