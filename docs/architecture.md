# 系统架构说明

本项目采用分层架构，把“音频输入、转写、说话人标注、文本后处理、摘要评估、文件输出、GUI 展示”拆成相对独立的模块。这样做的目的是让课程项目既能运行，也方便解释每一步用了哪些机器学习或工程方法。

## 目录结构

```text
project/asr_meeting_assistant/
  app.py                         # Streamlit 图形界面
  run_pipeline.py                # 命令行入口
  check_asr_setup.py             # ASR 环境检查
  test_whisper_load.py           # faster-whisper 加载诊断
  src/asr_mvp/
    schemas.py                   # Segment、AudioInfo、PipelineConfig 数据结构
    audio_io.py                  # 音频元数据检查、文本读取、时间格式化
    transcription.py             # 人工文本解析、faster-whisper/openai-whisper ASR
    diarization.py               # turns baseline、MFCC + KMeans、pyannote/trained-model 接口
    text_processing.py           # 文本清理、短句合并、摘要、关键词、WER/CER
    outputs.py                   # transcript/summary/report/json/metrics 输出
    pipeline.py                  # 主流程编排层
    audio_features.py            # 音频特征与 MFCC-like speaker embedding
    model_training.py            # 可选监督学习扩展，需使用自备带标签音频
  docs/
    architecture.md              # 本文件
    final_report.md              # 项目报告草稿
```

## 主流程

```text
Audio + optional transcript/reference
  -> inspect_audio
  -> transcribe_audio
  -> generic text cleanup
  -> speaker diarization
  -> short-segment merge
  -> summary + keywords
  -> WER/CER metrics
  -> markdown/json reports
```

## 模块边界

- `schemas.py`：只定义数据结构，不做业务逻辑。
- `audio_io.py`：只负责读取和检查输入，不依赖 ASR 或模型。
- `transcription.py`：负责把音频或人工文本转换为带时间戳的 `Segment`。
- `diarization.py`：负责给每个 `Segment` 分配匿名 speaker 标签。
- `text_processing.py`：负责文本清理、合并、摘要和评估指标。
- `outputs.py`：负责把内存结果写成可展示文件。
- `pipeline.py`：只负责连接上述模块，避免单文件越来越大。

## Speaker Diarization 设计

项目目前支持三种层次：

- `turns`：简单轮流标签，用于最小演示。
- `cluster`：从每个 ASR 片段切出对应音频，提取频谱特征和 MFCC-like 统计特征，再用 KMeans 聚类得到 `SPEAKER_00`、`SPEAKER_01` 等匿名标签。
- `pyannote` / `trained-model`：保留扩展接口，但不作为默认依赖。

`cluster` 方法可以展示课程中的特征提取、标准化、无监督聚类思想。但它不能识别真实姓名，只能把声音相似的片段分到同一匿名 speaker。

## 输出文件

每次运行会在指定输出目录生成：

```text
raw_transcript.md   # 原始 ASR 输出，如果有
transcript.md       # 清理/合并后的展示文本
summary.md          # 抽取式摘要、关键词、行动项
report.md           # 输入信息、流程、指标、局限性
segments.json       # 结构化片段
metrics.json        # CER/WER、speaker 数量、运行时间等
```

## 架构原则

- 不把老师提供的范例项目材料纳入本项目成果。
- 默认只使用用户提供的音频和文本。
- 生成输出放在 `outputs/`，作为运行产物，不作为源码核心。
- 真实 ASR、pyannote、监督分类器都设计成可选组件，避免环境依赖阻塞基础流程。
- 报告中诚实区分“匿名 speaker 聚类”和“真实身份识别”。
