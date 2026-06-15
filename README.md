# 会议音频智能助手 MVP

这是一个机器学习课程项目原型，用于把多人对话音频整理成可阅读、可评估、可展示的结构化材料。

核心输出包括：

- 原始 ASR 转写文本
- 清理后的带时间戳 transcript
- 匿名说话人标签，例如 `SPEAKER_00`
- 自动摘要、关键词和行动项
- WER/CER 等评估指标
- Markdown/JSON 实验报告

## 项目思路

本项目对应课程项目中的 ASR、多说话人对话、speaker diarization 和本地音频处理方向。整体流程是：

```text
音频输入 -> ASR/人工转写 -> 文本清理 -> 说话人聚类 -> 摘要与评估 -> 报告输出
```

说话人标签分为两个层次：

- `turns`：轮流标签基线，适合快速演示完整流程。
- `cluster`：从 ASR 时间片中切出音频，提取频谱/MFCC-like 特征，用 KMeans 聚类成匿名说话人标签。

注意：当前系统做的是匿名 speaker diarization，不是识别真实姓名。如果要识别真实身份，需要每个人的参考声音样本或人工标注。

## 项目架构

主代码已经拆成模块化结构：

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

详细说明见：

```text
project/asr_meeting_assistant/docs/architecture.md
```

## 图形界面 GUI

在项目根目录运行：

```powershell
python -m streamlit run project\asr_meeting_assistant\app.py
```

然后打开：

```text
http://localhost:8501
```

GUI 支持：

- 运行真实多人对话样本
- 上传自定义音频
- 使用已有 transcript 或自动 ASR
- 选择说话人标注方法：`cluster`、`turns`、`pyannote`、`trained-model`
- 查看摘要、转写文本、报告和 JSON 片段

## 真实多人对话样本

项目默认使用你放入的音频和文本：

```text
project/多人对话.wav
project/多人对话文本.txt
```

人工 transcript 模式，运行快、稳定，适合检查流程：

```powershell
python project\asr_meeting_assistant\run_pipeline.py `
  --audio project\多人对话.wav `
  --transcript-file project\多人对话文本.txt `
  --reference-file project\多人对话文本.txt `
  --output project\asr_meeting_assistant\outputs\real_dialogue_human_transcript `
  --engine demo `
  --language zh `
  --speakers 5 `
  --diarizer cluster
```

真实 ASR + 匿名说话人聚类模式：

```powershell
python project\asr_meeting_assistant\run_pipeline.py `
  --audio project\多人对话.wav `
  --reference-file project\多人对话文本.txt `
  --output project\asr_meeting_assistant\outputs\real_asr_base_cluster `
  --engine faster-whisper `
  --model base `
  --language zh `
  --speakers 5 `
  --diarizer cluster
```

## ASR 环境检查

如果使用 `faster-whisper`：

```powershell
python -m pip install faster-whisper -i https://pypi.tuna.tsinghua.edu.cn/simple
python project\asr_meeting_assistant\check_asr_setup.py
python project\asr_meeting_assistant\test_whisper_load.py
```

如果 `test_whisper_load.py` 能成功加载模型，就可以运行真实 ASR。

## 输出文件

每次运行会在 `--output` 指定目录生成：

```text
raw_transcript.md   原始 ASR 文本
transcript.md       清理后的带 speaker 标签文本
summary.md          摘要、关键词、行动项
report.md           实验报告
segments.json       结构化片段
metrics.json        指标和运行元数据
```

`outputs/` 是运行产物，已经加入 `.gitignore`，最终提交时重点提交源码和文档。

## 可选扩展

项目保留了监督学习训练脚本：

```powershell
python project\asr_meeting_assistant\train_audio_classifier.py `
  --data-dir project\asr_meeting_assistant\data\labeled_audio `
  --output project\asr_meeting_assistant\outputs\audio_classifier `
  --test-size 0.3
```

这个脚本只应该用于我们自己准备的带标签音频数据，不使用老师提供的范例项目材料。没有自备标签数据时，不把它作为最终项目核心成果。

## 视频展示建议

1. 展示问题：多人会议/访谈音频难以检索、总结和复盘。
2. 展示系统流程：音频、ASR、匿名 speaker 标签、摘要、指标、报告。
3. 展示 GUI：运行真实多人对话样本。
4. 展示课程 ML 连接：MFCC-like 特征、标准化、KMeans 聚类、CER/WER。
5. 诚实说明局限：当前是课程级 diarization 原型，不是商业级 ASR/真实身份识别系统。
