# 论文驱动创新：说话人感知的情绪与音频事件会议分析

## 论文依据

An 等人在 *FunAudioLLM: Voice Understanding and Generation Foundation Models for Natural Interaction Between Humans and LLMs* 中提出，SenseVoice 将多语言语音识别、语音情绪识别（SER）和音频事件检测（AED）统一在同一语音理解模型中。

- 论文页面：https://arxiv.org/abs/2407.04051
- 论文 PDF：https://arxiv.org/pdf/2407.04051
- 官方实现：https://github.com/FunAudioLLM/SenseVoice

SenseVoice 支持 `HAPPY`、`SAD`、`ANGRY`、`NEUTRAL`、`FEARFUL`、`DISGUSTED`、`SURPRISED` 等情绪标签，以及掌声、笑声、哭声、咳嗽、喷嚏、呼吸声和背景音乐等音频事件。

## 本项目的创新

论文提供的是语音块级情绪和事件理解。本项目进一步将这些声学标签与 pyannote 的匿名说话人分离结果对齐，形成“谁在什么时候、以什么情绪、说了什么、伴随什么声音事件”的联合时间线。

```text
SenseVoice 文本/语言/情绪/事件
                +
pyannote 匿名说话人标签
                ↓
说话人感知的情绪与事件会议时间线
```

相较于普通会议转写，本项目新增：

1. 保留并结构化解析 SenseVoice 标签，而不是从文本中删除标签。
2. 为每个转写片段记录 `language`、`emotion` 和 `events`。
3. 将情绪与事件绑定到 `SPEAKER_XX`，统计每位说话人的主导情绪。
4. 检测同一说话人的情绪转折，并在摘要中生成时间线。
5. 在 GUI、Markdown 报告和 JSON 中同时呈现声学与文本信息。

这属于模型能力融合和会议场景方法创新，而不是声称重新训练或发明了 SenseVoice、pyannote 模型。

## 实验设计

### 对照组

- Baseline：文本 + 说话人标签。
- Proposed：文本 + 说话人标签 + 情绪 + 音频事件。

### 指标

- ASR：CER/WER。
- 说话人分离：人工抽查说话人归属，条件允许时计算 DER。
- 情绪识别：人工标注部分片段，计算 Accuracy 和 Macro-F1。
- 事件检测：对笑声、掌声、咳嗽等事件计算 Precision、Recall 和 F1。
- 摘要质量：由多名评价者对完整性、可读性和情绪信息帮助程度进行 1-5 分评分。

### 消融实验

1. 仅文本。
2. 文本 + 说话人。
3. 文本 + 说话人 + 情绪。
4. 文本 + 说话人 + 情绪 + 音频事件。

## 局限性

- 情绪标签是粗粒度声学分类，不代表对说话人心理状态的判断。
- 嘈杂、重叠和远场语音会同时影响说话人、情绪和事件结果。
- SenseVoice 返回标签块但不直接返回每个标签块的时间戳；本项目按标签块文本长度映射到音频时长，因此时间点是近似值。
- 情绪与事件结果应作为会议检索和摘要辅助信息，不应用于高风险人员评价。
