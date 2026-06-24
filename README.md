# 会议音频智能助手

本项目将会议、访谈等音频转换为带时间戳和匿名说话人标签的文本，并生成摘要、关键词、CER/WER 指标及 Markdown/JSON 报告。

## 功能

- FunASR SenseVoiceSmall 中文语音识别
- SenseVoice 语言、情绪与音频事件标签解析
- pyannote.audio 多说话人分离（可选）
- MFCC + KMeans 说话人聚类（无需 Hugging Face Token）
- 单人音频快速模式：人数设为 `1` 时跳过说话人模型
- NVIDIA CUDA 自动加速，无可用显卡时回退 CPU
- Streamlit 可视化界面
- 原始文本、整理文本、摘要、报告和结构化 JSON 输出
- 本地离线大模型全局语义参会角色智能识别，区分主持人、汇报人、参会提问人、旁听人员
- Ollama 本地 LLM 逐句纠错（可选），自动拉取模型，离线可用

### 研究参考文献（说话人感知的情绪与音频事件会议分析模块）
本项目将 SenseVoice 的语言、情绪和音频事件标签与匿名说话人结果对齐，形成“谁在什么时候、以什么情绪、说了什么、伴随什么声音事件”的联合时间线：

**FunAudioLLM: Voice Understanding and Generation Foundation Models for Natural Interaction Between Humans and LLMs**
- 创新说明：[说话人感知的情绪与音频事件会议分析模块创新说明](docs/innovation_emotion_events.md)
- 论文来源：arXiv 2024
- 论文页面：[https://arxiv.org/abs/2407.04051](https://arxiv.org/abs/2407.04051)
- 论文 PDF：[https://arxiv.org/pdf/2407.04051](https://arxiv.org/pdf/2407.04051)
- 官方实现：[FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)
- 对应创新点：保留并结构化解析 SenseVoice 的语言、情绪和音频事件标签，再与 pyannote 或聚类得到的 `SPEAKER_XX` 对齐。
- 核心支撑：论文提出 SenseVoice 在同一模型中联合提供多语言 ASR、语音情绪识别和音频事件检测能力。本项目进一步生成说话人级主导情绪、声音事件统计和情绪转折时间线，并在 GUI、Markdown 与 JSON 中统一呈现。

### 研究参考文献（双VAD融合模块）  
本项目中双VAD融合与重叠语音检测优化方案参考以下学术文献：  
 **Attention Is Not Always the Answer: Optimizing Voice Activity Detection with Simple Feature Fusion**  
    - 发表会议：INTERSPEECH 2025  
    - 论文链接：[https://arxiv.org/pdf/2506.01365.pdf](https://arxiv.org/pdf/2506.01365.pdf)  
    - 对应创新点：双VAD决策级交集融合  
    - 核心支撑：文献验证了融合不同架构的VAD特征可有效降低检测错误率（DER），提升噪声场景鲁棒性。本项目基于该结论，实现SenseVoice声学VAD与pyannote分割VAD的交叉校验，减少端点误判。

### 研究参考文献（LLM纠错模块）  
本项目中的 ASR 后处理逐句纠错方案参考以下学术文献：  
 **ASR-EC Benchmark: Evaluating Large Language Models on Chinese ASR Error Correction**  
    - 发表会议：arXiv 预印本，2024 年 12 月  
    - 论文链接：[https://arxiv.org/abs/2412.03075](https://arxiv.org/abs/2412.03075)  
    - 对应创新点：Ollama 本地 LLM 对 ASR 转录文本逐句错别字校对  
    - 核心支撑：该基准系统评估了 LLM 在中文 ASR 纠错任务上的表现，涵盖提示工程（zero-shot/few-shot）、微调和多模态增强三种范式。实验结果验证了通过精心设计的 prompt 引导 LLM 进行 ASR 后处理纠错的可行性，为本地化部署小参数模型（如 Qwen2.5-1.5B）进行逐句校对提供了方法论依据。

**LLM纠错模块设计思路**

该模块作为 ASR pipeline 的可选后处理步骤，位于文本清洗（polish）之后、说话人分离（diarization）之前：

```
ASR转录 → 去除标签/文本清洗 → [LLM逐句纠错] → 说话人分离 → 合并/去重 → 输出
```

**关键设计决策：**

- **逐句纠错而非批量拼接**：每条 ASR 片段独立发送给 LLM。批量拼接要求模型严格遵循输出格式（编号/分隔符），小参数模型（1.5B）对此类指令的遵循能力不足，容易产生片段串位、漏行或幻觉拼接。逐句模式虽增加 API 调用次数，但保证了每条输出与输入片段的一一对应，不破坏时间戳对齐。
- **Few-shot 示例驱动**：prompt 中嵌入两组"原文→校对"示例，用具体样例约束模型的输出行为，比纯规则描述更有效地抑制了模型自由改写和增删字词的倾向。
- **长度安全校验**：若纠错输出长度偏离原文 ±50%，自动丢弃并保留原文，防止幻觉输出破坏转录内容。
- **自动模型拉取**：通过 Ollama `/api/pull` 端点实现首次使用时自动下载模型，用户无需手动执行 `ollama pull`。
- **降级容错**：Ollama 不可用或模型下载失败时，纠错步骤自动跳过，pipeline 正常完成，不影响核心转录功能。
- **GUI 状态指示**：Streamlit 侧边栏实时检测 Ollama 连接状态和模型就绪情况，以绿/黄/红色状态框反馈。

**已知局限：** 小参数模型（<2B）的指令遵循能力有限，偶尔会出现漏改或轻微改写。换用 7B 及以上模型可显著提升纠错质量和一致性。

说话人标签如 `SPEAKER_00` 只区分声音，不代表真实姓名。

### 研究参考文献（离线大模型参会角色识别模块）
本项目全局上下文驱动的参会角色自动识别方案参考以下学术文献：
1. CASCA: Leveraging Role-Based Lexical Cues for Meeting Speaker Role Classification
- 发表会议：ICNLSP 2024
- 论文链接：https://aclanthology.org/2024.icnlsp-1.42.pdf
- 对应创新点：基于完整会议全局对话上下文无监督区分四类会议发言角色
- 核心支撑：文献证明仅依靠单句片段极易误判角色，完整全局对话能大幅提升分类准确率；本项目采用本地离线LLM完成推理，无需上传会议原始音频文本至外部云端，保障数据隐私，且完全独立于底层音频分割、语音识别模块，仅对输出文本做后处理拓展，不改动原有音频处理逻辑。
2. ASR-Synchronized Speaker Role Detection for Offline Meeting Analysis
- 论文链接：https://arxiv.org/pdf/2507.17765
- 对应创新点：离线本地部署大模型完成会后文本角色分析，无需在线API
- 核心支撑：区分“说话人分割（谁在说话）”与“发言角色识别（此人在会议中职能）”为两个独立任务，本模块作为上层拓展功能，与原有双VAD、多说话人分割模块分层解耦，无代码、功能重叠冲突。

## 环境要求

- Windows 10/11
- Python 3.12（推荐）
- 约 8 GB 可用磁盘空间；首次使用时模型会下载到用户缓存目录
- NVIDIA 显卡可选，本项目测试环境为 RTX 4060 Laptop GPU

不要提交 `.venv`、模型缓存或 Token。虚拟环境包含本机路径和平台相关二进制，无法保证换一台电脑后仍可运行。

## 依赖文件


| 文件                          | 用途                                       |
| --------------------------- | ---------------------------------------- |
| `requirements.txt`          | GUI、FunASR、ModelScope 和基础 PyTorch        |
| `requirements-pyannote.txt` | 在基础依赖上增加 pyannote、TorchCodec、torchvision |
| `requirements-cuda.txt`     | CUDA 12.8 版 torch、torchvision、torchaudio |


主要版本：

- FunASR `1.3.10`
- ModelScope `1.37.1`
- pyannote.audio `4.0.4`（可选）
- PyTorch `2.11.0`
- Streamlit `1.58.0`
- scikit-learn `1.9.0`

## 一键安装

### 方案 A：使用项目虚拟环境

基础 FunASR，CPU 或由本机 PyTorch 自动选择设备：

```powershell
cd C:\Users\你的用户名\ASR
.\setup.ps1
```

FunASR + NVIDIA CUDA：

```powershell
.\setup.ps1 -Cuda
```

FunASR + pyannote：

```powershell
.\setup.ps1 -Pyannote
```

完整 CUDA + pyannote 环境：

```powershell
.\setup.ps1 -Cuda -Pyannote
```

CUDA 官方源较慢时可切换南京大学镜像：

```powershell
.\setup.ps1 -Cuda -Pyannote `
  -TorchIndexUrl "https://mirror.nju.edu.cn/pytorch/whl/cu128"
```

脚本默认创建 `.venv`。pyannote 模式还会把 Windows shared FFmpeg 下载到 `.runtime/`，这两个目录都已被 Git 忽略。

### 方案 B：使用已有 Python/Conda 环境

先激活自己的环境，然后运行：

```powershell
conda activate 你的环境名
.\setup.ps1 -UseCurrentEnvironment
```

根据需要组合参数：

```powershell
.\setup.ps1 -UseCurrentEnvironment -Cuda
.\setup.ps1 -UseCurrentEnvironment -Pyannote
.\setup.ps1 -UseCurrentEnvironment -Cuda -Pyannote
```

也可以手动安装：

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-pyannote.txt
python -m pip install -r requirements-cuda.txt `
  --extra-index-url https://download.pytorch.org/whl/cu128
```

只安装实际需要的文件。未使用 pyannote 时，不需要安装 `requirements-pyannote.txt`，也不需要 Hugging Face Token。

## pyannote 与 Hugging Face Token（可选）

只有在界面选择 `pyannote` 说话人标注时才需要以下步骤。使用 `cluster`、`turns`，或者将说话人数设为 `1` 时均不需要 Token。

1. 注册并登录 [Hugging Face](https://huggingface.co/)。
2. 使用同一个账号分别打开并接受以下仓库的使用条件：
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
  - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
3. 打开 [Access Tokens](https://huggingface.co/settings/tokens)，创建一个 `Read` Token。
4. 若创建 Fine-grained Token，需要允许读取已获准访问的 gated repositories。
5. 在启动程序的同一个 PowerShell 窗口中设置：

```powershell
$env:HF_TOKEN="hf_你的Token"
```

Token 相当于密码。不要把它写进代码、`.env`、截图、聊天记录或 Git 提交。每位协作者应使用自己的 Token。

## 启动可视化界面

使用项目 `.venv`：

```powershell
.\run_gui.ps1
```

使用当前已激活的 Python/Conda 环境：

```powershell
.\run_gui.ps1 -UseCurrentEnvironment
```

打开：

```text
http://localhost:8501
```

界面操作：

1. 选择项目样本或上传音频。
2. 选择 `FunASR SenseVoiceSmall`。
3. 按录音内容填写语言代码；代码含义见下方“语言标签选择”。
4. `cluster` 和 `pyannote` 均可勾选“自动估计说话人数量”；取消勾选后可用滑块指定固定人数。
5. 确定是单人音频时取消自动估计并将人数设为 `1`，程序会跳过说话人分离以节省时间。
6. 多人音频可选择 `cluster` 或 `pyannote`；pyannote 通常更准确，但需要额外依赖和 Token。
7. 点击“运行处理流程”。

### 语言标签选择

SenseVoiceSmall 支持中文、英语、粤语、日语和韩语。在界面的“语言代码”中填写：


| 语言代码   | 录音语言   | 建议使用场景           |
| ------ | ------ | ---------------- |
| `auto` | 自动识别   | 不确定语言，或录音中包含多种语言 |
| `zh`   | 普通话/中文 | 主要内容为普通话时使用      |
| `en`   | 英语     | 主要内容为英语时使用       |
| `yue`  | 粤语     | 主要内容为粤语时使用       |
| `ja`   | 日语     | 主要内容为日语时使用       |
| `ko`   | 韩语     | 主要内容为韩语时使用       |


已知录音的主要语言时，建议填写对应代码；中英混合或语言不确定时填写 `auto`。输出中的 `ZH`、`EN`、`YUE`、`JA`、`KO` 是 SenseVoice 对各片段生成的语言标签，不是说话人标签。

首次运行会下载 SenseVoice 和 pyannote 模型，之后从本地缓存加载。

真实示例录音不会提交到公开仓库。clone 后请在界面上传自己的音频；如已通过其他受控渠道取得 `多人对话.wav`，可将它放在项目根目录。

## 输出

运行结果写入 `outputs/gui_last_run/`：

```text
raw_transcript.md
transcript.md
summary.md
report.md
segments.json
metrics.json
```

`outputs/` 已加入 `.gitignore`。

## 项目结构

```text
app.py                     Streamlit GUI
setup.ps1                  可选虚拟环境/当前环境安装脚本
run_gui.ps1                GUI 启动脚本
requirements*.txt          分层依赖清单
src/asr_mvp/transcription.py
src/asr_mvp/diarization.py
src/asr_mvp/pipeline.py
src/asr_mvp/text_processing.py
```

## Git 协作

推荐在功能分支提交：

```powershell
git switch -c codex/gpu-pyannote
git add .gitignore README.md app.py setup.ps1 run_gui.ps1 requirements*.txt src
git add "多人对话文本.txt"
git status
git diff --cached
git commit -m "Improve FunASR and pyannote workflow"
git push -u origin codex/gpu-pyannote
```

真实录音可能包含声音、姓名和谈话内容。根目录的 `多人对话.wav` 已被忽略，请通过获得授权的受控渠道分享测试音频。

## 已知限制

- pyannote 输出的是匿名说话人，不会自动识别真实姓名。
- 重叠语音、远场录音和噪声会降低识别与分人准确率。
- CER 是逐字指标，经过人工润色但非逐字记录的参考文本会导致指标偏低。
- CUDA 主要提升速度，不直接提高模型准确率。

