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

论文驱动的创新设计与实验方案见 [说话人感知的情绪与音频事件会议分析](docs/innovation_emotion_events.md)。

说话人标签如 `SPEAKER_00` 只区分声音，不代表真实姓名。

## 环境要求

- Windows 10/11
- Python 3.12（推荐）
- 约 8 GB 可用磁盘空间；首次使用时模型会下载到用户缓存目录
- NVIDIA 显卡可选，本项目测试环境为 RTX 4060 Laptop GPU

不要提交 `.venv`、模型缓存或 Token。虚拟环境包含本机路径和平台相关二进制，无法保证换一台电脑后仍可运行。

## 依赖文件

| 文件 | 用途 |
| --- | --- |
| `requirements.txt` | GUI、FunASR、ModelScope 和基础 PyTorch |
| `requirements-pyannote.txt` | 在基础依赖上增加 pyannote、TorchCodec、torchvision |
| `requirements-cuda.txt` | CUDA 12.8 版 torch、torchvision、torchaudio |

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

| 语言代码 | 录音语言 | 建议使用场景 |
| --- | --- | --- |
| `auto` | 自动识别 | 不确定语言，或录音中包含多种语言 |
| `zh` | 普通话/中文 | 主要内容为普通话时使用 |
| `en` | 英语 | 主要内容为英语时使用 |
| `yue` | 粤语 | 主要内容为粤语时使用 |
| `ja` | 日语 | 主要内容为日语时使用 |
| `ko` | 韩语 | 主要内容为韩语时使用 |

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
