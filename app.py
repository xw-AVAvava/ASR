from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from asr_mvp import PipelineConfig, run_pipeline

REAL_AUDIO = ROOT / "多人对话.wav"
REAL_TRANSCRIPT = ROOT / "多人对话文本.txt"
MODEL_BASE = ROOT / "models" / "faster-whisper-base"
MODEL_MEDIUM = ROOT / "models" / "faster-whisper-medium"
GUI_OUTPUT = ROOT / "outputs" / "gui_last_run"
MEDIUM_OUTPUT = ROOT / "outputs" / "medium_no_fix"
BASE_OUTPUT = ROOT / "outputs" / "check_accuracy"
UPLOAD_DIR = ROOT / "outputs" / "gui_uploads"

FUNASR_SENSEVOICE_LOCAL = Path("D:/asr_models/SenseVoiceSmall")
FUNASR_PARAFORMER_ZH_LOCAL = Path("D:/asr_models/paraformer-zh")
FUNASR_SENSEVOICE = str(FUNASR_SENSEVOICE_LOCAL) if FUNASR_SENSEVOICE_LOCAL.exists() else "iic/SenseVoiceSmall"
FUNASR_PARAFORMER_ZH = str(FUNASR_PARAFORMER_ZH_LOCAL) if FUNASR_PARAFORMER_ZH_LOCAL.exists() else "paraformer-zh"

st.set_page_config(page_title="会议音频智能助手", page_icon="M", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #f7f8f5; }
    section[data-testid="stSidebar"] { background: #edf1ea; }
    .status-box { border: 1px solid #cfd8cc; background: #fff; border-radius: 8px; padding: 12px 14px; margin: 8px 0 14px; }
    .status-ok { border-left: 4px solid #1f7a5b; }
    .status-warn { border-left: 4px solid #b7791f; }
    .small-muted { color: #5f6b61; font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_uploaded_audio(uploaded_file) -> Path | None:
    if uploaded_file is None:
        return None
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / uploaded_file.name
    target.write_bytes(uploaded_file.getvalue())
    return target


def save_text_to_temp(text: str, suffix: str = ".txt") -> Path | None:
    if not text.strip():
        return None
    tmp = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=suffix)
    with tmp:
        tmp.write(text.strip())
    return Path(tmp.name)


def format_metric(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def format_percent(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, (float, int)):
        return f"{value * 100:.2f}%"
    return str(value)


def accuracy_value(metrics: dict[str, Any], accuracy_key: str, cer_key: str) -> float | None:
    value = metrics.get(accuracy_key)
    if isinstance(value, (float, int)):
        return float(value)
    cer = metrics.get(cer_key)
    if isinstance(cer, (float, int)):
        return max(0.0, min(1.0, 1.0 - float(cer)))
    return None


def model_options() -> dict[str, dict[str, str]]:
    return {
        "FunASR SenseVoiceSmall（推荐）": {
            "engine": "funasr",
            "model": FUNASR_SENSEVOICE,
            "note": "推荐先试。FunASR 新一代通用语音模型，支持中文，通常比 Whisper base 更适合中文口语。首次运行会下载模型。",
        },
        "FunASR Paraformer-zh（中文专用）": {
            "engine": "funasr",
            "model": FUNASR_PARAFORMER_ZH,
            "note": "中文专用 ASR 模型。专门用于中文普通话识别，适合你问的“专门用于中文”的场景。首次运行会下载模型。",
        },
        "faster-whisper medium 本地模型": {
            "engine": "faster-whisper",
            "model": str(MODEL_MEDIUM),
            "note": "已下载的 medium 模型，不联网，准确率比 base 高但 CPU 上较慢。",
        },
        "faster-whisper base 本地模型": {
            "engine": "faster-whisper",
            "model": str(MODEL_BASE),
            "note": "已下载的 base 模型，速度较快但准确率一般。",
        },
    }


def local_whisper_model_ok(path_text: str) -> bool:
    path = Path(path_text)
    required = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]
    return path.exists() and all((path / name).exists() for name in required)


def selected_model_available(option: dict[str, str]) -> bool:
    if option["engine"] != "faster-whisper":
        return True
    return local_whisper_model_ok(option["model"])


def show_model_status(label: str, option: dict[str, str]) -> None:
    ok = selected_model_available(option)
    box_class = "status-ok" if ok else "status-warn"
    status = "模型可用" if ok else "本地模型文件不完整"
    st.markdown(
        f"""
        <div class="status-box {box_class}">
          <strong>{status}</strong><br>
          <span class="small-muted">{label}<br>{option['note']}<br>engine={option['engine']}, model={option['model']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_metrics(output_dir: Path) -> None:
    metrics = read_json(output_dir / "metrics.json")
    if not metrics:
        st.info("还没有可展示的运行结果。")
        return

    cols = st.columns(6)
    cols[0].metric("Raw CER", format_metric(metrics.get("raw_cer")))
    cols[1].metric("展示 CER", format_metric(metrics.get("display_cer")))
    cols[2].metric("展示字符准确率", format_percent(accuracy_value(metrics, "display_accuracy", "display_cer")))
    cols[3].metric("运行时间", f"{metrics.get('runtime_seconds', 0):.1f}s")
    cols[4].metric("原始片段", metrics.get("raw_segment_count", "n/a"))
    cols[5].metric("展示片段", metrics.get("segment_count", "n/a"))

    emotion_counts = metrics.get("emotion_counts") or {}
    event_counts = metrics.get("event_counts") or {}
    if emotion_counts or event_counts:
        dominant_emotion = max(emotion_counts.items(), key=lambda item: item[1])[0] if emotion_counts else "n/a"
        voice_cols = st.columns(3)
        voice_cols[0].metric("总体主导情绪", dominant_emotion)
        voice_cols[1].metric("情绪标签片段", sum(emotion_counts.values()))
        voice_cols[2].metric("音频事件", sum(event_counts.values()))

    with st.expander("完整 metrics.json", expanded=False):
        st.json(metrics)


def show_output_files(output_dir: Path, key_prefix: str) -> None:
    show_metrics(output_dir)
    summary = read_text(output_dir / "summary.md")
    transcript = read_text(output_dir / "transcript.md")
    raw_transcript = read_text(output_dir / "raw_transcript.md")
    report = read_text(output_dir / "report.md")
    segments = read_json(output_dir / "segments.json")

    tabs = st.tabs(["转写文本", "情绪与事件", "原始 ASR", "摘要", "报告", "Segments JSON"])
    with tabs[0]:
        st.markdown(transcript or "还没有生成 transcript.md")
    with tabs[1]:
        voice_rows = []
        if isinstance(segments, list):
            for seg in segments:
                if not isinstance(seg, dict):
                    continue
                emotion = seg.get("emotion")
                events = seg.get("events") or []
                if not emotion and not events:
                    continue
                voice_rows.append(
                    {
                        "开始(s)": round(float(seg.get("start", 0.0)), 2),
                        "说话人": seg.get("speaker", "SPEAKER_00"),
                        "语言": seg.get("language") or "",
                        "情绪": emotion or "",
                        "事件": ", ".join(events),
                        "文本": seg.get("text", ""),
                    }
                )
        if voice_rows:
            st.dataframe(voice_rows, width="stretch", hide_index=True)
        else:
            st.info("当前结果没有 SenseVoice 情绪或音频事件标签。")
    with tabs[2]:
        st.markdown(raw_transcript or "还没有生成 raw_transcript.md")
    with tabs[3]:
        st.markdown(summary or "还没有生成 summary.md")
    with tabs[4]:
        st.markdown(report or "还没有生成 report.md")
    with tabs[5]:
        st.json(segments or {})

    transcript_path = output_dir / "transcript.md"
    if transcript_path.exists():
        st.download_button(
            "下载 transcript.md",
            data=read_text(transcript_path),
            file_name="transcript.md",
            mime="text/markdown",
            key=f"{key_prefix}_download_transcript",
        )


st.title("会议音频智能助手")
st.caption("现在支持 faster-whisper 和 FunASR。中文专用模型是 FunASR Paraformer-zh。")

with st.sidebar:
    st.header("运行设置")
    input_mode = st.radio(
        "输入来源",
        ["项目自带多人对话", "上传自定义音频"],
        index=0 if REAL_AUDIO.exists() else 1,
    )

    if input_mode == "项目自带多人对话":
        audio_path = REAL_AUDIO if REAL_AUDIO.exists() else None
        default_reference = read_text(REAL_TRANSCRIPT)
        default_transcript = default_reference
        default_speakers = 5
        if audio_path is None:
            st.warning("本地示例音频不存在，请上传音频文件。")
    else:
        uploaded_audio = st.file_uploader("上传音频", type=["wav", "mp3", "m4a", "mp4", "flac"])
        audio_path = save_uploaded_audio(uploaded_audio)
        default_reference = ""
        default_transcript = ""
        default_speakers = 2

    mode = st.radio("转写模式", ["真实 ASR", "使用已有 transcript"], index=0)
    options = model_options()
    model_label = st.selectbox("ASR 模型", list(options.keys()), index=0, disabled=mode != "真实 ASR")
    selected_model = options[model_label]

    language = st.text_input("语言代码", value="zh")
    diarizer = st.selectbox("说话人标注", ["cluster", "turns", "pyannote"], index=0)
    supports_auto_speakers = diarizer in {"cluster", "pyannote"}
    if "auto_speakers" not in st.session_state:
        st.session_state["auto_speakers"] = True
    if not supports_auto_speakers:
        st.session_state["auto_speakers"] = False
    auto_speakers = st.checkbox(
        "自动估计说话人数量",
        disabled=not supports_auto_speakers,
        key="auto_speakers",
        help="cluster 和 pyannote 支持自动估计；turns 需要指定人数。",
    )
    speakers = st.slider(
        "说话人数量",
        min_value=1,
        max_value=8,
        value=default_speakers,
        disabled=supports_auto_speakers and auto_speakers,
        help="取消自动估计后，使用这里指定的固定人数。",
    )
    compute_type = st.selectbox("Whisper 计算精度", ["int8", "float32"], index=0, disabled=mode != "真实 ASR" or selected_model["engine"] != "faster-whisper")
    cpu_threads = st.slider("CPU 线程", min_value=1, max_value=4, value=1, disabled=mode != "真实 ASR" or selected_model["engine"] != "faster-whisper")
    remove_repeated_text = st.checkbox("去除重复语句", value=True)

    st.divider()
    st.subheader("LLM 纠错 (Ollama)")
    use_llm_correct = st.checkbox("使用 LLM 纠正 ASR 识别错误", value=False)
    if use_llm_correct:
        llm_model = st.text_input(
            "Ollama 模型名称",
            value="qwen2.5:1.5b",
            help="需要先在 Ollama 中拉取该模型。首次使用时会自动下载。",
        )
        try:
            from asr_mvp.llm_correction import check_ollama_status, OLLAMA_DEFAULT_BASE_URL
            ok, msg, models = check_ollama_status(OLLAMA_DEFAULT_BASE_URL)
            model_installed = llm_model in models if models else False
            if ok:
                if model_installed:
                    st.markdown(
                        f'<div class="status-box status-ok"><strong>Ollama:</strong> 已连接 · 模型已就绪 · {msg}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="status-box status-warn"><strong>Ollama:</strong> 已连接 · 模型"{llm_model}"未安装，运行时会自动下载 · {msg}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    f'<div class="status-box status-warn"><strong>Ollama:</strong> {msg}</div>',
                    unsafe_allow_html=True,
                )
        except Exception as exc:
            st.warning(f"无法检查 Ollama 状态: {exc}")
    else:
        llm_model = "qwen2.5:1.5b"

    st.divider()
    run_clicked = st.button("运行处理流程", type="primary", width="stretch")

if audio_path is None:
    st.warning("请先上传音频文件。")
else:
    st.markdown(
        f"""
        <div class="status-box status-ok">
          <strong>当前音频</strong><br>
          <span class="small-muted">{audio_path}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

if mode == "真实 ASR":
    show_model_status(model_label, selected_model)

main_tab, compare_tab, docs_tab = st.tabs(["运行", "实验对比", "项目说明"])

with main_tab:
    left, right = st.columns([1, 1])
    with left:
        transcript_text = st.text_area("已有 transcript（仅在使用已有 transcript 模式生效）", value=default_transcript, height=220)
    with right:
        reference_text = st.text_area("参考文本（用于计算 WER/CER）", value=default_reference, height=220)

    if run_clicked:
        if audio_path is None:
            st.error("还没有选择音频文件。")
        elif mode == "真实 ASR" and not selected_model_available(selected_model):
            st.error("所选本地 faster-whisper 模型不完整。")
        else:
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            transcript_file = save_text_to_temp(transcript_text) if mode == "使用已有 transcript" else None
            reference_file = save_text_to_temp(reference_text)
            engine = "demo" if mode == "使用已有 transcript" else selected_model["engine"]
            model = "tiny" if mode == "使用已有 transcript" else selected_model["model"]

            config = PipelineConfig(
                audio=audio_path,
                output=GUI_OUTPUT,
                engine=engine,
                model=model,
                compute_type=compute_type,
                cpu_threads=int(cpu_threads),
                language=language or None,
                speakers=0 if supports_auto_speakers and auto_speakers else int(speakers),
                transcript_file=transcript_file,
                reference_file=reference_file,
                diarizer=diarizer,
                remove_repeated_text=remove_repeated_text,
                speaker_model=None,
                use_llm_correct=use_llm_correct,
                llm_model=llm_model,
            )

            with st.spinner("正在运行处理流程。首次使用 FunASR 会下载模型，请耐心等待..."):
                try:
                    result = run_pipeline(config)
                    st.success(
                        "处理完成："
                        f"{result['metadata']['asr_engine_used']}，"
                        f"Raw CER={format_metric(result['metadata'].get('raw_cer'))}，"
                        f"Display CER={format_metric(result['metadata'].get('display_cer'))}，"
                        f"Display Accuracy={format_percent(result['metadata'].get('display_accuracy'))}"
                    )
                except Exception as exc:
                    st.error(f"处理失败：{exc}")

    st.subheader("最近一次 GUI 运行结果")
    show_output_files(GUI_OUTPUT, "gui_last_run")

with compare_tab:
    st.subheader("已有实验结果对比")
    rows = []
    for name, path in [
        ("base baseline", BASE_OUTPUT),
        ("medium no fix", MEDIUM_OUTPUT),
        ("GUI last run", GUI_OUTPUT),
    ]:
        metrics = read_json(path / "metrics.json")
        if metrics:
            rows.append(
                {
                    "实验": name,
                    "Raw CER": metrics.get("raw_cer"),
                    "Display CER": metrics.get("display_cer"),
                    "Display Accuracy": accuracy_value(metrics, "display_accuracy", "display_cer"),
                    "运行时间(s)": metrics.get("runtime_seconds"),
                    "原始片段": metrics.get("raw_segment_count"),
                    "展示片段": metrics.get("segment_count"),
                    "输出目录": str(path),
                }
            )
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("还没有可对比的实验结果。")

    st.markdown("### Medium 实验结果")
    show_output_files(MEDIUM_OUTPUT, "medium_no_fix")

with docs_tab:
    st.subheader("项目说明")
    st.markdown(read_text(ROOT / "README.md") or "README.md 不存在。")
