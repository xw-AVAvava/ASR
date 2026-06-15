from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from asr_mvp import PipelineConfig, run_pipeline


LOCAL_SAMPLE_AUDIO = ROOT / "data" / "sample_audio.wav"
LOCAL_SAMPLE_TRANSCRIPT = ROOT / "data" / "sample_transcript.txt"
LEGACY_SAMPLE_AUDIO = ROOT.parent / "多人对话.wav"
LEGACY_SAMPLE_TRANSCRIPT = ROOT.parent / "多人对话文本.txt"
REAL_AUDIO = LOCAL_SAMPLE_AUDIO if LOCAL_SAMPLE_AUDIO.exists() else LEGACY_SAMPLE_AUDIO
REAL_TRANSCRIPT = LOCAL_SAMPLE_TRANSCRIPT if LOCAL_SAMPLE_TRANSCRIPT.exists() else LEGACY_SAMPLE_TRANSCRIPT
DEMO_TRANSCRIPT = ROOT / "data" / "demo_transcript.txt"
GUI_OUTPUT = ROOT / "outputs" / "gui_last_run"
UPLOAD_DIR = ROOT / "outputs" / "gui_uploads"
REAL_OUTPUT = ROOT / "outputs" / "real_dialogue_human_transcript"


st.set_page_config(
    page_title="会议音频智能助手",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #16211f;
        --muted: #5d6965;
        --paper: #f7f3ea;
        --panel: #fffaf0;
        --line: #d9cdb7;
        --teal: #176b66;
        --amber: #c47a1b;
        --leaf: #6b8f3a;
    }
    .stApp {
        background:
            linear-gradient(120deg, rgba(23, 107, 102, .10), transparent 34%),
            radial-gradient(circle at 82% 8%, rgba(196, 122, 27, .18), transparent 24%),
            var(--paper);
        color: var(--ink);
    }
    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
    }
    .hero {
        border: 1px solid var(--line);
        background: linear-gradient(135deg, rgba(255,250,240,.92), rgba(236,244,236,.88));
        padding: 24px 28px;
        border-radius: 8px;
        margin-bottom: 18px;
        box-shadow: 0 14px 35px rgba(22, 33, 31, .08);
    }
    .hero-title {
        font-size: 34px;
        line-height: 1.1;
        font-weight: 760;
        margin: 0 0 8px 0;
    }
    .hero-copy {
        color: var(--muted);
        font-size: 16px;
        max-width: 980px;
        margin: 0;
    }
    .metric-card {
        border: 1px solid var(--line);
        background: rgba(255,250,240,.9);
        border-radius: 8px;
        padding: 14px 16px;
        min-height: 92px;
    }
    .metric-label {
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: .06em;
    }
    .metric-value {
        color: var(--teal);
        font-size: 24px;
        font-weight: 760;
        margin-top: 6px;
    }
    .small-note {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.45;
    }
    .status-ok {
        border-left: 4px solid var(--teal);
        background: rgba(23,107,102,.08);
        padding: 10px 12px;
        border-radius: 6px;
        color: var(--ink);
    }
    section[data-testid="stSidebar"] {
        background: #efe5d1;
    }
    .stButton > button {
        border-radius: 6px;
        border: 1px solid var(--teal);
        background: var(--teal);
        color: white;
        font-weight: 700;
    }
    .stDownloadButton > button {
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric_card(label: str, value: object) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_metric(value: object, suffix: str = "") -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}{suffix}"
    return f"{value}{suffix}"


def show_output_files(output_dir: Path, key_prefix: str) -> None:
    metrics = read_json(output_dir / "metrics.json")
    if metrics:
        cols = st.columns(5)
        with cols[0]:
            metric_card("音频时长", f"{metrics.get('duration_seconds', 0):.1f}s" if metrics.get("duration_seconds") else "n/a")
        with cols[1]:
            metric_card("文本片段", metrics.get("segment_count", "n/a"))
        with cols[2]:
            metric_card("说话人标签", metrics.get("speaker_count", "n/a"))
        with cols[3]:
            metric_card("CER", format_metric(metrics.get("cer")))
        with cols[4]:
            runtime = metrics.get("runtime_seconds")
            metric_card("运行时间", f"{runtime:.2f}s" if runtime is not None else "n/a")

    summary = read_text(output_dir / "summary.md")
    transcript = read_text(output_dir / "transcript.md")
    report = read_text(output_dir / "report.md")
    segments = read_json(output_dir / "segments.json")

    result_tabs = st.tabs(["摘要", "转写文本", "报告", "片段 JSON"])
    with result_tabs[0]:
        st.markdown(summary or "还没有生成摘要。")
    with result_tabs[1]:
        st.markdown(transcript or "还没有生成转写文本。")
    with result_tabs[2]:
        st.markdown(report or "还没有生成报告。")
    with result_tabs[3]:
        st.json(segments or {})

    if output_dir.exists():
        st.download_button(
            "下载 transcript.md",
            data=read_text(output_dir / "transcript.md"),
            file_name="transcript.md",
            mime="text/markdown",
            disabled=not (output_dir / "transcript.md").exists(),
            key=f"{key_prefix}_download_transcript",
        )


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


st.markdown(
    """
    <div class="hero">
        <div class="hero-title">会议音频智能助手</div>
        <p class="hero-copy">
            一个用于课程项目展示的图形界面：支持音频转写、时间戳解析、匿名说话人标签、
            自动摘要和实验报告生成。
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("运行设置")
    sample_mode = st.radio(
        "输入来源",
        ["真实多人对话样本", "上传自定义音频"],
        index=0,
    )

    if sample_mode == "真实多人对话样本":
        if REAL_AUDIO.exists():
            audio_path = REAL_AUDIO
            transcript_default = read_text(REAL_TRANSCRIPT)
            reference_default = transcript_default
            st.markdown(
                f'<div class="small-note">使用本地样本：{REAL_AUDIO.name}。GitHub 仓库不包含音频文件。</div>',
                unsafe_allow_html=True,
            )
        else:
            audio_path = None
            transcript_default = ""
            reference_default = ""
            st.warning("当前仓库不包含演示音频。请切换到“上传自定义音频”，或在本地 data/ 中放入 sample_audio.wav。")
        default_language = "zh"
        default_speakers = 5
    else:
        uploaded_audio = st.file_uploader("音频文件", type=["wav", "mp3", "m4a", "mp4", "flac"])
        audio_path = save_uploaded_audio(uploaded_audio)
        transcript_default = ""
        reference_default = ""
        default_language = "zh"
        default_speakers = 2

    engine_label = st.selectbox(
        "转写模式",
        ["使用已有文本", "自动 ASR", "faster-whisper", "openai-whisper"],
        index=0,
    )
    engine_map = {
        "使用已有文本": "demo",
        "自动 ASR": "auto",
        "faster-whisper": "faster-whisper",
        "openai-whisper": "openai-whisper",
    }
    engine = engine_map[engine_label]
    language = st.text_input("语言代码", value=default_language)
    speakers = st.slider("基线说话人标签数量", min_value=1, max_value=8, value=default_speakers)
    diarizer = st.selectbox("说话人标注方式", ["cluster", "turns", "pyannote"], index=0)

    st.divider()
    run_clicked = st.button("运行处理流程", type="primary", use_container_width=True)


if audio_path is None:
    st.warning("请上传音频文件后再运行自定义流程。")
else:
    st.markdown(
        f"""
        <div class="status-ok">
        当前音频：<strong>{audio_path.name}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

main_tabs = st.tabs(["处理流程", "真实对话", "项目报告"])

with main_tabs[0]:
    st.subheader("处理流程运行器")
    left, right = st.columns([1, 1])
    with left:
        transcript_text = st.text_area("输入转写文本（仅“使用已有文本”模式生效）", value=transcript_default, height=260)
    with right:
        reference_text = st.text_area("参考文本，用于计算 WER/CER", value=reference_default, height=260)

    if run_clicked:
        if audio_path is None:
            st.error("还没有选择音频文件。")
        else:
            transcript_file = save_text_to_temp(transcript_text) if engine == "demo" else None
            reference_file = save_text_to_temp(reference_text)
            config = PipelineConfig(
                audio=audio_path,
                output=GUI_OUTPUT,
                engine=engine,
                language=language or None,
                speakers=int(speakers),
                transcript_file=transcript_file,
                reference_file=reference_file,
                diarizer=diarizer,
                speaker_model=None,
            )
            with st.spinner("正在运行处理流程..."):
                try:
                    result = run_pipeline(config)
                    st.success(f"处理完成，使用的转写来源：{result['metadata']['asr_engine_used']}。")
                except Exception as exc:
                    st.error(f"处理失败：{exc}")

    show_output_files(GUI_OUTPUT if GUI_OUTPUT.exists() else REAL_OUTPUT, "latest_result")

with main_tabs[1]:
    st.subheader("真实多人对话样本")
    st.markdown("这里展示本地样本的处理结果。GitHub 仓库默认不包含音频文件，请用自己的音频复现实验。")
    show_output_files(REAL_OUTPUT, "real_dialogue")

with main_tabs[2]:
    st.subheader("项目报告草稿")
    st.markdown(read_text(ROOT / "docs" / "final_report.md"))
