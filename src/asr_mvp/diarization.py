from __future__ import annotations

import os
import pickle
from pathlib import Path
import sys

import numpy as np

from .audio_features import extract_features_from_array, extract_speaker_embedding_from_array, load_wav_mono, slice_audio
from .schemas import PipelineConfig, Segment


_DLL_DIRECTORY_HANDLES: list[object] = []
_DLL_DIRECTORY_PATHS: set[str] = set()


def ensure_windows_audio_dlls() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    venv_root = Path(sys.executable).resolve().parent.parent
    candidate_dirs = list((venv_root / "ffmpeg-shared").glob("*/bin"))

    try:
        import torch

        candidate_dirs.append(Path(torch.__file__).resolve().parent / "lib")
    except Exception:
        pass

    for path_part in os.environ.get("PATH", "").split(os.pathsep):
        if not path_part:
            continue
        path_dir = Path(path_part)
        try:
            if any(path_dir.glob("avcodec-*.dll")):
                candidate_dirs.append(path_dir)
        except OSError:
            continue

    for candidate_dir in candidate_dirs:
        try:
            resolved = candidate_dir.resolve()
            key = str(resolved).lower()
            if key in _DLL_DIRECTORY_PATHS or not resolved.is_dir():
                continue
            handle = os.add_dll_directory(str(resolved))
        except OSError:
            continue
        _DLL_DIRECTORY_HANDLES.append(handle)
        _DLL_DIRECTORY_PATHS.add(key)
        os.environ["PATH"] = str(resolved) + os.pathsep + os.environ.get("PATH", "")


def assign_speakers_turn_based(segments: list[Segment], speakers: int) -> list[Segment]:
    speakers = max(1, speakers)
    seen_prefix_labels = {seg.speaker for seg in segments if seg.speaker != "SPEAKER_00"}
    if seen_prefix_labels:
        return segments
    for i, seg in enumerate(segments):
        seg.speaker = f"SPEAKER_{i % speakers:02d}"
    return segments

def estimate_speaker_count(X: np.ndarray, max_speakers: int = 8) -> int:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    sample_count = X.shape[0]
    if sample_count <= 1:
        return 1
    if sample_count == 2:
        return 2

    max_k = max(2, min(max_speakers, sample_count - 1))
    best_k = 2
    best_score = -1.0
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        if len(set(int(label) for label in labels)) < 2:
            continue
        score = float(silhouette_score(X, labels))
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

def assign_speakers_pyannote(config: PipelineConfig, segments: list[Segment]) -> list[Segment]:
    ensure_windows_audio_dlls()
    from pyannote.audio import Pipeline  # type: ignore
    import torch

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for pyannote.audio diarization.")
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    if pipeline is None:
        raise RuntimeError("Could not load pyannote pipeline. Check HF_TOKEN and model access permissions.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    print(f"[Diarization] Running pyannote on {device}.", flush=True)

    try:
        audio = load_wav_mono(config.audio)
        waveform = torch.from_numpy(audio.samples).unsqueeze(0)
        pipeline_input: object = {"waveform": waveform, "sample_rate": audio.sample_rate}
    except Exception:
        pipeline_input = str(config.audio)

    if config.speakers > 0:
        diarization = pipeline(pipeline_input, num_speakers=config.speakers)
    else:
        diarization = pipeline(pipeline_input)

    annotation = getattr(diarization, "speaker_diarization", diarization)
    if not hasattr(annotation, "itertracks"):
        raise RuntimeError(f"Unsupported pyannote output type: {type(diarization).__name__}")

    diarized_turns = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        diarized_turns.append((float(turn.start), float(turn.end), str(speaker)))

    for seg in segments:
        best_speaker = seg.speaker
        best_overlap = 0.0
        for start, end, speaker in diarized_turns:
            overlap = max(0.0, min(seg.end, end) - max(seg.start, start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        seg.speaker = best_speaker
    return segments

def assign_speakers_with_trained_model(config: PipelineConfig, segments: list[Segment]) -> list[Segment]:
    if config.speaker_model is None:
        raise RuntimeError("--speaker-model is required when --diarizer trained-model is used.")
    with config.speaker_model.open("rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    label_names = artifact["label_names"]
    audio = load_wav_mono(config.audio)

    for seg in segments:
        audio_slice = slice_audio(audio, seg.start, seg.end)
        features = extract_features_from_array(audio_slice).reshape(1, -1)
        label_id = int(model.predict(features)[0])
        label = str(label_names[label_id]).upper()
        seg.speaker = f"MODEL_{label}"
    return segments

def assign_speakers_with_clustering(config: PipelineConfig, segments: list[Segment]) -> list[Segment]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    if not segments:
        return segments

    audio = load_wav_mono(config.audio)
    embeddings = []
    valid_indexes = []
    for index, seg in enumerate(segments):
        audio_slice = slice_audio(audio, seg.start, seg.end)
        if audio_slice.samples.size == 0:
            continue
        embeddings.append(extract_speaker_embedding_from_array(audio_slice))
        valid_indexes.append(index)

    if not embeddings:
        return assign_speakers_turn_based(segments, max(1, config.speakers))

    X = np.vstack(embeddings).astype(np.float32)
    X = StandardScaler().fit_transform(X)
    if config.speakers <= 0:
        speakers = estimate_speaker_count(X, max_speakers=8)
    else:
        speakers = max(1, min(config.speakers, len(embeddings)))

    if speakers == 1:
        for seg in segments:
            seg.speaker = "SPEAKER_00"
        return segments

    labels = KMeans(n_clusters=speakers, random_state=42, n_init=10).fit_predict(X)

    # KMeans cluster IDs are arbitrary, so rename them by first appearance in time.
    label_map: dict[int, str] = {}
    for index, label_id in zip(valid_indexes, labels):
        label_int = int(label_id)
        if label_int not in label_map:
            label_map[label_int] = f"SPEAKER_{len(label_map):02d}"
        segments[index].speaker = label_map[label_int]

    for seg in segments:
        if not seg.speaker:
            seg.speaker = "SPEAKER_00"
    return segments

def assign_speakers(config: PipelineConfig, segments: list[Segment]) -> tuple[list[Segment], str]:
    if config.speakers == 1:
        for seg in segments:
            seg.speaker = "SPEAKER_00"
        return segments, "single speaker (diarization skipped)"

    if config.diarizer == "cluster":
        try:
            clustered = assign_speakers_with_clustering(config, segments)
            label = "MFCC + KMeans speaker clustering"
            if config.speakers <= 0:
                speaker_count = len({seg.speaker for seg in clustered})
                label += f" (auto estimated {speaker_count} speakers)"
            return clustered, label
        except Exception as exc:
            return assign_speakers_turn_based(segments, config.speakers), f"turn baseline (speaker clustering unavailable: {exc})"
    if config.diarizer == "trained-model":
        try:
            return assign_speakers_with_trained_model(config, segments), "trained logistic-regression audio classifier"
        except Exception as exc:
            return assign_speakers_turn_based(segments, config.speakers), f"turn baseline (trained model unavailable: {exc})"
    if config.diarizer == "pyannote":
        try:
            return assign_speakers_pyannote(config, segments), "pyannote.audio"
        except Exception as exc:
            return assign_speakers_turn_based(segments, config.speakers), f"turn baseline (pyannote unavailable: {exc})"
    return assign_speakers_turn_based(segments, config.speakers), "turn baseline"
from typing import List, Tuple

def merge_vad_segments(
    sensevoice_segments: List[Tuple[float, float]],
    pyannote_segments: List[Tuple[float, float]],
    overlap_threshold: float = 0.3
) -> List[Tuple[float, float]]:
    """
    双VAD结果融合：取两个VAD结果的交集作为最终有效语音段
    降低噪声、静音误判对识别准确率的影响
    :param sensevoice_segments: SenseVoice输出的语音段列表 [(start, end), ...]
    :param pyannote_segments: pyannote输出的语音段列表 [(start, end), ...]
    :param overlap_threshold: 最小重叠比例阈值
    :return: 融合后的有效语音段列表
    """
    merged = []
    for sv_start, sv_end in sensevoice_segments:
        sv_duration = sv_end - sv_start
        for py_start, py_end in pyannote_segments:
            overlap_start = max(sv_start, py_start)
            overlap_end = min(sv_end, py_end)
            if overlap_end <= overlap_start:
                continue
            overlap_duration = overlap_end - overlap_start
            if overlap_duration / sv_duration >= overlap_threshold:
                merged.append((overlap_start, overlap_end))
                break
    return _merge_close_segments(merged, gap=0.2)


def detect_overlap_speech(
    speaker_segments: List[dict],
    min_overlap_duration: float = 0.3
) -> List[Tuple[float, float]]:
    """
    检测多人重叠说话时间段，对应cross-speech研究方向
    :param speaker_segments: 说话人分段列表，每个元素含start, end, speaker
    :param min_overlap_duration: 最小时长阈值
    :return: 重叠语音段列表
    """
    timestamps = []
    for seg in speaker_segments:
        timestamps.append((seg["start"], 1))
        timestamps.append((seg["end"], -1))
    timestamps.sort(key=lambda x: (x[0], x[1]))
    
    overlap_segments = []
    current_speakers = 0
    overlap_start = None
    for time, delta in timestamps:
        current_speakers += delta
        if current_speakers >= 2 and overlap_start is None:
            overlap_start = time
        elif current_speakers < 2 and overlap_start is not None:
            if time - overlap_start >= min_overlap_duration:
                overlap_segments.append((overlap_start, time))
            overlap_start = None
    return overlap_segments


def _merge_close_segments(segments: List[Tuple[float, float]], gap: float) -> List[Tuple[float, float]]:
    """合并间隔小于gap的相邻片段，避免碎片化"""
    if not segments:
        return []
    segments.sort()
    merged = [list(segments[0])]
    for start, end in segments[1:]:
        if start - merged[-1][1] <= gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]