from __future__ import annotations

import os
import pickle

import numpy as np

from .audio_features import extract_features_from_array, extract_speaker_embedding_from_array, load_wav_mono, slice_audio
from .schemas import PipelineConfig, Segment


def assign_speakers_turn_based(segments: list[Segment], speakers: int) -> list[Segment]:
    speakers = max(1, speakers)
    seen_prefix_labels = {seg.speaker for seg in segments if seg.speaker != "SPEAKER_00"}
    if seen_prefix_labels:
        return segments
    for i, seg in enumerate(segments):
        seg.speaker = f"SPEAKER_{i % speakers:02d}"
    return segments

def assign_speakers_pyannote(config: PipelineConfig, segments: list[Segment]) -> list[Segment]:
    from pyannote.audio import Pipeline  # type: ignore

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for pyannote.audio diarization.")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    diarization = pipeline(str(config.audio))

    diarized_turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
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

    speakers = max(1, min(config.speakers, len(segments)))
    if speakers == 1:
        for seg in segments:
            seg.speaker = "SPEAKER_00"
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

    if len(embeddings) < speakers:
        return assign_speakers_turn_based(segments, speakers)

    X = np.vstack(embeddings).astype(np.float32)
    X = StandardScaler().fit_transform(X)
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
    if config.diarizer == "cluster":
        try:
            return assign_speakers_with_clustering(config, segments), "MFCC + KMeans speaker clustering"
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
