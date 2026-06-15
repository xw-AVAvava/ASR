from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FEATURE_NAMES = [
    "duration_seconds",
    "mean_amplitude",
    "std_amplitude",
    "rms_mean",
    "rms_std",
    "rms_max",
    "zcr_mean",
    "zcr_std",
    "centroid_mean",
    "centroid_std",
    "bandwidth_mean",
    "bandwidth_std",
    "rolloff_mean",
    "rolloff_std",
    "low_high_ratio_mean",
    "low_high_ratio_std",
    "silence_ratio",
]


SPEAKER_EMBEDDING_NAMES = [
    *FEATURE_NAMES[1:],
    *[f"mfcc_{i:02d}_mean" for i in range(1, 13)],
    *[f"mfcc_{i:02d}_std" for i in range(1, 13)],
]


@dataclass
class AudioArray:
    samples: np.ndarray
    sample_rate: int


def _pcm_to_float(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (data - 128.0) / 128.0
    if sample_width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        return data / 32768.0
    if sample_width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        return data / 2147483648.0
    raise ValueError(f"Unsupported WAV sample width: {sample_width}")


def load_wav_mono(path: Path) -> AudioArray:
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())
    samples = _pcm_to_float(raw, sample_width)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    return AudioArray(samples=samples.astype(np.float32), sample_rate=sample_rate)


def slice_audio(audio: AudioArray, start: float, end: float) -> AudioArray:
    start_i = max(0, int(start * audio.sample_rate))
    end_i = min(len(audio.samples), int(end * audio.sample_rate))
    return AudioArray(samples=audio.samples[start_i:end_i], sample_rate=audio.sample_rate)


def frame_audio(samples: np.ndarray, sample_rate: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    frame_size = max(1, int(sample_rate * frame_ms / 1000.0))
    hop_size = max(1, int(sample_rate * hop_ms / 1000.0))
    if len(samples) < frame_size:
        pad_width = frame_size - len(samples)
        samples = np.pad(samples, (0, pad_width))
    frame_count = 1 + max(0, (len(samples) - frame_size) // hop_size)
    frames = np.empty((frame_count, frame_size), dtype=np.float32)
    for i in range(frame_count):
        start = i * hop_size
        frames[i] = samples[start : start + frame_size]
    return frames


def _safe_stats(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int = 26) -> np.ndarray:
    freq_count = n_fft // 2 + 1
    f_min = 80.0
    f_max = min(sample_rate / 2.0, 7600.0)
    mel_points = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, freq_count - 1)

    filters = np.zeros((n_mels, freq_count), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = int(bins[i - 1]), int(bins[i]), int(bins[i + 1])
        if center <= left:
            center = min(left + 1, freq_count - 1)
        if right <= center:
            right = min(center + 1, freq_count - 1)
        if center > left:
            filters[i - 1, left:center] = (np.arange(left, center) - left) / float(center - left)
        if right > center:
            filters[i - 1, center:right] = (right - np.arange(center, right)) / float(right - center)
    return filters


def _dct_basis(n_mels: int, n_mfcc: int) -> np.ndarray:
    n = np.arange(n_mels, dtype=np.float32)
    k = np.arange(n_mfcc, dtype=np.float32)[:, None]
    basis = np.cos(np.pi / n_mels * (n + 0.5) * k)
    basis[0] *= math.sqrt(1.0 / n_mels)
    basis[1:] *= math.sqrt(2.0 / n_mels)
    return basis.astype(np.float32)


def extract_features_from_array(audio: AudioArray) -> np.ndarray:
    samples = audio.samples
    sample_rate = audio.sample_rate
    if samples.size == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    duration = samples.size / float(sample_rate) if sample_rate else 0.0
    mean_amp = float(np.mean(np.abs(samples)))
    std_amp = float(np.std(samples))

    frames = frame_audio(samples, sample_rate)
    window = np.hanning(frames.shape[1]).astype(np.float32)
    windowed = frames * window

    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)

    spectra = np.abs(np.fft.rfft(windowed, axis=1))
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sample_rate)
    spectral_sum = np.sum(spectra, axis=1) + 1e-12
    centroid = np.sum(spectra * freqs, axis=1) / spectral_sum
    bandwidth = np.sqrt(np.sum(spectra * (freqs - centroid[:, None]) ** 2, axis=1) / spectral_sum)

    cumsum = np.cumsum(spectra, axis=1)
    rolloff_idx = np.argmax(cumsum >= 0.85 * spectral_sum[:, None], axis=1)
    rolloff = freqs[np.clip(rolloff_idx, 0, len(freqs) - 1)]

    low_mask = freqs <= 1000
    high_mask = freqs > 1000
    low_energy = np.sum(spectra[:, low_mask], axis=1)
    high_energy = np.sum(spectra[:, high_mask], axis=1) + 1e-12
    low_high_ratio = low_energy / high_energy

    rms_threshold = max(0.005, float(np.percentile(rms, 20)))
    silence_ratio = float(np.mean(rms <= rms_threshold))

    rms_mean, rms_std = _safe_stats(rms)
    zcr_mean, zcr_std = _safe_stats(zcr)
    centroid_mean, centroid_std = _safe_stats(centroid)
    bandwidth_mean, bandwidth_std = _safe_stats(bandwidth)
    rolloff_mean, rolloff_std = _safe_stats(rolloff)
    ratio_mean, ratio_std = _safe_stats(low_high_ratio)

    features = [
        duration,
        mean_amp,
        std_amp,
        rms_mean,
        rms_std,
        float(np.max(rms)),
        zcr_mean,
        zcr_std,
        centroid_mean,
        centroid_std,
        bandwidth_mean,
        bandwidth_std,
        rolloff_mean,
        rolloff_std,
        ratio_mean,
        ratio_std,
        silence_ratio,
    ]
    return np.array(features, dtype=np.float32)


def extract_speaker_embedding_from_array(audio: AudioArray) -> np.ndarray:
    samples = audio.samples
    sample_rate = audio.sample_rate
    if samples.size == 0 or sample_rate <= 0:
        return np.zeros(len(SPEAKER_EMBEDDING_NAMES), dtype=np.float32)

    base_features = extract_features_from_array(audio)[1:]
    frames = frame_audio(samples, sample_rate, frame_ms=25.0, hop_ms=10.0)
    frames = frames - np.mean(frames, axis=1, keepdims=True)
    window = np.hanning(frames.shape[1]).astype(np.float32)
    windowed = frames * window
    power = np.abs(np.fft.rfft(windowed, axis=1)) ** 2

    filters = _mel_filterbank(sample_rate, frames.shape[1])
    mel_energy = np.maximum(power @ filters.T, 1e-10)
    log_mel = np.log(mel_energy)
    mfcc = log_mel @ _dct_basis(filters.shape[0], 13).T

    # Drop coefficient 0 because it mostly captures loudness/channel energy.
    voice_mfcc = mfcc[:, 1:13]
    mfcc_mean = np.mean(voice_mfcc, axis=0)
    mfcc_std = np.std(voice_mfcc, axis=0)
    embedding = np.concatenate([base_features, mfcc_mean, mfcc_std]).astype(np.float32)
    return np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)


def extract_features(path: Path) -> np.ndarray:
    return extract_features_from_array(load_wav_mono(path))
