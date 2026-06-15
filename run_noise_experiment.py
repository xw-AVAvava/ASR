from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from asr_mvp.audio_features import AudioArray, extract_features_from_array, load_wav_mono
from asr_mvp.model_training import load_dataset


def add_noise(audio: AudioArray, snr_db: float, rng: np.random.Generator) -> AudioArray:
    signal = audio.samples.astype(np.float32)
    signal_power = float(np.mean(signal**2) + 1e-12)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=signal.shape).astype(np.float32)
    mixed = np.clip(signal + noise, -1.0, 1.0)
    return AudioArray(samples=mixed, sample_rate=audio.sample_rate)


def run_noise_experiment(data_dir: Path, model_path: Path, output_dir: Path, test_size: float = 0.3) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    with model_path.open("rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    labels = artifact["label_names"]

    dataset, skipped = load_dataset(data_dir)
    class_counts = {label: dataset.labels.count(label) for label in sorted(set(dataset.labels))}
    stratify = dataset.y if min(class_counts.values()) >= 2 else None
    _, X_test, _, y_test, _, paths_test = train_test_split(
        dataset.X,
        dataset.y,
        dataset.paths,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    rng = np.random.default_rng(42)
    snr_levels: list[float | None] = [None, 30.0, 20.0, 10.0, 5.0]
    rows = []
    summary = []
    for snr in snr_levels:
        predictions = []
        for path in paths_test:
            audio = load_wav_mono(path)
            if snr is not None:
                audio = add_noise(audio, snr, rng)
            features = extract_features_from_array(audio).reshape(1, -1)
            pred = int(model.predict(features)[0])
            predictions.append(pred)
        acc = float(accuracy_score(y_test, predictions))
        level_name = "clean" if snr is None else f"{int(snr)}dB"
        summary.append({"snr": level_name, "accuracy": round(acc, 4)})
        for path, actual, pred in zip(paths_test, y_test, predictions):
            rows.append([level_name, str(path), labels[int(actual)], labels[int(pred)]])

    with (output_dir / "noise_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["snr", "path", "actual", "predicted"])
        writer.writerows(rows)

    metrics = {
        "data_dir": str(data_dir),
        "model_path": str(model_path),
        "test_count": int(len(y_test)),
        "summary": summary,
        "skipped_files": skipped,
    }
    (output_dir / "noise_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    table = ["| SNR | 准确率 |", "| --- | --- |"]
    for item in summary:
        table.append(f"| {item['snr']} | {item['accuracy']} |")
    report = [
        "# 噪声鲁棒性实验",
        "",
        "## 目标",
        "",
        "向测试集音频片段注入高斯噪声，压力测试训练好的音频分类器。",
        "",
        "## 结果",
        "",
        *table,
        "",
        "## 结果解释",
        "",
        "- SNR 越低，通常噪声越强，准确率越容易下降。",
        "- 如果准确率保持稳定，说明当前手工特征对这个小数据集比较鲁棒。",
        "- 如果准确率明显下降，说明后续应加入降噪、预处理或训练时的数据增强。",
    ]
    (output_dir / "noise_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a noise robustness experiment for the audio classifier")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    metrics = run_noise_experiment(args.data_dir, args.model, args.output)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
