from __future__ import annotations

import csv
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .audio_features import FEATURE_NAMES, extract_features


@dataclass
class Dataset:
    paths: list[Path]
    labels: list[str]
    X: np.ndarray
    y: np.ndarray


def infer_label(path: Path) -> str | None:
    match = re.match(r"([A-Za-z]+)_\d+\.wav$", path.name)
    if match:
        return match.group(1).lower()
    parent = path.parent.name.lower()
    if parent not in {"", ".", "audio", "data", "wav"}:
        return parent
    return None


def load_dataset(data_dir: Path, min_bytes: int = 1024) -> tuple[Dataset, list[str]]:
    paths: list[Path] = []
    labels: list[str] = []
    features: list[np.ndarray] = []
    skipped: list[str] = []

    for path in sorted(data_dir.rglob("*.wav")):
        label = infer_label(path)
        if label is None:
            skipped.append(f"{path}: could not infer label")
            continue
        if path.stat().st_size < min_bytes:
            skipped.append(f"{path}: file too small")
            continue
        try:
            vector = extract_features(path)
        except Exception as exc:
            skipped.append(f"{path}: {exc}")
            continue
        paths.append(path)
        labels.append(label)
        features.append(vector)

    if not features:
        raise RuntimeError(f"No valid wav files found under {data_dir}")

    unique_labels = sorted(set(labels))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_to_id[label] for label in labels], dtype=np.int64)
    X = np.vstack(features).astype(np.float32)
    return Dataset(paths=paths, labels=labels, X=X, y=y), skipped


def _markdown_confusion(labels: list[str], matrix: np.ndarray) -> str:
    header = "| 真实类别 \\ 预测类别 | " + " | ".join(labels) + " |"
    divider = "| --- | " + " | ".join(["---"] * len(labels)) + " |"
    rows = [header, divider]
    for label, row in zip(labels, matrix):
        rows.append("| " + label + " | " + " | ".join(str(int(v)) for v in row) + " |")
    return "\n".join(rows)


def train_audio_classifier(data_dir: Path, output_dir: Path, test_size: float = 0.3, random_state: int = 42) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset, skipped = load_dataset(data_dir)
    label_names = sorted(set(dataset.labels))

    class_counts = {label: dataset.labels.count(label) for label in label_names}
    stratify = dataset.y if min(class_counts.values()) >= 2 else None
    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        dataset.X,
        dataset.y,
        dataset.paths,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    base_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
        ]
    )

    min_train_class = min(np.bincount(y_train))
    if min_train_class >= 2:
        cv_splits = min(5, int(min_train_class))
        search = GridSearchCV(
            base_pipeline,
            param_grid={"classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
            cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state),
            scoring="f1_macro",
        )
        model = search.fit(X_train, y_train)
        best_model = model.best_estimator_
        cv_summary = {
            "best_params": model.best_params_,
            "best_cv_f1_macro": float(model.best_score_),
            "cv_splits": cv_splits,
        }
    else:
        best_model = base_pipeline.fit(X_train, y_train)
        cv_summary = {
            "best_params": {"classifier__C": 1.0},
            "best_cv_f1_macro": None,
            "cv_splits": 0,
        }

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    train_accuracy = float(accuracy_score(y_train, y_train_pred))
    test_accuracy = float(accuracy_score(y_test, y_test_pred))
    test_f1_macro = float(f1_score(y_test, y_test_pred, average="macro"))
    matrix = confusion_matrix(y_test, y_test_pred, labels=list(range(len(label_names))))

    artifact = {
        "model": best_model,
        "feature_names": FEATURE_NAMES,
        "label_names": label_names,
    }
    with (output_dir / "audio_classifier.pkl").open("wb") as f:
        pickle.dump(artifact, f)

    predictions_path = output_dir / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "actual", "predicted"])
        for path, actual, predicted in zip(paths_test, y_test, y_test_pred):
            writer.writerow([str(path), label_names[int(actual)], label_names[int(predicted)]])

    feature_path = output_dir / "features.csv"
    with feature_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", *FEATURE_NAMES])
        for path, label, row in zip(dataset.paths, dataset.labels, dataset.X):
            writer.writerow([str(path), label, *[float(v) for v in row]])

    metrics = {
        "data_dir": str(data_dir),
        "sample_count": len(dataset.paths),
        "class_counts": class_counts,
        "train_count": int(len(y_train)),
        "test_count": int(len(y_test)),
        "train_accuracy": round(train_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4),
        "test_f1_macro": round(test_f1_macro, 4),
        "confusion_matrix": matrix.tolist(),
        "labels": label_names,
        "skipped_files": skipped,
        **cv_summary,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    report = [
        "# 音频分类器训练报告",
        "",
        "## 目标",
        "",
        "从音频片段中训练一个小型监督学习模型。这个部分把项目和课程内容连接起来：特征提取、训练/测试划分、标准化、逻辑回归、正则化、模型选择和分类指标。",
        "",
        "## 数据集",
        "",
        f"- 数据目录: `{data_dir}`",
        f"- 有效样本数: `{len(dataset.paths)}`",
        f"- 跳过文件数: `{len(skipped)}`",
        f"- 类别数量: `{class_counts}`",
        "",
        "## 模型",
        "",
        "- 特征: 波形能量、过零率、频谱中心、频谱带宽、频谱滚降、低/高频能量比、静音比例。",
        "- 分类器: Logistic Regression。",
        "- 预处理: StandardScaler。",
        "- 模型选择: 样本足够时，用 GridSearchCV 搜索 L2 正则化强度 `C`。",
        f"- 最优参数: `{cv_summary['best_params']}`",
        "",
        "## 指标",
        "",
        f"- 训练准确率: `{metrics['train_accuracy']}`",
        f"- 测试准确率: `{metrics['test_accuracy']}`",
        f"- 测试集宏平均 F1: `{metrics['test_f1_macro']}`",
        "",
        "## 混淆矩阵",
        "",
        _markdown_confusion(label_names, matrix),
        "",
        "## 结果解释",
        "",
        "- 如果训练准确率明显高于测试准确率，说明模型可能过拟合。",
        "- 如果两者都低，说明手工音频特征可能不够表达任务差异。",
        "- 后续可以加入 MFCC 或神经网络说话人嵌入等更强特征。",
        "",
        "## 分类报告",
        "",
        "```text",
        classification_report(y_test, y_test_pred, target_names=label_names, zero_division=0),
        "```",
    ]
    (output_dir / "training_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return metrics


def load_classifier(model_path: Path) -> dict:
    with model_path.open("rb") as f:
        return pickle.load(f)


def predict_audio_label(model_path: Path, wav_path: Path) -> tuple[str, dict[str, float]]:
    artifact = load_classifier(model_path)
    model = artifact["model"]
    label_names = artifact["label_names"]
    x = extract_features(wav_path).reshape(1, -1)
    label_id = int(model.predict(x)[0])
    probabilities: dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]
        probabilities = {label: float(prob) for label, prob in zip(label_names, probs)}
    return label_names[label_id], probabilities
