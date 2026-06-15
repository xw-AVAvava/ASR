from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from asr_mvp.model_training import predict_audio_label


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict the label of one wav file with the trained audio classifier")
    parser.add_argument("--model", required=True, type=Path, help="Path to audio_classifier.pkl")
    parser.add_argument("--audio", required=True, type=Path, help="Path to a wav file")
    args = parser.parse_args()

    label, probabilities = predict_audio_label(args.model, args.audio)
    print(json.dumps({"label": label, "probabilities": probabilities}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

