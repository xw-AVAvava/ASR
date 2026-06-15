from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from asr_mvp.model_training import train_audio_classifier


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a small audio classifier for the ASR project")
    parser.add_argument("--data-dir", required=True, type=Path, help="Directory containing labeled .wav files")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for model and reports")
    parser.add_argument("--test-size", default=0.3, type=float)
    args = parser.parse_args()

    metrics = train_audio_classifier(args.data_dir, args.output, test_size=args.test_size)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

