from __future__ import annotations

import importlib.util
import os
import site
import sys


PACKAGES = ["faster_whisper", "ctranslate2", "av"]


def main() -> int:
    try:
        user_site = site.getusersitepackages()
        if user_site and os.path.isdir(user_site) and user_site not in sys.path:
            sys.path.append(user_site)
    except Exception:
        user_site = None

    print("ASR dependency check")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Python executable: {sys.executable}")
    if user_site:
        print(f"User site-packages: {user_site}")
    missing = []
    for package in PACKAGES:
        ok = importlib.util.find_spec(package) is not None
        status = "OK" if ok else "MISSING"
        print(f"- {package}: {status}")
        if not ok:
            missing.append(package)

    if missing:
        print()
        print("Install command:")
        print("python -m pip install faster-whisper -i https://pypi.tuna.tsinghua.edu.cn/simple")
        return 1

    print()
    print("faster-whisper is ready. You can run:")
    print(
        "python project\\asr_meeting_assistant\\run_pipeline.py "
        "--audio project\\多人对话.wav "
        "--reference-file project\\多人对话文本.txt "
        "--output project\\asr_meeting_assistant\\outputs\\real_asr "
        "--engine faster-whisper --model tiny --language zh --speakers 5"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
