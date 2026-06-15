from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path


LOAD_TIMEOUT_SECONDS = 45


def resolve_tiny_model() -> str | Path:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--Systran--faster-whisper-tiny" / "snapshots"
    if not cache_root.exists():
        return "tiny"
    candidates = []
    for snapshot in cache_root.iterdir():
        if not snapshot.is_dir():
            continue
        required = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]
        if all((snapshot / name).exists() for name in required):
            candidates.append(snapshot)
    if not candidates:
        return "tiny"
    return max(candidates, key=lambda path: path.stat().st_mtime)


def configure_safe_cpu_runtime(force_cpu_isa: str | None) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    if force_cpu_isa:
        os.environ["CT2_FORCE_CPU_ISA"] = force_cpu_isa
    else:
        os.environ.pop("CT2_FORCE_CPU_ISA", None)


def load_worker(model_ref: str, compute_type: str, force_cpu_isa: str | None, queue: mp.Queue) -> None:
    try:
        configure_safe_cpu_runtime(force_cpu_isa)
        from faster_whisper import WhisperModel

        start = time.time()
        WhisperModel(
            model_ref,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=1,
            num_workers=1,
        )
        queue.put(("ok", round(time.time() - start, 2)))
    except Exception as exc:
        queue.put(("error", repr(exc)))


def try_load(model_ref: str, compute_type: str, force_cpu_isa: str | None) -> bool:
    label = force_cpu_isa or "auto"
    print(f"[Test] Trying compute_type={compute_type}, CT2_FORCE_CPU_ISA={label}", flush=True)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=load_worker, args=(model_ref, compute_type, force_cpu_isa, queue))
    process.start()
    process.join(LOAD_TIMEOUT_SECONDS)
    if process.is_alive():
        process.terminate()
        process.join(5)
        print(f"[Test] TIMEOUT after {LOAD_TIMEOUT_SECONDS}s. This setting hangs on this machine.", flush=True)
        return False

    if queue.empty():
        print(f"[Test] FAILED without a Python error. Exit code: {process.exitcode}", flush=True)
        return False

    status, value = queue.get()
    if status == "ok":
        print(f"[Test] OK. Model loaded in {value}s.", flush=True)
        print(f"[Test] Recommended pipeline options: --compute-type {compute_type}", flush=True)
        if force_cpu_isa:
            print(f"[Test] Also add: --force-cpu-isa {force_cpu_isa}", flush=True)
        return True

    print(f"[Test] ERROR: {value}", flush=True)
    return False


def main() -> int:
    from faster_whisper import WhisperModel
    del WhisperModel

    model_ref = resolve_tiny_model()
    print(f"[Test] Model path: {model_ref}", flush=True)
    cases = [
        ("int8", None),
        ("int8", "GENERIC"),
        ("int8_float32", "GENERIC"),
        ("float32", "GENERIC"),
    ]
    for compute_type, force_cpu_isa in cases:
        if try_load(str(model_ref), compute_type, force_cpu_isa):
            return 0

    print("[Test] No tested faster-whisper CPU setting loaded successfully.", flush=True)
    print("[Test] Next fallback: use the provided transcript mode, or switch ASR backend.", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
