"""One-shot native SELECT-only BEST reload for the official 4B source arm.

This runs only after the sealed 466-step trainer completes. All output stays
private. It never reads CAL, typed DEV, CSS, formal, or public benchmark data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
import sys
import time
from pathlib import Path

from scripts.admit_qwen35_4b_posttrained import IMAGE_ID, _require_gpu, compare
from training.model.data import load_partition
from training.model.decision_model import DecisionModel, encode
from training.model.train import evaluate

LOCK_SHA256 = "7a3fe34eef8a4845edb71bead0aa12660993d0b81ad3a99184714f69e37809cd"
MAX_SECONDS = 180
FIRST_ROWS = 32


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_lock(path: Path) -> dict:
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o600
        or sha(path) != LOCK_SHA256
    ):
        raise ValueError("Full-arm lock changed")
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("status") != "LOCKED_NO_GPU"
        or result.get("image_id") != IMAGE_ID
        or result.get("planned_updates") != 466
    ):
        raise ValueError("Full-arm lock does not admit SELECT reload")
    return result


def _write_once(path: Path, value: dict) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        json.dump(value, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())


def _best(run: Path) -> tuple[int, Path, Path]:
    complete = json.loads((run / "COMPLETE.json").read_text(encoding="utf-8"))
    if (
        complete.get("status") != "complete"
        or complete.get("step") != 466
        or complete.get("planned_updates") != 466
        or complete.get("calibration_status") != "untouched"
    ):
        raise ValueError("Full training did not complete its frozen budget")
    checkpoints = sorted(path for path in run.glob("checkpoint-*") if path.is_dir())
    steps = [int(path.name.rsplit("-", 1)[1]) for path in checkpoints]
    if steps != [64, 128, 192, 256, 320, 384, 448, 466]:
        raise ValueError("SELECT checkpoint roster differs from frozen schedule")
    scored = [
        (
            json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))[
                "dev_metrics"
            ],
            step,
            path,
        )
        for step, path in zip(steps, checkpoints, strict=True)
    ]
    _, step, checkpoint = max(
        scored,
        key=lambda item: (
            item[0]["family_macro_accuracy"],
            -item[0]["family_macro_brier"],
            -item[1],
        ),
    )
    best = json.loads((run / "BEST.json").read_text(encoding="utf-8"))
    if (
        best.get("checkpoint") != checkpoint.name
        or complete.get("best") != checkpoint.name
    ):
        raise ValueError("BEST differs from fixed SELECT-only selector")
    predictions = run / f"select-step-{step:07d}-predictions.jsonl"
    if not predictions.is_file():
        raise ValueError("BEST SELECT predictions are absent")
    return step, checkpoint, predictions


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    lock = read_lock(args.lock)
    _require_gpu()
    import torch

    if sys.executable != "/usr/bin/python":
        raise ValueError("Reload interpreter differs from Base-compatible runtime")
    run_root = args.run.resolve(strict=True)
    select = args.select.resolve(strict=True)
    source = args.source.resolve(strict=True)
    if sha(select) != lock["data_sha256"]["select"]:
        raise ValueError("SELECT bytes differ from full-arm lock")
    if source.name != lock["source_fingerprint"]["source_name"]:
        raise ValueError("Official source identity differs")
    provenance = json.loads((run_root / "provenance.json").read_text(encoding="utf-8"))
    if (
        provenance.get("model_source") != lock["source_fingerprint"]
        or provenance.get("contract", {}).get("planned_updates") != 466
        or provenance.get("contract", {}).get("data_sha256") != lock["data_sha256"]
        or provenance.get("cal_examples_audited_only") != 700
    ):
        raise ValueError("Completed run provenance differs from sealed full arm")
    step, checkpoint, predictions = _best(run_root)
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("SELECT reload output must be new")
    out.mkdir(mode=0o700)

    def _timeout(_sig: int, _frame: object) -> None:
        raise TimeoutError("SELECT reload exceeded 180 seconds")

    old_handler = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(MAX_SECONDS)
    try:
        model, tokenizer = DecisionModel.from_checkpoint(checkpoint, source_path=source)
        model = model.float().to(torch.device("cuda:0"))
        model.backbone.config.use_cache = False
        rows = load_partition(select, "select")[:FIRST_ROWS]
        encoded = [encode(row, tokenizer, 8192) for row in rows]
        evaluate(
            model,
            encoded,
            pad_id=tokenizer.pad_token_id,
            batch_size=2,
            device=torch.device("cuda:0"),
            output=out,
            tag="best-reload32",
        )
        torch.cuda.synchronize()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    original = out / "best-original32.jsonl"
    fd = os.open(original, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        for line in predictions.read_text(encoding="utf-8").splitlines()[:FIRST_ROWS]:
            stream.write(line + "\n")
    comparison = compare(original, out / "best-reload32-predictions.jsonl", FIRST_ROWS)
    elapsed = time.monotonic() - started
    result = {
        "schema_version": "decision2-qwen35-4b-posttrained-full-select-reload/1",
        "status": comparison["status"] if elapsed <= MAX_SECONDS else "FAIL",
        "lock_sha256": LOCK_SHA256,
        "step": step,
        "checkpoint": checkpoint.name,
        "elapsed_seconds": elapsed,
        "comparison": comparison,
        "provenance_sha256": sha(run_root / "provenance.json"),
        "complete_sha256": sha(run_root / "COMPLETE.json"),
        "best_sha256": sha(run_root / "BEST.json"),
        "original_full_select_sha256": sha(predictions),
        "reload32_sha256": sha(out / "best-reload32-predictions.jsonl"),
    }
    _write_once(out / "select-reload-receipt.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("SELECT BEST reload parity failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "run", "source", "select", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    result = run(parser.parse_args())
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "step", "elapsed_seconds", "comparison")
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
