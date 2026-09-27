"""Run the frozen Nox v3 CSS continuation or public command, with a receipt.

The CSS continuation requires a completed, SHA-bound over-budget recovery
receipt. It appends ``--resume`` to the original native collector command;
all successful answers still come from the same pinned native adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def run(
    plan_path: Path,
    expected_plan_sha: str,
    panel: str,
    recovery_path: Path | None,
    log_path: Path,
    receipt_path: Path,
) -> dict:
    if panel not in {"css", "public"} or os.environ.get("GPU_ID") != "1":
        raise ValueError("unexpected Nox panel or GPU")
    if sha(plan_path) != expected_plan_sha:
        raise ValueError("frozen plan digest changed")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    model = next(item for item in plan["inference"] if item["key"] == "nox")
    if model["group"] != "decision1" or model["model_id"] != (
        "llm-semantic-router/Decision-1.0-Nox-4B"
    ):
        raise ValueError("Nox identity changed")
    predictions = Path(model["paths"][panel])
    command = model["commands"][("typed", "css", "public").index(panel)]
    if panel == "css":
        if recovery_path is None or not predictions.is_file():
            raise ValueError("CSS continuation needs a recovery receipt and prefix")
        recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
        if (
            recovery.get("schema_version") != "decision2-v3-nox-overbudget-recovery/1"
            or recovery.get("plan_sha256") != expected_plan_sha
            or recovery.get("recovered_predictions_sha256") != sha(predictions)
        ):
            raise ValueError("recovery receipt does not match the partial predictions")
        command += " --resume"
    elif recovery_path is not None or predictions.exists():
        raise ValueError("public run must use a fresh output and the original command")
    if log_path.exists() or receipt_path.exists():
        raise FileExistsError("this continuation attempt already exists")
    started = time.monotonic()
    start_utc = datetime.now(timezone.utc).isoformat()
    with log_path.open("xb") as output:
        os.chmod(log_path, 0o600)
        completed = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            stdout=output,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            check=False,
            timeout=14400,
        )
    receipt = {
        "schema_version": "decision2-v3-nox-native-continuation/1",
        "plan_sha256": expected_plan_sha,
        "script_sha256": sha(Path(__file__)),
        "model_id": model["model_id"],
        "model_revision": model["revision"],
        "panel": panel,
        "gpu": "1",
        "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
        "recovery_receipt_sha256": sha(recovery_path) if recovery_path else None,
        "start_utc": start_utc,
        "end_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_hours": (time.monotonic() - started) / 3600,
        "exit_code": completed.returncode,
        "log_sha256": sha(log_path),
        "predictions_sha256": sha(predictions) if predictions.is_file() else None,
    }
    with receipt_path.open("x", encoding="utf-8") as output:
        os.chmod(receipt_path, 0o600)
        json.dump(receipt, output, sort_keys=True, indent=2)
        output.write("\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--panel", choices=("css", "public"), required=True)
    parser.add_argument("--recovery", type=Path)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.plan,
        args.plan_sha256,
        args.panel,
        args.recovery,
        args.log,
        args.receipt,
    )
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
