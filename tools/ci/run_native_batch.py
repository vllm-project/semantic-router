#!/usr/bin/env python3
"""Execute compatible native contracts with shared models and separate receipts."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from execution_batches import validate_execution_batch

ROOT = Path(__file__).resolve().parents[2]


def run_command(command: list[str], *, cwd: Path, env: dict, check: bool, timeout: int):
    process = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True)
    try:
        status = process.wait(timeout=timeout)
    finally:
        # A completed or timed-out contract must not leave compiler/test children
        # running in the next contract's worker. Kill the whole private group.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    result = subprocess.CompletedProcess(command, status)
    if check:
        result.check_returncode()
    return result


def run_batch(batch: dict, output: Path, *, run=run_command) -> bool:
    validate_execution_batch(batch, "native")
    if output.exists() and any(output.iterdir()):
        raise ValueError("native worker evidence directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    results = output / "results"
    results.mkdir(exist_ok=True)
    outcomes = []
    for record in batch["verifications"]:
        directory = output / record["id"]
        directory.mkdir(exist_ok=True)
        receipt = results / f"{record['id']}.json"
        receipt.unlink(missing_ok=True)
        env = {
            **os.environ,
            "MODEL_TEST_PROVIDER": record["runtime"],
            "MODEL_TEST_DEVICE": record["device"],
            "MODEL_TEST_REPORT_DIR": str(directory.resolve()),
            "MODEL_TEST_MANIFEST": str((directory / "models.json").resolve()),
            "MODEL_TEST_MODELS_DIR": str(ROOT / "models"),
            "VLLM_SR_REQUIRE_MODEL_TESTS": "1",
        }
        commands = [
            ["make", record["target"]],
            [
                sys.executable,
                "tools/ci/runtime_evidence.py",
                record.get("evidence", "native"),
                "--directory",
                str(directory),
                "--output",
                str(directory / "evidence.json"),
            ],
            [
                sys.executable,
                "tools/ci/ci_results.py",
                "record",
                "--verification",
                json.dumps(record),
                "--evidence",
                str(directory / "evidence.json"),
                "--output",
                str(receipt),
            ],
        ]
        status = "success"
        print(f"::group::{record['category']} / {record['display_name']}", flush=True)
        deadline = time.monotonic() + record.get("timeout_minutes", 120) * 60
        try:
            for command in commands:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(
                        command, record.get("timeout_minutes", 120) * 60
                    )
                result = run(
                    command,
                    cwd=ROOT,
                    env=env,
                    check=False,
                    timeout=remaining,
                )
                if result.returncode:
                    raise RuntimeError(
                        f"command failed ({result.returncode}): {command[0:2]}"
                    )
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            status = "failure"
            receipt.unlink(missing_ok=True)
            (directory / "failure.txt").write_text(str(error) + "\n")
            print(f"::error::{record['id']}: {error}", flush=True)
        finally:
            print("::endgroup::", flush=True)
        outcomes.append(
            {"id": record["id"], "category": record["category"], "result": status}
        )
    (output / "execution.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a") as stream:
            stream.write("| Category | Contract | Result |\n| --- | --- | --- |\n")
            for outcome in outcomes:
                stream.write(
                    f"| {outcome['category']} | {outcome['id']} | {outcome['result']} |\n"
                )
    return all(row["result"] == "success" for row in outcomes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        return 0 if run_batch(json.loads(args.batch), args.output.resolve()) else 1
    except (ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Native worker rejected: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
