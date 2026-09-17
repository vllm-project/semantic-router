#!/usr/bin/env python3
"""Run compatible component contracts with isolated evidence and shared setup."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from ci_plan import digest
from ci_results import actual_platform, make_receipt
from python_test_evidence import summarize
from verification_catalog import load_catalog
from workflow_evidence import go_cases, junit_cases

ROOT = Path(__file__).resolve().parents[2]


def validate_batch(batch: dict) -> None:
    catalog = load_catalog()
    worker = catalog["component_workers"].get(batch.get("id"))
    if not worker or any(batch.get(key) != value for key, value in worker.items()):
        raise ValueError("component batch has an unknown worker contract")
    records = batch.get("verifications", [])
    if not records or len({row["id"] for row in records}) != len(records):
        raise ValueError("component batch must contain distinct verification IDs")
    for row in records:
        declared = catalog["verifications"].get(row["id"], {})
        if (
            declared.get("executor") != "tools"
            or declared.get("worker") != batch["id"]
            or row.get("executor") != "tools"
            or row.get("worker") != batch["id"]
            or row.get("target") != declared.get("target")
            or row.get("contract_sha256")
            != digest(
                {key: value for key, value in row.items() if key != "contract_sha256"}
            )
        ):
            raise ValueError(
                f"component contract differs from its planned worker: {row['id']}"
            )


def commands(target: str, output: Path) -> list[list[str]]:
    """Keep the maintained Make entrypoints as the owners of discovery."""
    python = str(ROOT / ".venv-agent/bin/python")
    if target == "test-learning-tools":
        return [["make", target], ["make", "test-calibration"]]
    if target == "soak-test":
        return [
            ["make", target, f"SOAK_TEST_REPORT_DIR={output}"],
            [
                python,
                "-m",
                "pytest",
                "bench/test_openai_fault_proxy.py",
                "-q",
                f"--junitxml={output / 'proxy.xml'}",
            ],
        ]
    if target == "test-e2e-unit":
        return [["make", target, f"E2E_UNIT_REPORT_DIR={output}"]]
    if target in {
        "vllm-sr-test",
        "vllm-sr-sim-test",
        "harness-check",
        "onnx-artifact-test",
        "ck-rewrite-test",
        "test-training-contracts",
        "test-provider-simulator",
    }:
        return [["make", target]]
    raise ValueError(f"unknown component target: {target}")


def run_command(command: list[str], env: dict, log: Path) -> None:
    with log.open("a") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        print("\n".join(log.read_text().splitlines()[-100:]), flush=True)
        result.check_returncode()


def contract_evidence(record: dict, output: Path) -> dict:
    if record["target"] == "test-e2e-unit":
        return json.loads((output / "evidence.json").read_text())
    if record["target"] == "soak-test":
        cases, expected = go_cases(
            output / "tests.jsonl", output / "inventory.jsonl", set()
        )
        proxy = junit_cases(output / "proxy.xml")
        cases.extend({**case, "id": "proxy:" + case["id"]} for case in proxy)
        expected.extend("proxy:" + case["id"] for case in proxy)
        return {
            "runtime": "none",
            "device": "none",
            "platform": actual_platform(),
            "expected_cases": expected,
            "cases": cases,
        }
    return summarize(output / "python-events.jsonl")


def run_batch(batch: dict, output: Path) -> bool:
    validate_batch(batch)
    # A rerun must not qualify stale output from a prior execution.
    output.mkdir(parents=True, exist_ok=False)
    results = output / "results"
    results.mkdir()
    observer = output / "observer"
    observer.mkdir()
    (observer / "sitecustomize.py").write_text(
        "from python_test_evidence import observe_unittest\nobserve_unittest()\n"
    )
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"CI_PYTHON_TEST_EVENTS", "PYTEST_PLUGINS", "PYTHONPATH"}
    }
    failed = []
    for record in batch["verifications"]:
        identity, target = record["id"], record["target"]
        raw = output / "raw" / identity
        raw.mkdir(parents=True)
        (raw / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
        log = raw / "commands.log"
        print(f"::group::{record['display_name']}", flush=True)
        try:
            if target == "test-training-contracts":
                python = str(ROOT / ".venv-agent/bin/python")
                for command in (
                    [
                        python,
                        "-m",
                        "pip",
                        "install",
                        "torch==2.10.0",
                        "--index-url",
                        "https://download.pytorch.org/whl/cpu",
                    ],
                    [
                        python,
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        "tools/ci/training-test-requirements.txt",
                    ],
                ):
                    run_command(command, env, log)
            test_env = dict(env)
            if target not in {"test-e2e-unit", "soak-test"}:
                test_env.update(
                    CI_REPO_ROOT=str(ROOT),
                    CI_PYTHON_TEST_EVENTS=str(raw / "python-events.jsonl"),
                    PYTHONPATH=os.pathsep.join((str(observer), str(ROOT / "tools/ci"))),
                    PYTEST_PLUGINS="python_test_evidence",
                )
            if target == "test-training-contracts":
                test_env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            for command in commands(target, raw):
                run_command(command, test_env, log)
            evidence = contract_evidence(record, raw)
            (raw / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
            receipt = make_receipt(
                record,
                evidence,
                source_sha=source,
                execution_platform=actual_platform(),
                environ=env,
            )
            (results / f"{identity}.json").write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
            print(
                f"{identity}: {len(evidence['cases'])} required cases passed",
                flush=True,
            )
        except (
            ValueError,
            KeyError,
            TypeError,
            OSError,
            subprocess.CalledProcessError,
        ) as exc:
            failed.append(identity)
            (raw / "failure.txt").write_text(str(exc) + "\n")
            print(f"::error::{identity}: {exc}", flush=True)
        finally:
            print("::endgroup::", flush=True)
    if failed:
        print(f"Failed component contracts: {', '.join(failed)}", flush=True)
    return not failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return 0 if run_batch(json.loads(args.batch), args.output.resolve()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
