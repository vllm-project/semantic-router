#!/usr/bin/env python3
"""Run the published-model contract through the model runtime; reject skips and empty selection.

Every listed Go test serves its Vela 1.0 package through a managed model
runtime (``servingtest.Managed``) and reads the package from an environment
variable. This runner provisions the runtime's pinned packages as plain
directories, points those variables at them, and requires every listed test to
pass: a skip, a failure or a missing test fails the contract.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROUTER = ROOT / "src/semantic-router"
# Environment variable -> the runtime's built-in package it names.
PACKAGES = {
    "VLLM_SR_DOMAIN_MODEL": "vllm-sr/Vela-1.0-Encoder-307M-Domain",
    "VLLM_SR_PII_MODEL": "vllm-sr/Vela-1.0-Encoder-307M-PII",
    "VLLM_SR_JAILBREAK_MODEL": "vllm-sr/Vela-1.0-Encoder-307M-Guard",
    "VLLM_SR_FACTCHECK_MODEL": "vllm-sr/Vela-1.0-Encoder-307M-FactCheck",
    "VLLM_SR_FEEDBACK_MODEL": "vllm-sr/Vela-1.0-Encoder-307M-Feedback",
    "VLLM_SR_MMBERT_TEST_MODEL": "vllm-sr/Vela-1.0-Encoder-307M-Embedding",
}
# Router consumers read their label mappings from the package directory; the
# runtime's pinned file list leaves them out.
ROUTER_FILES = ["*_mapping.json"]
CLASSIFIER_TESTS = (
    "TestJailbreakDistributionRealModel",
    "TestJailbreakRiskRealModelContract",
    "TestClassifyPIIWithDetails_RealModelFindsPIIPastTheWindow",
    "TestFeedbackPredictionAndAbstentionRealModel",
    "TestFeedbackForcedAbstentionPreservesPredictionRealModel",
    "TestFactCheckPolicyRealModel",
    "TestFactCheckEmptyInputRealModel",
    "TestLocalClassifierMaintainedCPU",
    "TestUnifiedClassifierPublishedModels",
)
CACHE_TESTS = ("TestNegationFalseHitRegressionInMemory",)
# The pinned Omni Nano snapshot (VELA_OMNI_ARTIFACT) serves these.
OMNI_TESTS = {
    "classification": (
        "TestEmbeddingClassifier_IntegrationImageQueryEndToEnd",
        "TestEmbeddingClassifier_IntegrationTextRulesIgnoredOnImagePath",
    ),
    "cache": ("TestOmniStorageIntegrationUsesArtifactDimensionAndIdentity",),
}
SELECTIONS = (
    ("./pkg/classification", CLASSIFIER_TESTS, "classification.jsonl"),
    ("./pkg/cache", CACHE_TESTS, "cache.jsonl"),
    *(
        ("./pkg/" + package, names, "omni-" + package + ".jsonl")
        for package, names in OMNI_TESTS.items()
    ),
)


def required_inventory() -> set[tuple[str, str]]:
    """The receipt cannot shorten the checked-in mandatory case inventory."""
    return {(package, name) for package, names, _ in SELECTIONS for name in names}


def provision(models_dir: Path) -> dict[str, dict]:
    """Copy each pinned package into ``models_dir`` as plain files (the runtime refuses links)."""
    from vllm_srun.registry.resolve import (  # noqa: PLC0415 - only the model lane installs the runtime
        fetch,
        resolve,
    )

    cache = models_dir / ".runtime-cache"
    models = {}
    for env, repo in PACKAGES.items():
        target = models_dir / repo.split("/", 1)[1]
        ref = fetch(resolve(repo, cache_dir=cache), ROUTER_FILES, cache_dir=cache)
        if not _complete(target, ref):
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(ref.root, target, symlinks=False)
            (target / ".complete").write_text(ref.revision + "\n")
        models[env] = {
            "env": env,
            "repo_id": repo,
            "revision": ref.revision,
            "path": str(target),
        }
    return models


def _complete(target: Path, ref) -> bool:
    marker = target / ".complete"
    if not marker.is_file() or marker.read_text().strip() != ref.revision:
        return False
    return all(
        (target / path.relative_to(ref.root)).is_file()
        for path in ref.root.rglob("*")
        if path.is_file()
    )


def validate_results(events: list[dict], expected: set[str]) -> dict:
    passed = {e["Test"] for e in events if e.get("Action") == "pass" and "Test" in e}
    skipped = sorted(
        {e["Test"] for e in events if e.get("Action") == "skip" and "Test" in e}
    )
    failed = sorted(
        {
            e.get("Test", e.get("Package", "unknown"))
            for e in events
            if e.get("Action") == "fail"
        }
    )
    missing = sorted(expected - passed)
    return {
        "passed": sorted(passed),
        "skipped": skipped,
        "failed": failed,
        "missing": missing,
        "success": bool(expected) and not (skipped or failed or missing),
    }


def run_suite(package: str, names: tuple[str, ...], env: dict, output: Path) -> dict:
    args = [
        "go",
        "test",
        "-json",
        "-count=1",
        "-timeout=45m",
        "-p=1",
        "-run",
        "^(" + "|".join(names) + ")$",
        package,
    ]
    events = []
    with output.open("w") as log:
        process = subprocess.Popen(
            args,
            cwd=ROUTER,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                print(line, end="", flush=True)
                continue
            events.append(event)
            if event.get("Output"):
                print(event["Output"], end="", flush=True)
        code = process.wait()
    result = validate_results(events, set(names))
    result.update(package=package, exit_code=code, expected=sorted(names))
    result["success"] = result["success"] and code == 0
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument(
        "--omni", type=Path, required=True, help="the pinned Vela Omni Nano snapshot"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    models = provision(args.models_dir.resolve())
    env = {
        **os.environ,
        "VLLM_SR_REQUIRE_MODEL_TESTS": "1",
        "REQUIRE_OMNI_TESTS": "1",
        "VELA_OMNI_ARTIFACT": str(args.omni.resolve()),
        **{env: model["path"] for env, model in models.items()},
    }
    suites = [
        run_suite(package, names, env, args.output / filename)
        for package, names, filename in SELECTIONS
    ]
    report = {
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "runtime": "model-runtime",
        "device": "cpu",
        "models": [
            *models.values(),
            {"env": "VELA_OMNI_ARTIFACT", "path": str(args.omni.resolve())},
        ],
        "suites": suites,
        "success": all(s["success"] for s in suites),
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
