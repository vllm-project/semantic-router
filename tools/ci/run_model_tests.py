#!/usr/bin/env python3
"""Run the published-model contract through the model runtime; reject skips and empty selection.

Every listed Vela 1.0 test serves its package through a managed model runtime
(``servingtest.Managed``) and reads the package from an environment variable;
this runner provisions the runtime's pinned packages as plain directories and
points those variables at them. The Vela 2.0 tests ask a runtime that serves
one pinned size on CPU: the runner downloads each size's pinned files into the
runtime cache, serves it offline from there at the profile the Router's
implicit CPU deployment runs it with, and names the endpoint for the size's
tests. Every listed test must pass: a skip, a failure or a missing test fails
the contract.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
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
# Log -> the Vela 2.0 size a runtime of its own serves for the tests that log
# there: its repository, the profile the Router's implicit CPU deployment runs
# it with (config.ImplicitModelRuntimeDeployment), the variable that names the
# endpoint and the tests.
VELA2_SUITES = {
    "vela2-0.3b.jsonl": (
        "vllm-sr/Vela-2.0-0.3B",
        "max_speed",
        "VLLM_SRUN_VELA2_ENDPOINT",
        ("TestVela2PublishedAnswers03BRealModel", "TestVela2RouterMatchesSystemOne"),
    ),
    "vela2-0.8b.jsonl": (
        "vllm-sr/Vela-2.0-0.8B",
        "exact",
        "VLLM_SRUN_VELA2_08B_ENDPOINT",
        ("TestVela2PublishedAnswers08BRealModel",),
    ),
}
SELECTIONS = (
    ("./pkg/classification", CLASSIFIER_TESTS, "classification.jsonl"),
    ("./pkg/cache", CACHE_TESTS, "cache.jsonl"),
    *(
        ("./pkg/" + package, names, "omni-" + package + ".jsonl")
        for package, names in OMNI_TESTS.items()
    ),
    *(("./pkg/classification", suite[3], log) for log, suite in VELA2_SUITES.items()),
)
# A cold CPU load reads the weights and runs the golden check; the 0.8B takes
# about half a minute on four cores.
VELA2_READY_TIMEOUT_SECONDS = 900


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


def provision_vela2(cache: Path) -> dict[str, dict]:
    """Download each Vela 2.0 size's pinned files into the runtime cache, which serves them offline."""
    from vllm_srun.registry.resolve import (  # noqa: PLC0415 - only the model lane installs the runtime
        resolve,
    )

    models = {}
    for log, (repo, profile, env, _) in VELA2_SUITES.items():
        started = time.monotonic()
        ref = resolve(repo, cache_dir=cache)
        models[log] = {
            "env": env,
            "repo_id": repo,
            "revision": ref.revision,
            "profile": profile,
            "bytes": sum(
                path.stat().st_size for path in ref.root.rglob("*") if path.is_file()
            ),
            "provision_seconds": round(time.monotonic() - started, 1),
        }
    return models


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wait_ready(
    process: subprocess.Popen, endpoint: str, started: float, runtime: str
) -> None:
    """Poll a starting runtime's /health until it answers; fail when it exits or takes too long."""
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"{runtime}: the runtime exited {process.returncode}")
        if time.monotonic() - started > VELA2_READY_TIMEOUT_SECONDS:
            raise RuntimeError(
                f"{runtime}: not ready in {VELA2_READY_TIMEOUT_SECONDS} s"
            )
        try:
            with urllib.request.urlopen(endpoint + "/health", timeout=5):
                return
        except OSError:
            time.sleep(0.5)


@contextlib.contextmanager
def serve_vela2(model: dict, cache: Path, log: Path) -> Iterator[str]:
    """Serve one pinned Vela 2.0 size on CPU, offline, until the block ends; yield its endpoint.

    The result cache is off, so a repeated request runs the model again.
    """
    port = free_port()
    command = [
        *shlex.split(os.environ.get("VLLM_SRUN_COMMAND", "vllm-srun")),
        "serve",
        model["repo_id"],
        "--revision",
        model["revision"],
        "--device",
        "cpu",
        "--profile",
        model["profile"],
        "--cache-dir",
        str(cache),
        "--offline",
        "--result-cache-entries",
        "0",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    endpoint = f"http://127.0.0.1:{port}"
    started = time.monotonic()
    with log.open("w") as output:
        process = subprocess.Popen(
            command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            wait_ready(process, endpoint, started, f"{model['repo_id']} ({log.name})")
            model["ready_seconds"] = round(time.monotonic() - started, 1)
            print(
                f"{model['repo_id']}@{model['revision'][:12]} ready on CPU "
                f"({model['profile']}) after {model['ready_seconds']} s",
                flush=True,
            )
            yield endpoint
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()


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


def run_vela2_suite(
    package: str,
    names: tuple[str, ...],
    env: dict,
    output: Path,
    model: dict,
    cache: Path,
) -> dict:
    """Run a Vela 2.0 size's tests against a runtime serving it; a runtime that never gets ready fails them."""
    try:
        with serve_vela2(model, cache, output.with_suffix(".runtime.log")) as endpoint:
            return run_suite(package, names, {**env, model["env"]: endpoint}, output)
    except RuntimeError as error:
        print(f"::error::{error}", flush=True)
        result = validate_results([], set(names))
        result.update(
            package=package, exit_code=1, expected=sorted(names), error=str(error)
        )
        return result


def record_vela2(models_dir: Path, output: Path) -> int:
    """Rewrite the recorded answers of every Vela 2.0 size from the served pinned size."""
    cache = models_dir / ".runtime-cache"
    env = {
        **os.environ,
        "VLLM_SR_REQUIRE_MODEL_TESTS": "1",
        "VLLM_SR_VELA2_RECORD": "1",
    }
    suites = [
        run_vela2_suite(
            "./pkg/classification",
            VELA2_SUITES[log][3][:1],
            env,
            output / log,
            model,
            cache,
        )
        for log, model in provision_vela2(cache).items()
    ]
    return 0 if all(suite["success"] for suite in suites) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument(
        "--omni", type=Path, required=True, help="the pinned Vela Omni Nano snapshot"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--record-vela2",
        action="store_true",
        help="rewrite the Vela 2.0 tests' recorded answers on this CPU instead of running the contract",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.record_vela2:
        return record_vela2(args.models_dir.resolve(), args.output)
    models = provision(args.models_dir.resolve())
    cache = args.models_dir.resolve() / ".runtime-cache"
    vela2 = provision_vela2(cache)
    env = {
        **os.environ,
        "VLLM_SR_REQUIRE_MODEL_TESTS": "1",
        "REQUIRE_OMNI_TESTS": "1",
        "VELA_OMNI_ARTIFACT": str(args.omni.resolve()),
        **{env: model["path"] for env, model in models.items()},
    }
    suites = []
    for package, names, filename in SELECTIONS:
        output = args.output / filename
        if filename in vela2:
            suites.append(
                run_vela2_suite(package, names, env, output, vela2[filename], cache)
            )
        else:
            suites.append(run_suite(package, names, env, output))
    report = {
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "runtime": "model-runtime",
        "device": "cpu",
        "models": [
            *models.values(),
            *vela2.values(),
            {"env": "VELA_OMNI_ARTIFACT", "path": str(args.omni.resolve())},
        ],
        "suites": suites,
        "success": all(s["success"] for s in suites),
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
