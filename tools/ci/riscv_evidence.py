#!/usr/bin/env python3
"""Qualify Candle's emulated RISC-V execution without claiming physical hardware."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
from http import HTTPStatus
from pathlib import Path

from ci_results import actual_platform

ROOT = Path(__file__).resolve().parents[2]
TARGET = "riscv64gc-unknown-linux-gnu"
ELF_MACHINE_HEADER_SIZE = 20
ELF_MACHINE_RISCV = 243
RUST_TESTS = (
    "model_architectures::traditional::modernbert_test::test_candle_context_classifier_loaders_execute_beyond_default",
    "model_architectures::traditional::candle_models::modernbert::tests::test_chunked_attention_matches_dense",
)


def go_cases(path: Path, required: set[str], prefix: str) -> list[dict]:
    events = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.startswith("{")
    ]
    active, started, packages, terminals = set(), set(), set(), []
    cases = []
    for event in events:
        name, action = event.get("Test"), event.get("Action")
        if action not in {"run", "pass", "skip", "fail"}:
            continue
        packages.add(event.get("Package"))
        if not name:
            if action != "run":
                terminals.append(action)
            if active:
                raise ValueError("RISC-V Go process ended before its tests")
            continue
        root = name.split("/", 1)[0]
        if root not in required:
            raise ValueError(f"unexpected RISC-V test: {name}")
        if action == "run":
            if name in started or (name != root and root not in active):
                raise ValueError(f"duplicate or orphan RISC-V test: {name}")
            started.add(name)
            active.add(name)
        else:
            if name not in active or action != "pass":
                raise ValueError(f"RISC-V test missing, failed or skipped: {name}")
            active.remove(name)
            if any(child.startswith(name + "/") for child in active):
                raise ValueError("RISC-V test finished before its children")
            cases.append({"id": prefix + ":" + name, "status": "passed"})
    if (
        not required
        or not required <= started
        or active
        or terminals != ["pass"]
        or len(packages) != 1
        or None in packages
    ):
        raise ValueError("incomplete RISC-V Go execution inventory")
    return cases


def rust_cases(directory: Path) -> list[dict]:
    required = (directory / "rust-required.txt").read_text().splitlines()
    if required != list(RUST_TESTS):
        raise ValueError("RISC-V synthetic classifier/attention inventory changed")
    discovered = (directory / "rust-list.txt").read_text()
    cases = []
    for index, name in enumerate(required):
        if name + ": test" not in discovered:
            raise ValueError(f"RISC-V Rust case not discovered: {name}")
        text = (directory / f"rust-{index}.log").read_text()
        if (
            name not in text
            or "running 1 test" not in text
            or len(
                re.findall(r"test result: ok\. 1 passed; 0 failed; 0 ignored;", text)
            )
            != 1
            or "test result: FAILED" in text
        ):
            raise ValueError(f"RISC-V Rust case did not pass exactly once: {name}")
        cases.append({"id": "rust:" + name, "status": "passed"})
    return cases


def target_binary(path: Path) -> dict:
    with path.open("rb") as stream:
        header = stream.read(ELF_MACHINE_HEADER_SIZE)
        if (
            len(header) < ELF_MACHINE_HEADER_SIZE
            or header[:6] != b"\x7fELF\x02\x01"
            or int.from_bytes(header[18:20], "little") != ELF_MACHINE_RISCV
        ):
            raise ValueError(f"not a little-endian ELF64 RISC-V binary: {path}")
        stream.seek(0)
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "platform": "linux/riscv64",
    }


def router_cases(report: dict, source: str) -> list[dict]:
    responses = report.get("responses", [])
    if report.get("source_sha") != source or [
        item.get("path") for item in responses
    ] != ["health", "ready", "classify", "preview"]:
        raise ValueError("RISC-V router response source or inventory differs")
    for item in responses:
        if item.get("http_status") != HTTPStatus.OK or not item.get("body"):
            raise ValueError(f"RISC-V router request failed: {item}")
    response = responses[2]["body"]
    if (
        not response.get("classification", {}).get("category")
        or response.get("routing_decision") == "placeholder_response"
        or response.get("signal_errors")
    ):
        raise ValueError("RISC-V router did not execute Domain inference")
    preview = responses[3]["body"]
    domain = (preview.get("metrics") or {}).get("domain") or {}
    score = domain.get("confidence")
    latency = domain.get("execution_time_ms")
    if (
        preview.get("signal_errors")
        or domain.get("confidence_available") is not True
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0 < score <= 1
        or not isinstance(latency, (int, float))
        or not math.isfinite(latency)
        or latency <= 0
    ):
        raise ValueError("RISC-V Preview did not observe scored Domain inference")
    return [
        {"id": "router:" + item["path"], "status": "passed", "response": item}
        for item in responses
    ]


def evidence(directory: Path) -> dict:
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
    ).strip()
    manifest = json.loads((directory / "models.json").read_text())
    models = manifest.get("models", [])
    if manifest.get("provider") != "candle" or len(models) != 1:
        raise ValueError("RISC-V requires one registered Candle Domain checkpoint")
    model = models[0]
    if (
        model.get("name") != "Domain"
        or not re.fullmatch(r"[0-9a-f]{40}", model.get("revision", ""))
        or Path(model["path"]).name != model["revision"]
    ):
        raise ValueError("RISC-V checkpoint has no immutable identity")
    pattern = (directory / "minimal-pattern.txt").read_text().strip()
    if pattern != (
        "^Test(Owned.*|NewRegexProvider|RegexProvider_.*|UtilityFunctions|"
        "EmbeddingCapabilitiesConformance|EmbeddingDimensionStateValidation)$"
    ):
        raise ValueError(
            "RISC-V minimal bindings must use the maintained owned fixture selection"
        )
    excluded = "TestOwnedNativeMaintainedHallucinationWithoutLabelMetadata"
    skip = (directory / "minimal-skip.txt").read_text().strip()
    ownership = json.loads((ROOT / "tools/ci/core_test_profiles.json").read_text())[
        "excluded"
    ]
    if skip != "^" + excluded + "$" or not any(
        row["package"] == "candle-binding"
        and row["test"] == excluded
        and row["profile"] == "legacy-hallucination-checkpoints"
        for row in ownership
    ):
        raise ValueError("RISC-V fixture exclusion has no declared checkpoint owner")
    listed = (directory / "binding-list.txt").read_text().splitlines()
    required = {
        name for name in listed if re.fullmatch(pattern, name) and name != excluded
    }
    if not {
        "TestUtilityFunctions",
        "TestNewRegexProvider",
        "TestEmbeddingCapabilitiesConformance",
        "TestEmbeddingDimensionStateValidation",
    } <= required or not any(name.startswith("TestOwned") for name in required):
        raise ValueError("RISC-V owned binding discovery is incomplete")
    cases, expected = [], []
    for filename, names, prefix in (
        ("host-parity.jsonl", {"TestCandleClassifierParity"}, "host"),
        ("qemu-ffi.jsonl", {"TestNativeClassifierFFIIsLinked"}, "qemu-ffi"),
        ("qemu-parity.jsonl", {"TestCandleClassifierParity"}, "qemu-parity"),
        ("qemu-minimal.jsonl", required, "qemu-binding"),
    ):
        cases.extend(go_cases(directory / filename, names, prefix))
        expected.extend(prefix + ":" + name for name in sorted(names))
    extra = rust_cases(directory) + router_cases(
        json.loads((directory / "router.json").read_text()), source
    )
    cases.extend(extra)
    expected.extend(item["id"] for item in extra)
    binaries = [
        target_binary(ROOT / path)
        for path in (
            f"candle-binding/target/{TARGET}/release/libcandle_semantic_router.so",
            f"candle-binding/target/{TARGET}/candle-riscv64.test",
            "bin/router-riscv64",
        )
    ]
    emulator = shutil.which("qemu-riscv64-static") or shutil.which("qemu-riscv64")
    if not emulator:
        raise ValueError("RISC-V evidence requires qemu-user")
    version = subprocess.check_output([emulator, "--version"], text=True).splitlines()[
        0
    ]
    if "qemu-riscv64" not in version:
        raise ValueError("unexpected RISC-V emulator identity")
    return {
        "source_sha": source,
        "runtime": "candle",
        "device": "cpu",
        "platform": "linux/riscv64",
        "execution": {"mode": "qemu-user", "host_platform": actual_platform()},
        "emulator_version": version,
        "models": models,
        "binaries": binaries,
        "cases": cases,
        "expected_cases": expected,
    }
