#!/usr/bin/env python3
"""Qualify the pure-Go router on emulated RISC-V without claiming physical hardware.

The riscv64 router runs no model code: it attaches to a model runtime on the
host, so its classification must equal the runtime's own answer.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from http import HTTPStatus
from pathlib import Path

from ci_results import actual_platform

ROOT = Path(__file__).resolve().parents[2]
ROUTER_BINARY = Path("bin/router-riscv64")
ELF_MACHINE_HEADER_SIZE = 20
ELF_MACHINE_RISCV = 243


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
    ] != ["health", "ready", "classify"]:
        raise ValueError("RISC-V router response source or inventory differs")
    for item in responses:
        if item.get("http_status") != HTTPStatus.OK or not item.get("body"):
            raise ValueError(f"RISC-V router request failed: {item}")
    classified = responses[2]["body"]
    category = classified.get("classification", {}).get("category")
    if (
        not category
        or classified.get("routing_decision") == "placeholder_response"
        or classified.get("signal_errors")
    ):
        raise ValueError("RISC-V router did not classify through the runtime")
    results = report.get("runtime", {}).get("results") or [{}]
    if category != results[0].get("label"):
        raise ValueError(
            f"RISC-V router category {category!r} differs from the runtime's "
            f"{results[0].get('label')!r}"
        )
    return [
        {"id": "router:" + item["path"], "status": "passed", "response": item}
        for item in responses
    ]


def evidence(directory: Path) -> dict:
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
    ).strip()
    cases = router_cases(json.loads((directory / "router.json").read_text()), source)
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
        "runtime": "model-runtime",
        "device": "cpu",
        "platform": "linux/riscv64",
        "execution": {"mode": "qemu-user", "host_platform": actual_platform()},
        "emulator_version": version,
        "binaries": [target_binary(ROOT / ROUTER_BINARY)],
        "cases": cases,
        "expected_cases": [case["id"] for case in cases],
    }
