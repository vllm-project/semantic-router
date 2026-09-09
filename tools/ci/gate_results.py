#!/usr/bin/env python3
"""Require every selected CI job to succeed; only unselected jobs may skip."""

from __future__ import annotations

import json
import os
import sys
from typing import Any


def validate_results(
    results: dict[str, Any],
    signals: dict[str, str],
    *,
    active: bool = True,
    lifecycle: str = "pr",
) -> list[str]:
    errors: list[str] = []
    if results.get("changes", {}).get("result") != "success":
        return ["changes must succeed before its selection can be trusted"]
    ignored = {"full", "docs_only", "website", "helm", "domains"}
    if lifecycle == "main":
        ignored.add("images")  # Main builds/publishes images after this gate.
    for signal, value in signals.items():
        if signal in ignored or value != "true" or not active:
            continue
        job = (
            ("core" if lifecycle == "main" else "core-tests")
            if signal == "core_test"
            else signal.replace("_", "-")
        )
        if job not in results:
            errors.append(f"{job}: selected but absent from gate results")
    if "quality" not in results:
        errors.append("quality is absent from gate results")
    for job, data in results.items():
        if job == "changes":
            continue
        signal = "core_test" if job in ("core", "core-tests") else job.replace("-", "_")
        if job == "quality":
            selected = active
        else:
            value = signals.get(signal)
            if value not in ("true", "false"):
                errors.append(f"{job}: missing or invalid selection output {signal!r}")
                continue
            selected = active and value == "true"
        status = data.get("result")
        if selected and status != "success":
            errors.append(f"{job}: selected but {status or 'missing'}")
        elif not selected and status not in ("success", "skipped"):
            errors.append(f"{job}: unexpected {status or 'missing'}")
    return errors


def main() -> int:
    results = json.loads(os.environ["DOMAIN_RESULTS"])
    signals = results.get("changes", {}).get("outputs", {})
    errors = validate_results(
        results,
        signals,
        active=os.environ.get("ACTIVE", "true") == "true",
        lifecycle=os.environ.get("LIFECYCLE", "pr"),
    )
    summary = "\n".join(
        f"- {job}: {data.get('result', 'missing')}" for job, data in results.items()
    )
    print(summary)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(summary + "\n")
    for error in errors:
        print(f"::error::{error}", file=sys.stderr)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
