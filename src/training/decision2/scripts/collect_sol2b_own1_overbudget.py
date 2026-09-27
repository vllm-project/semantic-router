"""Complete a frozen Sol-1 CSS panel after native over-budget failures.

The original native collector aborts on a request longer than its 16,384-token
limit. This helper identifies those requests with the model's own tokenizer and
encoder, reruns only in-budget remaining requests through the *unchanged*
native collector, and stitches full-denominator predictions. It never loads
answers, truncates text, or changes any native probability. This amendment is
limited to the exact already-frozen CSS panel and own-Sol-1 revision.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inference.run import (
    ADAPTER_VERSION,
    DECISION_BACKENDS,
    completed_rows,
    digest,
    file_digest,
    load_prompts,
    local_revision,
    verify_model_family,
)
from scripts.plan_final_eval import sha_file
from scripts.preflight_sol2b_v3 import OWN_REVISION, PANELS

SCHEMA = "decision2-sol2b-own1-css-overbudget-amendment/1"
MAX_LENGTH = 16384
OVER_LENGTH = re.compile(
    r"[^:]+: (\d+) tokens exceeds max_length=16384; no truncation allowed\Z"
)


def _exclusive(path: Path, payload: bytes) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ValueError("Output must be a new absolute file")
    parent = path.parent.resolve(strict=True)
    if stat.S_IMODE(parent.stat().st_mode) != 0o700:
        raise ValueError("Output parent must be private mode 0700")
    with os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600),
        "wb",
    ) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def _module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load pinned native source {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(
    prompt: Path, own1: Path, prefix: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    if sha_file(prompt) != PANELS["css"][2]:
        raise ValueError("Frozen CSS prompt bytes changed")
    if not local_revision(own1, OWN_REVISION):
        raise ValueError("Own Sol source revision unattested")
    verify_model_family(own1, "sol")
    rows = load_prompts(prompt)
    if len(rows) != PANELS["css"][0] or any(
        set(row["questions"]) != {"label"} for row in rows
    ):
        raise ValueError("Expected exact CSS evaluation shape")
    previous = [
        json.loads(line) for line in prefix.read_text(encoding="utf-8").splitlines()
    ]
    if not 0 < len(previous) < len(rows):
        raise ValueError("Expected a nonempty, incomplete native prefix")
    config_sha = file_digest(own1 / "bundle-manifest.json")
    validated = completed_rows(
        prefix,
        rows,
        "sol",
        OWN_REVISION,
        config_sha,
        DECISION_BACKENDS["sol"][0],
        True,
    )
    if len(validated) != len(previous) or [row["id"] for row in previous] != [
        row["id"] for row in rows[: len(previous)]
    ]:
        raise ValueError("Native prefix must match a contiguous frozen CSS prefix")
    return rows, previous, config_sha


def plan(
    prompt: Path, own1: Path, prefix: Path, filtered: Path, receipt: Path
) -> dict[str, Any]:
    rows, previous, config_sha = _inputs(prompt, own1, prefix)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(own1), local_files_only=True)
    api = _module(own1 / "code/decision_api.py", "sol1_frozen_decision_api")
    native = _module(own1 / "code/decision_model.py", "sol1_frozen_decision_model")
    over_budget = []
    valid = []
    for row in rows[len(previous) :]:
        question = api.question_row(row["state"], "label", row["questions"]["label"])
        try:
            encoded = native.encode(question, tokenizer, MAX_LENGTH)
        except ValueError as exc:
            match = OVER_LENGTH.fullmatch(str(exc))
            if match is None:
                raise
            over_budget.append(
                {
                    "id": row["id"],
                    "input_tokens": int(match.group(1)),
                    "source_input_sha256": digest(
                        {"state": row["state"], "questions": row["questions"]}
                    ),
                }
            )
        else:
            if len(encoded["ids"]) > MAX_LENGTH:
                raise AssertionError("Native encoder returned an over-budget row")
            valid.append(row)
    if not over_budget or not valid:
        raise ValueError("Expected at least one over-budget and one valid suffix item")
    filtered_bytes = b"".join(_json_bytes(row) for row in valid)
    _exclusive(filtered, filtered_bytes)
    value = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "status": "gold_free_overbudget_plan; no_labels_or_scores_read",
        "native_revision": OWN_REVISION,
        "native_adapter_version": ADAPTER_VERSION,
        "native_config_sha256": config_sha,
        "native_api_sha256": sha_file(own1 / "code/decision_api.py"),
        "native_encoder_sha256": sha_file(own1 / "code/decision_model.py"),
        "prompt_sha256": PANELS["css"][2],
        "prefix_sha256": sha_file(prefix),
        "prefix_items": len(previous),
        "filtered_sha256": hashlib.sha256(filtered_bytes).hexdigest(),
        "filtered_items": len(valid),
        "over_budget": over_budget,
        "over_budget_items": len(over_budget),
        "total_items": len(rows),
    }
    _exclusive(receipt, _json_bytes(value))
    return {
        "prefix_items": len(previous),
        "filtered_items": len(valid),
        "over_budget_items": len(over_budget),
        "plan_sha256": sha_file(receipt),
    }


def stitch(
    prompt: Path,
    own1: Path,
    prefix: Path,
    filtered: Path,
    suffix: Path,
    plan_path: Path,
    output: Path,
    receipt: Path,
) -> dict[str, Any]:
    rows, previous, config_sha = _inputs(prompt, own1, prefix)
    plan_value = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan_value.get("schema_version") != SCHEMA
        or plan_value.get("prefix_sha256") != sha_file(prefix)
        or plan_value.get("prompt_sha256") != PANELS["css"][2]
        or plan_value.get("native_config_sha256") != config_sha
        or plan_value.get("filtered_sha256") != sha_file(filtered)
        or plan_value.get("total_items") != len(rows)
        or plan_value.get("prefix_items") != len(previous)
    ):
        raise ValueError("Gold-free continuation plan changed")
    filtered_rows = load_prompts(filtered)
    suffix_rows = [
        json.loads(line) for line in suffix.read_text(encoding="utf-8").splitlines()
    ]
    if len(filtered_rows) != plan_value["filtered_items"] or len(suffix_rows) != len(
        filtered_rows
    ):
        raise ValueError("Filtered native suffix incomplete")
    suffix_validated = completed_rows(
        suffix,
        filtered_rows,
        "sol",
        OWN_REVISION,
        config_sha,
        DECISION_BACKENDS["sol"][0],
        True,
    )
    if len(suffix_validated) != len(filtered_rows) or [
        row["id"] for row in suffix_rows
    ] != [row["id"] for row in filtered_rows]:
        raise ValueError("Filtered native suffix order differs")
    over_budget = {row["id"]: row for row in plan_value["over_budget"]}
    native_suffix = {row["id"]: row for row in suffix_rows}
    if len(over_budget) != plan_value["over_budget_items"] or set(over_budget) & set(
        native_suffix
    ):
        raise ValueError("Over-budget and valid suffix IDs overlap")
    if set(over_budget) | set(native_suffix) != {
        row["id"] for row in rows[len(previous) :]
    }:
        raise ValueError("Continuation does not cover all frozen CSS rows")
    first = previous[0]
    combined = list(previous)
    for row in rows[len(previous) :]:
        if row["id"] in native_suffix:
            combined.append(native_suffix[row["id"]])
            continue
        refused = over_budget[row["id"]]
        expected_digest = digest({"state": row["state"], "questions": row["questions"]})
        if (
            refused["source_input_sha256"] != expected_digest
            or refused["input_tokens"] <= MAX_LENGTH
        ):
            raise ValueError("Over-budget refusal does not match frozen prompt")
        invalid = {
            "id": row["id"],
            "answers": {"label": None},
            "latency_ms": 0.0,
            "usage": {"input_tokens": refused["input_tokens"], "scored_questions": 0},
            "model": first["model"],
            "backend": "sol",
            "model_id": DECISION_BACKENDS["sol"][0],
            "adapter_version": ADAPTER_VERSION,
            "model_revision": OWN_REVISION,
            "revision_attested": True,
            "model_config_sha256": config_sha,
            "source_input_sha256": expected_digest,
            "runtime_matches_validated": first["runtime_matches_validated"],
            "runtime_differences": first["runtime_differences"],
            "invalid_reason": "native_max_length_exceeded_no_truncation",
        }
        combined.append(invalid)
    if len(combined) != len(rows) or [item["id"] for item in combined] != [
        row["id"] for row in rows
    ]:
        raise ValueError("Full CSS output order differs")
    payload = b"".join(_json_bytes(row) for row in combined)
    _exclusive(output, payload)
    value = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "status": "full_denominator_native_output; overbudget_counted_invalid",
        "label_exposure": "none_in_this_amendment",
        "plan_sha256": sha_file(plan_path),
        "prompt_sha256": PANELS["css"][2],
        "prefix_sha256": sha_file(prefix),
        "suffix_sha256": sha_file(suffix),
        "full_predictions_sha256": hashlib.sha256(payload).hexdigest(),
        "prefix_items": len(previous),
        "native_suffix_items": len(suffix_rows),
        "over_budget_items": len(over_budget),
        "total_items": len(combined),
    }
    _exclusive(receipt, _json_bytes(value))
    return {
        "total_items": len(combined),
        "over_budget_items": len(over_budget),
        "full_predictions_sha256": sha_file(output),
        "receipt_sha256": sha_file(receipt),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "stitch"):
        sub = commands.add_parser(name)
        sub.add_argument("--prompt", type=Path, required=True)
        sub.add_argument("--own1", type=Path, required=True)
        sub.add_argument("--prefix", type=Path, required=True)
        sub.add_argument("--filtered", type=Path, required=True)
        if name == "plan":
            sub.add_argument("--plan-receipt", type=Path, required=True)
        else:
            sub.add_argument("--suffix", type=Path, required=True)
            sub.add_argument("--plan-receipt", type=Path, required=True)
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = (
        plan(args.prompt, args.own1, args.prefix, args.filtered, args.plan_receipt)
        if args.cmd == "plan"
        else stitch(
            args.prompt,
            args.own1,
            args.prefix,
            args.filtered,
            args.suffix,
            args.plan_receipt,
            args.output,
            args.receipt,
        )
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
