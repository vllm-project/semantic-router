"""Offline, append-only accounting reconciliation from retained stream evidence."""

from __future__ import annotations

import hashlib
import json

from .contracts import digest
from .store import TERMINAL
from .transport import CallFailure, cost_for, normalize_usage

VERSION = "sr-bench-accounting-v1"
BUCKETS = ("input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens")


def _stream_receipt(path, byte_limit):
    """Read bounded evidence without interpreting generated text as instructions."""
    checksum = hashlib.sha256()
    size = 0
    lines = []
    receipt = {"raw_usage": None, "model": None, "finish_reason": None, "done": 0}

    def consume():
        data = "\n".join(
            line[5:].lstrip() for line in lines if line.startswith("data:")
        )
        lines.clear()
        if not data:
            return
        if data == "[DONE]":
            receipt["done"] += 1
            return
        event = json.loads(data)
        if event.get("error"):
            raise ValueError("Saved stream contains an error event")
        receipt["raw_usage"] = event.get("usage") or receipt["raw_usage"]
        receipt["model"] = event.get("model") or receipt["model"]
        for choice in event.get("choices", []):
            if choice.get("index", 0) == 0:
                receipt["finish_reason"] = (
                    choice.get("finish_reason") or receipt["finish_reason"]
                )

    with path.open("rb") as stream:
        while raw := stream.readline(byte_limit + 1):
            size += len(raw)
            if size > byte_limit:
                raise ValueError("Saved stream exceeds frozen evidence bound")
            checksum.update(raw)
            line = raw.decode("utf-8").rstrip("\r\n")
            if line:
                lines.append(line)
            else:
                consume()
        if lines:
            consume()
    receipt["stream_sha256"] = checksum.hexdigest()
    return receipt


def _target(manifest, call):
    inventory = {
        **manifest.get("auxiliary_targets", {}),
        **{target["id"]: target for target in manifest["targets"]},
    }
    if call["role"] == "subject":
        return inventory[call["target_id"]]
    case = next(c for c in manifest["cases"] if c["id"] == call["case_id"])
    ref = (
        manifest.get("benchmark_options", {})
        .get(case["benchmark"], {})
        .get(call["role"])
    )
    if ref not in inventory:
        raise ValueError("Saved auxiliary target identity is unavailable")
    return inventory[ref]


def _reconcile_call(store, run_id, manifest, call):
    row = {
        "call_id": call["id"],
        "original_receipt_sha256": digest(call),
        "original_usage": call.get("usage"),
        "original_cost_usd": call.get("cost_usd"),
        "usage": None,
        "cost_usd": None,
        "verified": False,
    }
    try:
        path = (
            store.root
            / "runs"
            / run_id
            / digest([call["case_id"], call["target_id"]])[:24]
            / (call["id"] + ".sse")
        )
        receipt = _stream_receipt(
            path, max(manifest["limits"]["max_output_chars"] * 30, 1048576)
        )
        row["stream_sha256"] = receipt["stream_sha256"]
        if (
            call["status"] != "completed"
            or receipt["done"] != 1
            or receipt["finish_reason"]
            not in {"stop", "length", "tool_calls", "function_call"}
            or receipt["finish_reason"] != call.get("finish_reason")
            or receipt["raw_usage"] != call.get("raw_usage")
            or receipt["model"] != call.get("model")
        ):
            raise ValueError("Saved stream and terminal call receipt do not agree")
        target = _target(manifest, call)
        if target["kind"] == "mom" and (
            call.get("inference_call_count") != 1
            or call.get("model_usage")
            or not call.get("selected_model")
            or call["selected_model"] != receipt["model"]
        ):
            raise ValueError("Only a proven direct MoM call can use final-stream usage")
        if target.get("expected_response_model") and (
            target["expected_response_model"] != receipt["model"]
        ):
            raise ValueError("Saved response model differs from the frozen target")
        usage = normalize_usage(receipt["raw_usage"] or {})
        cost = cost_for(usage, receipt["model"], target.get("prices", {}))
        if usage is None or cost is None:
            raise ValueError("Saved usage or frozen model prices are incomplete")
        row.update(usage=usage, cost_usd=cost, verified=True)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, CallFailure) as exc:
        # File paths and provider payloads must never escape through error strings.
        row["unknown_reason"] = (
            str(exc)
            if isinstance(exc, (ValueError, CallFailure))
            and not isinstance(exc, json.JSONDecodeError)
            else "Saved stream evidence could not be verified"
        )
    return row


def reconcile_usage(store, run_id):
    run = store.get(run_id)
    if run["status"] not in TERMINAL or run["manifest"]["mode"] != "live":
        raise ValueError("Usage reconciliation requires a terminal live run")
    calls = store.calls(run_id)
    if not calls:
        raise ValueError("Run has no saved calls to reconcile")
    rows = [_reconcile_call(store, run_id, run["manifest"], call) for call in calls]
    evidence = digest(
        {
            "version": VERSION,
            "plan_sha256": run["manifest"]["plan_sha256"],
            "calls": rows,
        }
    )
    verified = sum(row["verified"] for row in rows)
    artifact = {
        "id": "accounting-" + evidence[:24],
        "version": VERSION,
        "run_id": run_id,
        "evidence_sha256": evidence,
        "plan_sha256": run["manifest"]["plan_sha256"],
        "qualified": verified == len(rows),
        "verified_call_count": verified,
        "unverifiable_call_count": len(rows) - verified,
        "corrected_call_count": sum(
            row["verified"]
            and (
                row["usage"] != row["original_usage"]
                or row["cost_usd"] != row["original_cost_usd"]
            )
            for row in rows
        ),
        "original_known_spend_usd": sum(row["original_cost_usd"] or 0 for row in rows),
        "corrected_known_spend_usd": sum(row["cost_usd"] or 0 for row in rows),
        "original_receipts_preserved": True,
        "model_requests": 0,
        "calls": rows,
    }
    return store.append_accounting_correction(run_id, artifact)


def correction_metadata(store, run_id):
    correction = store.accounting_correction(run_id)
    return (
        {key: value for key, value in correction.items() if key != "calls"}
        if correction
        else None
    )


def effective_calls(store, run_id, summary=False):
    """Return a derived view; original call/detail APIs retain original receipts."""
    calls = store.calls(run_id, summary=summary)
    correction = store.accounting_correction(run_id)
    if not correction:
        return calls
    rows = {row["call_id"]: row for row in correction["calls"]}
    return [
        {
            **call,
            "usage": rows[call["id"]]["usage"],
            "cost_usd": rows[call["id"]]["cost_usd"],
            "cost_complete": rows[call["id"]]["verified"],
            "accounting_correction_id": correction["id"],
        }
        for call in calls
    ]


def cache_neutral_cost(manifest, call):
    """Counterfactual token-equivalent cost; this is never a billing receipt."""
    if call.get("usage") is None:
        return None
    try:
        target = _target(manifest, call)
        parts = call.get("model_usage") or [call]
        total = 0.0
        for part in parts:
            usage = part.get("usage")
            model = part.get("selected_model") or part.get("model")
            price = target.get("prices", {}).get(model)
            if usage is None or price is None:
                return None
            prompt = sum(usage[k] for k in BUCKETS if k != "output_tokens")
            total += (
                prompt * price["input"] + usage["output_tokens"] * price["output"]
            ) / 1_000_000
        return total
    except (KeyError, ValueError):
        return None
