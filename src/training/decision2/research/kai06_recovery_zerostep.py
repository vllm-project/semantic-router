"""Verify own-Kai2 package versus its selected checkpoint before recovery training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from inference.kai_continuation import verify_export
from inference.kai_lex import MODELS, verify_native_bundle
from inference.run import file_digest

NATIVE_SHA = "03d90e449078bacd18390ad52c3ba3af7cff86bb507c2c2f9088f0260305d1ab"
SELECT_SHA = "429285511fe5737dabfc799f01e7715155ad0f913994d5d10f5fdb627f6b0976"


def compare(reference: list[dict], actual: list[dict]) -> dict:
    if len(reference) != 700 or len(actual) != len(reference):
        raise ValueError("SELECT prediction cardinality differs")
    max_drift = 0.0
    changed = 0
    for before, after in zip(reference, actual):
        for key in ("id", "question_id", "type", "candidate_ids", "input_tokens"):
            if before[key] != after[key]:
                raise ValueError(f"Native prediction identity changed: {key}")
        if max(
            range(len(before["probabilities"])), key=before["probabilities"].__getitem__
        ) != max(
            range(len(after["probabilities"])), key=after["probabilities"].__getitem__
        ):
            changed += 1
        for key in ("logits", "probabilities"):
            if len(before[key]) != len(after[key]):
                raise ValueError(f"Native prediction shape changed: {key}")
            for a, b in zip(before[key], after[key]):
                if not math.isfinite(a) or not math.isfinite(b):
                    raise ValueError("Nonfinite prediction")
                max_drift = max(max_drift, abs(a - b))
        for key in ("confidence", "probability"):
            a, b = before[key], after[key]
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError("Nonfinite prediction")
            max_drift = max(max_drift, abs(a - b))
    return {
        "items": len(reference),
        "argmax_changes": changed,
        "max_abs_drift": max_drift,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "run", "native", "select", "source_predictions", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if (
        file_digest(args.select) != SELECT_SHA
        or file_digest(args.native / "MANIFEST.json") != NATIVE_SHA
    ):
        raise ValueError("Frozen Kai2 export or SELECT changed")
    identity = verify_export(args.run, args.native, NATIVE_SHA)
    verify_native_bundle(args.base, "kai", MODELS["kai"]["revision"])
    complete = json.loads((args.run / "COMPLETE.json").read_text(encoding="utf-8"))
    if (
        complete.get("selected", {}).get("step") != 98
        or complete.get("identity", {}).get("dev_sha256") != SELECT_SHA
    ):
        raise ValueError("Kai2 selected checkpoint or SELECT identity differs")
    sys.path.insert(0, str(args.base))
    import torch
    import decision_runtime as api

    if torch.version.hip is None or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one isolated AMD GPU required")
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.use_deterministic_algorithms(True)
    rows = [json.loads(line) for line in args.select.open(encoding="utf-8")]
    source = [
        json.loads(line) for line in args.source_predictions.open(encoding="utf-8")
    ]
    if len(rows) != 700 or len(source) != 700:
        raise ValueError("Frozen SELECT or source predictions have wrong cardinality")
    gold_free = [
        {
            key: value
            for key, value in row.items()
            if key not in ("target", "hard_target_id")
        }
        for row in rows
    ]
    native = api.load_native(
        args.native, expected_manifest_sha256=NATIVE_SHA, device="cuda:0"
    )
    native.collator = type(native.collator)(
        native.collator.tokenizer, max_length=1024, state_truncation="error"
    )
    package = api.predict(native, gold_free, batch_size=8)
    source_comparison = compare(source, package)
    api.configure_training(native, max_input_tokens=1024)
    api.restore_native_policy_for_export(native)
    restored = api.predict(native, gold_free, batch_size=8)
    configured_comparison = compare(package, restored)
    passed = all(
        result["argmax_changes"] == 0 and result["max_abs_drift"] <= 1e-6
        for result in (source_comparison, configured_comparison)
    )
    report = {
        "schema": "decision2-kai06-recovery-zerostep/1",
        "status": "PASS" if passed else "HOLD_drift",
        "native_manifest_sha256": NATIVE_SHA,
        "native_lineage": identity,
        "select_sha256": SELECT_SHA,
        "source_predictions_sha256": file_digest(args.source_predictions),
        "source_checkpoint_vs_export": source_comparison,
        "export_vs_configured_restored": configured_comparison,
        "gold_or_scores_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.chmod(0o600)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "source_checkpoint_vs_export",
                    "export_vs_configured_restored",
                )
            },
            sort_keys=True,
        )
    )
    if not passed:
        raise ValueError("Kai2 zero-step package parity failed")


if __name__ == "__main__":
    main()
