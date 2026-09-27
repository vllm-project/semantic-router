"""Record one native Nox CSS over-budget failure before resuming the same run.

This is a pre-key execution repair. It never reads gold labels. It preserves
the native collector's successful prefix and inserts a deliberately invalid
answer for the exact request that made the pinned native loader raise its
documented no-truncation error. All other errors remain fatal. The receipt
and original failure log must be retained alongside the final predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inference.run import completed_rows, digest

PLAN_VERSION = "decision2-first-release-v3-plan/2"
ERROR = re.compile(
    r"^ValueError: label: (?P<tokens>\d+) tokens exceeds "
    r"max_length=(?P<limit>\d+); no truncation allowed$"
)


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def recover(
    plan_path: Path,
    expected_plan_sha: str,
    failure_log: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    if sha(plan_path) != expected_plan_sha:
        raise ValueError("frozen plan digest changed")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("plan_version") != PLAN_VERSION:
        raise ValueError("unknown frozen v3 plan")
    model = next(item for item in plan["inference"] if item["key"] == "nox")
    if model["group"] != "decision1" or model["revision"] is None:
        raise ValueError("Nox baseline identity is not fixed")
    prompts = Path(plan["css_prompts"]["path"])
    if sha(prompts) != plan["css_prompts"]["sha256"]:
        raise ValueError("gold-free CSS prompts changed")
    predictions = Path(model["paths"]["css"])
    if not predictions.is_file() or predictions.is_symlink():
        raise ValueError("native partial prediction file is absent")
    all_prompts = rows(prompts)
    prior = rows(predictions)
    if not prior or len(prior) >= len(all_prompts):
        raise ValueError("native prefix is empty or complete")
    if [row["id"] for row in prior] != [row["id"] for row in all_prompts[: len(prior)]]:
        raise ValueError("native predictions are not an exact prompt prefix")
    template = prior[-1]
    fixed = {
        "adapter_version": template.get("adapter_version"),
        "backend": template.get("backend"),
        "model": template.get("model"),
        "model_config_sha256": template.get("model_config_sha256"),
        "model_id": template.get("model_id"),
        "model_revision": template.get("model_revision"),
        "revision_attested": template.get("revision_attested"),
        "runtime_matches_validated": template.get("runtime_matches_validated"),
    }
    if (
        fixed["backend"] != "nox"
        or fixed["model_id"] != model["model_id"]
        or fixed["model_revision"] != model["revision"]
        or fixed["revision_attested"] is not True
        or fixed["runtime_matches_validated"] is not True
        or not isinstance(fixed["model_config_sha256"], str)
    ):
        raise ValueError("native prefix is not the attested Nox package")
    completed = completed_rows(
        predictions,
        all_prompts,
        "nox",
        model["revision"],
        fixed["model_config_sha256"],
        model["model_id"],
        True,
    )
    if completed != {row["id"] for row in prior}:
        raise ValueError("native prefix has stale rows")
    tail = failure_log.read_text(encoding="utf-8").splitlines()
    match = ERROR.fullmatch(tail[-1]) if tail else None
    if match is None:
        raise ValueError("failure is not the exact native over-budget exception")
    observed, limit = int(match["tokens"]), int(match["limit"])
    if observed <= limit or limit != 16384:
        raise ValueError("native over-budget numbers disagree")
    failed = all_prompts[len(prior)]
    if set(failed) != {"id", "state", "questions"} or set(failed["questions"]) != {
        "label"
    }:
        raise ValueError("next CSS request is not a single label decision")
    old_sha = sha(predictions)
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    intent_path = receipt_path.with_name(receipt_path.name + ".intent")
    if intent_path.exists():
        raise FileExistsError(intent_path)
    answer = {
        **template,
        "id": failed["id"],
        "answers": {"label": {"type": "choice", "error": "input_over_budget"}},
        "latency_ms": None,
        "usage": None,
        "source_input_sha256": digest(
            {"state": failed["state"], "questions": failed["questions"]}
        ),
        "invalid_reason": "native_input_over_budget_no_truncation",
        "observed_tokens": observed,
        "max_length": limit,
    }
    encoded = json.dumps(answer, ensure_ascii=False, separators=(",", ":")) + "\n"
    intent = {
        "schema_version": "decision2-v3-nox-overbudget-recovery-intent/1",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": expected_plan_sha,
        "failure_log_sha256": sha(failure_log),
        "prior_predictions_sha256": old_sha,
        "appended_row_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "failed_prompt_id": failed["id"],
    }
    with intent_path.open("x", encoding="utf-8") as output:
        os.chmod(intent_path, 0o600)
        json.dump(intent, output, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    with predictions.open("a", encoding="utf-8") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    result = {
        "schema_version": "decision2-v3-nox-overbudget-recovery/1",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": expected_plan_sha,
        "prompts_sha256": sha(prompts),
        "failure_log_sha256": sha(failure_log),
        "intent_sha256": sha(intent_path),
        "script_sha256": sha(Path(__file__)),
        "model_id": model["model_id"],
        "model_revision": model["revision"],
        "failed_prompt_id": failed["id"],
        "failed_prompt_index": len(prior),
        "observed_tokens": observed,
        "max_length": limit,
        "prior_predictions_sha256": old_sha,
        "recovered_predictions_sha256": sha(predictions),
        "appended_row_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "policy": "invalid_answer_counts_as_failure; native run resumes without truncation",
    }
    with receipt_path.open("x", encoding="utf-8") as output:
        json.dump(result, output, sort_keys=True, indent=2)
        output.write("\n")
    os.chmod(receipt_path, 0o600)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--failure-log", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = recover(args.plan, args.plan_sha256, args.failure_log, args.receipt)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
