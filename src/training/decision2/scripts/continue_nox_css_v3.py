"""Resume a frozen Nox CSS run across exact native over-budget failures.

Every exception is retained. Only the signed one-row invalid-answer recovery
may unblock the next native continuation. This never opens CSS gold or scores.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.recover_nox_overbudget_v3 import ERROR, recover, sha
from scripts.run_nox_resume_v3 import run


def continue_css(
    plan: Path,
    plan_sha: str,
    first_failure_log: Path,
    log_root: Path,
    max_recoveries: int,
) -> dict:
    if not 1 <= max_recoveries <= 100:
        raise ValueError("invalid recovery ceiling")
    pending = first_failure_log
    events = []
    for _ in range(max_recoveries):
        tail = pending.read_text(encoding="utf-8").splitlines()
        if not tail or ERROR.fullmatch(tail[-1]) is None:
            raise ValueError("native failure is not an exact over-budget event")
        frozen = json.loads(plan.read_text(encoding="utf-8"))
        model = next(item for item in frozen["inference"] if item["key"] == "nox")
        predictions = Path(model["paths"]["css"])
        with predictions.open(encoding="utf-8") as stream:
            failed_index = sum(1 for _ in stream) + 1
        recovery_path = log_root / f"nox.css.overbudget-{failed_index}.receipt.json"
        recovery = recover(plan, plan_sha, pending, recovery_path)
        process_log = log_root / f"nox.css.resume-{failed_index}.process.log"
        execution_receipt = log_root / f"nox.css.resume-{failed_index}.receipt.json"
        outcome = run(
            plan,
            plan_sha,
            "css",
            recovery_path,
            process_log,
            execution_receipt,
        )
        events.append(
            {
                "failed_prompt_index": failed_index,
                "recovery_receipt_sha256": sha(recovery_path),
                "recovery_intent_sha256": sha(
                    recovery_path.with_name(recovery_path.name + ".intent")
                ),
                "failed_prompt_id": recovery["failed_prompt_id"],
                "native_continuation_receipt_sha256": sha(execution_receipt),
                "native_exit_code": outcome["exit_code"],
            }
        )
        if outcome["exit_code"] == 0:
            return {
                "status": "css_complete",
                "plan_sha256": plan_sha,
                "events": events,
                "predictions_sha256": sha(predictions),
            }
        pending = process_log
    raise RuntimeError("native over-budget recovery ceiling reached")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--first-failure-log", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--max-recoveries", type=int, default=40)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.summary.exists():
        raise FileExistsError(args.summary)
    result = continue_css(
        args.plan,
        args.plan_sha256,
        args.first_failure_log,
        args.log_root,
        args.max_recoveries,
    )
    with args.summary.open("x", encoding="utf-8") as output:
        os.chmod(args.summary, 0o600)
        json.dump(result, output, sort_keys=True, indent=2)
        output.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
