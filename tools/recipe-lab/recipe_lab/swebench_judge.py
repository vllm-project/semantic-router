"""SWE-bench judging for agentic runs (AG2, Z14).

A separate batch step after ``run-agentic``: the driver only produces the
client's patches and journals; correctness is judged here by the SWE-bench
official harness (the environment AG0-W4 verified on the 121 box: docker,
``swebench`` pip package, per-task evaluation images). The judge command is
a fully parameterized template - endpoints, paths and workers are all
explicit, nothing is hard-coded.

Output is sr-bench-shaped per-task results (the engine's completed-case
row: ``case_id`` / binary ``correct`` / 0/1 ``score`` / ``cost_usd`` /
``call_count`` / ``latency_s``), so the M2 paired machinery and the AG7
agentic gates consume a ``judge-agentic`` run the same way they consume an
sr-bench run. A task the report does not name stays ``correct: None`` -
missing evidence is never graded as zero.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .agentic_driver import task_result_rows
from .contract import ContractError

DEFAULT_JUDGE_COMMAND = (
    "{python} -m swebench.harness.run_evaluation"
    " --predictions_path {predictions_path}"
    " --dataset_name {dataset_name}"
    " --run_id {run_id}"
    " --max_workers {max_workers}"
    " --report_dir {log_dir}"
)


class JudgeError(ContractError):
    """The judge command failed or produced an unparsable report."""


def build_predictions(run_dir: str | Path, path: Path) -> int:
    """Write the official harness predictions JSONL from a run's results.

    Returns the row count; tasks without a status (never driven) are not
    predicted - the harness would charge image hours for an empty patch.
    Tasks whose run failed keep their (possibly empty) patch so the harness
    records them as unresolved rather than silently dropping them.
    """
    rows: list[dict[str, str]] = []
    for result_path in sorted((Path(run_dir) / "results").glob("*.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") == "done":
            rows.append(
                {
                    "instance_id": str(result["instance_id"]),
                    # Official harness submission identifier. One run can use
                    # multiple routed models, so label the whole submission.
                    "model_name_or_path": "recipe-lab",
                    "model_patch": str(result.get("model_patch") or ""),
                }
            )
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows)


def parse_report(payload: Any) -> dict[str, dict[str, Any]]:
    """Normalize one official-harness report into per-instance verdicts.

    Accepts the shapes the harness has shipped: ``{id: {"resolved": bool,
    "report": {...}}}`` (current), ``{id: bool}``, ``{"resolved_ids": [...],
    "unresolved_ids": [...]}``, or a list of ``{"instance_id", "resolved"}``.
    Anything else is an error, never a guess.
    """
    verdicts: dict[str, dict[str, Any]] = {}
    if isinstance(payload, dict):
        if isinstance(payload.get("resolved_ids"), list):
            resolved = {str(i) for i in payload["resolved_ids"]}
            unresolved = {str(i) for i in payload.get("unresolved_ids") or [] if i}
            for inst in resolved | unresolved:
                verdicts[inst] = {"resolved": inst in resolved}
            return verdicts
        for inst, entry in payload.items():
            if isinstance(entry, bool):
                verdicts[str(inst)] = {"resolved": entry}
            elif isinstance(entry, dict) and isinstance(entry.get("resolved"), bool):
                verdicts[str(inst)] = {"resolved": entry["resolved"], "report": entry}
    elif isinstance(payload, list):
        for entry in payload:
            if (
                isinstance(entry, dict)
                and isinstance(entry.get("resolved"), bool)
                and entry.get("instance_id")
            ):
                verdicts[str(entry["instance_id"])] = {"resolved": entry["resolved"]}
    if not verdicts:
        raise JudgeError(
            f"report is not a recognized harness report shape: {str(payload)[:400]}"
        )
    return verdicts


def locate_report(
    log_dir: Path, explicit: str | None, run_id: str, *, official: bool
) -> Path:
    """Prefer the harness aggregate; keep report.json for custom commands."""
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise JudgeError(f"--report-path {path} does not exist")
        return path
    candidates = [
        path for path in log_dir.glob("*.json") if path.name.endswith(f".{run_id}.json")
    ]
    if not candidates and not official:
        candidates = sorted(log_dir.rglob("report.json"))
    if not candidates:
        raise JudgeError(f"no judge report for {run_id!r} found under {log_dir}")
    if len(candidates) > 1:
        raise JudgeError(
            f"{len(candidates)} judge reports under {log_dir}; pass --report-path"
        )
    return candidates[0]


def judge_run(args: Any) -> int:
    """The ``judge-agentic`` command body."""
    run_dir = Path(args.run_dir).resolve()
    results_dir = run_dir / "results"
    if not results_dir.is_dir() or not any(results_dir.glob("*.json")):
        raise JudgeError(f"{run_dir} has no results/ - run-agentic first")
    log_dir = run_dir / "judge"
    log_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = (
        Path(args.predictions_path).resolve()
        if args.predictions_path
        else log_dir / "predictions.jsonl"
    )
    if args.predictions_path is None or args.rebuild_predictions:
        count = build_predictions(run_dir, predictions_path)
        print(f"predictions: {count} tasks -> {predictions_path}", flush=True)

    official = args.judge_command == DEFAULT_JUDGE_COMMAND
    # The official harness reuses per-instance outcomes from cwd/logs on
    # repeat run IDs, even if the submitted patch changed. Keep each call's
    # execution logs and aggregate together and independent of older calls.
    judge_dir = (
        Path(tempfile.mkdtemp(prefix="evaluation-", dir=log_dir))
        if official
        else log_dir
    )
    dataset = Path(args.dataset_name)
    fields = {
        "python": sys.executable,
        "predictions_path": str(predictions_path),
        "dataset_name": (
            str(dataset.resolve())
            if official and dataset.exists()
            else args.dataset_name
        ),
        "run_id": args.run_id,
        "max_workers": args.max_workers,
        "log_dir": str(judge_dir),
        "report_path": str(judge_dir / "report.json"),
    }
    # Substitute after tokenization so a path with whitespace stays one argument.
    tokens = [
        token.strip('"').format(**fields)
        for token in shlex.split(args.judge_command, posix=os.name != "nt")
    ]
    print(f"judge: {shlex.join(tokens)}", flush=True)
    proc = subprocess.run(
        tokens,
        cwd=judge_dir if official else None,
        capture_output=True,
        text=True,
        timeout=args.judge_timeout_s,
        check=False,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise JudgeError(f"judge command returned {proc.returncode}: {tail}")

    report_path = locate_report(
        judge_dir, args.report_path, args.run_id, official=official
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    verdicts = parse_report(payload)

    rows = task_result_rows(run_dir)
    judged = 0
    for row in rows:
        verdict = verdicts.get(str(row["case_id"]))
        if verdict is None:
            row["details"]["judge"] = "missing_from_report"
            continue
        resolved = verdict["resolved"]
        row["correct"] = resolved
        row["score"] = 1.0 if resolved else 0.0
        row["details"]["judge"] = verdict.get("report", {"resolved": resolved})
        judged += 1

    results_path = run_dir / "judge-results.jsonl"
    results_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    resolved_count = sum(1 for row in rows if row["correct"] is True)
    summary = {
        "run_id": args.run_id,
        "dataset_name": args.dataset_name,
        "report_path": str(report_path),
        "total": len(rows),
        "judged": judged,
        "unjudged": len(rows) - judged,
        "resolved": resolved_count,
        "accuracy": resolved_count / len(rows) if rows else None,
    }
    (run_dir / "judge-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(
        f"JUDGE-DONE resolved={resolved_count}/{len(rows)} "
        f"unjudged={summary['unjudged']} -> {results_path}",
        flush=True,
    )
    return 0
