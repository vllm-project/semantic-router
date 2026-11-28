"""C3: per-model profiles grouped by the router's own labels or lanes.

Two grouping modes share this module:

- **label mode** (legacy domain-template path): tasks are rewritten so
  ``category`` is the router's own domain label from ``observe-labels``
  (``--labels``). Labels observed on fewer than ``profile.min_label_cases``
  cases merge into ``other`` and the merge is recorded in the profile.
  Cases without an observed label join ``other`` too and are counted
  (``uncovered``).
- **lane mode** (template path, sub-issue-8 §2.2 step 4): ``category`` is
  the template lane observed by ``observe-lanes`` (``--lanes``). Rare lanes
  are never merged - lanes observed on fewer than
  ``bootstrap.min_lane_cases`` cases are recorded in the profile as
  *unmeasured* and the bindings generator binds them by the default rule
  and marks them untested. Cases without an observed lane get the
  ``unobserved`` category and are counted (``unobserved_cases``).

Verified facts this module encodes (plan doc §3.2, checked 2026-11-01
against src/training/model_eval sources):

- result_to_config.py ``collect_model_accuracies`` (L214-241) derives a
  model name from the *basename* of each analysis.json parent dir:
  ``_cot`` is stripped when present, otherwise ``_direct``, then the
  first ``_`` becomes ``/``. ``auto`` is skipped (L249). recipe-lab
  therefore writes each model's results into ``<base>_direct`` where
  ``base`` is the alias with its single ``/`` replaced by ``_``; aliases
  the convention cannot round-trip (two slashes, an underscore without a
  slash, or a `_cot`/`_direct` collision) are refused with exit 2.
- multi_benchmark_eval.py takes ``--split`` tasks JSONL (``task_id`` /
  ``prompt_user``; ``category`` is carried through to the records
  verbatim), ``--out`` records JSONL, ``--endpoint`` (OpenAI v1 base),
  ``--model`` served name and ``--params`` generation JSON. It resumes
  by default (records with ``error: null`` count as done) and restarts
  with ``--force``. Record fields used here: ``prompt_tokens``,
  ``completion_tokens``, ``finish_reason``, ``latency_s``.
- mcq_judge.py takes ``--records`` + ``--tasks`` and writes
  ``judged.jsonl`` (``is_correct``, ``no_answer``, ``finish_reason``,
  ``completion_tokens``, ``latency_s``), ``summary.json`` and - with
  ``--analysis`` - the result_to_config-format ``analysis.json``
  (``category_accuracy``). An unextractable letter is a wrong answer,
  never unevaluated.
- cost_aware_calib.py ``--token-costs-only --token-costs-dir D`` writes
  one percent-encoded ``{D}/{model}.json`` per model, mapping category
  -> avg tokens; result_to_config.py decodes it back (``unquote``), so
  the alias used here must be the same one bootstrap (C4) ranks with.
  §3.2-5: without that dir result_to_config only prints a Warning and
  silently treats ``--cost-lambda`` as 0 - the C4 caller must reject
  that warning, which is why the profile always produces the dir.
- The scripts are invoked, never copied: each one runs as a subprocess
  (``<python> <model_eval_dir>/<script> ...``) and the profile provenance
  records the script path, its content sha256, its git blob hash
  (``git hash-object``, available without a repo checkout) and the full
  command line of every invocation.

Labels come from C2 (``observe-labels``): tasks are rewritten so
``category`` is the router's own label - never the dataset's original
category, which the router's domain classifier cannot emit (§3.2-4).
Labels observed on fewer than ``profile.min_label_cases`` cases merge
into ``other`` and the merge is recorded in the profile. Cases without
an observed label join ``other`` too and are counted (``uncovered``).

Anti-leak: only ``calib`` outputs feed bootstrap (C4/C5); ``dev``
outputs only feed the answer bank (C6). ``frozen_test`` is refused.

Quality gate: per model and split, truncation (``finish_reason ==
"length"``) and no-answer rates are checked against the contract's
``evidence_quality`` ceilings (M3 defaults 2%); exceeding either fails
with exit 2 and a hint to adjust the generation params.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .contract import ContractError, ExperimentSpec, dataset_content_hash
from .model_eval_tasks import (
    load_source_rows,
    rows_to_model_eval_tasks,
    write_tasks,
)
from .provenance import portable_path as _portable_path

PROFILE_SCHEMA = "recipe-lab-profile-v1"
OTHER_LABEL = "other"
TRUNCATION_FINISH = "length"
# Lane mode: cases without an observed lane get this category (a lost
# record is not a lane observation); bindings never use it.
UNOBSERVED_LANE = "unobserved"

MODEL_EVAL_SCRIPTS = (
    "multi_benchmark_eval.py",
    "mcq_judge.py",
    "cost_aware_calib.py",
)

# The Wilson 95% two-sided normal quantile (deterministic, no scipy).
_Z95 = 1.959963984540054


def model_from_results_dirname(dir_name: str) -> str:
    """Replicate result_to_config.collect_model_accuracies naming (§3.2-1)."""
    if "_cot" in dir_name:
        model_name = dir_name.replace("_cot", "")
    else:
        model_name = dir_name.replace("_direct", "")
    return model_name.replace("_", "/", 1) if "_" in model_name else model_name


def results_dirname(alias: str) -> str:
    """The results subdir whose basename round-trips to the alias.

    collect_model_accuracies reads the *basename* of the analysis.json
    parent dir, so the convention must hold on the basename even for a
    slashed alias: one ``/`` becomes the first ``_``; more than one
    cannot round-trip and is refused by the contract already.
    """
    dir_name = f"{alias.replace('/', '_', 1)}_direct"
    if "/" in dir_name or model_from_results_dirname(dir_name) != alias:
        raise ContractError(
            f"model alias {alias!r} cannot round-trip through the result_to_config "
            "dirname convention (must contain exactly one '/' or no '_', and must "
            "not collide with the _cot/_direct suffix stripping)"
        )
    return dir_name


def load_labels(path: str | Path) -> dict[str, str]:
    """Load C2 labels JSONL as case_id -> router_label."""
    labels: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"{path}:{line_no} is not valid JSON: {exc}"
                ) from exc
            case_id = row.get("case_id")
            label = row.get("router_label")
            if not isinstance(case_id, str) or not case_id:
                raise ContractError(f"{path}:{line_no} has no case_id")
            if not isinstance(label, str) or not label:
                raise ContractError(f"{path}:{line_no} has no router_label")
            if case_id in labels:
                raise ContractError(f"{path}:{line_no}: duplicate case_id {case_id!r}")
            labels[case_id] = label
    if not labels:
        raise ContractError(f"{path} contains no labels")
    return labels


def load_lanes(path: str | Path) -> dict[str, str]:
    """Load C2 (lane mode) lanes JSONL as case_id -> lane."""
    lanes: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"{path}:{line_no} is not valid JSON: {exc}"
                ) from exc
            case_id = row.get("case_id")
            lane = row.get("lane")
            if not isinstance(case_id, str) or not case_id:
                raise ContractError(f"{path}:{line_no} has no case_id")
            if not isinstance(lane, str) or not lane:
                raise ContractError(f"{path}:{line_no} has no lane")
            if case_id in lanes:
                raise ContractError(f"{path}:{line_no}: duplicate case_id {case_id!r}")
            lanes[case_id] = lane
    if not lanes:
        raise ContractError(f"{path} contains no lanes")
    return lanes


def rewrite_task_lanes(
    rows: list[dict[str, Any]], lanes: dict[str, str]
) -> tuple[list[dict[str, Any]], int]:
    """Rewrite tasks' ``category`` to the observed template lane.

    Returns ``(rewritten_rows, unobserved_count)``. Cases without an
    observed lane get the ``unobserved`` category and are counted - a lost
    record is never folded into a real lane. Rare lanes are NOT merged:
    the profile records them as unmeasured (see ``run_profile``).
    """
    assigned: dict[str, str] = {}
    unobserved = 0
    for row in rows:
        case_id = row.get("case_id") or row.get("task_id")
        lane = lanes.get(case_id)
        if lane is None:
            unobserved += 1
            lane = UNOBSERVED_LANE
        assigned[case_id] = lane
    out: list[dict[str, Any]] = []
    for row in rows:
        rewritten = dict(row)
        case_id = row.get("case_id") or row.get("task_id")
        rewritten["category"] = assigned[case_id]
        out.append(rewritten)
    return out, unobserved


def rewrite_task_categories(
    rows: list[dict[str, Any]],
    labels: dict[str, str],
    *,
    min_label_cases: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    """Rewrite tasks' ``category`` to the router label; merge rare labels.

    Returns ``(rewritten_rows, merge_record, uncovered_count)``. Labels
    observed on fewer than ``min_label_cases`` cases merge into ``other``
    (the merge record goes into the profile provenance); cases without an
    observed label join ``other`` too and are counted as uncovered.
    """
    assigned: dict[str, str] = {}
    uncovered = 0
    for row in rows:
        case_id = row.get("case_id") or row.get("task_id")
        label = labels.get(case_id)
        if label is None:
            uncovered += 1
            label = OTHER_LABEL
        assigned[case_id] = label
    counts = Counter(assigned.values())
    dropped = {
        label
        for label, n in counts.items()
        if label != OTHER_LABEL and n < min_label_cases
    }
    merge_record: dict[str, Any] = {}
    if dropped:
        merge_record = {
            "min_label_cases": min_label_cases,
            "merged": {label: counts[label] for label in sorted(dropped)},
            "merged_into": OTHER_LABEL,
        }
    out: list[dict[str, Any]] = []
    for row in rows:
        rewritten = dict(row)
        case_id = row.get("case_id") or row.get("task_id")
        label = assigned[case_id]
        rewritten["category"] = OTHER_LABEL if label in dropped else label
        out.append(rewritten)
    return out, merge_record, uncovered


def _portable_commands(commands: dict[str, Any]) -> dict[str, Any]:
    """Render plain paths and cost_aware_calib's --model alias,path pairs."""

    def portable_argv(command: str, argv: list[str]) -> list[str]:
        result = []
        for index, arg in enumerate(argv):
            if command == "token_costs" and index and argv[index - 1] == "--model":
                alias, separator, path = arg.partition(",")
                result.append(alias + separator + _portable_path(path))
            else:
                result.append(_portable_path(str(arg)))
        return result

    return {
        split: {key: portable_argv(key, argv) for key, argv in entries.items()}
        for split, entries in commands.items()
    }


def script_provenance(path: Path) -> dict[str, Any]:
    """Path + content sha256 + git blob hash for one invoked script."""
    data = path.read_bytes()
    entry: dict[str, Any] = {
        "path": _portable_path(str(path)),
        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
    }
    # git hash-object is repo-independent (the blob hash is content-only),
    # but a broken checkout in the cwd must not break it - run from a
    # neutral temp directory.
    try:
        proc = subprocess.run(
            ["git", "hash-object", "--", str(path)],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
            cwd=tempfile.gettempdir(),
        )
        entry["git_blob"] = proc.stdout.strip() if proc.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        entry["git_blob"] = None
    return entry


def wilson_interval(correct: int, n: int, z: float = _Z95) -> list[float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n == 0:
        return [0.0, 0.0]
    p = correct / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def _percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    idx = (len(sorted_values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (idx - lo)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _run(argv: list[str], *, timeout_s: float, what: str) -> None:
    try:
        proc = subprocess.run(
            argv,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise ContractError(
            f"{what} exceeded the trial budget ({timeout_s:.0f}s): {exc}"
        ) from exc
    if proc.returncode != 0:
        tail = (proc.stdout or "")[-4000:]
        raise ContractError(f"{what} failed (exit {proc.returncode}):\n{tail}")


def check_evidence_quality(
    judged_rows: list[dict[str, Any]],
    *,
    max_truncation_pct: float,
    max_no_answer_pct: float,
    where: str,
) -> dict[str, Any]:
    """M3 gate for model_eval evidence (§3.3 hard rule 3).

    Truncation means ``finish_reason == "length"``; no-answer means the
    judge could not extract a letter. Either rate beyond its ceiling
    fails with exit 2 and a generation-params hint.
    """
    n = len(judged_rows)
    truncated = sum(
        1 for row in judged_rows if row.get("finish_reason") == TRUNCATION_FINISH
    )
    no_answer = sum(1 for row in judged_rows if row.get("no_answer"))
    truncation_rate = truncated / n if n else 0.0
    no_answer_rate = no_answer / n if n else 0.0
    if truncation_rate * 100 > max_truncation_pct:
        raise ContractError(
            f"{where}: truncation rate {truncation_rate:.1%} exceeds the "
            f"{max_truncation_pct:g}% ceiling ({truncated}/{n} cases). Adjust the "
            "generation params - e.g. raise generation.max_tokens or disable "
            "thinking via generation.chat_template_kwargs - and rerun."
        )
    if no_answer_rate * 100 > max_no_answer_pct:
        raise ContractError(
            f"{where}: no-answer rate {no_answer_rate:.1%} exceeds the "
            f"{max_no_answer_pct:g}% ceiling ({no_answer}/{n} cases). Adjust the "
            "generation params (e.g. the answer-protocol prompt or max_tokens) "
            "and rerun."
        )
    return {
        "cases": n,
        "truncation_rate": truncation_rate,
        "no_answer_rate": no_answer_rate,
    }


def _question_cost(
    prompt_tokens: float | None,
    completion_tokens: float | None,
    pricing: dict[str, float] | None,
) -> float | None:
    """Cost of one question from the contract's assumed_pricing rates.

    Only ``assumed_pricing`` rates are available to a direct-model eval
    (provider billing never sees these calls); any cost computed from
    them is annotated "assumed" downstream (M4 rule). No rates -> None,
    never a fake 0.
    """
    if pricing is None or prompt_tokens is None or completion_tokens is None:
        return None
    return (
        prompt_tokens / 1e6 * pricing["prompt_per_1m"]
        + completion_tokens / 1e6 * pricing["completion_per_1m"]
    )


def summarize_judged(
    judged_rows: list[dict[str, Any]],
    records_by_task: dict[str, dict[str, Any]],
    *,
    pricing: dict[str, float] | None,
) -> dict[str, Any]:
    """Per-label and total stats for one model on one split.

    Label grouping uses the rewritten ``category`` (the router label).
    Latency percentiles are per-question wall time from the records.
    """
    labels: dict[str, list[dict[str, Any]]] = {}
    for row in judged_rows:
        labels.setdefault(row.get("category") or OTHER_LABEL, []).append(row)

    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        correct = sum(1 for row in rows if row.get("is_correct"))
        prompts: list[float] = []
        completions: list[float] = []
        latencies: list[float] = []
        costs: list[float] = []
        for row in rows:
            record = records_by_task.get(row.get("task_id"), {})
            for key, sink in (
                ("prompt_tokens", prompts),
                ("completion_tokens", completions),
                ("latency_s", latencies),
            ):
                value = record.get(key)
                if isinstance(value, (int, float)):
                    sink.append(float(value))
            cost = _question_cost(
                (
                    record.get("prompt_tokens")
                    if isinstance(record.get("prompt_tokens"), (int, float))
                    else None
                ),
                (
                    row.get("completion_tokens")
                    if isinstance(row.get("completion_tokens"), (int, float))
                    else None
                ),
                pricing,
            )
            if cost is not None:
                costs.append(cost)
        out: dict[str, Any] = {
            "n": n,
            "accuracy": correct / n if n else 0.0,
            "accuracy_ci95": wilson_interval(correct, n),
            "avg_prompt_tokens": _mean(prompts),
            "avg_completion_tokens": _mean(completions),
            "p50_latency_s": _percentile(sorted(latencies), 0.50),
            "p95_latency_s": _percentile(sorted(latencies), 0.95),
            "cost_per_question": _mean(costs),
        }
        if pricing is not None and out["cost_per_question"] is not None:
            out["cost_basis"] = "assumed_pricing"
        return out

    by_label = {label: block(rows) for label, rows in sorted(labels.items())}
    return {"labels": by_label, "total": block(judged_rows)}


def run_profile(
    *,
    spec: ExperimentSpec,
    cases_paths: dict[str, Path],
    cases_hashes: dict[str, str],
    labels_dir: str | Path | None = None,
    lanes_dir: str | Path | None = None,
    splits: list[str],
    out_dir: str | Path,
    model_eval_dir: str | Path,
    python_bin: str,
    concurrency: int = 8,
    request_timeout: float = 1800.0,
    force: bool = False,
) -> Path:
    """Run the C3 chain and write ``<out_dir>/profile.json`` (byte-stable).

    Exactly one grouping source is required: ``labels_dir`` (legacy label
    mode) or ``lanes_dir`` (template lane mode). Per split: rewrite tasks
    to the grouping labels/lanes, then per model run
    multi_benchmark_eval (direct endpoint, resume-capable) and mcq_judge
    ``--analysis``; per split run cost_aware_calib ``--token-costs-only``
    once for all models. Quality gate after each judge. Lane mode records
    the unmeasured lanes (below ``bootstrap.min_lane_cases``) instead of
    merging them.
    """
    if (labels_dir is None) == (lanes_dir is None):
        raise ContractError(
            "profile needs exactly one grouping source: --labels (legacy "
            "domain-template path) or --lanes (template lane mode)"
        )
    lane_mode = lanes_dir is not None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    grouping_dir = Path(lanes_dir if lane_mode else labels_dir)  # type: ignore[arg-type]
    model_eval_dir = Path(model_eval_dir)
    scripts = {name: model_eval_dir / name for name in MODEL_EVAL_SCRIPTS}
    for path in scripts.values():
        if not path.is_file():
            raise ContractError(f"model_eval script not found: {path}")
    if spec.generation is None:
        raise ContractError(
            "spec declares no generation section: profile needs the shared "
            "generation params (temperature/max_tokens/...)"
        )

    script_prov = {name: script_provenance(path) for name, path in scripts.items()}
    # One params file shared by every model here and the later live arms (C7).
    params_path = out / "generation.json"
    params_path.write_text(
        json.dumps(spec.generation, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )

    models = {model.name: model for model in spec.models}
    aliases = [model.name for model in spec.models]
    timeout_s = spec.budget.max_minutes_per_trial * 60

    label_merges: dict[str, Any] = {}
    uncovered_by_split: dict[str, int] = {}
    unmeasured_by_split: dict[str, dict[str, int]] = {}
    quality: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    commands: dict[str, Any] = {}
    tasks_hashes: dict[str, str] = {}
    source_name = "lanes" if lane_mode else "labels"
    labels_hashes = {
        split: dataset_content_hash(grouping_dir / f"{source_name}-{split}.jsonl")
        for split in splits
    }

    for split in splits:
        rows = load_source_rows(cases_paths[split])
        grouping_path = grouping_dir / f"{source_name}-{split}.jsonl"
        if lane_mode:
            lanes = load_lanes(grouping_path)
            tasks, unobserved = rewrite_task_lanes(rows, lanes)
            uncovered_by_split[split] = unobserved
            counts = Counter(lanes.values())
            unmeasured_by_split[split] = {
                lane: n
                for lane, n in sorted(counts.items())
                if n < spec.bootstrap_config.min_lane_cases
            }
        else:
            labels = load_labels(grouping_path)
            tasks, merge_record, uncovered = rewrite_task_categories(
                rows, labels, min_label_cases=spec.profile_config.min_label_cases
            )
            if merge_record:
                label_merges[split] = merge_record
            uncovered_by_split[split] = uncovered
        # The rewrite works on case rows; the direct-model eval consumes
        # model_eval task rows (task_id/prompt_user/answer). Split manifests
        # may already hold mirrored task lines (from `cases to-model-eval`),
        # so the conversion is idempotent per row.
        tasks = [
            (
                row
                if isinstance(row, dict) and "task_id" in row and "prompt_user" in row
                else rows_to_model_eval_tasks([row])[0]
            )
            for row in tasks
        ]
        tasks_path = write_tasks(tasks, out / split / "tasks.jsonl")
        tasks_hashes[split] = dataset_content_hash(tasks_path)

        split_commands: dict[str, Any] = {}
        split_stats: dict[str, Any] = {}
        split_quality: dict[str, Any] = {}
        judged_paths: list[tuple[str, Path]] = []
        for alias in aliases:
            model = models[alias]
            rdir = out / split / "results" / results_dirname(alias)
            rdir.mkdir(parents=True, exist_ok=True)
            records_path = rdir / "records.jsonl"
            judged_path = rdir / "judged.jsonl"
            gen_argv = [
                str(python_bin),
                str(scripts["multi_benchmark_eval.py"]),
                "--split",
                str(tasks_path),
                "--out",
                str(records_path),
                "--endpoint",
                model.endpoint,
                "--model",
                model.served_name,
                "--params",
                str(params_path),
                "--concurrency",
                str(concurrency),
                "--request-timeout",
                str(request_timeout),
            ]
            if force:
                gen_argv.append("--force")
            _run(
                gen_argv,
                timeout_s=timeout_s,
                what=f"multi_benchmark_eval for {alias} on {split}",
            )
            judge_argv = [
                str(python_bin),
                str(scripts["mcq_judge.py"]),
                "--records",
                str(records_path),
                "--tasks",
                str(tasks_path),
                "--out",
                str(judged_path),
                "--summary",
                str(rdir / "summary.json"),
                "--analysis",
                str(rdir / "analysis.json"),
            ]
            _run(
                judge_argv,
                timeout_s=timeout_s,
                what=f"mcq_judge for {alias} on {split}",
            )
            split_commands[f"{alias}/generate"] = gen_argv
            split_commands[f"{alias}/judge"] = judge_argv
            judged_paths.append((alias, judged_path))

            judged_rows = _read_jsonl(judged_path)
            split_quality[alias] = check_evidence_quality(
                judged_rows,
                max_truncation_pct=spec.evidence_quality.max_truncation_pct,
                max_no_answer_pct=spec.evidence_quality.max_no_answer_pct,
                where=f"profile {alias} on {split}",
            )
            records_by_task = {
                row.get("task_id"): row for row in _read_jsonl(records_path)
            }
            pricing = None
            if spec.cost_basis is not None:
                pricing = (spec.cost_basis.assumed_pricing or {}).get(alias)
            split_stats[alias] = summarize_judged(
                judged_rows, records_by_task, pricing=pricing
            )

        # One token-costs invocation per split, all models together (C4's
        # --cost-lambda input; filenames percent-encode the alias).
        cost_argv = [str(python_bin), str(scripts["cost_aware_calib.py"])]
        for alias, judged_path in judged_paths:
            cost_argv += ["--model", f"{alias},{judged_path}"]
        cost_argv += [
            "--token-costs-dir",
            str(out / split / "token_costs"),
            "--token-costs-only",
        ]
        _run(
            cost_argv,
            timeout_s=timeout_s,
            what=f"cost_aware_calib token costs for {split}",
        )
        split_commands["token_costs"] = cost_argv

        commands[split] = split_commands
        stats[split] = split_stats
        quality[split] = split_quality

    profile = {
        "schema": PROFILE_SCHEMA,
        "grouping": "lanes" if lane_mode else "labels",
        "splits": list(splits),
        "models": aliases,
        "label_merges": label_merges,
        "uncovered_cases": uncovered_by_split,
        "unmeasured_lanes": unmeasured_by_split if lane_mode else {},
        "quality": quality,
        "model_split_stats": stats,
        "inputs": {
            "cases_sha256": dict(cases_hashes),
            "labels_sha256": labels_hashes,
            "tasks_sha256": tasks_hashes,
            "generation": dict(spec.generation),
            "generation_sha256": dataset_content_hash(params_path),
            "scripts": script_prov,
            "commands": _portable_commands(commands),
        },
    }
    profile_path = out / "profile.json"
    profile_path.write_text(
        json.dumps(
            profile, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    return profile_path


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"{path}:{line_no} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ContractError(f"{path}:{line_no} must be a JSON object")
            rows.append(row)
    return rows
