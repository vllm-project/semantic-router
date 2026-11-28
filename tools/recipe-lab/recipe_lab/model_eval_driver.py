"""model_eval driver for the live `verify` tier (C7).

When a workload declares `driver: model_eval`, verify stops using sr-bench
and drives every arm itself with the same C3 scripts the profile uses:

- the baseline and candidate arms each hit one ROUTER endpoint (one router
  process per config; the model name is the router entrypoint, default
  `vllm-sr/auto`);
- the S/W reference arms hit the declared models' DIRECT endpoints from
  `spec.models`;
- `mcq_judge.py` turns every arm's records into per-case correctness, and
  that correctness feeds the same machinery the sr-bench driver uses: M2
  paired statistics (via `verdict.compute_paired_stats`, the sr-bench
  compare field shape), M3 evidence quality (the same rate keys and
  ceilings), M4 reference arms with the same run-cache rule (keyed by the
  rewritten tasks hash + alias instead of dataset + manifest hash), and
  the shared three-state verdict.

Y3 task binding: the arms answer the contract-declared `model_eval_tasks`
rendering of the split (the caller hash-checks `--tasks` against the
contract), and the router-label category rewrite happens inside the
driver from the C2 labels file - an externally rewritten task file can
never enter the evidence path.

AB/BA alternation follows `abba.schedule` exactly like the sr-bench
driver. One deliberate difference: an arm whose generation or judging
subprocess fails is an operational error (ContractError -> exit 2), not
statistical incompleteness - the scripts resume completed records, so a
crashed arm is rerun, not reasoned around. An arm that completes but
produces no judged rows IS incompleteness (the pair is skipped and the
verdict turns INCONCLUSIVE).

Accuracy is the planned-denominator rate (correct / tasks count), matching
sr-bench: an unanswered or missing case counts as incorrect, never as
absent evidence in the score. The paired delta, by contrast, is computed
over the cases both arms judged (the sr-bench compare convention).

Router-arm spend is unknown and stays None; reference arms get an
"assumed" estimate from `cost_basis.assumed_pricing` when the records
carry token counts. The candidate arm's routing distribution comes from
the candidate router's replay store, not from the generation records:
after the arms run the driver fetches the router's replay records via
its management endpoint and joins them by prompt (the verify candidate
arm is the only traffic that sends the task prompts on that router).
The shares give the cost-matched random mix R: the share-weighted
mixture of the reference arms' per-case accuracy and cost, a paired
C-vs-R bootstrap interval where every resample redraws R's assignment
by the shares, a Pareto row, and the `ref:random_mix` constraint bound.
Missing evidence (no management endpoint, no matching replay, a routed
model outside `spec.models`, an unmeasured reference) keeps R
unevaluable with a reason - never guessed.
"""

from __future__ import annotations

import bisect
import json
import random
import statistics
from pathlib import Path
from typing import Any

from . import abba
from .adapters.router_replay import (
    ReplayFetchError,
    fetch_config_hash,
    fetch_replays,
    routing_shares_from_replays,
)
from .contract import ContractError, ExperimentSpec, dataset_content_hash
from .live import (
    _evidence_entry,
    _load_run_cache,
    _merge_paired_stats,
    _save_run_cache,
    _worst_rates,
    resolve_gates,
)
from .model_eval_tasks import write_tasks
from .profile import (
    _percentile,
    _read_jsonl,
    _run,
    load_labels,
    load_lanes,
    rewrite_task_categories,
    rewrite_task_lanes,
    wilson_interval,
)
from .verdict import (
    _BOOTSTRAP_SAMPLES,
    _BOOTSTRAP_SEED,
    PairedStats,
    compute_paired_stats,
    evaluate_verdict,
)

_GEN_SCRIPT = "multi_benchmark_eval.py"
_JUDGE_SCRIPT = "mcq_judge.py"


def _read_optional(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL sidecar that may legitimately be absent (records)."""
    return _read_jsonl(path) if path.is_file() else []


def run_arm(
    *,
    python_bin: str,
    model_eval_dir: str | Path,
    tasks_path: str | Path,
    out_dir: str | Path,
    endpoint: str,
    model: str,
    params_path: str | Path,
    concurrency: int,
    request_timeout: float,
    timeout_s: float,
    force: bool = False,
) -> Path:
    """Run one arm: multi_benchmark_eval + mcq_judge -> judged.jsonl.

    Mirrors the profile's per-model invocation (same argv shape, same
    resume semantics); `--analysis` is skipped - the result_to_config
    analysis format only matters for bootstrap. Returns the judged path.
    """
    model_eval_dir = Path(model_eval_dir)
    scripts = {name: model_eval_dir / name for name in (_GEN_SCRIPT, _JUDGE_SCRIPT)}
    for path in scripts.values():
        if not path.is_file():
            raise ContractError(f"model_eval script not found: {path}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records_path = out / "records.jsonl"
    judged_path = out / "judged.jsonl"
    gen_argv = [
        str(python_bin),
        str(scripts[_GEN_SCRIPT]),
        "--split",
        str(tasks_path),
        "--out",
        str(records_path),
        "--endpoint",
        str(endpoint),
        "--model",
        str(model),
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
        what=f"multi_benchmark_eval for {model} on {endpoint}",
    )
    _run(
        [
            str(python_bin),
            str(scripts[_JUDGE_SCRIPT]),
            "--records",
            str(records_path),
            "--tasks",
            str(tasks_path),
            "--out",
            str(judged_path),
            "--summary",
            str(out / "summary.json"),
        ],
        timeout_s=timeout_s,
        what=f"mcq_judge for {model} on {endpoint}",
    )
    return judged_path


def paired_stats_from_judged(
    base_rows: list[dict[str, Any]],
    cand_rows: list[dict[str, Any]],
) -> PairedStats | None:
    """M2 paired statistics from two arms' judged rows.

    Rows join on `task_id`; the diff per case is candidate-correct minus
    baseline-correct (0/1 booleans). One implicit group (plain mean) - the
    mixed task set has no declared benchmark weights, and the profile's
    total accuracy is the same unweighted mean. Zero common cases yield
    None (the verdict then says why instead of inventing a delta).
    """
    base_by_task = {row.get("task_id"): row for row in base_rows if row.get("task_id")}
    diffs: list[int] = []
    for row in cand_rows:
        task_id = row.get("task_id")
        base = base_by_task.get(task_id)
        if base is None:
            continue
        cand_ok = bool(row.get("is_correct"))
        base_ok = bool(base.get("is_correct"))
        diffs.append(int(cand_ok) - int(base_ok))
    if not diffs:
        return None
    return compute_paired_stats(diffs)


def arm_quality_rates(
    judged_rows: list[dict[str, Any]],
    planned_cases: int,
) -> dict[str, float | None] | None:
    """M3 rates for one arm from its judged rows (same keys as sr-bench).

    Truncation and no-answer come from the judge's per-case fields
    (`finish_reason == "length"`, `no_answer`); each question is one
    request, so the call-level finish-reason share equals the truncation
    rate. The missing-result rate is the planned-denominator gap
    (missing cases were never judged). No rows and a positive plan means
    the whole arm is missing (rate 1.0); a non-positive plan yields None.
    """
    if planned_cases <= 0:
        return None
    n = len(judged_rows)
    if n:
        truncated = sum(
            1 for row in judged_rows if row.get("finish_reason") == "length"
        )
        no_answer = sum(1 for row in judged_rows if row.get("no_answer"))
        truncation_rate: float | None = truncated / n
        no_answer_rate: float | None = no_answer / n
    else:
        truncation_rate = None
        no_answer_rate = None
    return {
        "truncation_rate": truncation_rate,
        "missing_result_rate": max(0, planned_cases - n) / planned_cases,
        "no_answer_rate": no_answer_rate,
        # reference only (never gated), same as the sr-bench call-level share
        "finish_reason_length_rate": truncation_rate,
    }


def _arm_accuracy(judged_rows: list[dict[str, Any]], planned: int) -> float:
    """Planned-denominator accuracy: missing and unanswered count as wrong."""
    correct = sum(1 for row in judged_rows if row.get("is_correct"))
    return correct / planned if planned else 0.0


def _arm_token_costs(
    records_rows: list[dict[str, Any]], pricing: dict[str, float] | None
) -> list[float] | None:
    """Per-request assumed costs from one arm's records' token counts.

    Only `assumed_pricing` rates are available here (provider billing never
    sees these calls); rows without numeric token counts are skipped. None
    means pricing itself is undeclared - unknown, never zero.
    """
    if pricing is None:
        return None
    costs: list[float] = []
    for row in records_rows:
        prompt = row.get("prompt_tokens")
        completion = row.get("completion_tokens")
        if not isinstance(prompt, (int, float)) or isinstance(prompt, bool):
            continue
        if not isinstance(completion, (int, float)) or isinstance(completion, bool):
            continue
        costs.append(
            prompt / 1e6 * pricing["prompt_per_1m"]
            + completion / 1e6 * pricing["completion_per_1m"]
        )
    return costs


def _arm_cost_usd(
    records_rows: list[dict[str, Any]], pricing: dict[str, float] | None
) -> float | None:
    """Total assumed spend of one direct arm from its records' token counts.

    No rates or unknown token counts -> None.
    """
    costs = _arm_token_costs(records_rows, pricing)
    return sum(costs) if costs else None


def _arm_mean_case_cost(
    records_rows: list[dict[str, Any]], pricing: dict[str, float] | None
) -> float | None:
    """Mean per-request assumed cost of one direct arm (None without prices)."""
    costs = _arm_token_costs(records_rows, pricing)
    return sum(costs) / len(costs) if costs else None


def random_mix_paired_stats(
    cand_rows: list[dict[str, Any]],
    ref_rows_by_alias: dict[str, list[dict[str, Any]]],
    shares: dict[str, float],
    *,
    alpha: float = 0.05,
    bootstrap_samples: int = _BOOTSTRAP_SAMPLES,
    seed: int = _BOOTSTRAP_SEED,
) -> dict[str, Any] | None:
    """C8: paired statistics of the candidate arm against the random mix R.

    R answers task i with model m drawn with probability p_m (the
    candidate arm's routing shares over `spec.models` aliases), so its
    per-task correctness is a random variable: every bootstrap resample
    redraws R's assignment as well as the task indices, and the interval
    is the bootstrap quantiles with the same fixed seed and quantile
    indexing as `compute_paired_stats`. The point delta is analytic - the
    candidate accuracy minus the share-weighted reference accuracies over
    the paired cases. Cases pair only when the candidate row and every
    weighted reference arm judged them; an empty pairing yields None.
    """
    cand_by_task = {
        row.get("task_id"): bool(row.get("is_correct"))
        for row in cand_rows
        if row.get("task_id")
    }
    ref_by_task = {
        alias: {
            row.get("task_id"): bool(row.get("is_correct"))
            for row in rows
            if row.get("task_id")
        }
        for alias, rows in ref_rows_by_alias.items()
    }
    aliases = sorted(a for a, p in shares.items() if p > 0)
    if not aliases or not cand_by_task:
        return None
    paired = [
        task_id
        for task_id in sorted(cand_by_task)
        if all(task_id in ref_by_task[a] for a in aliases)
    ]
    if not paired:
        return None
    weight_sum = sum(shares[a] for a in aliases)
    probs = {a: shares[a] / weight_sum for a in aliases}
    cand_ok = [1.0 if cand_by_task[task_id] else 0.0 for task_id in paired]
    ref_ok = {
        a: [1.0 if ref_by_task[a][task_id] else 0.0 for task_id in paired]
        for a in aliases
    }
    delta = statistics.mean(cand_ok) - sum(
        probs[a] * statistics.mean(ref_ok[a]) for a in aliases
    )
    # Joint bootstrap: resample task indices with replacement and redraw
    # R's per-task assignment by the shares on every resample.
    cumulative: list[float] = []
    total = 0.0
    for a in aliases:
        total += probs[a]
        cumulative.append(total)
    rng = random.Random(seed)
    n = len(paired)
    samples: list[float] = []
    for _ in range(bootstrap_samples):
        idx = rng.choices(range(n), k=n)
        c_sum = 0.0
        r_sum = 0.0
        for i in idx:
            c_sum += cand_ok[i]
            draw = rng.random() * total
            alias = aliases[bisect.bisect_left(cumulative, draw)]
            r_sum += ref_ok[alias][i]
        samples.append(c_sum / n - r_sum / n)
    samples.sort()
    low_i = max(0, int(0.025 * bootstrap_samples) - 1)
    high_i = min(bootstrap_samples - 1, int(0.975 * bootstrap_samples) - 1)
    return {
        "paired_cases": n,
        "delta": delta,
        # the C8 validation check: the resampled mean converges to the
        # analytic delta (report cross-check, P3 evidence)
        "bootstrap_mean": statistics.mean(samples),
        "bootstrap_ci95": (samples[low_i], samples[high_i]),
    }


def _random_mix_evidence(
    *,
    spec: ExperimentSpec,
    planned: int,
    cand_judged_rows: list[dict[str, Any]],
    cand_replay_shares: dict[str, float] | None,
    cand_replay_note: str | None,
    ref_rows_by_role: dict[str, list[dict[str, Any]]],
    ref_case_cost_by_role: dict[str, float | None],
) -> dict[str, Any]:
    """C8: the cost-matched random mix R for the candidate arm.

    Shares come from the candidate router's replay records (joined by
    prompt by the caller), mapped to `spec.models` aliases via the
    models' names or served names; expected accuracy and cost mix the
    reference arms' per-case values by those shares. Every gap - no
    replay evidence, a routed model outside `spec.models`, an unmeasured
    needed reference - keeps R unevaluable with a reason; missing
    evidence is never counted as zero.
    """
    evidence: dict[str, Any] = {
        "evaluable": False,
        "reason": None,
        "shares": {},
        "expected_accuracy": None,
        "expected_cost": None,
        "cost_note": None,
        "paired": None,
    }
    if spec.references is None:
        evidence["reason"] = "the spec declares no references section"
        return evidence
    if cand_replay_shares is None:
        evidence["reason"] = "candidate routing distribution unknown: " + (
            cand_replay_note or "no replay evidence"
        )
        return evidence
    # a replay's selected model is the routed model: the spec's model
    # name, or the backend's served name - either form maps to the alias,
    # and anything else stays unknown
    model_to_alias = {model.name: model.name for model in spec.models}
    model_to_alias.update({model.served_name: model.name for model in spec.models})
    shares: dict[str, float] = {}
    unknown: list[str] = []
    for routed, share in cand_replay_shares.items():
        alias = model_to_alias.get(routed)
        if alias is None:
            unknown.append(routed)
        else:
            shares[alias] = shares.get(alias, 0.0) + share
    if unknown:
        evidence["reason"] = (
            "candidate arm routed to models outside spec.models: "
            + ", ".join(sorted(unknown))
        )
        return evidence
    ref_rows_by_alias: dict[str, list[dict[str, Any]]] = {}
    case_cost_by_alias: dict[str, float | None] = {}
    missing: list[str] = []
    for role in ("strong", "weak"):
        alias = getattr(spec.references, role)
        rows = ref_rows_by_role.get(role)
        if rows:
            ref_rows_by_alias[alias] = rows
            case_cost_by_alias[alias] = ref_case_cost_by_role.get(role)
        else:
            missing.append(f"{role} ({alias})")
    needed = sorted(a for a, p in shares.items() if p > 0)
    unmeasured = [a for a in needed if a not in ref_rows_by_alias]
    if missing or unmeasured:
        gaps = sorted(set(missing) | set(unmeasured))
        evidence["reason"] = (
            "reference arms missing or unmeasured for the random mix: "
            + ", ".join(gaps)
        )
        return evidence
    case_costs = {a: case_cost_by_alias.get(a) for a in needed}
    expected_case_cost: float | None = None
    if all(cost is not None for cost in case_costs.values()):
        expected_case_cost = sum(shares[a] * case_costs[a] for a in needed)
    evidence.update(
        {
            "evaluable": True,
            "shares": dict(sorted(shares.items())),
            "expected_accuracy": sum(
                shares[a] * _arm_accuracy(ref_rows_by_alias[a], planned) for a in needed
            ),
            "expected_cost": (
                expected_case_cost * planned if expected_case_cost is not None else None
            ),
            "cost_note": "assumed" if expected_case_cost is not None else None,
        }
    )
    evidence["paired"] = random_mix_paired_stats(
        cand_judged_rows, ref_rows_by_alias, shares
    )
    return evidence


def _latency_percentiles(
    records_rows: list[dict[str, Any]],
) -> tuple[float | None, float | None]:
    values = sorted(
        float(row["latency_s"])
        for row in records_rows
        if isinstance(row.get("latency_s"), (int, float))
        and not isinstance(row.get("latency_s"), bool)
    )
    return _percentile(values, 0.50), _percentile(values, 0.95)


def _pareto_row(
    *,
    arm: str,
    arm_id: str,
    role: str | None,
    judged_rows: list[dict[str, Any]],
    records_rows: list[dict[str, Any]],
    planned: int,
    cost_usd: float | None,
    cost_note: str | None,
) -> dict[str, Any]:
    correct = sum(1 for row in judged_rows if row.get("is_correct"))
    p50, p95 = _latency_percentiles(records_rows)
    return {
        "arm": arm,
        "id": arm_id,
        "role": role,
        "accuracy": _arm_accuracy(judged_rows, planned),
        "accuracy_ci95": wilson_interval(correct, planned),
        "cost_usd": cost_usd,
        "cost_note": cost_note,
        "latency_p50_s": p50,
        "latency_p95_s": p95,
    }


def verify_live_model_eval(
    *,
    spec: ExperimentSpec,
    tasks_path: str | Path,
    labels_path: str | Path | None,
    candidate_config: str | Path,
    baseline_endpoint: str,
    candidate_endpoint: str,
    management_endpoint: str | None = None,
    router_model: str,
    repetitions: int,
    measurements: list[dict[str, Any]],
    model_eval_dir: str | Path,
    python_bin: str,
    work_dir: str | Path,
    concurrency: int = 8,
    request_timeout: float = 1800.0,
    force: bool = False,
    run_cache: str | Path | None = None,
    incomplete_reasons: list[str] | None = None,
    lanes_path: str | Path | None = None,
) -> dict[str, Any]:
    """Drive the model_eval harness for one candidate and decide the verdict.

    `repetitions` AB/BA pairs (abba.schedule): the baseline arm hits
    `baseline_endpoint`, the candidate arm `candidate_endpoint` - both
    routers serving their respective configs on the same frozen task set
    (`tasks_path`, the contract-declared `model_eval_tasks` rendering of
    the split, hash-checked against the contract by the caller). Y3: the
    grouping rewrite the profile used to do externally now happens here -
    `labels_path` (legacy label mode) or `lanes_path` (template lane mode)
    is the C2 JSONL for the split; every arm answers the rewritten file
    written into `work_dir`, and no externally rewritten task file can
    enter the evidence path. Both rewrite inputs' hashes are recorded with
    the result.

    Per pair, the judged rows feed the M2 paired statistics, the M3
    evidence-quality rates and the per-arm accuracies; the merged stats
    and shared `evaluate_verdict` decide.

    M4: `spec.references` arms run once each on the model's direct
    endpoint (`spec.models` entry), cached by (rewritten tasks hash,
    alias) in `run_cache`; their values anchor `ref:`-bound constraints.

    C8: the cost-matched random mix R mixes the reference arms' per-case
    accuracy and cost by the candidate arm's routing distribution. The
    distribution is read from the candidate router's replay store
    (`management_endpoint`, the router management base URL) and joined by
    prompt - the candidate arm is the only traffic that sends the task
    prompts on that router. It anchors `ref:random_mix` constraint bounds
    and the Pareto row; without matching replays R stays unevaluable with
    a reason. (The endpoint itself is mandatory: the Y4 config binding
    refuses to run the driver without it.)

    The result dict mirrors the sr-bench driver's shape (verdict/reasons/
    metrics/gates/constraints/bench_refs/trials/abba_summary/paired/
    evidence_quality/references/random_mix/pareto) plus `driver`,
    `tasks_hash` (the contract-declared rendering's hash),
    `labeled_tasks_sha256`/`labels_sha256` (the in-driver rewrite's
    inputs and output), `label_merges`, `uncovered_cases`,
    `config_sha256` (the candidate config the router's management API
    acknowledged as active - Y4), `planned_cases` and `distribution`
    (the alias-keyed routing shares when R is evaluable); `preview` and
    `baseline_run_id` are None - there is no sr-bench run behind this
    evidence.
    """
    if repetitions < 1:
        raise ContractError("verify needs at least one repetition")
    tasks = Path(tasks_path)
    if not tasks.is_file():
        raise ContractError(f"model_eval tasks file not found: {tasks}")
    task_rows = _read_jsonl(tasks)
    planned = len(task_rows)
    if planned == 0:
        raise ContractError(f"model_eval tasks file is empty: {tasks}")
    if spec.generation is None:
        raise ContractError(
            "the model_eval driver needs spec.generation: the shared "
            "generation params every arm is sent"
        )
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    params_path = work_dir / "generation.json"
    params_path.write_text(
        json.dumps(spec.generation, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    timeout_s = spec.budget.max_minutes_per_trial * 60
    tasks_hash = dataset_content_hash(tasks)
    # Y3: rewrite the bound tasks' categories to the router's own grouping
    # (the C2 labels file, legacy mode, or the C2 lanes file, template lane
    # mode), exactly like the profile does - but here, so the arms never
    # answer an externally rewritten file. Rare labels merge into `other`
    # under spec.profile_config.min_label_cases; uncovered cases join `other`
    # and are counted. Lane mode never merges: lanes below
    # bootstrap.min_lane_cases stay in the tasks and the profile records
    # them unmeasured; unobserved cases get the `unobserved` category.
    if (labels_path is None) == (lanes_path is None):
        raise ContractError(
            "exactly one grouping source is required: labels_path (legacy) "
            "or lanes_path (template lane mode)"
        )
    lane_mode = lanes_path is not None
    grouping_file = Path(lanes_path if lane_mode else labels_path)
    if not grouping_file.is_file():
        raise ContractError(
            f"{'lanes' if lane_mode else 'labels'} file not found: {grouping_file}"
        )
    label_merges: dict[str, Any] = {}
    if lane_mode:
        lanes = load_lanes(grouping_file)
        labeled_rows, uncovered = rewrite_task_lanes(task_rows, lanes)
    else:
        labels = load_labels(grouping_file)
        labeled_rows, label_merges, uncovered = rewrite_task_categories(
            task_rows, labels, min_label_cases=spec.profile_config.min_label_cases
        )
    labeled_tasks_path = write_tasks(labeled_rows, work_dir / "tasks-labeled.jsonl")
    labeled_tasks_hash = dataset_content_hash(labeled_tasks_path)
    labels_hash = dataset_content_hash(grouping_file)
    # Y4: the candidate arm's verdict must name the config it ran. The
    # candidate config is bound two ways before any arm is dispatched: the
    # local file's sha256 is recorded, and the router's management API
    # (GET /api/v1/config/hash) must acknowledge an ACTIVE config whose
    # source hash is that same file - a router still serving an older
    # document fails here instead of grading the wrong recipe.
    config_file = Path(candidate_config)
    if not config_file.is_file():
        raise ContractError(f"candidate config file not found: {config_file}")
    config_sha256 = dataset_content_hash(config_file)
    if management_endpoint is None:
        raise ContractError(
            "--candidate-management-endpoint is required: the driver must "
            "check GET /api/v1/config/hash to confirm the candidate router "
            "has loaded --candidate-config"
        )
    try:
        activation = fetch_config_hash(management_endpoint)
    except ReplayFetchError as exc:
        raise ContractError(
            f"cannot check the candidate router's config: {exc}"
        ) from exc
    status = activation.get("activation_status")
    if status != "active":
        raise ContractError(
            f"the candidate router's config activation is {status!r}, not "
            "active; load the candidate config before verifying"
        )
    served = activation.get("source_config_hash")
    if not isinstance(served, str) or "sha256:" + served != config_sha256:
        raise ContractError(
            "the candidate router serves config "
            f"{('sha256:' + served) if isinstance(served, str) else '<unknown>'}, "
            f"but --candidate-config hashes to {config_sha256}"
        )
    metric_key = spec.objective.metric_key

    # AB/BA pairs: both arms live, alternating execution order per repetition.
    trials: dict[int, dict[str, dict[str, Any]]] = {}
    for label, rep in abba.schedule(repetitions):
        endpoint = baseline_endpoint if label == abba.BASELINE else candidate_endpoint
        arm_dir = work_dir / f"{label}-rep{rep}"
        try:
            judged = run_arm(
                python_bin=python_bin,
                model_eval_dir=model_eval_dir,
                tasks_path=labeled_tasks_path,
                out_dir=arm_dir,
                endpoint=endpoint,
                model=router_model,
                params_path=params_path,
                concurrency=concurrency,
                request_timeout=request_timeout,
                timeout_s=timeout_s,
                force=force,
            )
        except ContractError as exc:
            raise ContractError(f"{label} arm (rep {rep}): {exc}") from exc
        trials.setdefault(rep, {})[label] = {
            "judged": str(judged),
            "endpoint": endpoint,
            "status": "completed",
        }

    # per-pair statistics + evidence quality
    pair_stats: list[Any] = []
    pair_evidence: list[
        tuple[dict[str, float | None] | None, dict[str, float | None] | None]
    ] = []
    rows: list[dict[str, Any]] = []
    judged_by_arm: dict[tuple[int, str], list[dict[str, Any]]] = {}
    incomplete = list(incomplete_reasons or [])
    for rep in sorted(trials):
        base_judged_rows = _read_jsonl(Path(trials[rep][abba.BASELINE]["judged"]))
        cand_judged_rows = _read_jsonl(Path(trials[rep][abba.CANDIDATE]["judged"]))
        if not base_judged_rows or not cand_judged_rows:
            incomplete.append(
                f"pair {rep}: an arm produced no judged results "
                f"(baseline {len(base_judged_rows)}, candidate "
                f"{len(cand_judged_rows)} rows)"
            )
            continue
        judged_by_arm[(rep, abba.BASELINE)] = base_judged_rows
        judged_by_arm[(rep, abba.CANDIDATE)] = cand_judged_rows
        stats = paired_stats_from_judged(base_judged_rows, cand_judged_rows)
        if stats is not None:
            pair_stats.append(stats)
        pair_evidence.append(
            (
                arm_quality_rates(base_judged_rows, planned),
                arm_quality_rates(cand_judged_rows, planned),
            )
        )
        for label, judged_rows in (
            (abba.BASELINE, base_judged_rows),
            (abba.CANDIDATE, cand_judged_rows),
        ):
            rows.append(
                {
                    "label": label,
                    "metrics": {metric_key: _arm_accuracy(judged_rows, planned)},
                }
            )

    summary = abba.paired_summary(rows, metric_key)
    baseline_metric = (
        summary[abba.BASELINE]["median"] if summary[abba.BASELINE] else None
    )
    candidate_metric = (
        summary[abba.CANDIDATE]["median"] if summary[abba.CANDIDATE] else None
    )

    # constraints: the objective metric is measurable per arm; anything else
    # (e.g. replay-tier cost proxies) is not and stays unevaluated.
    constraint_evidence: list[dict[str, Any]] = []
    for constraint in spec.objective.constraints:
        if constraint.metric == metric_key:
            per_arm: dict[str, list[float]] = {abba.BASELINE: [], abba.CANDIDATE: []}
            for (_rep, label), judged_rows in judged_by_arm.items():
                per_arm[label].append(_arm_accuracy(judged_rows, planned))
            baseline_c = (
                statistics.median(per_arm[abba.BASELINE])
                if per_arm[abba.BASELINE]
                else None
            )
            candidate_c = (
                statistics.median(per_arm[abba.CANDIDATE])
                if per_arm[abba.CANDIDATE]
                else None
            )
        else:
            incomplete.append(
                f"constraint {constraint.name}: metric {constraint.metric!r} is "
                "not measurable under the model_eval driver"
            )
            baseline_c = candidate_c = None
        constraint_evidence.append(
            {
                "name": constraint.name,
                "metric": constraint.metric,
                "op": constraint.op,
                "bound": constraint.bound,
                "tolerance_pct": constraint.tolerance_pct,
                "tolerance_pp": constraint.tolerance_pp,
                "factor": constraint.factor,
                "baseline": baseline_c,
                "candidate": candidate_c,
            }
        )

    # M4: reference arms - one direct-model eval per declared alias, cached
    # by (rewritten tasks hash, alias). A failed or missing reference is
    # incomplete evidence, never a zero.
    references_evidence: dict[str, Any] = {}
    reference_values: dict[str, float | None] = {"strong": None, "weak": None}
    ref_pareto: list[dict[str, Any]] = []
    # C8: per-role judged rows and mean per-case costs feed the random mix
    ref_rows_by_role: dict[str, list[dict[str, Any]]] = {}
    ref_case_cost_by_role: dict[str, float | None] = {}
    if spec.references is not None:
        models_by_name = {model.name: model for model in spec.models}
        cache = _load_run_cache(run_cache)
        cache_dirty = False
        pricing_by_alias: dict[str, dict[str, float]] = {}
        if spec.cost_basis is not None:
            pricing_by_alias = dict(spec.cost_basis.assumed_pricing or {})
        for role in ("strong", "weak"):
            alias = getattr(spec.references, role)
            entry: dict[str, Any] = {"alias": alias, "cached": False}
            key = f"{labeled_tasks_hash}:{alias}"
            judged_str = cache.get(key)
            judged_path = Path(judged_str) if judged_str else None
            if judged_path is None or not judged_path.is_file():
                model = models_by_name.get(alias)
                if model is None:
                    incomplete.append(
                        f"reference arm {role!r}: model {alias!r} is not "
                        "declared in spec.models"
                    )
                    references_evidence[role] = entry
                    continue
                ref_dir = work_dir / f"reference-{role}"
                try:
                    judged_path = run_arm(
                        python_bin=python_bin,
                        model_eval_dir=model_eval_dir,
                        tasks_path=labeled_tasks_path,
                        out_dir=ref_dir,
                        endpoint=model.endpoint,
                        model=model.served_name,
                        params_path=params_path,
                        concurrency=concurrency,
                        request_timeout=request_timeout,
                        timeout_s=timeout_s,
                        force=force,
                    )
                except ContractError as exc:
                    incomplete.append(f"reference arm {role!r}: {exc}")
                    references_evidence[role] = entry
                    continue
                cache[key] = str(judged_path)
                cache_dirty = True
            else:
                entry["cached"] = True
            ref_rows = _read_jsonl(judged_path)
            ref_records = _read_optional(judged_path.parent / "records.jsonl")
            ref_rows_by_role[role] = ref_rows
            entry.update(
                {
                    "judged": str(judged_path),
                    "metric": _arm_accuracy(ref_rows, planned),
                }
            )
            pricing = pricing_by_alias.get(alias)
            ref_case_cost_by_role[role] = _arm_mean_case_cost(ref_records, pricing)
            cost = _arm_cost_usd(ref_records, pricing)
            entry.update(
                {
                    "cost_usd": cost,
                    "cost_note": "assumed" if cost is not None else None,
                }
            )
            reference_values[role] = entry["metric"]
            references_evidence[role] = entry
            ref_pareto.append(
                _pareto_row(
                    arm="reference",
                    arm_id=alias,
                    role=role,
                    judged_rows=ref_rows,
                    records_rows=ref_records,
                    planned=planned,
                    cost_usd=cost,
                    cost_note=entry["cost_note"],
                )
            )
        if cache_dirty:
            _save_run_cache(run_cache, cache)

    # M3: conservative merge - each rate takes the worst observed pair.
    evidence = {
        "baseline": _evidence_entry(
            _worst_rates([base for base, _ in pair_evidence]),
            "baseline",
            spec.evidence_quality,
            incomplete,
        ),
        "candidate": _evidence_entry(
            _worst_rates([cand for _, cand in pair_evidence]),
            "candidate",
            spec.evidence_quality,
            incomplete,
        ),
    }

    # Pareto rows: the last completed pair's arms plus the references.
    pareto: list[dict[str, Any]] = []
    last_rep = max(judged_by_arm, key=lambda k: k[0], default=None)
    if last_rep is not None:
        rep = last_rep[0]
        for label in (abba.BASELINE, abba.CANDIDATE):
            judged_rows = judged_by_arm.get((rep, label), [])
            endpoint = trials[rep][label]["endpoint"]
            records_rows = _read_optional(
                Path(trials[rep][label]["judged"]).parent / "records.jsonl"
            )
            pareto.append(
                _pareto_row(
                    arm=label,
                    arm_id=f"router:{endpoint}",
                    role=None,
                    judged_rows=judged_rows,
                    records_rows=records_rows,
                    planned=planned,
                    # the router hides the spend: unknown, never zero
                    cost_usd=None,
                    cost_note=None,
                )
            )
    pareto.extend(ref_pareto)

    # C8: the cost-matched random mix R. Its expected accuracy anchors
    # `ref:random_mix` constraint bounds; a Pareto row and a paired C-vs-R
    # bootstrap interval carry the P3 comparison. The candidate arm's
    # routing shares come from the candidate router's replay store (the
    # generation records carry no responding model), joined by prompt -
    # the candidate arm is the only traffic that sends the task prompts,
    # and failed records are no routing decision. Any gap - a failed
    # fetch, no matching record - keeps R unevaluable with a reason,
    # never guessed. The endpoint itself is mandatory upstream: the Y4
    # config binding already refused to run without it.
    replay_shares: dict[str, float] | None = None
    replay_note: str | None = None
    if spec.references is not None:
        task_prompts = {
            row["prompt_user"]
            for row in task_rows
            if isinstance(row.get("prompt_user"), str) and row["prompt_user"]
        }
        if not task_prompts:
            replay_note = "the tasks file declares no prompt_user values"
        else:
            try:
                replay_records = fetch_replays(management_endpoint)
            except ReplayFetchError as exc:
                replay_note = f"replay fetch failed: {exc}"
            else:
                replay_shares, replay_stats = routing_shares_from_replays(
                    replay_records, task_prompts
                )
                if replay_shares is None:
                    replay_note = (
                        "no replay record with a selected model matches "
                        "the task prompts "
                        f"(matched={replay_stats['matched']} of "
                        f"{replay_stats['total']} records, "
                        f"{replay_stats['failed']} failed, "
                        f"{replay_stats['no_model']} without a model)"
                    )
    random_mix = _random_mix_evidence(
        spec=spec,
        planned=planned,
        cand_judged_rows=(
            judged_by_arm[(last_rep[0], abba.CANDIDATE)] if last_rep is not None else []
        ),
        cand_replay_shares=replay_shares,
        cand_replay_note=replay_note,
        ref_rows_by_role=ref_rows_by_role,
        ref_case_cost_by_role=ref_case_cost_by_role,
    )
    if random_mix["evaluable"]:
        reference_values["random_mix"] = random_mix["expected_accuracy"]
        pareto.append(
            {
                "arm": "reference",
                "id": "ref:random_mix",
                "role": "random_mix",
                "accuracy": random_mix["expected_accuracy"],
                # a share-weighted mixture has no single binomial count:
                # the paired bootstrap interval is the reported interval
                "accuracy_ci95": None,
                "cost_usd": random_mix["expected_cost"],
                "cost_note": random_mix["cost_note"],
                "latency_p50_s": None,
                "latency_p95_s": None,
            }
        )

    guards, undeclared = resolve_gates(spec.gates, measurements)
    stats = _merge_paired_stats(pair_stats) if pair_stats else None
    verdict = evaluate_verdict(
        baseline_metric=baseline_metric,
        candidate_metric=candidate_metric,
        direction=spec.objective.direction,
        min_improvement_pct=spec.objective.min_improvement_pct,
        guards=guards,
        incomplete_reasons=incomplete,
        constraints=constraint_evidence,
        stats=stats,
        min_cases=spec.objective.min_cases,
        min_improvement_pp=spec.objective.min_improvement_pp,
        # This driver measures accuracy. A constraint on another metric must
        # never read the numerical accuracy of a reference as its bound.
        references={
            role: {metric_key: value} for role, value in reference_values.items()
        },
    )
    if undeclared:
        verdict.reasons.append(
            "undeclared guard measurements ignored (not declared in spec.gates): "
            + ", ".join(undeclared)
        )
    bench_refs = [
        f"meval:tasks:{tasks_hash}",
        f"meval:labeled-tasks:{labeled_tasks_hash}",
        f"meval:labels:{labels_hash}",
    ]
    for rep in sorted(trials):
        for label in (abba.BASELINE, abba.CANDIDATE):
            bench_refs.append(f"meval:{label}-rep{rep}")
    for role in ("strong", "weak"):
        if role in references_evidence:
            bench_refs.append(f"meval:reference:{role}")
    if random_mix["evaluable"]:
        bench_refs.append("meval:reference:random_mix")
    result: dict[str, Any] = {
        "verdict": verdict.verdict,
        "reasons": list(verdict.reasons),
        "improvement_pct": verdict.improvement_pct,
        "metrics": {
            "baseline": baseline_metric,
            "candidate": candidate_metric,
        },
        "gates": {
            g.name: (
                "pass" if g.passed else ("incomplete" if not g.evaluated else "fail")
            )
            for g in guards
        },
        "constraints": verdict.constraints,
        "undeclared_measurements": undeclared,
        "bench_refs": bench_refs,
        "baseline_run_id": None,
        "trials": {str(rep): trials[rep] for rep in sorted(trials)},
        "abba_summary": summary,
        "preview": None,
        "paired": stats.to_dict() if stats else None,
        "evidence_quality": evidence,
        "references": references_evidence,
        "random_mix": random_mix,
        "distribution": random_mix["shares"] if random_mix["evaluable"] else None,
        "pareto": pareto,
        "driver": "model_eval",
        "tasks_hash": tasks_hash,
        "labeled_tasks_sha256": labeled_tasks_hash,
        "labels_sha256": labels_hash,
        "label_merges": label_merges,
        "uncovered_cases": uncovered,
        # Y4: the candidate config identity the management API acknowledged.
        "config_sha256": config_sha256,
        "planned_cases": planned,
    }
    if not spec.gates:
        result["gates_note"] = "no guard declared"
    return result
