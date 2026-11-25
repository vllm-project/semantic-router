"""recipe-lab CLI.

Subcommands:
  plan                    load and validate an experiment spec, print it normalized
  hash                    compute the frozen dataset identity: sha256 of raw bytes (A4)
  split                   split a labeled case list into calib/dev/frozen_test (B)
  attribute               E1/E2/E3 misroute attribution over labeled records
  collect-replays         pull router_replay records from the management API
  diagnose                coverage / distribution / reachability over unlabeled rows (D)
  bank                    build or apply the per-case/per-model answer bank
  propose-thresholds      perturb one decision-signal threshold
  propose-rule-thresholds move every rule-level predicate site of one signal (C)
  propose-calibrated      rule threshold candidates at observed quantiles (D)
  run                     evaluate candidate configs in the replay tier -> ledger
  ledger                  inspect or append to the evidence ledger (append/show/top)
  live-verdict            compare two sr-bench runs -> PASS/FAIL/INCONCLUSIVE
                          (guard floors from spec.gates (A3), --guard-measurements
                          carry measurements only; --insight-refs (G2); economic
                          constraints from objective.constraints (E); frozen_test
                          only with --final)
  verify                  drive sr-bench end to end for one candidate (I):
                          preview routing + repetitions of AB/BA pairs (fresh
                          baseline manifest run vs fresh candidate-plan) -> per-pair
                          compare -> medians/ranges -> PASS/FAIL/INCONCLUSIVE
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

from . import __version__
from .adapters.router_replay import (
    fetch_replays,
    records_to_cases,
    records_to_rows,
    write_cases,
)
from .attribution import (
    attribute,
    load_records,
    render_markdown,
    summarize,
    write_jsonl,
)
from .bank import apply_bank, build_bank, load_bank, write_bank
from .contract import (
    ContractError,
    ExperimentSpec,
    Workload,
    check_runnable_split,
    dataset_content_hash,
)
from .insights import merge_insight_refs, parse_insight_refs, resolve_insight_refs
from .layout import select_recipe
from .ledger import Ledger, LedgerRecord
from .live import live_verdict_from_runs, measurements_from_rows, verify_live
from .proposers import rule_threshold, threshold_search
from .runner import (
    COST_PROXY_METRIC,
    HotReloadSession,
    RunnerEnv,
    RunnerError,
    RunnerTimeoutError,
    attribution_delta,
    evaluate_candidate,
    replay_constraint_evidence,
    replay_metric_value,
    routed_cost_index,
)
from .splitting import (
    default_split_rule,
    load_cases,
    parse_ratios,
    split_cases,
    write_splits,
)
from .srbench import SrbenchCli, cases_to_srbench_rows
from .srbench import evaluate as srbench_evaluate


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _add_recipe_selectors(parser: argparse.ArgumentParser) -> None:
    """Name the recipe to edit when the base config declares several."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--recipe",
        default=None,
        help="recipe to edit (recipes[].name); defaults to the only recipe",
    )
    group.add_argument(
        "--public-model",
        default=None,
        help="public model name; the recipe is the one its entrypoint serves",
    )


def _fail(message: str) -> None:
    """Print the error and exit with the contract-error code (2)."""
    print(message, file=sys.stderr)
    raise SystemExit(2)


def _load_spec(path: str) -> ExperimentSpec:
    try:
        return ExperimentSpec.from_yaml(path)
    except ContractError as exc:
        _fail(f"contract error: {exc}")
    except OSError as exc:
        _fail(f"cannot read spec: {exc}")


def _select_split(
    spec: ExperimentSpec, split_name: str, workload: Workload | None = None
) -> str:
    """B: resolve --split against the workload; frozen_test is quarantined and
    legacy single-dataset workloads only accept 'dev'. M5: `workload` is the
    selected Workload (default: the spec's first)."""
    workload = workload if workload is not None else spec.workload
    try:
        check_runnable_split(split_name)
        workload.split_hash(split_name)
    except ContractError as exc:
        raise ContractError(str(exc)) from exc
    if workload.legacy:
        print(
            "warning: workload uses the legacy single-dataset form; treated as split 'dev' only; "
            "migrate to workload.splits for calib/dev/frozen_test separation",
            file=sys.stderr,
        )
    return split_name


def cmd_plan(args: argparse.Namespace) -> int:
    spec = _load_spec(args.spec)
    _print_json(spec.to_dict())
    return 0


def cmd_hash(args: argparse.Namespace) -> int:
    """Compute the frozen dataset identity of a JSONL file (A4)."""
    try:
        content_hash = dataset_content_hash(args.file)
    except OSError as exc:
        raise ContractError(f"cannot read dataset file: {exc}") from exc
    _print_json({"file": str(args.file), "content_hash": content_hash})
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    """B: split a labeled case list into calib/dev/frozen_test deterministically."""
    ratios = parse_ratios(args.ratios)
    try:
        cases = load_cases(args.cases)
    except OSError as exc:
        raise ContractError(f"cannot read cases: {exc}") from exc
    split_map = split_cases(cases, ratios, args.group_key, args.seed)
    splits = write_splits(split_map, args.output_dir)
    split_rule = args.split_rule or default_split_rule(
        args.group_key, args.seed, ratios
    )
    _print_json({"split_rule": split_rule, "splits": splits})
    return 0


def cmd_cases_to_srbench(args: argparse.Namespace) -> int:
    """X3: render the recipe-lab cases as the split's sr-bench dataset.

    Deterministic (same cases => byte-identical output), so the printed
    srbench_sha256 can be frozen into the contract's split declaration and
    the same file re-hashes to it for as long as the campaign runs.
    """
    try:
        cases = load_cases(args.cases)
    except OSError as exc:
        raise ContractError(f"cannot read cases: {exc}") from exc
    rows = cases_to_srbench_rows(cases, benchmark=args.benchmark)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    _print_json(
        {
            "cases": str(args.cases),
            "output": str(output),
            "cases_count": len(rows),
            "srbench_sha256": dataset_content_hash(output),
        }
    )
    return 0


def cmd_attribute(args: argparse.Namespace) -> int:
    spec = _load_spec(args.spec)
    # M5: transfer workloads are quarantined from attribution like frozen_test.
    workload = spec.require_tunable_workload(getattr(args, "workload", None))
    split_name = _select_split(spec, args.split, workload)
    # A4 contract gates: re-hash the selected split, then any --cases input.
    # ContractError propagates to main() -> exit 2.
    split_hash = spec.verify_workload(split=split_name, workload=workload.name)
    if args.cases:
        spec.require_dataset_file(args.cases, "cases", expected_hash=split_hash)
    try:
        cases = load_records(args.records)
    except (ContractError, OSError) as exc:
        _fail(f"cannot load attribution records: {exc}")

    # coverage pre-check only applies to string (label) truths; signal-fact
    # truths are resolved per case (uncovered ones become policy_gap rows)
    labels = sorted({c.truth for c in cases if isinstance(c.truth, str)})
    gaps = spec.validate_policy_covers(labels)
    if gaps:
        print(
            "policy_gap warning: oracle policy does not cover these labels: "
            + ", ".join(gaps),
            file=sys.stderr,
        )

    rows = attribute(cases, spec.oracle_policy, dataset_hash=split_hash)
    summary = summarize(rows)

    out_dir = Path(args.output_dir)
    jsonl_path = write_jsonl(rows, out_dir / "attribution.jsonl")
    # G3: the summary is written next to attribution.jsonl so `next` can
    # resolve a record's attribution_ref, verify the artifact hash and read it.
    summary_path = out_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path = out_dir / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(summary, split_hash), encoding="utf-8")

    _print_json(
        {
            "records": len(cases),
            "attribution_jsonl": str(jsonl_path),
            "summary_json": str(summary_path),
            "report": str(report_path),
            "summary": summary,
        }
    )
    return 0


def cmd_collect_replays(args: argparse.Namespace) -> int:
    """Pull replay records from the router management API and either join
    them against the labeled case list -> attribution input records, or with
    --unlabeled emit one raw diagnostic row per record (no labels needed)."""
    if args.unlabeled and args.cases:
        _fail("collect-replays: --unlabeled and --cases are mutually exclusive")
    if not args.unlabeled and not args.cases:
        _fail(
            "collect-replays: one of --cases (labeled join) or --unlabeled (raw rows) is required"
        )
    try:
        records = fetch_replays(
            args.url,
            show_details=not args.no_details,
            timeout=args.timeout,
            api_key_env=args.api_key_env,
        )
    except RuntimeError as exc:
        _fail(f"replay fetch error: {exc}")

    output = Path(args.output)
    result: dict = {"replay_records": len(records), "output": str(output)}
    if args.unlabeled:
        rows = records_to_rows(records)
        result["rows"] = len(rows)
    else:
        try:
            cases = load_cases(args.cases)
        except ContractError as exc:
            _fail(f"cases error: {exc}")
        rows, stats = records_to_cases(records, cases, family=args.family)
        write_cases(cases, output.with_suffix(".cases.jsonl"))
        result["rows"] = len(rows)
        result["join_stats"] = stats
    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    _print_json(result)
    return 0


def cmd_bank(args: argparse.Namespace) -> int:
    if args.bank_action == "build":
        cases = []
        with Path(args.cases).open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if line:
                    cases.append(json.loads(line))
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        if not models:
            _fail("--models must list at least one model name")
        try:
            entries = build_bank(cases, models, args.endpoint, timeout=args.timeout)
        except ContractError as exc:
            _fail(f"answer bank error: {exc}")
        out = write_bank(entries, args.output)
        _print_json({"entries": len(entries), "models": models, "output": str(out)})
        return 0

    if args.bank_action == "apply":
        try:
            bank = load_bank(args.bank)
            rows = [
                json.loads(line)
                for line in Path(args.records).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (ContractError, OSError, json.JSONDecodeError) as exc:
            _fail(f"cannot load bank or records: {exc}")
        merged, stats = apply_bank(rows, bank)
        output = Path(args.output) if args.output else Path(args.records)
        with output.open("w", encoding="utf-8") as fh:
            for row in merged:
                fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        _print_json({"rows": len(merged), "stats": stats, "output": str(output)})
        return 0

    _fail(f"unknown bank action: {args.bank_action}")  # pragma: no cover


def cmd_propose(args: argparse.Namespace) -> int:
    if args.spec:
        spec = _load_spec(args.spec)
        # A4/B contract gate: optional spec; verifies the selected split hash.
        # M5: transfer workloads are quarantined from proposals.
        workload = spec.require_tunable_workload(getattr(args, "workload", None))
        split_name = _select_split(spec, args.split, workload)
        spec.verify_workload(split=split_name, workload=workload.name)
    try:
        with Path(args.base_config).open("r", encoding="utf-8") as fh:
            base = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        _fail(f"cannot read base config: {exc}")
    deltas = [float(part) for part in str(args.deltas).split(",") if part.strip()]
    if not deltas:
        _fail("--deltas must list at least one perturbation")
    try:
        recipe = select_recipe(base, recipe=args.recipe, public_model=args.public_model)
        candidates = threshold_search.propose_thresholds(
            base, args.signal, deltas, lo=args.lo, hi=args.hi, recipe=recipe
        )
        paths = threshold_search.write_candidates(candidates, args.output_dir)
    except ContractError as exc:
        _fail(f"proposer error: {exc}")
    _print_json(
        {
            "recipe": recipe.name,
            "signal": args.signal,
            "candidates": [c.summary() for c in candidates],
            "files": [str(p) for p in paths],
        }
    )
    return 0


def cmd_propose_rule(args: argparse.Namespace) -> int:
    try:
        with Path(args.base_config).open("r", encoding="utf-8") as fh:
            base = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"cannot read base config: {exc}") from exc
    try:
        recipe = select_recipe(base, recipe=args.recipe, public_model=args.public_model)
        if args.list_sites:
            sites = rule_threshold.find_threshold_sites(base, recipe)
            if args.signal:
                sites = [s for s in sites if s.signal == args.signal]
            _print_json(
                {
                    "recipe": recipe.name,
                    "sites": [
                        {
                            "decision": s.decision,
                            "signal": s.signal,
                            "op": s.op,
                            "value": s.value,
                            "negated": s.negated,
                            "path": list(s.path),
                        }
                        for s in sites
                    ],
                }
            )
            return 0
    except ContractError as exc:
        raise SystemExit(f"proposer error: {exc}") from exc
    if not args.signal:
        raise SystemExit("propose-rule-thresholds: --signal is required")
    values = (
        [float(part) for part in str(args.values).split(",") if part.strip()]
        if args.values
        else None
    )
    deltas = (
        [float(part) for part in str(args.deltas).split(",") if part.strip()]
        if args.deltas
        else None
    )
    if values is None and deltas is None:
        raise SystemExit(
            "propose-rule-thresholds: one of --values or --deltas is required"
        )
    if values is not None and deltas is not None:
        raise SystemExit(
            "propose-rule-thresholds: --values and --deltas are mutually exclusive"
        )
    if not args.output_dir:
        raise SystemExit("propose-rule-thresholds: --output-dir is required")
    decisions = (
        [d.strip() for d in str(args.decisions).split(",") if d.strip()]
        if args.decisions
        else None
    )
    try:
        candidates = rule_threshold.propose_rule_thresholds(
            base,
            args.signal,
            values=values,
            deltas=deltas,
            decisions=decisions,
            allow_partial=args.allow_partial,
            from_value=args.from_value,
            op=args.op,
            lo=args.lo,
            hi=args.hi,
            recipe=recipe,
        )
        paths = rule_threshold.write_candidates(candidates, args.output_dir)
    except ContractError as exc:
        raise SystemExit(f"proposer error: {exc}") from exc
    _print_json(
        {
            "recipe": recipe.name,
            "signal": args.signal,
            "candidates": [c.summary() for c in candidates],
            "files": [str(p) for p in paths],
        }
    )
    return 0


def _resolve_candidates(value: str) -> list[Path]:
    path = Path(value)
    if path.is_dir():
        return sorted(path.glob("*.yaml"))
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


# S5: run --proposer is a closed vocabulary plus the agent-authored form.
_PROPOSER_TAGS = (
    "threshold-search",
    "rule-threshold",
    "distribution-calibration",
    "bootstrap",
)


def _proposer_arg(value: str) -> str:
    """Validate the run --proposer provenance tag (S5).

    A candidate the agent authored itself must be tagged ``agent:<id>`` with
    id matching [a-z0-9._-]+, so ledger rows stay filterable per author.
    """
    if value in _PROPOSER_TAGS or re.fullmatch(r"agent:[a-z0-9._-]+", value):
        return value
    raise argparse.ArgumentTypeError(
        "invalid --proposer "
        + repr(value)
        + ": expected one of "
        + ", ".join(_PROPOSER_TAGS)
        + " or agent:<id> (id: [a-z0-9._-]+)"
    )


def _candidate_entry(
    path: Path, fallback_proposer: str, index: int, total: int
) -> dict:
    """C: prefer the sidecar metadata the proposer wrote next to the config
    (`<stem>.candidate.json`); fall back to the bare-path form for configs
    that come without one."""
    sidecar = path.with_suffix(".candidate.json")
    if not sidecar.is_file():
        return {
            "kind": "config",
            "ref": str(path),
            "change_summary": f"candidate {index}/{total}",
            "provenance": {"proposer": fallback_proposer, "insight_refs": []},
        }
    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read candidate sidecar {sidecar}: {exc}") from exc
    return {
        "kind": raw.get("kind", "config"),
        "ref": raw.get("ref", str(path)),
        "change_summary": raw.get("change_summary", ""),
        "provenance": raw.get(
            "provenance", {"proposer": fallback_proposer, "insight_refs": []}
        ),
    }


def _validate_config(path: Path, validate_bin: str) -> str | None:
    """C: returns None when the config validates, else the rejection reason."""
    try:
        proc = subprocess.run(
            [validate_bin, "config", "validate", "--config", str(path)],
            capture_output=True,
            text=True,
            timeout=300.0,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ContractError(f"validator binary not found: {validate_bin}") from exc
    except subprocess.TimeoutExpired:
        return "config validate timed out"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else "no output"
        return f"config validate exit {proc.returncode}: {tail}"
    return None


def cmd_run(args: argparse.Namespace) -> int:
    spec = _load_spec(args.spec)
    # A5 budget gate: rounds beyond the contract are refused up front.
    if args.round > spec.budget.rounds:
        raise ContractError(
            f"budget exhausted: round {args.round} > budget.rounds {spec.budget.rounds}"
        )
    # B: resolve the split; frozen_test is quarantined, legacy is dev-only.
    # M5: transfer workloads are quarantined from replay like frozen_test.
    workload = spec.require_tunable_workload(getattr(args, "workload", None))
    split_name = _select_split(spec, args.split, workload)
    split_hash = workload.split_hash(split_name)
    # A4 contract gates: re-hash the selected split, then the --cases input
    # must be byte-identical to it (cross-round comparability).
    spec.verify_workload(split=split_name, workload=workload.name)
    spec.require_dataset_file(args.cases, "cases", expected_hash=split_hash)
    try:
        bank = load_bank(args.bank)
        cases = [
            json.loads(line)
            for line in Path(args.cases).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot load bank or cases: {exc}")
    candidates = _resolve_candidates(args.candidates)
    if not candidates:
        _fail(f"no candidate configs found: {args.candidates}")

    # C: optional pre-flight validation; rejected candidates never reach the ledger.
    rejected: list[dict] = []
    if args.validate:
        kept: list[Path] = []
        for path in candidates:
            reason = _validate_config(path, args.validate_bin)
            if reason is None:
                kept.append(path)
            else:
                rejected.append({"path": str(path), "reason": reason})
                print(f"[run] rejected {path.name}: {reason}", file=sys.stderr)
        candidates = kept
    if not candidates:
        _print_json(
            {
                "evaluated": 0,
                "timed_out": [],
                "rejected": rejected,
                "results": [],
                "best": None,
            }
        )
        return 2

    output_dir = Path(args.output_dir)
    ledger = Ledger(args.ledger)
    # G2: fail fast on unverifiable insight references (ContractError -> exit 2).
    insight_refs = resolve_insight_refs(parse_insight_refs(args.insight_refs), ledger)
    env = RunnerEnv(
        router_bin=args.router_bin,
        mock_backend_script=args.mock_backend,
        python_bin=args.python_bin,
        work_dir=output_dir,
        backend_port=args.backend_port,
        ready_timeout_s=args.ready_timeout,
        hf_home=args.hf_home,
        mock_extra_args=(
            tuple(args.mock_extra_args.split()) if args.mock_extra_args else ()
        ),
        reload_mode=args.reload,
        api_key_env=args.api_key_env,
    )

    # A5: the per-trial budget is the contract's max_minutes_per_trial; it
    # bounds readiness wait + request loop + replay fetch for every trial.
    trial_timeout_s = spec.budget.max_minutes_per_trial * 60

    # H: hot mode keeps one router alive across the whole round; the session
    # boots from the baseline config, or the first candidate when there is
    # no baseline, and swaps every further candidate in via PUT.
    session: HotReloadSession | None = None
    run_results: list[dict] = []
    if env.reload_mode == "hot":
        boot_source = args.baseline_config or str(candidates[0])
        session = HotReloadSession(
            env=env,
            boot_source=boot_source,
            router_model=args.router_model,
            family=args.family,
        )

    baseline_summary: dict | None = None
    # E2: baseline arm's routed_cost_index (None when no baseline was evaluated)
    baseline_cost_index: float | None = None
    if args.baseline_config:
        print(
            f"[run] evaluating baseline config: {args.baseline_config}", file=sys.stderr
        )
        try:
            if session is not None:
                session.ensure_started()
                result = session.evaluate(
                    candidate_config=args.baseline_config,
                    cases=cases,
                    oracle_policy=spec.oracle_policy,
                    bank=bank,
                    trial_timeout_s=trial_timeout_s,
                )
            else:
                result = evaluate_candidate(
                    candidate_config=args.baseline_config,
                    cases=cases,
                    oracle_policy=spec.oracle_policy,
                    bank=bank,
                    env=env,
                    router_model=args.router_model,
                    family=args.family,
                    trial_timeout_s=trial_timeout_s,
                )
        except RunnerTimeoutError as exc:
            if session is not None:
                session.stop()  # H: no leaked router/backend on the exit path
            raise SystemExit(
                f"baseline evaluation exceeded the trial budget: {exc}"
            ) from exc
        except (RunnerError, ContractError, OSError) as exc:
            if session is not None:
                session.stop()  # H: no leaked router/backend on the exit path
            raise SystemExit(f"baseline evaluation failed: {exc}") from exc
        if result.get("incomplete"):
            # H: a partial baseline capture has no summary to read, and it
            # cannot anchor the attribution deltas the candidates' records
            # would cite - surface the capture reason instead of a KeyError
            # on the missing summary.
            if session is not None:
                session.stop()  # no leaked router/backend on the exit path
            raise SystemExit(
                "baseline replay capture incomplete: "
                f"{result.get('incomplete_reason', '')}"
            )
        baseline_summary = result["summary"]
        run_results.append(
            {
                "path": str(args.baseline_config),
                "reload": result.get("reload", "restart"),
                "status": "baseline",
                "timings": result.get("timings"),
            }
        )
        # E2: the baseline arm's cost proxy anchors `bound: baseline` constraints.
        if spec.cost_model is not None:
            baseline_cost_index, baseline_unknown = routed_cost_index(
                result["rows"], spec.cost_model
            )
            if baseline_cost_index is None and baseline_unknown:
                print(
                    "[run] baseline cost proxy incomplete: routed model(s) not in cost_model: "
                    + ", ".join(baseline_unknown),
                    file=sys.stderr,
                )

    appended = []
    timeouts = []
    for index, candidate_path in enumerate(candidates, 1):
        print(
            f"[run] candidate {index}/{len(candidates)}: {candidate_path}",
            file=sys.stderr,
        )
        entry = _candidate_entry(candidate_path, args.proposer, index, len(candidates))
        # G2: carry the validated insight references on the record provenance.
        merge_insight_refs(entry["provenance"], insight_refs)
        try:
            if session is not None:
                session.ensure_started()  # no-op while up; re-boots after a fallback
                result = session.evaluate_with_fallback(
                    candidate_config=candidate_path,
                    cases=cases,
                    oracle_policy=spec.oracle_policy,
                    bank=bank,
                    trial_timeout_s=trial_timeout_s,
                )
            else:
                result = evaluate_candidate(
                    candidate_config=candidate_path,
                    cases=cases,
                    oracle_policy=spec.oracle_policy,
                    bank=bank,
                    env=env,
                    router_model=args.router_model,
                    family=args.family,
                    trial_timeout_s=trial_timeout_s,
                )
        except RunnerTimeoutError as exc:
            # A5: a hung stack is never a worse recipe - record status=timeout
            # with no metrics; top_replay excludes it from the ranking.
            timeouts.append(str(candidate_path))
            record = LedgerRecord(
                round=args.round,
                ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                candidate=entry,
                tier="replay",
                baseline_ref=spec.stack.baseline_ref if spec.stack else "unknown",
                metrics={},
                verdict="RANKED",
                dataset_hash=split_hash,
                split=split_name,
                reasons=[f"trial timeout: {exc}"],
                # A3: replay never evaluates guards; the declared gates are
                # recorded as explicitly not evaluated.
                gates={gate.name: "not_evaluated" for gate in spec.gates},
                bench_refs=[],
                attribution_delta=None,
                attribution_ref=args.attribution_ref,
                coverage_warning=True,
                notes="",
                status="timeout",
            )
            ledger.append(record)
            run_results.append(
                {
                    "path": str(candidate_path),
                    "reload": env.reload_mode,
                    "status": "timeout",
                    "timings": None,
                }
            )
            print(
                f"[run] {candidate_path.name}: TRIAL TIMEOUT -> ledger (status=timeout)",
                file=sys.stderr,
            )
            continue
        except (RunnerError, ContractError, OSError) as exc:
            if session is not None:
                session.stop()  # H: no leaked router/backend on the exit path
            _fail(f"candidate evaluation failed for {candidate_path}: {exc}")

        reload_used = result.get("reload", "restart")
        notes = f"statuses={json.dumps(result['statuses'], ensure_ascii=False)} reload={reload_used}"
        if result.get("reload") == "restart-fallback":
            # H: the PUT was rejected; the ledger must say why this candidate
            # was evaluated on a fresh router instead of the live session.
            notes += f" fallback_reason={result.get('fallback_reason', '')}"

        if result.get("incomplete"):
            # H: the shared replay store lost or duplicated records for this
            # candidate's request ids; no metric is trusted from a partial pull.
            record = LedgerRecord(
                round=args.round,
                ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                candidate=entry,
                tier="replay",
                baseline_ref=spec.stack.baseline_ref if spec.stack else "unknown",
                metrics={},
                verdict="RANKED",
                dataset_hash=split_hash,
                split=split_name,
                workload=workload.name,
                reasons=[f"incomplete replay capture: {result['incomplete_reason']}"],
                gates=None,
                bench_refs=[],
                attribution_delta=None,
                attribution_ref=args.attribution_ref,
                coverage_warning=True,
                notes=notes,
                status="incomplete",
            )
            ledger.append(record)
            run_results.append(
                {
                    "path": str(candidate_path),
                    "reload": reload_used,
                    "status": "incomplete",
                    "timings": result.get("timings"),
                }
            )
            print(
                f"[run] {candidate_path.name}: INCOMPLETE replay capture -> ledger (status=incomplete)",
                file=sys.stderr,
            )
            continue

        summary = result["summary"]
        # A1: the replay metric is contracted in the spec (not the live metric_key)
        replay_metric = spec.objective.replay_metric
        metric_value = replay_metric_value(summary, replay_metric)
        coverage_warning = (
            result["bank_stats"].get("no_bank_entry", 0) > 0 or metric_value is None
        )
        # E2: replay cost proxy - a secondary metric plus constraint warnings.
        # Without cost_model the replay tier does no constraint work at all:
        # cost is adjudicated only in the live tier (the next-context export
        # carries that hint).
        reasons: list[str] = []
        metrics: dict = (
            {replay_metric: metric_value} if metric_value is not None else {}
        )
        constraint_evidence: dict | None = None
        constraint_warning = False
        cost_text = ""
        if spec.cost_model is not None:
            cost_index, unknown_models = routed_cost_index(
                result["rows"], spec.cost_model
            )
            if cost_index is not None:
                metrics[COST_PROXY_METRIC] = cost_index
                cost_text = f" {COST_PROXY_METRIC}={cost_index}"
            elif unknown_models:
                reasons.append(
                    "cost proxy incomplete: routed model(s) not in cost_model: "
                    + ", ".join(unknown_models)
                    + " (unknown cost is never counted as zero)"
                )
            if spec.objective.constraints:
                evidence, constraint_warning = replay_constraint_evidence(
                    spec.objective.constraints, cost_index, baseline_cost_index
                )
                constraint_evidence = {c["name"]: c for c in evidence}
                reasons.extend(
                    c["reason"] for c in evidence if c["status"] == "violated"
                )
        record = LedgerRecord(
            round=args.round,
            ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            candidate=entry,
            tier="replay",
            baseline_ref=spec.stack.baseline_ref if spec.stack else "unknown",
            metrics=metrics,
            verdict="RANKED",
            dataset_hash=split_hash,
            split=split_name,
            workload=workload.name,
            reasons=reasons,
            # A3: replay never evaluates guards; the declared gates are
            # recorded as explicitly not evaluated.
            gates={gate.name: "not_evaluated" for gate in spec.gates},
            bench_refs=[],
            attribution_delta=(
                attribution_delta(baseline_summary, summary)
                if baseline_summary
                else None
            ),
            attribution_ref=args.attribution_ref,
            coverage_warning=coverage_warning,
            notes=notes,
            constraint_warning=constraint_warning,
            constraints=constraint_evidence,
        )
        stored = ledger.append(record)
        appended.append(stored)
        run_results.append(
            {
                "path": str(candidate_path),
                "reload": reload_used,
                "status": "ranked",
                "timings": result.get("timings"),
            }
        )
        accuracy_text = "n/a" if metric_value is None else f"{metric_value * 100:.2f}%"
        warning_text = " CONSTRAINT_WARNING" if constraint_warning else ""
        print(
            f"[run] {candidate_path.name}: {replay_metric}={accuracy_text}{cost_text} -> ledger{warning_text}",
            file=sys.stderr,
        )

    if session is not None:
        session.stop()

    best = ledger.top_replay(spec.objective.replay_metric)
    _print_json(
        {
            "evaluated": len(appended),
            "timed_out": timeouts,
            "rejected": rejected,
            "results": run_results,
            "best": best,
        }
    )
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    ledger = Ledger(args.file)
    if args.ledger_action == "append":
        try:
            raw = json.loads(Path(args.record).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _fail(f"cannot read record file: {exc}")
        # G2: merge --insight-refs into the record's candidate provenance;
        # unresolvable refs raise ContractError -> main() exits 2.
        if args.insight_refs:
            refs = resolve_insight_refs(parse_insight_refs(args.insight_refs), ledger)
            merge_insight_refs(
                raw.setdefault("candidate", {}).setdefault("provenance", {}), refs
            )
        try:
            stored = ledger.append(LedgerRecord.from_dict(raw))
        except ContractError as exc:
            _fail(f"ledger contract error: {exc}")
        _print_json({"appended": True, "records": len(ledger.read()), "record": stored})
        return 0

    if args.ledger_action == "show":
        try:
            rows = ledger.filter(
                tier=args.tier,
                verdict=args.verdict,
                round_no=args.round,
                proposer=args.proposer,
            )
        except ContractError as exc:
            _fail(f"ledger contract error: {exc}")
        if args.json:
            _print_json(rows)
        else:
            for row in rows:
                metrics = json.dumps(row.get("metrics", {}), ensure_ascii=False)
                print(
                    f"round={row.get('round')}\ttier={row.get('tier')}\tverdict={row.get('verdict')}"
                    f"\tmetrics={metrics}"
                )
        return 0

    if args.ledger_action == "top":
        # A1: --spec takes the metric from objective.replay_metric;
        # --metric-key stays for ad-hoc ranking. Exactly one is required.
        if args.spec:
            metric_key = _load_spec(args.spec).objective.replay_metric
        else:
            metric_key = args.metric_key
        try:
            best = ledger.top_replay(metric_key)
        except ContractError as exc:
            _fail(f"ledger contract error: {exc}")
        _print_json(best if best is not None else {})
        return 0

    _fail(f"unknown ledger action: {args.ledger_action}")  # pragma: no cover


def verdict_exit_code(verdict: str) -> int:
    """0 = PASS, 1 = FAIL, 2 = INCONCLUSIVE (or tool/usage error elsewhere)."""
    return {"PASS": 0, "FAIL": 1}.get(verdict, 2)


def cmd_live_verdict(args: argparse.Namespace) -> int:
    """Compare two sr-bench runs and record the live-tier verdict."""
    spec = _load_spec(args.spec)
    # B: frozen_test is quarantined; only the --final promotion verdict may
    # cite it. Every other split goes through the normal A4 re-hash gate.
    if args.split == "frozen_test" and not args.final:
        raise ContractError(
            "frozen_test is quarantined; pass --final for the promotion verdict"
        )
    # M5: a transfer workload is final-verify only (the frozen_test rule
    # applies to the whole workload).
    workload = spec.select_workload(getattr(args, "workload", None))
    if workload.role == "transfer" and not args.final:
        raise ContractError(
            f"workload {workload.name!r} has role 'transfer' (the frozen_test rule "
            "applies to the whole workload): pass --final for the promotion verdict"
        )
    dataset_hash = spec.verify_workload(split=args.split, workload=workload.name)
    try:
        raw_measurements = (
            [
                json.loads(line)
                for line in Path(args.guard_measurements)
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            if args.guard_measurements
            else []
        )
        candidate_row = (
            json.loads(Path(args.candidate).read_text(encoding="utf-8"))
            if args.candidate
            else None
        )
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot load guard measurements or candidate: {exc}")
    # A3: measurement schema violations are contract errors (exit 2); a
    # `floor` key in the file must never silently override spec.gates.
    measurements = measurements_from_rows(raw_measurements)
    # G2: fail fast on unverifiable insight references (ContractError -> exit
    # 2); a rec- reference needs the ledger it is resolved against.
    ledger = Ledger(args.ledger) if args.ledger else None
    insight_refs = resolve_insight_refs(parse_insight_refs(args.insight_refs), ledger)
    candidate_row = candidate_row or {
        "kind": "config",
        "ref": args.candidate_run_id,
        "change_summary": "",
        "provenance": {"proposer": "external", "insight_refs": []},
    }
    merge_insight_refs(candidate_row.setdefault("provenance", {}), insight_refs)

    cli = SrbenchCli(
        bin=args.srbench_bin,
        url=args.srbench_url,
        store=args.srbench_store,
        timeout_s=args.timeout,
    )
    try:
        result = live_verdict_from_runs(
            spec=spec,
            cli=cli,
            baseline_run_id=args.baseline_run_id,
            candidate_run_id=args.candidate_run_id,
            measurements=measurements,
            workload_split=args.split,
        )
    except (ContractError, RuntimeError) as exc:
        _fail(f"live verdict error: {exc}")

    if args.ledger:
        record = LedgerRecord(
            round=args.round,
            ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            candidate=candidate_row,
            tier="live",
            baseline_ref=args.baseline_run_id,
            metrics={spec.objective.metric_key: result["metrics"]["candidate"]},
            verdict=result["verdict"],
            dataset_hash=dataset_hash,
            # M1: the legacy single-dataset form has no real split.
            split=None if workload.legacy else args.split,
            legacy_workload=workload.legacy,
            workload=workload.name,
            reasons=result["reasons"],
            gates=result["gates"],
            # E: the declared constraints' measured evidence, for report replay.
            constraints={c["name"]: c for c in result["constraints"]},
            bench_refs=result["bench_refs"],
            # M2: paired statistics from the compare payload, verbatim.
            paired=result.get("paired"),
            attribution_delta=None,
            attribution_ref=args.attribution_ref,
            coverage_warning=False,
            notes=f"baseline={result['metrics']['baseline']}",
        )
        ledger.append(record)
    _print_json(result)
    return verdict_exit_code(result["verdict"])


def cmd_verify(args: argparse.Namespace) -> int:
    """Drive sr-bench end to end for one candidate (I).

    Preview routing first, then `--repetitions` AB/BA pairs (each pair
    re-runs the baseline's frozen manifest AND submits a fresh
    candidate-plan), per-pair compare, descriptive medians/ranges, and the
    shared three-state verdict. Any incomplete pair -> INCONCLUSIVE.
    """
    spec = _load_spec(args.spec)
    # B: frozen_test is quarantined; only the --final promotion verdict may
    # cite it. Every other split goes through the normal A4 re-hash gate.
    if args.split == "frozen_test" and not args.final:
        raise ContractError(
            "frozen_test is quarantined; pass --final for the promotion verdict"
        )
    # M5: a transfer workload is final-verify only (the frozen_test rule
    # applies to the whole workload).
    workload = spec.select_workload(getattr(args, "workload", None))
    if workload.role == "transfer" and not args.final:
        raise ContractError(
            f"workload {workload.name!r} has role 'transfer' (the frozen_test rule "
            "applies to the whole workload): pass --final for the promotion verdict"
        )
    dataset_hash = spec.verify_workload(split=args.split, workload=workload.name)
    try:
        raw_measurements = (
            [
                json.loads(line)
                for line in Path(args.guard_measurements)
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            if args.guard_measurements
            else []
        )
        candidate_row = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot load guard measurements or candidate: {exc}") from exc
    # A3: measurement schema violations are contract errors (exit 2); a
    # `floor` key in the file must never silently override spec.gates.
    measurements = measurements_from_rows(raw_measurements)
    # G2: fail fast on unverifiable insight references (ContractError -> exit
    # 2); a rec- reference needs the ledger it is resolved against.
    ledger = Ledger(args.ledger) if args.ledger else None
    insight_refs = resolve_insight_refs(parse_insight_refs(args.insight_refs), ledger)
    merge_insight_refs(candidate_row.setdefault("provenance", {}), insight_refs)
    # sr-bench MoM target ids the router is serving the candidate on: the
    # CLI flag wins; otherwise the candidate file must carry srbench_targets.
    if args.candidate_targets:
        targets = [t.strip() for t in args.candidate_targets.split(",") if t.strip()]
    else:
        targets = candidate_row.get("srbench_targets")
    if (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(t, str) and t.strip() for t in targets)
    ):
        raise ContractError(
            "verify needs sr-bench MoM target ids: pass --candidate-targets "
            "(comma-separated) or add a non-empty srbench_targets list to the candidate JSON"
        )
    work_dir = (
        Path(args.work_dir)
        if args.work_dir
        else Path(tempfile.mkdtemp(prefix="rl-verify-"))
    )
    cli = SrbenchCli(
        bin=args.srbench_bin,
        url=args.srbench_url,
        store=args.srbench_store,
        timeout_s=args.timeout,
    )
    try:
        # M4: reference-arm run IDs are cached so repeated verify invocations
        # do not re-run the (expensive) reference arms; default location
        # follows the ledger when present, else the work dir.
        run_cache = args.run_cache or (
            Path(args.ledger).parent / "run-id-cache.json" if args.ledger else None
        )
        result = verify_live(
            spec=spec,
            cli=cli,
            baseline_run_id=args.baseline_run_id,
            baseline_manifest=args.baseline_manifest,
            targets=targets,
            repetitions=args.repetitions,
            measurements=measurements,
            work_dir=work_dir,
            workload_split=args.split,
            run_cache=run_cache,
        )
    except (ContractError, RuntimeError) as exc:
        raise SystemExit(f"verify error: {exc}") from exc

    if args.ledger:
        preview = result.get("preview") or {}
        notes_parts = [
            f"baseline={result['metrics']['baseline']}",
            f"repetitions={args.repetitions}",
        ]
        if preview.get("run_id"):
            notes_parts.append(f"preview_run={preview['run_id']}")
        distribution = preview.get("distribution")
        if distribution:
            notes_parts.append(
                "routing_distribution="
                + json.dumps(distribution, sort_keys=True, ensure_ascii=False)
            )
        record = LedgerRecord(
            round=args.round,
            ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            candidate=candidate_row,
            tier="live",
            # R2: when the baseline came from --baseline-manifest the run id
            # is resolved inside verify_live and echoed in the result.
            baseline_ref=args.baseline_run_id or result.get("baseline_run_id"),
            metrics={spec.objective.metric_key: result["metrics"]["candidate"]},
            verdict=result["verdict"],
            dataset_hash=dataset_hash,
            # M1: the legacy single-dataset form has no real split.
            split=None if workload.legacy else args.split,
            legacy_workload=workload.legacy,
            workload=workload.name,
            reasons=result["reasons"],
            gates=result["gates"],
            # E: the declared constraints' measured evidence, for report replay.
            constraints={c["name"]: c for c in result["constraints"]},
            bench_refs=result["bench_refs"],
            # M2: the pairs' merged paired statistics, verbatim.
            paired=result.get("paired"),
            attribution_delta=None,
            attribution_ref=args.attribution_ref,
            coverage_warning=False,
            notes="; ".join(notes_parts),
        )
        ledger.append(record)
    _print_json(result)
    return verdict_exit_code(result["verdict"])


def cmd_score_candidate(args: argparse.Namespace) -> int:
    """Score one deployed candidate against a saved sr-bench baseline.

    The channel a coding agent uses to submit its own proposal: the receipt
    (schema ``recipe-lab-sr-bench-run-v1``) is printed as one JSON line and
    saved under ``<workspace>/sr-bench``. The exit code follows the verdict.
    """
    receipt = srbench_evaluate(args)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return verdict_exit_code(receipt["verdict"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recipe-lab",
        description="Closed-loop experiments for semantic-router routing recipes (RSI workbench).",
    )
    parser.add_argument(
        "--version", action="version", version=f"recipe-lab {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser(
        "plan", help="load and validate an experiment spec, print it normalized"
    )
    p_plan.add_argument(
        "--spec", required=True, help="path to the experiment spec YAML"
    )
    p_plan.set_defaults(func=cmd_plan)

    p_hash = sub.add_parser(
        "hash",
        help="compute the frozen dataset identity: sha256 of the JSONL raw bytes (A4)",
    )
    p_hash.add_argument("--file", required=True, help="dataset JSONL file")
    p_hash.set_defaults(func=cmd_hash)

    p_split = sub.add_parser(
        "split",
        help="split a labeled case list into calib/dev/frozen_test (B)",
        description="Deterministic group-level 3-way split: same cases + group key + seed => byte-identical output.",
    )
    p_split.add_argument(
        "--cases",
        required=True,
        help="labeled case list JSONL (needs case_id + prompt)",
    )
    p_split.add_argument(
        "--group-key",
        default="source_id",
        help="case field to group by (leakage guard)",
    )
    p_split.add_argument(
        "--ratios",
        default="0.4,0.3,0.3",
        help="calib/dev/frozen_test ratios, e.g. 0.4,0.3,0.3",
    )
    p_split.add_argument(
        "--seed", type=int, default=42, help="shuffle seed for the group assignment"
    )
    p_split.add_argument(
        "--split-rule", default=None, help="override the provenance split_rule text"
    )
    p_split.add_argument(
        "--output-dir",
        required=True,
        help="directory for calib.jsonl / dev.jsonl / frozen_test.jsonl",
    )
    p_split.set_defaults(func=cmd_split)

    p_cases = sub.add_parser(
        "cases", help="case-file utilities (X3: one contract, two dataset formats)"
    )
    cases_sub = p_cases.add_subparsers(dest="cases_command", required=True)
    p_tosr = cases_sub.add_parser(
        "to-srbench",
        help="deterministically render a cases JSONL as an sr-bench dataset "
        "(case_id->id, prompt->messages, truth->answer)",
    )
    p_tosr.add_argument(
        "--cases", required=True, help="cases JSONL (case_id/prompt/truth)"
    )
    p_tosr.add_argument(
        "--output", required=True, help="output sr-bench dataset JSONL path"
    )
    p_tosr.add_argument(
        "--benchmark",
        default="recipe-lab",
        help="benchmark label recorded on every row (default: recipe-lab)",
    )
    p_tosr.set_defaults(func=cmd_cases_to_srbench)

    p_attr = sub.add_parser(
        "attribute", help="run E1/E2/E3 misroute attribution over labeled records"
    )
    p_attr.add_argument(
        "--spec",
        required=True,
        help="path to the experiment spec YAML (provides oracle policy)",
    )
    p_attr.add_argument(
        "--records", required=True, help="path to the attribution records JSONL"
    )
    p_attr.add_argument(
        "--split",
        default="dev",
        choices=["calib", "dev", "frozen_test"],
        help="workload split the records were produced on (frozen_test is quarantined; B)",
    )
    p_attr.add_argument(
        "--workload",
        default=None,
        help="workload from spec.workloads the records belong to (default: the first workload; "
        "transfer workloads are quarantined; M5)",
    )
    p_attr.add_argument(
        "--cases",
        default=None,
        help="optional labeled case list JSONL; when given, its hash must match the selected split (A4)",
    )
    p_attr.add_argument(
        "--output-dir",
        default=".",
        help="directory for attribution.jsonl and report.md",
    )
    p_attr.set_defaults(func=cmd_attribute)

    p_collect = sub.add_parser(
        "collect-replays",
        help="pull router_replay records from the management API and join them with labeled cases (--unlabeled for raw diagnostic rows)",
    )
    p_collect.add_argument(
        "--url",
        required=True,
        help="router management base URL, e.g. http://127.0.0.1:8080",
    )
    p_collect.add_argument(
        "--cases",
        default=None,
        help="labeled case list JSONL (case_id/prompt/truth); omit with --unlabeled",
    )
    p_collect.add_argument(
        "--unlabeled",
        action="store_true",
        help="skip the labeled join and emit one raw diagnostic row per replay record (no --cases needed)",
    )
    p_collect.add_argument(
        "--output", required=True, help="output attribution-records JSONL path"
    )
    p_collect.add_argument(
        "--family", default="d3", help="signal family tag for the records"
    )
    p_collect.add_argument(
        "--no-details", action="store_true", help="do not request showDetails=true"
    )
    p_collect.add_argument(
        "--timeout", type=float, default=30.0, help="management API timeout in seconds"
    )
    p_collect.add_argument(
        "--api-key-env",
        default=None,
        help="env var name holding the management API bearer token",
    )
    p_collect.set_defaults(func=cmd_collect_replays)

    p_bank = sub.add_parser(
        "bank", help="build or apply the per-case/per-model answer bank"
    )
    bank_sub = p_bank.add_subparsers(dest="bank_action", required=True)

    p_bbuild = bank_sub.add_parser(
        "build", help="query each case against each model and grade"
    )
    p_bbuild.add_argument(
        "--cases", required=True, help="cases JSONL (case_id/prompt/expected_contains)"
    )
    p_bbuild.add_argument(
        "--models", required=True, help="comma-separated model (served) names"
    )
    p_bbuild.add_argument(
        "--endpoint",
        required=True,
        help="OpenAI-compatible endpoint, e.g. http://127.0.0.1:8000",
    )
    p_bbuild.add_argument("--output", required=True, help="output bank JSONL path")
    p_bbuild.add_argument("--timeout", type=float, default=180.0)
    p_bbuild.set_defaults(func=cmd_bank)

    p_bapply = bank_sub.add_parser(
        "apply", help="join bank correctness into attribution records"
    )
    p_bapply.add_argument("--bank", required=True, help="answer bank JSONL")
    p_bapply.add_argument(
        "--records",
        required=True,
        help="attribution records JSONL (answer_ok joined by case_id+routed)",
    )
    p_bapply.add_argument(
        "--output", default=None, help="output path (default: overwrite records)"
    )
    p_bapply.set_defaults(func=cmd_bank)

    p_prop = sub.add_parser(
        "propose-thresholds",
        help="generate candidate configs by perturbing one decision-signal threshold",
    )
    p_prop.add_argument("--base-config", required=True, help="base recipe config YAML")
    p_prop.add_argument(
        "--spec",
        default=None,
        help="optional experiment spec; when given, the selected split hash is verified first (A4/B)",
    )
    p_prop.add_argument(
        "--split",
        default="dev",
        choices=["calib", "dev", "frozen_test"],
        help="workload split to verify against (frozen_test is quarantined; B)",
    )
    p_prop.add_argument(
        "--workload",
        default=None,
        help="workload from spec.workloads to calibrate on (default: the first workload; "
        "transfer workloads are quarantined; M5)",
    )
    p_prop.add_argument(
        "--signal", required=True, help="decision signal name whose gte to perturb"
    )
    p_prop.add_argument(
        "--deltas", default="-0.1,-0.05,0.05,0.1", help="comma-separated deltas"
    )
    p_prop.add_argument("--lo", type=float, default=0.0, help="threshold lower clamp")
    p_prop.add_argument("--hi", type=float, default=1.0, help="threshold upper clamp")
    p_prop.add_argument(
        "--output-dir", required=True, help="directory for candidate config YAMLs"
    )
    _add_recipe_selectors(p_prop)
    p_prop.set_defaults(func=cmd_propose)

    p_rule = sub.add_parser(
        "propose-rule-thresholds",
        help="generate candidate configs by moving every rule-level predicate site of one signal (C)",
        description=(
            "Finds predicate occurrences inside decision rules (including under NOT branches), "
            "optionally filters them, and moves all selected sites to each target value. "
            "Writes one YAML config plus a sidecar metadata JSON per candidate."
        ),
    )
    p_rule.add_argument("--base-config", required=True, help="base recipe config YAML")
    p_rule.add_argument(
        "--signal",
        default=None,
        help="decision signal name whose rule predicates to move (optional with --list-sites)",
    )
    p_rule.add_argument(
        "--from",
        dest="from_value",
        type=float,
        default=None,
        help="only move sites currently at this value (e.g. the 2.5 -> 1.75 BUG-3 calibration)",
    )
    p_rule.add_argument(
        "--values", default=None, help="comma-separated target threshold values"
    )
    p_rule.add_argument(
        "--deltas", default=None, help="comma-separated deltas from the current value"
    )
    p_rule.add_argument(
        "--decisions",
        default=None,
        help="comma-separated decision names to restrict the move to",
    )
    p_rule.add_argument(
        "--allow-partial",
        action="store_true",
        help="skip decisions without a matching site (warn) instead of failing",
    )
    p_rule.add_argument(
        "--op",
        choices=["gte", "gt", "lte", "lt"],
        default=None,
        help="restrict to one predicate operator",
    )
    p_rule.add_argument(
        "--lo",
        type=float,
        default=None,
        help="threshold lower clamp (default: inferred from the signal)",
    )
    p_rule.add_argument(
        "--hi",
        type=float,
        default=None,
        help="threshold upper clamp (default: inferred from the signal)",
    )
    p_rule.add_argument(
        "--list-sites",
        action="store_true",
        help="print the matching threshold sites as JSON and exit",
    )
    p_rule.add_argument(
        "--output-dir",
        default=None,
        help="directory for candidate config YAMLs (required when proposing)",
    )
    _add_recipe_selectors(p_rule)
    p_rule.set_defaults(func=cmd_propose_rule)

    p_run = sub.add_parser(
        "run", help="evaluate candidate configs in the replay tier (RANKED -> ledger)"
    )
    p_run.add_argument(
        "--tier",
        default="replay",
        choices=["replay"],
        help="replay tier (RANKED records); live tier uses the live-verdict command",
    )
    p_run.add_argument("--spec", required=True, help="experiment spec YAML")
    p_run.add_argument("--cases", required=True, help="labeled case list JSONL")
    p_run.add_argument(
        "--split",
        default="dev",
        choices=["calib", "dev", "frozen_test"],
        help="workload split to evaluate on (frozen_test is quarantined; B)",
    )
    p_run.add_argument(
        "--workload",
        default=None,
        help="workload from spec.workloads to replay on (default: the first workload; "
        "transfer workloads are quarantined; M5)",
    )
    p_run.add_argument("--bank", required=True, help="answer bank JSONL")
    p_run.add_argument(
        "--candidates",
        required=True,
        help="directory of candidate config YAMLs, or comma-separated config paths",
    )
    p_run.add_argument(
        "--baseline-config",
        default=None,
        help="evaluate this config first to anchor attribution deltas",
    )
    p_run.add_argument(
        "--proposer",
        type=_proposer_arg,
        default="threshold-search",
        help="provenance tag for the candidates: threshold-search | rule-threshold | distribution-calibration | bootstrap | agent:<id>",
    )
    p_run.add_argument(
        "--attribution-ref",
        default=None,
        help="attribution artifact this round is based on (path@hash) from recipe-lab attribute; recorded in the ledger",
    )
    p_run.add_argument(
        "--insight-refs",
        default=None,
        help="comma-separated provenance refs: rec-<id>, bench:<run_id>, attr:<path>@sha256:<hex>, diag:<path>@sha256:<hex>#<anchor> (G2)",
    )
    p_run.add_argument(
        "--ledger", required=True, help="ledger JSONL to append RANKED records to"
    )
    p_run.add_argument(
        "--output-dir",
        required=True,
        help="per-candidate evaluation artifacts directory",
    )
    p_run.add_argument(
        "--router-bin", required=True, help="path to the semantic-router binary"
    )
    p_run.add_argument(
        "--mock-backend",
        required=True,
        help="path to a mock/real OpenAI backend launcher script",
    )
    p_run.add_argument(
        "--python-bin", default=sys.executable, help="python used to launch the backend"
    )
    p_run.add_argument("--router-model", default="vllm-sr/auto")
    p_run.add_argument("--family", default="d3")
    p_run.add_argument("--backend-port", type=int, default=8000)
    p_run.add_argument(
        "--ready-timeout",
        type=int,
        default=300,
        help="model-runtime readiness timeout in seconds",
    )
    p_run.add_argument(
        "--hf-home",
        default=None,
        help="HF_HOME for the router process (model artifacts)",
    )
    p_run.add_argument(
        "--mock-extra-args",
        default="",
        help="extra args passed to the mock backend (split on spaces)",
    )
    p_run.add_argument(
        "--reload",
        choices=["restart", "hot"],
        default="restart",
        help="candidate switching: one router restart per candidate (default) or one live router fed by PUT /api/v1/config (H)",
    )
    p_run.add_argument(
        "--api-key-env",
        default=None,
        help="env var name holding the management API bearer token (config PUT in hot mode + replay fetch)",
    )
    p_run.add_argument("--round", type=int, default=1)
    p_run.add_argument(
        "--validate",
        action="store_true",
        help="pre-flight each candidate config with --validate-bin; rejected candidates never reach the ledger (C)",
    )
    p_run.add_argument(
        "--validate-bin",
        default="vllm-sr",
        help="validator invoked as '<bin> config validate --config <path>' (C)",
    )
    p_run.set_defaults(func=cmd_run)

    p_live = sub.add_parser(
        "live-verdict",
        help="compare two sr-bench runs and record the live-tier three-state verdict",
    )
    p_live.add_argument("--spec", required=True, help="experiment spec YAML")
    p_live.add_argument(
        "--split",
        default="dev",
        choices=["calib", "dev", "frozen_test"],
        help="workload split the sr-bench runs were produced on (B)",
    )
    p_live.add_argument(
        "--workload",
        default=None,
        help="workload from spec.workloads the runs belong to (default: the first workload; "
        "transfer workloads need --final; M5)",
    )
    p_live.add_argument(
        "--final",
        action="store_true",
        help="promotion verdict; the only consumer allowed to cite the frozen_test split (B)",
    )
    p_live.add_argument(
        "--baseline-run-id", required=True, help="sr-bench baseline run ID"
    )
    p_live.add_argument(
        "--candidate-run-id", required=True, help="sr-bench candidate run ID"
    )
    p_live.add_argument(
        "--guard-measurements",
        default=None,
        help="guard measurements JSONL (name/candidate/baseline only; floors come from spec.gates, A3)",
    )
    p_live.add_argument(
        "--candidate", default=None, help="candidate JSON file for the ledger record"
    )
    p_live.add_argument(
        "--insight-refs",
        default=None,
        help="comma-separated provenance refs merged into the candidate: rec-<id>, bench:<run_id>, attr:<path>@sha256:<hex>, diag:<path>@sha256:<hex>#<anchor> (G2)",
    )
    p_live.add_argument(
        "--attribution-ref",
        default=None,
        help="attribution artifact this round is based on (path@hash); recorded in the ledger",
    )
    p_live.add_argument(
        "--ledger", default=None, help="ledger JSONL to append the live record to"
    )
    p_live.add_argument("--srbench-bin", default="vllm-sr")
    p_live.add_argument("--srbench-url", default=None)
    p_live.add_argument("--srbench-store", default=None)
    p_live.add_argument("--timeout", type=float, default=3600.0)
    p_live.add_argument("--round", type=int, default=1)
    p_live.set_defaults(func=cmd_live_verdict)

    p_verify = sub.add_parser(
        "verify",
        help="drive sr-bench end to end: preview + AB/BA repetitions -> live verdict (I)",
    )
    p_verify.add_argument("--spec", required=True, help="experiment spec YAML")
    p_verify.add_argument(
        "--split",
        default="dev",
        choices=["calib", "dev", "frozen_test"],
        help="workload split the sr-bench baseline run was produced on (B)",
    )
    p_verify.add_argument(
        "--workload",
        default=None,
        help="workload from spec.workloads to verify (default: the first workload; "
        "transfer workloads need --final; M5)",
    )
    p_verify.add_argument(
        "--final",
        action="store_true",
        help="promotion verdict; the only consumer allowed to cite the frozen_test split (B)",
    )
    p_verify.add_argument(
        "--baseline-run-id",
        required=False,
        default=None,
        help="terminal completed sr-bench baseline run ID (exactly one of "
        "--baseline-run-id / --baseline-manifest)",
    )
    p_verify.add_argument(
        "--baseline-manifest",
        default=None,
        help="sr-bench baseline manifest JSON to submit as the baseline run "
        "instead of citing an existing one (exactly one of --baseline-run-id / "
        "--baseline-manifest); a MoM-only manifest gets the spec's strong "
        "reference target injected as a single segment",
    )
    p_verify.add_argument(
        "--candidate", required=True, help="candidate JSON file for the ledger record"
    )
    p_verify.add_argument(
        "--candidate-targets",
        default=None,
        help="comma-separated sr-bench MoM target ids serving the candidate; falls back to the candidate JSON's srbench_targets",
    )
    p_verify.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="number of AB/BA pairs (default 3); each pair re-runs the baseline manifest and a fresh candidate-plan",
    )
    p_verify.add_argument(
        "--work-dir",
        default=None,
        help="scratch directory for the extracted baseline manifest and preview report (default: a temp dir)",
    )
    p_verify.add_argument(
        "--guard-measurements",
        default=None,
        help="guard measurements JSONL (name/candidate/baseline only; floors come from spec.gates, A3)",
    )
    p_verify.add_argument(
        "--insight-refs",
        default=None,
        help="comma-separated provenance refs merged into the candidate: rec-<id>, bench:<run_id>, attr:<path>@sha256:<hex>, diag:<path>@sha256:<hex>#<anchor> (G2)",
    )
    p_verify.add_argument(
        "--attribution-ref",
        default=None,
        help="attribution artifact this round is based on (path@hash); recorded in the ledger",
    )
    p_verify.add_argument(
        "--ledger", default=None, help="ledger JSONL to append the live record to"
    )
    p_verify.add_argument(
        "--run-cache",
        default=None,
        help=(
            "JSON file caching completed reference-arm run IDs (M4); default: "
            "next to the ledger when --ledger is given, else no cache"
        ),
    )
    p_verify.add_argument("--srbench-bin", default="vllm-sr")
    p_verify.add_argument("--srbench-url", default=None)
    p_verify.add_argument("--srbench-store", default=None)
    p_verify.add_argument("--timeout", type=float, default=3600.0)
    p_verify.add_argument("--round", type=int, default=1)
    p_verify.set_defaults(func=cmd_verify)

    # The agent's own proposal: one deployed candidate against a saved
    # baseline, through the installed sr-bench worker (sub-issue 3).
    p_score = sub.add_parser(
        "score-candidate",
        help="score one deployed candidate against a saved sr-bench baseline",
    )
    p_score.add_argument("--workspace", required=True, type=Path)
    p_score.add_argument(
        "--candidate", required=True, type=Path, help="authored v0.5 source YAML"
    )
    p_score.add_argument(
        "--deployed-config",
        type=Path,
        help="CLI-owned runtime-config[.STACK].yaml and adjacent provenance",
    )
    p_score.add_argument(
        "--proposer", required=True, help="agent:<id> identifying the proposer"
    )
    p_score.add_argument("--baseline-run", required=True)
    p_score.add_argument(
        "--baseline-target", required=True, help="explicit target in the baseline run"
    )
    p_score.add_argument(
        "--candidate-target",
        required=True,
        help="registered MoM with capture_recipe enabled",
    )
    p_score.add_argument(
        "--active-runtime-hash", required=True, help="expected active runtime SHA-256"
    )
    p_score.add_argument("--candidate-run", help="reuse a saved run without dispatch")
    p_score.add_argument(
        "--guard-case",
        action="append",
        default=[],
        help="protected case ID; default protects every case",
    )
    p_score.add_argument(
        "--min-speedup-pct",
        type=float,
        default=0,
        help="minimum reduction in summed common-success case time",
    )
    p_score.add_argument(
        "--wait-seconds",
        type=float,
        default=3600,
        help="time to wait for an active candidate",
    )
    p_score.add_argument(
        "--vllm-sr", default="vllm-sr", help="installed vllm-sr executable"
    )
    p_score.set_defaults(func=cmd_score_candidate)

    p_ledger = sub.add_parser("ledger", help="inspect or append to the evidence ledger")
    ledger_sub = p_ledger.add_subparsers(dest="ledger_action", required=True)

    p_show = ledger_sub.add_parser("show", help="list ledger records with filters")
    p_show.add_argument("--file", required=True, help="path to the ledger JSONL")
    p_show.add_argument("--tier", choices=["replay", "preview", "live"], default=None)
    p_show.add_argument(
        "--verdict", choices=["PASS", "FAIL", "INCONCLUSIVE", "RANKED"], default=None
    )
    p_show.add_argument("--round", type=int, default=None)
    p_show.add_argument("--proposer", default=None)
    p_show.add_argument(
        "--json", action="store_true", help="print full records as JSON"
    )
    p_show.set_defaults(func=cmd_ledger)

    p_append = ledger_sub.add_parser(
        "append", help="append one validated record to the ledger"
    )
    p_append.add_argument("--file", required=True, help="path to the ledger JSONL")
    p_append.add_argument(
        "--record", required=True, help="path to a JSON file containing one record"
    )
    p_append.add_argument(
        "--insight-refs",
        default=None,
        help="comma-separated provenance refs merged into the record: rec-<id>, bench:<run_id>, attr:<path>@sha256:<hex>, diag:<path>@sha256:<hex>#<anchor> (G2)",
    )
    p_append.set_defaults(func=cmd_ledger)

    p_top = ledger_sub.add_parser(
        "top", help="best replay-tier candidate (hand-off to live tier)"
    )
    p_top.add_argument("--file", required=True, help="path to the ledger JSONL")
    top_metric = p_top.add_mutually_exclusive_group(required=True)
    top_metric.add_argument(
        "--metric-key", help="objective metric key to rank by (ad-hoc)"
    )
    top_metric.add_argument(
        "--spec",
        help="experiment spec YAML; rank by its objective.replay_metric (A1)",
    )
    p_top.set_defaults(func=cmd_ledger)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SystemExit:
        raise
    except (ContractError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
