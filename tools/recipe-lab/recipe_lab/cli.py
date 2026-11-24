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
  suggest-depth           propose a Speed depth threshold from development evidence
  draft                   create a two-backend Speed recipe from declared cards (starter path)
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
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import NoReturn

import yaml

from . import __version__
from .adapters.router_replay import (
    fetch_replays,
    records_to_cases,
    records_to_rows,
    write_cases,
)
from .agentic_bootstrap import (
    CLIENTS,
    PROFILE_KIND,
    bootstrap_agentic_config,
    profile_from_reports,
)
from .agentic_driver import run_driver
from .attribution import (
    attribute,
    load_records,
    render_markdown,
    summarize,
    write_jsonl,
)
from .bank import apply_bank, build_bank, import_judged, load_bank, write_bank
from .bindings import run_bindings
from .bootstrap import (
    DEFAULT_LAMBDAS,
    DEFAULT_LISTENER_PORT,
    run_bootstrap,
)
from .bundle import build_bundle
from .classification import load_classification_source
from .contract import (
    DEFAULT_AGENTIC_GATES,
    RUNNABLE_SPLITS,
    ContractError,
    ExperimentSpec,
    Workload,
    check_runnable_split,
    dataset_content_hash,
)
from .diagnose import (
    DEFAULT_MIN_SAMPLE,
    DEFAULT_RARE_PCT,
    DEFAULT_UTILIZATION_WARN,
    build_diagnosis,
    load_diagnosis,
    load_diagnostic_rows,
    write_diagnosis,
)
from .insights import merge_insight_refs, parse_insight_refs, resolve_insight_refs
from .layout import select_recipe
from .ledger import Ledger, LedgerRecord
from .live import live_verdict_from_runs, measurements_from_rows, verify_live
from .model_eval_driver import verify_live_model_eval
from .model_eval_tasks import load_source_rows, rows_to_model_eval_tasks, write_tasks
from .next import write_next_context
from .observe_labels import (
    build_seed_config,
    observe_split,
    seed_config_fingerprint,
    write_labels,
    write_labels_summary,
    write_seed_config,
)
from .observe_lanes import (
    OBSERVE_MODEL_NAME,
    build_observation_config,
    observation_config_fingerprint,
    observe_lanes_split,
    publish_observation_snapshot,
    template_info,
    write_lanes,
    write_lanes_summary,
)
from .profile import results_dirname, run_profile
from .proposers import distribution_calibration, rule_threshold, threshold_search
from .proposers import session_policy as session_policy_proposer
from .provenance import promotion_candidate, write_run_manifest
from .report import render_promotion_draft, write_run_report
from .report_delivery import write_report
from .runner import (
    COST_PROXY_METRIC,
    HotReloadSession,
    RunnerEnv,
    RunnerError,
    RunnerTimeoutError,
    attribution_delta,
    candidate_tag,
    evaluate_candidate,
    evaluate_route_replay,
    replay_constraint_evidence,
    replay_metric_value,
    routed_cost_index,
)
from .session_attribution import write_session_report
from .session_cases import (
    DEFAULT_DUPLICATE_WARN_PCT,
    GRANULARITIES,
    cases_from_replays,
    load_route_cases,
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
from .swebench_judge import DEFAULT_JUDGE_COMMAND, judge_run


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


def _router_model_for_proposer(override: str | None, proposer: str | None) -> str:
    return override or (
        OBSERVE_MODEL_NAME if proposer == "bindings" else "vllm-sr/auto"
    )


def _select_split(
    spec: ExperimentSpec,
    split_name: str,
    workload: Workload | None = None,
    *,
    allow_frozen_test: bool = False,
) -> str:
    """B: resolve --split against the workload; frozen_test is quarantined and
    legacy single-dataset workloads only accept 'dev'. M5: `workload` is the
    selected Workload (default: the spec's first).

    Y1: observe-labels passes ``allow_frozen_test`` - the model_eval verify
    driver rewrites the frozen split's categories from the C2 labels file, so
    the seed config must be able to label frozen_test too. Observation through
    the seed config is labeling, never tuning: the profile only ever consumes
    the calib/dev labels, so the quarantine's purpose (no frozen_test feedback
    into candidate selection) still holds."""
    workload = workload if workload is not None else spec.workload
    try:
        if not allow_frozen_test:
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


def cmd_cases_from_replays(args: argparse.Namespace) -> int:
    """AG3: productize the J3 rebuild - replay records -> route-replay cases
    at an explicit granularity, dedup rules written into provenance."""
    try:
        records = [
            json.loads(line)
            for line in Path(args.records).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read records: {exc}") from exc
    granularity = args.granularity
    if granularity is None:
        # agentic sources default to the session granularity; records without
        # conversation identities fall back to the J3 prompt convention
        has_identity = any(
            isinstance(r, dict)
            and (
                isinstance(r.get("conversation_id"), str)
                or isinstance(r.get("session_id"), str)
            )
            and (r.get("conversation_id") or r.get("session_id"))
            for r in records
        )
        granularity = "session" if has_identity else "prompt"
    cases, provenance = cases_from_replays(
        records,
        granularity,
        duplicate_warn_pct=args.duplicate_warn_pct,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    provenance_path = (
        Path(args.provenance)
        if args.provenance
        else output.with_suffix(".provenance.json")
    )
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if provenance["duplicate_warning"]:
        print(
            f"[cases] WARNING: duplicate share {provenance['duplicate_pct']}% exceeds "
            f"{args.duplicate_warn_pct:g}% at granularity {granularity!r}; consider a "
            "coarser granularity",
            file=sys.stderr,
        )
    _print_json(
        {
            "records": str(args.records),
            "output": str(output),
            "provenance": str(provenance_path),
            "granularity": granularity,
            "cases_count": len(cases),
            "duplicate_pct": provenance["duplicate_pct"],
            "cases_sha256": "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    )
    return 0


def _load_json_objects(paths: list[str], what: str) -> list[dict]:
    payloads: list[dict] = []
    for path in paths:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"cannot read {what} {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ContractError(f"{what} {path} is not a JSON object")
        payloads.append(payload)
    return payloads


def cmd_profile_agentic(args: argparse.Namespace) -> int:
    """AG5: aggregate per-model single-target sr-bench run reports into the
    agentic profile (recipe-lab-agentic-profile-v1)."""
    reports = _load_json_objects(args.reports, "run report")
    try:
        profile = profile_from_reports(reports)
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _print_json(
        {
            "profile": str(output),
            "kind": PROFILE_KIND,
            "models": [m["model"] for m in profile["models"]],
            "profile_sha256": "sha256:"
            + hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    )
    return 0


def cmd_bootstrap_agentic(args: argparse.Namespace) -> int:
    """AG5: generate an agentic candidate config from the agent recipe
    template, the profile and the client identity contract; every lane's
    choice rides a sidecar with its rule and evidence."""
    spec = _load_spec(args.spec)
    template_path = args.template
    if template_path is None:
        # the shipped agent recipe, resolved from the repository checkout
        template_path = str(
            Path(__file__).resolve().parents[3]
            / "config"
            / "recipes"
            / "agent"
            / "config.yaml"
        )
    profile_payload = _load_json_objects([args.profile], "profile")[0]
    try:
        template = yaml.safe_load(Path(template_path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot read template {template_path}: {exc}") from exc
    if not isinstance(template, dict):
        raise ContractError(f"template {template_path} is not a YAML mapping")
    local_models = list(spec.local_models)
    for raw in args.local_model or []:
        local_models.extend(m.strip() for m in raw.split(",") if m.strip())
    try:
        config, sidecar = bootstrap_agentic_config(
            template,
            profile_payload,
            client=args.client,
            local_models=local_models,
            lam=args.lam,
            recipe=args.recipe,
            public_model=args.public_model,
        )
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "agentic-config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    sidecar_path = output_dir / "agentic-config.bootstrap.json"
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rejection = None
    if args.validate:
        rejection = _validate_config(config_path, args.validate_bin)
        if rejection is not None:
            _fail(f"agentic candidate failed config validate: {rejection}")
    _print_json(
        {
            "config": str(config_path),
            "sidecar": str(sidecar_path),
            "client": args.client,
            "recipe": sidecar["recipe"],
            "lanes": {lane["decision"]: lane["model"] for lane in sidecar["lanes"]},
            "validated": args.validate and rejection is None,
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


def cmd_attribute_session(args: argparse.Namespace) -> int:
    """Session-aware agentic attribution over graded records (Sub-issue 6)."""
    try:
        rows = [
            json.loads(line)
            for line in Path(args.records).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot load attribution records: {exc}")
    summary = write_session_report(rows, args.output_dir)
    _print_json(summary)
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


def cmd_diagnose(args: argparse.Namespace) -> int:
    """D: three no-ground-truth checks over unlabeled replay rows — decision
    coverage (dead / rare), signal distributions (quantiles / histogram /
    range utilization) and threshold reachability. Writes diagnosis.json
    (recipe-lab-diagnosis-v1, input-hashed, byte-stable) and diagnosis.md."""
    rows = load_diagnostic_rows(args.records)
    try:
        with Path(args.config).open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"cannot read config: {exc}") from exc
    if not isinstance(config, dict):
        raise SystemExit(f"config is not a YAML object: {args.config}")
    expect_rare = [
        part.strip() for part in str(args.expect_rare).split(",") if part.strip()
    ]
    agent_requests = None
    if args.agent_requests:
        try:
            with Path(args.agent_requests).open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read agent-request manifest: {exc}") from exc
        entries = manifest.get("agent_requests") if isinstance(manifest, dict) else None
        if not isinstance(entries, list):
            raise SystemExit(
                'agent-request manifest must hold {"agent_requests": [...]}: '
                f"{args.agent_requests}"
            )
        agent_requests = entries
    diagnosis = build_diagnosis(
        rows,
        config,
        rows_path=str(args.records),
        rows_hash=dataset_content_hash(args.records),
        config_path=str(args.config),
        config_hash=dataset_content_hash(args.config),
        source=args.source,
        expect_rare=expect_rare,
        rare_pct=args.rare_pct,
        utilization_warn=args.utilization_warn,
        min_sample=args.min_sample,
        agent_requests=agent_requests,
        recipe=args.recipe,
        public_model=args.public_model,
    )
    json_path, md_path = write_diagnosis(diagnosis, args.output_dir)
    _print_json(
        {
            "diagnosis": str(json_path),
            "report": str(md_path),
            "recipe": diagnosis["inputs"]["recipe"],
            "samples": len(rows),
            "dead_decisions": diagnosis["dead_decisions"],
            "rare_decisions": diagnosis["rare_decisions"],
            "unreachable_sites": diagnosis["unreachable_sites"],
            "alerts": diagnosis["alerts"],
        }
    )
    return 0


def cmd_cases_to_model_eval(args: argparse.Namespace) -> int:
    """C1: convert an MCQ source into model_eval's tasks format.

    Deterministic (same source => byte-identical output), so the printed
    tasks_sha256 can be frozen into the contract like the sr-bench hash.
    """
    try:
        rows = load_source_rows(args.source)
    except OSError as exc:
        raise ContractError(f"cannot read source: {exc}") from exc
    tasks = rows_to_model_eval_tasks(rows, default_dataset=args.dataset)
    output = write_tasks(tasks, args.out)
    _print_json(
        {
            "source": str(args.source),
            "output": str(output),
            "tasks_count": len(tasks),
            "tasks_sha256": dataset_content_hash(output),
        }
    )
    return 0


def cmd_observe_labels(args: argparse.Namespace) -> int:
    """C2: observe the router's own domain labels for one split.

    Generates the seed config (14 domain decisions + catch-all, decision
    model pinned), boots the stack via the runner with tagged request ids,
    and writes labels-<split>.jsonl (case_id/router_label/confidence) plus a
    summary carrying the label coverage the provenance records.
    """
    spec = _load_spec(args.spec)
    # B/M5: the same split resolution as run/attribute - except frozen_test,
    # which the C2 labeler may observe (Y1: the model_eval verify driver
    # rewrites the frozen split's task categories from these labels; the
    # profile never sees them, so the quarantine still holds).
    workload = spec.require_tunable_workload(getattr(args, "workload", None))
    split_name = _select_split(spec, args.split, workload, allow_frozen_test=True)
    # A4: re-hash the selected split before touching it.
    spec.verify_workload(split=split_name, workload=workload.name)
    if spec.spec_dir is None:
        raise ContractError("spec must be loaded from a file to resolve its manifest")
    cases_path = Path(spec.spec_dir) / workload.split_manifest(split_name)
    try:
        cases = load_cases(cases_path)
    except OSError as exc:
        raise ContractError(f"cannot read split manifest: {exc}") from exc

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Y2: the seed classifies with the same classification sections every
    # bootstrap candidate will carry - one explicit source config.
    classification = load_classification_source(args.classification_config)
    seed_config = build_seed_config(
        backend_endpoint=f"127.0.0.1:{args.backend_port}",
        classification=classification,
    )
    config_path = write_seed_config(seed_config, out_dir / "seed-config.yaml")

    env = RunnerEnv(
        router_bin=args.router_bin,
        mock_backend_script=args.mock_backend,
        python_bin=args.python_bin,
        work_dir=out_dir / "work",
        backend_port=args.backend_port,
        ready_timeout_s=args.ready_timeout,
        hf_home=args.hf_home,
        mock_extra_args=(
            tuple(args.mock_extra_args.split()) if args.mock_extra_args else ()
        ),
    )
    # A5: the per-trial budget bounds readiness wait + requests + fetch.
    trial_timeout_s = spec.budget.max_minutes_per_trial * 60
    try:
        result = observe_split(
            env=env,
            config=config_path,
            cases=cases,
            router_model=args.router_model,
            request_timeout=args.request_timeout,
            trial_timeout_s=trial_timeout_s,
        )
    except RunnerTimeoutError as exc:
        raise SystemExit(f"observation exceeded the trial budget: {exc}") from exc
    except (RunnerError, ContractError) as exc:
        raise SystemExit(f"observation failed: {exc}") from exc

    labels_path = write_labels(result["rows"], out_dir / f"labels-{split_name}.jsonl")
    summary = {
        **result["summary"],
        "split": split_name,
        "workload": workload.name,
        "cases_file": str(cases_path),
        "cases_sha256": workload.split_hash(split_name),
        "seed_config": str(config_path),
        "seed_config_sha256": seed_config_fingerprint(config_path),
        "classification_source": str(args.classification_config),
        "classification_source_sha256": dataset_content_hash(
            args.classification_config
        ),
        "join_stats": result["stats"],
        "timings": result["timings"],
    }
    summary_path = write_labels_summary(
        summary, out_dir / f"labels-{split_name}-summary.json"
    )
    _print_json(
        {
            "labels": str(labels_path),
            "summary": str(summary_path),
            "cases": summary["cases"],
            "observed": summary["observed"],
            "missing": summary["missing"],
            "label_coverage": summary["label_coverage"],
            "label_counts": summary["label_counts"],
        }
    )
    return 0


def cmd_observe_lanes(args: argparse.Namespace) -> int:
    """C2 (template path): observe which template lane each case hits.

    Reads the template's lanes via ``vllm-sr recipe builtin list``, builds
    the observation config via ``recipe builtin init`` (template bound to
    placeholder models; the bundle's pinned decision model decides the
    lane), boots the stack with tagged request ids, and writes
    lanes-<split>.jsonl (case_id/lane/confidence) plus a summary carrying
    the lane coverage.
    """
    spec = _load_spec(args.spec)
    if spec.template is None:
        raise ContractError(
            "spec declares no template section: observe-lanes generates the "
            "observation config from a built-in recipe template (sub-issue-8 "
            "§2.1; the legacy path is observe-labels)"
        )
    # M5/Y1: observe-lanes is labeling, never tuning. A transfer workload may
    # be observed only on frozen_test (the final verify's grouping source);
    # the profile/bank/run commands keep their tunable guards, so no tuning
    # feedback path opens. frozen_test is allowed for the same reason
    # observe-labels allows it: the model_eval verify driver rewrites the
    # frozen split's task categories from the C2 lanes file, and the profile
    # only ever consumes calib/dev lanes.
    workload = spec.select_workload(getattr(args, "workload", None))
    if workload.role == "transfer" and args.split != "frozen_test":
        raise ContractError(
            f"workload {workload.name!r} has role 'transfer': observe-lanes "
            "accepts it only on --split frozen_test (the final verify's "
            "grouping source)"
        )
    split_name = _select_split(spec, args.split, workload, allow_frozen_test=True)
    spec.verify_workload(split=split_name, workload=workload.name)
    if spec.spec_dir is None:
        raise ContractError("spec must be loaded from a file to resolve its manifest")
    cases_path = Path(spec.spec_dir) / workload.split_manifest(split_name)
    try:
        cases = load_cases(cases_path)
    except OSError as exc:
        raise ContractError(f"cannot read split manifest: {exc}") from exc

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    template = template_info(
        bundle=spec.template.bundle,
        recipe=spec.template.recipe,
        vllm_sr_bin=args.vllm_sr_bin,
        catalog_version=args.catalog_version,
        timeout_s=args.init_timeout,
    )
    # builtin init exclusively creates its output. Keep the evaluated
    # config and router logs for each attempt, including repeated splits.
    run_dir = Path(tempfile.mkdtemp(prefix=f"observe-{split_name}-", dir=out_dir))
    try:
        observation = build_observation_config(
            template=template,
            out_dir=run_dir,
            backend_endpoint=f"127.0.0.1:{args.backend_port}",
            vllm_sr_bin=args.vllm_sr_bin,
            model_name=args.model_name,
            listener_port=args.listener_port,
            timeout_s=args.init_timeout,
        )
    except ContractError as exc:
        raise SystemExit(f"observation config generation failed: {exc}") from exc

    env = RunnerEnv(
        router_bin=args.router_bin,
        mock_backend_script=args.mock_backend,
        python_bin=args.python_bin,
        work_dir=run_dir / "work",
        backend_port=args.backend_port,
        ready_timeout_s=args.ready_timeout,
        hf_home=args.hf_home,
        mock_extra_args=(
            tuple(args.mock_extra_args.split()) if args.mock_extra_args else ()
        ),
    )
    trial_timeout_s = spec.budget.max_minutes_per_trial * 60
    try:
        result = observe_lanes_split(
            env=env,
            config=observation["config"],
            cases=cases,
            router_model=args.model_name,
            request_timeout=args.request_timeout,
            trial_timeout_s=trial_timeout_s,
        )
    except RunnerTimeoutError as exc:
        raise SystemExit(f"lane observation exceeded the trial budget: {exc}") from exc
    except (RunnerError, ContractError) as exc:
        raise SystemExit(f"lane observation failed: {exc}") from exc

    lanes_path = write_lanes(result["rows"], out_dir / f"lanes-{split_name}.jsonl")
    summary = {
        **result["summary"],
        "split": split_name,
        "workload": workload.name,
        "cases_file": str(cases_path),
        "cases_sha256": workload.split_hash(split_name),
        "observation_config": str(observation["config"]),
        "observation_config_sha256": observation_config_fingerprint(
            observation["config"]
        ),
        "template": {
            "bundle": template["bundle"],
            "recipe": template["recipe"],
            "sha256": template["sha256"],
            "catalog_version": template["catalog_version"],
        },
        "init_command": observation["init_command"],
        "cache_disabled": observation["cache_disabled"],
        "join_stats": result["stats"],
        "timings": result["timings"],
    }
    summary_path = write_lanes_summary(
        summary, out_dir / f"lanes-{split_name}-summary.json"
    )
    publish_observation_snapshot(out_dir, run_dir)
    _print_json(
        {
            "lanes": str(lanes_path),
            "summary": str(summary_path),
            "observation_config": str(observation["config"]),
            "cases": summary["cases"],
            "observed": summary["observed"],
            "missing": summary["missing"],
            "lane_coverage": summary["lane_coverage"],
            "lane_counts": summary["lane_counts"],
        }
    )
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    """C3: per-model profiles grouped by labels (legacy) or template lanes."""
    spec = _load_spec(args.spec)
    labels = getattr(args, "labels", None)
    lanes = getattr(args, "lanes", None)
    if (labels is None) == (lanes is None):
        raise ContractError(
            "profile needs exactly one grouping source: --labels (legacy "
            "domain-template path) or --lanes (template lane mode)"
        )
    if lanes is not None and spec.template is None:
        raise ContractError(
            "--lanes requires the spec's template section (lane mode is the "
            "template path; sub-issue-8 §2.1)"
        )
    workload = spec.require_tunable_workload(getattr(args, "workload", None))
    if spec.spec_dir is None:
        raise ContractError("spec must be loaded from a file to resolve its manifest")
    if not spec.models:
        raise ContractError(
            "spec declares no models section: profile needs the candidate "
            "models and their direct endpoints"
        )
    if spec.generation is None:
        raise ContractError(
            "spec declares no generation section: profile needs the shared "
            "generation params"
        )
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise ContractError("--splits is empty")
    for split in splits:
        if split not in RUNNABLE_SPLITS:
            raise ContractError(
                f"split {split!r} is not profileable: profile runs calib/dev only "
                "(frozen_test is quarantined; calib feeds bootstrap, dev the bank)"
            )
    model_eval_dir = Path(args.model_eval_dir)
    if not model_eval_dir.is_dir():
        raise ContractError(f"model_eval directory not found: {model_eval_dir}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_paths: dict[str, Path] = {}
    cases_hashes: dict[str, str] = {}
    for split in splits:
        cases_hashes[split] = spec.verify_workload(split=split, workload=workload.name)
        cases_paths[split] = Path(spec.spec_dir) / workload.split_manifest(split)

    # ContractError propagates to main() -> exit 2.
    profile_path = run_profile(
        spec=spec,
        cases_paths=cases_paths,
        cases_hashes=cases_hashes,
        labels_dir=Path(labels) if labels else None,
        lanes_dir=Path(lanes) if lanes else None,
        splits=splits,
        out_dir=out_dir,
        model_eval_dir=model_eval_dir,
        python_bin=args.python_bin,
        concurrency=args.concurrency,
        request_timeout=args.request_timeout,
        force=args.force,
    )
    _print_json(
        {
            "profile": str(profile_path),
            "splits": splits,
            "models": [model.name for model in spec.models],
            "grouping": "lanes" if lanes else "labels",
            "min_label_cases": spec.profile_config.min_label_cases,
            "min_lane_cases": spec.bootstrap_config.min_lane_cases,
        }
    )
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    """C4+C5 (legacy domain-template path): per-lambda scaffold via
    result_to_config, completed to candidates.

    This is the compatibility path (sub-issue-8 §2.5): it only runs with
    the explicit ``--legacy-domain-template`` flag and is not part of the
    cold-start completion criteria. The template path is ``bindings``.
    """
    if not getattr(args, "legacy_domain_template", False):
        raise ContractError(
            "bootstrap is the legacy domain-template path (result_to_config "
            "scaffold); pass --legacy-domain-template to run it explicitly, "
            "or use the template path: observe-lanes + profile --lanes + "
            "bindings (sub-issue-8 §2.5)"
        )
    spec = _load_spec(args.spec)
    if not spec.models:
        raise ContractError(
            "spec declares no models section: bootstrap needs the candidate "
            "models and their direct endpoints"
        )
    model_eval_dir = Path(args.model_eval_dir)
    if not model_eval_dir.is_dir():
        raise ContractError(f"model_eval directory not found: {model_eval_dir}")
    profile_dir = Path(args.profile)
    if not (profile_dir / "profile.json").is_file():
        raise ContractError(
            f"profile not found: {profile_dir / 'profile.json'} (bootstrap "
            "consumes the C3 profile directory)"
        )
    # Y2 ①: one explicit classification source for the observe seed and every
    # candidate; the classification sections are injected into both.
    classification = load_classification_source(args.classification_config)
    # Y2 ②: the label-agreement check observes the dev split (never calib,
    # which bootstrap consumes, and never frozen_test, which is quarantined).
    workload = spec.require_tunable_workload(getattr(args, "workload", None))
    if spec.spec_dir is None:
        raise ContractError("spec must be loaded from a file to resolve its manifest")
    if "dev" not in workload.splits:
        raise ContractError(
            f"workload {workload.name!r} declares no dev split: the bootstrap "
            "label-agreement check observes dev (declare it in the spec)"
        )
    spec.verify_workload(split="dev", workload=workload.name)
    cases_path = Path(spec.spec_dir) / workload.split_manifest("dev")
    try:
        dev_cases = load_cases(cases_path)
    except OSError as exc:
        raise ContractError(f"cannot read dev manifest: {exc}") from exc
    if not dev_cases:
        raise ContractError(f"dev split is empty: {cases_path}")
    env = RunnerEnv(
        router_bin=args.router_bin,
        mock_backend_script=args.mock_backend,
        python_bin=args.python_bin,
        work_dir=Path(args.out) / "label-check",
        backend_port=args.backend_port,
        ready_timeout_s=args.ready_timeout,
        hf_home=args.hf_home,
        mock_extra_args=(
            tuple(args.mock_extra_args.split()) if args.mock_extra_args else ()
        ),
    )
    result = run_bootstrap(
        spec=spec,
        profile_dir=profile_dir,
        model_eval_dir=model_eval_dir,
        lambdas=args.lambdas,
        out_dir=Path(args.out),
        python_bin=args.python_bin,
        validate_bin=args.validate_bin,
        classification=classification,
        classification_path=args.classification_config,
        dev_cases=dev_cases,
        env=env,
        request_timeout=args.request_timeout,
        listener_port=args.listener_port,
    )
    _print_json(result)
    return 0


def cmd_bindings(args: argparse.Namespace) -> int:
    """C4+C5 (template path): per-lane bindings + candidate configs.

    Runs the routing-space check, plans the per-lane bindings for every
    lambda (capability filter, accuracy - lambda * normalized cost,
    minimum_candidates, unmeasured lanes by the default rule), generates
    each candidate via ``recipe builtin init`` over the spec's
    providers_config, and validates it. A ``no_routing_space`` verdict ends
    the command with the report and no candidates (sub-issue-8 §2.2).
    """
    spec = _load_spec(args.spec)
    if not spec.models:
        raise ContractError(
            "spec declares no models section: bindings needs the candidate "
            "models, their direct endpoints and their declared capabilities"
        )
    profile_dir = Path(args.profile)
    if not (profile_dir / "profile.json").is_file():
        raise ContractError(
            f"profile not found: {profile_dir / 'profile.json'} (bindings "
            "consumes the C3 profile directory)"
        )
    result = run_bindings(
        spec=spec,
        profile_dir=profile_dir,
        out_dir=Path(args.out),
        lambdas_raw=args.lambdas,
        exclude=tuple(args.exclude_decision or ()),
        vllm_sr_bin=args.vllm_sr_bin,
        catalog_version=args.catalog_version,
        model_name=args.model_name,
        timeout_s=args.init_timeout,
    )
    _print_json(result)
    return 0


def cmd_bundle(args: argparse.Namespace) -> int:
    """C9: package final-verify evidence into a reviewable bundle."""
    revisions: dict[str, str] = {}
    for item in args.model_revision:
        name, sep, revision = item.partition("=")
        if not sep or not name.strip() or not revision.strip():
            raise ContractError(f"--model-revision must be NAME=REVISION, got {item!r}")
        revisions[name.strip()] = revision.strip()
    spec = _load_spec(args.spec)
    result = build_bundle(
        spec=spec,
        ledger_path=args.ledger,
        record_id=args.record,
        out_dir=args.out,
        config_path=args.config,
        skeleton_path=args.skeleton,
        bindings_path=args.bindings,
        profile_path=args.profile,
        validate_bin=args.validate_bin,
        router_commit=args.router_commit,
        model_revisions=revisions,
        model_eval_script=args.model_eval_script,
        candidate_sidecar=args.candidate_sidecar,
        extra_private_patterns=list(args.private_pattern),
    )
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

    if args.bank_action == "import":
        # C6: convert the profile's direct-model dev judged.jsonl files into
        # the answer bank format (no endpoint traffic - the evidence is the
        # profile run itself).
        try:
            spec = _load_spec(args.spec)
            if not spec.models:
                raise ContractError("spec declares no models: nothing to import")
            split = str(args.split)
            if split not in RUNNABLE_SPLITS:
                raise ContractError(
                    f"split {split!r} is not importable: profile runs "
                    f"{sorted(RUNNABLE_SPLITS)} only"
                )
            profile_dir = Path(args.profile_dir)
            stats: dict[str, object] = {}
            entries: list = []
            for model in spec.models:
                judged_path = (
                    profile_dir
                    / split
                    / "results"
                    / results_dirname(model.name)
                    / "judged.jsonl"
                )
                if not judged_path.is_file():
                    raise ContractError(
                        f"no profile judged.jsonl for model {model.name!r}: "
                        f"{judged_path} (run recipe-lab profile with this split first)"
                    )
                judged_rows = [
                    json.loads(line)
                    for line in judged_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                model_entries = import_judged(model.name, judged_rows)
                entries.extend(model_entries)
                stats[model.name] = {
                    "cases": len(model_entries),
                    "correct": sum(1 for e in model_entries if e.answer_ok),
                }
            out = write_bank(entries, args.output)
            _print_json(
                {
                    "entries": len(entries),
                    "models": stats,
                    "split": split,
                    "output": str(out),
                }
            )
        except (ContractError, OSError, json.JSONDecodeError) as exc:
            _fail(f"bank import error: {exc}")
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


def cmd_propose_calibrated(args: argparse.Namespace) -> int:
    """D: rule threshold candidates at observed distribution quantiles (or at
    a target traffic share). The diagnosis must have been produced over the
    same base config (hash-checked, exit 2 otherwise); each sidecar cites the
    diagnosis it came from."""
    if args.quantiles and args.target_share is not None:
        raise SystemExit(
            "propose-calibrated: --quantiles and --target-share are mutually exclusive"
        )
    quantiles = (
        [float(part) for part in str(args.quantiles).split(",") if part.strip()]
        if args.quantiles
        else None
    )
    if quantiles is None and args.target_share is None:
        raise SystemExit(
            "propose-calibrated: one of --quantiles or --target-share is required"
        )
    if not args.output_dir:
        raise SystemExit("propose-calibrated: --output-dir is required")
    try:
        diagnosis = load_diagnosis(args.diagnosis)
        diagnosis_hash = dataset_content_hash(args.diagnosis)
        config_hash = dataset_content_hash(args.base_config)
        with Path(args.base_config).open("r", encoding="utf-8") as fh:
            base = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"cannot read diagnosis or base config: {exc}") from exc
    try:
        proposals = distribution_calibration.propose_calibrated(
            diagnosis,
            base,
            args.signal,
            diagnosis_ref=f"diag:{args.diagnosis}@{diagnosis_hash}",
            config_hash=config_hash,
            quantiles=quantiles,
            target_share=args.target_share,
            from_value=args.from_value,
            op=args.op,
            lo=args.lo,
            hi=args.hi,
        )
        paths = distribution_calibration.write_calibrated_candidates(
            proposals, args.output_dir
        )
    except ContractError as exc:
        raise SystemExit(f"proposer error: {exc}") from exc
    _print_json(
        {
            "signal": args.signal,
            "candidates": [p.basis for p in proposals],
            "insight_refs": [list(p.insight_refs) for p in proposals],
            "files": [str(p) for p in paths],
        }
    )
    return 0


def cmd_propose_session_policy(args: argparse.Namespace) -> int:
    """AG6: diagnosis-triggered one-variable session-policy candidates."""
    try:
        base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"cannot read base config: {exc}") from exc
    if not isinstance(base, dict):
        raise SystemExit("propose-session-policy: base config is not a YAML mapping")
    diagnosis = None
    if args.diagnosis:
        diagnosis = _load_json_objects([args.diagnosis], "diagnosis")[0]
    summary = None
    if args.replay_summary:
        summary = _load_json_objects([args.replay_summary], "route-replay summary")[0]
    try:
        candidates = session_policy_proposer.propose_session_policy(
            base, diagnosis=diagnosis, replay_summary=summary
        )
        paths = session_policy_proposer.write_candidates(candidates, args.output_dir)
    except ValueError as exc:
        raise SystemExit(f"proposer error: {exc}") from exc
    _print_json(
        {
            "candidates": [c.summary() for c in candidates],
            # Z6: causes that must not become protection knobs (drift
            # calibration, candidate-set/lock checks, client-side identity)
            # ride the report as handoffs instead of config candidates
            "handoffs": session_policy_proposer.session_policy_handoffs(
                base, replay_summary=summary
            ),
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
    "bindings",
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
    """C: returns None when the config validates, else the rejection reason.

    --offline pins the gate to the CLI's structural rules: the default
    Router verdict depends on whichever image happens to sit on the
    machine, and a stale one rejects canonical configs.
    """
    try:
        proc = subprocess.run(
            [validate_bin, "config", "validate", "--offline", "--config", str(path)],
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


def cmd_run_route_replay(args: argparse.Namespace) -> int:
    """AG3: the route-replay tier - recorded sessions replayed per candidate
    for routing evidence only (RANKED or gate-violated records; no PASS/FAIL
    and no quality metric, because the recorded history cannot change with
    the candidate)."""
    spec = _load_spec(args.spec)
    if args.round > spec.budget.rounds:
        raise ContractError(
            f"budget exhausted: round {args.round} > budget.rounds {spec.budget.rounds}"
        )
    cases = load_route_cases(args.cases)
    cases_hash = "sha256:" + hashlib.sha256(Path(args.cases).read_bytes()).hexdigest()
    candidates = _resolve_candidates(args.candidates)
    if not candidates:
        _fail(f"no candidate configs found: {args.candidates}")
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
    output_dir = Path(args.output_dir)
    ledger = Ledger(args.ledger)
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
        reload_mode="restart",
        api_key_env=args.api_key_env,
    )
    trial_timeout_s = spec.budget.max_minutes_per_trial * 60
    agentic_workload = next((w for w in spec.workloads if w.kind == "agentic"), None)
    gates = (
        agentic_workload.resolved_agentic_gates()
        if agentic_workload is not None
        else DEFAULT_AGENTIC_GATES
    )
    cost_model = spec.cost_model
    strong_model = getattr(args, "strong_model", None)

    baseline_events: dict[str, int] | None = None
    if args.baseline_config:
        print(f"[run] route-replay baseline: {args.baseline_config}", file=sys.stderr)
        baseline = evaluate_route_replay(
            candidate_config=args.baseline_config,
            cases=cases,
            env=env,
            router_model=args.router_model,
            request_timeout=180.0,
            trial_timeout_s=trial_timeout_s,
            request_id_prefix=f"rl-rt-{candidate_tag(str(args.baseline_config))}-",
            gates=gates,
            cost_model=cost_model,
            strong_model=strong_model,
        )
        baseline_events = {
            "selection_failed": baseline["summary"]["selection_rejected"],
            "tool_loop_interrupted": baseline["summary"]["tool_loop_interrupted"],
        }

    appended = []
    for index, candidate_path in enumerate(candidates, 1):
        print(
            f"[run] candidate {index}/{len(candidates)}: {candidate_path}",
            file=sys.stderr,
        )
        entry = _candidate_entry(candidate_path, args.proposer, index, len(candidates))
        tag = candidate_tag(str(candidate_path))
        try:
            result = evaluate_route_replay(
                candidate_config=candidate_path,
                cases=cases,
                env=env,
                router_model=args.router_model,
                trial_timeout_s=trial_timeout_s,
                request_id_prefix=f"rl-rt-{tag}-",
                gates=gates,
                baseline_events=baseline_events,
                cost_model=cost_model,
                strong_model=strong_model,
            )
        except RunnerTimeoutError as exc:
            ledger.append(
                LedgerRecord(
                    round=args.round,
                    ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    candidate=entry,
                    tier="route_replay",
                    baseline_ref=spec.stack.baseline_ref if spec.stack else "unknown",
                    metrics={},
                    verdict="RANKED",
                    dataset_hash=cases_hash,
                    reasons=[f"trial timeout: {exc}"],
                    gates=dict.fromkeys(gates, "not_evaluated"),
                    bench_refs=[],
                    coverage_warning=True,
                    notes="",
                    status="timeout",
                )
            )
            print(
                f"[run] {candidate_path.name}: TRIAL TIMEOUT -> ledger", file=sys.stderr
            )
            continue
        except (RunnerError, ContractError, OSError) as exc:
            _fail(f"route-replay evaluation failed for {candidate_path}: {exc}")
        summary = result["summary"]
        gate_results = result["gate_results"]
        gate_status = {
            g.name: (
                "pass" if g.passed else ("incomplete" if not g.evaluated else "fail")
            )
            for g in gate_results
        }
        violated = any(g.evaluated and not g.passed for g in gate_results)
        reasons = list(result["reasons"])
        for g in gate_results:
            if g.evaluated and not g.passed:
                reasons.append(
                    f"agentic stability gate exceeded ({g.name}: {g.value:g} > bound {g.bound:g})"
                )
        metrics: dict = {}
        if summary["cost_index"] is not None:
            metrics[COST_PROXY_METRIC] = summary["cost_index"]
        record = LedgerRecord(
            round=args.round,
            ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            candidate=entry,
            tier="route_replay",
            baseline_ref=spec.stack.baseline_ref if spec.stack else "unknown",
            metrics=metrics,
            verdict="RANKED",
            dataset_hash=cases_hash,
            reasons=reasons,
            gates=gate_status,
            bench_refs=[],
            coverage_warning=any(
                status == "incomplete" for status in gate_status.values()
            ),
            notes="routing-evidence="
            + json.dumps(summary, ensure_ascii=False, sort_keys=True),
            status="gate-violated" if violated else None,
        )
        ledger.append(record)
        # the bare-path candidate form carries no name; fall back to the
        # config file's stem for the recorded-candidates summary
        appended.append(
            entry.get("name") or Path(entry["ref"]).stem
            if isinstance(entry, dict)
            else str(entry)
        )
        marker = "GATE-VIOLATED" if violated else "RANKED"
        print(f"[run] {candidate_path.name}: {marker} -> ledger", file=sys.stderr)
    _print_json(
        {
            "tier": "route_replay",
            "evaluated": len(candidates),
            "rejected": rejected,
            "recorded": appended,
        }
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    args.router_model = _router_model_for_proposer(args.router_model, args.proposer)
    if args.tier == "route-replay":
        return cmd_run_route_replay(args)
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
        replay_tags = {
            key: result[key]
            for key in ("candidate_tag", "request_tag")
            if key in result
        }
        notes = f"statuses={json.dumps(result['statuses'], ensure_ascii=False)} reload={reload_used}"
        if result.get("reload") == "restart-fallback":
            # H: the PUT was rejected; the ledger must say why this candidate
            # was evaluated on a fresh router instead of the live session.
            notes += f" fallback_reason={result.get('fallback_reason', '')}"

        if result.get("incomplete"):
            # Missing, duplicated and rejected requests all preclude grading.
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
                replay_failures=result.get("replay_failures", []),
                **replay_tags,
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
                    "replay_failures": result.get("replay_failures", []),
                    **replay_tags,
                }
            )
            print(
                f"[run] {candidate_path.name}: INCOMPLETE replay capture -> ledger (status=incomplete): {result['incomplete_reason']}",
                file=sys.stderr,
            )
            for failure in result.get("replay_failures", []):
                print(
                    f"[run] replay failure {json.dumps(failure, ensure_ascii=False)}",
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
            **replay_tags,
        )
        stored = ledger.append(record)
        appended.append(stored)
        run_results.append(
            {
                "path": str(candidate_path),
                "reload": reload_used,
                "status": "ranked",
                "timings": result.get("timings"),
                **replay_tags,
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
    # cite it. M5: a transfer workload is final-verify only (the frozen_test
    # rule applies to the whole workload).
    workload = spec.select_workload(getattr(args, "workload", None))
    workload.check_live_split(args.split, final=args.final)
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
            workload=workload,
        )
    except (ContractError, RuntimeError) as exc:
        _fail(f"live verdict error: {exc}")

    if args.ledger:
        record = LedgerRecord(
            round=args.round,
            ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            candidate=(
                promotion_candidate(candidate_row) if args.final else candidate_row
            ),
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
            agentic=result.get("agentic"),
            verification={
                "baseline": result["metrics"]["baseline"],
                "guard_measurements": measurements,
                "evidence_quality": result.get("evidence_quality"),
            },
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
    workload = spec.select_workload(getattr(args, "workload", None))
    workload.check_live_split(args.split, final=args.final)
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
    # M4: reference-arm run IDs are cached so repeated verify invocations do
    # not re-run the (expensive) reference arms; default location follows the
    # ledger when present, else the work dir.
    run_cache = args.run_cache or (
        Path(args.ledger).parent / "run-id-cache.json" if args.ledger else None
    )
    work_dir = (
        Path(args.work_dir)
        if args.work_dir
        else Path(tempfile.mkdtemp(prefix="rl-verify-"))
    )
    if workload.driver == "model_eval":
        # C7: the model_eval driver replaces sr-bench end to end - every arm
        # is a multi_benchmark_eval + mcq_judge pass (router arms hit the
        # router endpoints, reference arms the models' direct endpoints) and
        # the per-case correctness feeds the same M2/M3/M4 machinery.
        if args.baseline_run_id or args.baseline_manifest:
            raise ContractError(
                "the model_eval driver takes no sr-bench baseline: the "
                "baseline arm is --baseline-router-endpoint"
            )
        # Y3: the arms answer the contract-declared `model_eval_tasks`
        # rendering of the split - never an arbitrary file. verify_workload
        # above re-hashed the declared rendering; here the CLI file must be
        # byte-identical to it, and the C2 labels file drives the in-driver
        # category rewrite.
        if not args.tasks:
            raise ContractError(
                "the model_eval driver needs --tasks (the contract-declared "
                "model_eval_tasks rendering for this split)"
            )
        declared_tasks_hash = workload.split_model_eval_tasks_hash(args.split)
        if declared_tasks_hash is None:
            raise ContractError(
                f"workload {workload.name!r} split {args.split!r} must declare "
                "model_eval_tasks and model_eval_tasks_sha256 (the "
                "`cases to-model-eval` rendering) for the model_eval driver"
            )
        if not args.labels and not getattr(args, "lanes", None):
            raise ContractError(
                "the model_eval driver needs exactly one grouping source: "
                "--labels (the C2 labels JSONL) or --lanes (the C2 lane-mode "
                "lanes JSONL for this split; the driver rewrites task "
                "categories from it)"
            )
        if args.labels and getattr(args, "lanes", None):
            raise ContractError(
                "pass --labels or --lanes, not both (the driver rewrites task "
                "categories from exactly one grouping source)"
            )
        if getattr(args, "lanes", None) and spec.template is None:
            raise ContractError(
                "--lanes requires the spec's template section (lane mode is "
                "the template path; sub-issue-8 §2.1)"
            )
        spec.require_dataset_file(
            args.tasks, what="--tasks", expected_hash=declared_tasks_hash
        )
        missing_flags = [
            flag
            for flag, value in (
                ("--baseline-router-endpoint", args.baseline_router_endpoint),
                ("--candidate-router-endpoint", args.candidate_router_endpoint),
                # Y4: the verdict must name the config it graded.
                ("--candidate-config", args.candidate_config),
                ("--model-eval-dir", args.model_eval_dir),
            )
            if not value
        ]
        if missing_flags:
            raise ContractError(
                "the model_eval driver needs " + ", ".join(missing_flags)
            )
        try:
            result = verify_live_model_eval(
                spec=spec,
                tasks_path=args.tasks,
                labels_path=args.labels,
                lanes_path=getattr(args, "lanes", None),
                candidate_config=args.candidate_config,
                baseline_endpoint=args.baseline_router_endpoint,
                candidate_endpoint=args.candidate_router_endpoint,
                management_endpoint=args.candidate_management_endpoint,
                router_model=_router_model_for_proposer(
                    args.router_model, candidate_row["provenance"].get("proposer")
                ),
                repetitions=args.repetitions,
                measurements=measurements,
                model_eval_dir=args.model_eval_dir,
                python_bin=args.python_bin,
                work_dir=work_dir,
                concurrency=args.concurrency,
                request_timeout=args.request_timeout,
                force=args.force,
                run_cache=run_cache,
            )
        except (ContractError, OSError) as exc:
            raise SystemExit(f"verify error: {exc}") from exc
    else:
        # sr-bench MoM target ids the router is serving the candidate on: the
        # CLI flag wins; otherwise the candidate file must carry srbench_targets.
        if args.candidate_targets:
            targets = [
                t.strip() for t in args.candidate_targets.split(",") if t.strip()
            ]
        else:
            targets = candidate_row.get("srbench_targets")
        if (
            not isinstance(targets, list)
            or not targets
            or not all(isinstance(t, str) and t.strip() for t in targets)
        ):
            raise ContractError(
                "verify needs sr-bench MoM target ids: pass --candidate-targets "
                "(comma-separated) or add a non-empty srbench_targets list to the "
                "candidate JSON"
            )
        cli = SrbenchCli(
            bin=args.srbench_bin,
            url=args.srbench_url,
            store=args.srbench_store,
            timeout_s=args.timeout,
        )
        # Z4: the replay-sourced stability gates judge the arms' own router
        # replay records, and the CLI is their only source - an agentic
        # verify without the management endpoints could never evaluate
        # them, so fail fast before anything is spent on runs.
        if workload.kind == "agentic":
            resolved_gates = workload.resolved_agentic_gates()
            if (
                any(
                    gate in resolved_gates
                    for gate in ("max_selection_rejected", "max_tool_loop_interrupted")
                )
                and not args.candidate_management_endpoint
            ):
                raise ContractError(
                    "agentic verify needs --candidate-management-endpoint (the "
                    "candidate arm's router management base URL): the "
                    "replay-sourced stability gates judge the arm's own "
                    "replay records"
                )
            if (
                resolved_gates.get("max_tool_loop_interrupted") == "baseline"
                and not args.baseline_management_endpoint
            ):
                raise ContractError(
                    'agentic gate max_tool_loop_interrupted pins "baseline": '
                    "pass --baseline-management-endpoint (the baseline arm's "
                    "router management base URL), or declare a numeric "
                    "ceiling instead when the comparison has no comparable "
                    "baseline router (cold start)"
                )
        try:
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
                workload=workload,
                candidate_management_endpoint=args.candidate_management_endpoint,
                baseline_management_endpoint=args.baseline_management_endpoint,
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
        # the model_eval driver has no sr-bench preview: its distribution
        # rides the result (alias-keyed shares behind the random mix R)
        distribution = preview.get("distribution") or result.get("distribution")
        if distribution:
            notes_parts.append(
                "routing_distribution="
                + json.dumps(distribution, sort_keys=True, ensure_ascii=False)
            )
        # Y4: the ledger's candidate names the config the verdict graded -
        # sr-bench driver: the frozen config_hash of the MoM targets;
        # model_eval driver: the sha256 the management API acknowledged.
        candidate_row["config_sha256"] = result["config_sha256"]
        record = LedgerRecord(
            round=args.round,
            ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            candidate=(
                promotion_candidate(candidate_row) if args.final else candidate_row
            ),
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
            # AG1: agentic continuity evidence for agentic workloads.
            agentic=result.get("agentic"),
            attribution_delta=None,
            verification={
                "baseline": result["metrics"]["baseline"],
                "guard_measurements": measurements,
                "evidence_quality": result.get("evidence_quality"),
            },
            attribution_ref=args.attribution_ref,
            coverage_warning=False,
            # Y4: --final marks the promotion record the bundle reads.
            final=bool(args.final),
            notes="; ".join(notes_parts),
        )
        ledger.append(record)
    _print_json(result)
    return verdict_exit_code(result["verdict"])


def cmd_run_agentic(args: argparse.Namespace) -> int:
    """Drive a real agent client over a task file through one target (AG2).

    One independent working directory and session per task; a local header
    forwarding proxy journals every request's native identity headers (W2)
    and injects the per-request ``x-request-id`` the Z9 join needs. Judging
    is a separate batch step: ``judge-agentic`` on the same run directory.
    """
    return run_driver(args)


def cmd_judge_agentic(args: argparse.Namespace) -> int:
    """Judge a run-agentic directory with the SWE-bench official harness.

    Emits sr-bench-shaped per-task rows (``judge-results.jsonl``) so the M2
    paired machinery and the AG7 gates consume the run like an sr-bench run.
    """
    return judge_run(args)


def cmd_report(args: argparse.Namespace) -> int:
    """One help entry, two unambiguous inputs: the ledger replay (G1) or
    explicitly selected receipts (--evidence, rendered in supplied order)."""
    try:
        if args.evidence:
            if args.ledger:
                raise ValueError("pass either --ledger or --evidence, not both")
            evidence = [json.loads(path.read_bytes()) for path in args.evidence]
            report = write_report(
                Path(args.output), evidence, background=args.background
            )
            _print_json({"report": report, "evidence_count": len(evidence)})
            return 0
        if not args.ledger:
            raise ValueError("report needs --ledger or --evidence")
        report = write_run_report(args.ledger, args.output)
    except ValueError as exc:
        _fail(f"recipe-lab report: {exc}")
    _print_json({"report": args.output, "bytes": len(report)})
    return 0


def cmd_next(args: argparse.Namespace) -> int:
    """G3: full next-round context - contract summary, ledger tail, best
    replay candidate (A2 best_replay), last live verdict, latest attribution
    summary (hash-verified via attribution_ref), optional diagnosis highlights
    and casebook references, each with its citable insight-ref. Byte-stable
    for the same inputs; frozen_test case content is never exported."""
    spec = _load_spec(args.spec)
    context = write_next_context(
        args.ledger,
        args.output,
        spec=spec,
        diagnosis_path=args.diagnosis,
        symptoms=args.symptom,
        tail=args.tail,
    )
    _print_json({"context": args.output, "bytes": len(context)})
    return 0


def cmd_provenance(args: argparse.Namespace) -> int:
    manifest = write_run_manifest(args.ledger, args.output)
    _print_json(
        {
            "manifest": args.output,
            "kind": manifest["kind"],
            "rounds": manifest["rounds"]["count"],
        }
    )
    return 0


def cmd_record_draft(args: argparse.Namespace) -> int:
    ledger = Ledger(args.ledger)
    records = ledger.filter(tier="live", verdict="PASS")
    if not records:
        _fail("no PASS live record in the ledger; nothing to draft")
    record = records[-1]
    draft = render_promotion_draft(record)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(draft, encoding="utf-8")
    _print_json(
        {"draft": args.output, "candidate": (record.get("candidate") or {}).get("ref")}
    )
    return 0


def cmd_score_candidate(args: argparse.Namespace) -> int:
    """Score one deployed candidate against a saved sr-bench baseline.

    The channel a coding agent uses to submit its own proposal: the receipt
    (schema ``recipe-lab-sr-bench-run-v1``) is printed as one JSON line and
    saved under ``<workspace>/sr-bench``. The exit code follows the verdict.
    """
    receipt = srbench_evaluate(args)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, allow_nan=False))
    if "report_error" in receipt and receipt["verdict"] == "PASS":
        # The machine evidence is saved, but its report is not deliverable: a
        # PASS nobody can read is not a PASS.
        return 2
    return verdict_exit_code(receipt["verdict"])


class CommandParser(argparse.ArgumentParser):
    """Report a usage error in the JSON contract that agents consume.

    Every command prints one JSON document on stdout; argparse's default
    (prose on stderr) would leave a driving agent with nothing to parse. The
    receipt is INCONCLUSIVE, like any other run that could not be evaluated.
    """

    def error(self, message: str) -> NoReturn:
        print(
            json.dumps(
                {
                    "schema": "recipe-lab-cli-error-v1",
                    "verdict": "INCONCLUSIVE",
                    "reason": message,
                },
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = CommandParser(
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

    p_fromreplays = cases_sub.add_parser(
        "from-replays",
        help="AG3: build route-replay cases from router replay records "
        "(--granularity session|prompt|request; dedup rules into provenance; "
        "duplicate share above 50 percent warns)",
    )
    p_fromreplays.add_argument(
        "--records", required=True, help="router replay records JSONL"
    )
    p_fromreplays.add_argument(
        "--granularity",
        default=None,
        choices=list(GRANULARITIES),
        help="case granularity (default: session when the records carry "
        "conversation identities, else prompt - the J3 convention)",
    )
    p_fromreplays.add_argument(
        "--output", required=True, help="output route-replay cases JSONL path"
    )
    p_fromreplays.add_argument(
        "--provenance",
        default=None,
        help="output provenance JSON path (dedup rules and their effects); "
        "default: <output>.provenance.json",
    )
    p_fromreplays.add_argument(
        "--duplicate-warn-pct",
        type=float,
        default=DEFAULT_DUPLICATE_WARN_PCT,
        help="duplicate-share warning threshold in percent "
        f"(default {DEFAULT_DUPLICATE_WARN_PCT})",
    )
    p_fromreplays.set_defaults(func=cmd_cases_from_replays)

    p_boot = sub.add_parser(
        "bootstrap",
        help="legacy domain-template path (sub-issue-8 §2.5): per-lambda "
        "scaffold via result_to_config, completed to candidates; requires "
        "--legacy-domain-template and is not part of the cold-start "
        "completion criteria - the template path is observe-lanes + profile "
        "--lanes + bindings",
    )
    p_boot.add_argument(
        "--legacy-domain-template",
        action="store_true",
        help="run the legacy result_to_config scaffold path explicitly "
        "(sub-issue-8 §2.5: compatibility only)",
    )
    p_boot.add_argument(
        "--spec", required=True, help="experiment spec YAML (models + cost_basis)"
    )
    p_boot.add_argument(
        "--profile",
        required=True,
        help="C3 profile directory (profile.json + calib/results + calib/token_costs)",
    )
    p_boot.add_argument(
        "--model-eval-dir",
        required=True,
        help="directory holding result_to_config.py; the script runs as a "
        "subprocess and its hash lands in the provenance sidecar",
    )
    p_boot.add_argument(
        "--classification-config",
        required=True,
        help="classification source config whose global.model_catalog.modules "
        "and .embeddings replace the scaffold's copies in every candidate (Y2: "
        "same source as observe-labels' seed config)",
    )
    p_boot.add_argument(
        "--workload",
        default=None,
        help="workload from spec.workloads whose dev split the label-agreement "
        "check observes (default: the first workload)",
    )
    p_boot.add_argument(
        "--router-bin",
        required=True,
        help="path to the semantic-router binary (boots the Y2 label-agreement "
        "check: dev observed through the seed config and through every "
        "candidate)",
    )
    p_boot.add_argument(
        "--mock-backend",
        required=True,
        help="path to a mock/real OpenAI backend launcher script for the "
        "label-agreement check",
    )
    p_boot.add_argument(
        "--backend-port", type=int, default=8000, help="label-check backend port"
    )
    p_boot.add_argument(
        "--ready-timeout",
        type=int,
        default=300,
        help="model-runtime readiness timeout for the label-agreement check",
    )
    p_boot.add_argument(
        "--hf-home",
        default=None,
        help="HF_HOME for the label-check router process (model artifacts)",
    )
    p_boot.add_argument(
        "--mock-extra-args",
        default="",
        help="extra args passed to the label-check mock backend (split on spaces)",
    )
    p_boot.add_argument(
        "--request-timeout",
        type=float,
        default=180.0,
        help="per-request timeout for the label-agreement observations",
    )
    p_boot.add_argument(
        "--lambdas",
        default=DEFAULT_LAMBDAS,
        help="comma-separated cost-lambda values (0 = pure accuracy; a lambda "
        "whose token costs are missing is rejected, never silently ignored)",
    )
    p_boot.add_argument(
        "--out",
        required=True,
        help="output directory (scaffold-lambda-<lambda>.yaml + "
        "candidate-lambda-<lambda>.yaml + candidate-lambda-<lambda>.provenance.json)",
    )
    p_boot.add_argument(
        "--python-bin", default=sys.executable, help="python used to run the script"
    )
    p_boot.add_argument(
        "--validate-bin",
        default="vllm-sr",
        help="validator invoked as '<bin> config validate --offline --config <path>'; a "
        "rejected candidate fails the command (C5 completion criterion)",
    )
    p_boot.add_argument(
        "--listener-port",
        type=int,
        default=DEFAULT_LISTENER_PORT,
        help="gateway listener port written into every candidate",
    )
    p_boot.set_defaults(func=cmd_bootstrap)

    p_bindings = sub.add_parser(
        "bindings",
        help="per-lane bindings + candidate configs from the C3 profile "
        "(C4+C5 template path: routing-space check, capability filter, "
        "accuracy - lambda * normalized cost ranking, minimum_candidates, "
        "unmeasured lanes by the default rule, then recipe builtin init "
        "generates and config validate accepts every candidate; a "
        "no-routing-space verdict ends with the report and no candidates)",
    )
    p_bindings.add_argument(
        "--spec",
        required=True,
        help="experiment spec YAML (template + models with capabilities + "
        "providers_config + cost_basis)",
    )
    p_bindings.add_argument(
        "--profile",
        required=True,
        help="C3 profile directory (profile.json; lane mode, calib split only)",
    )
    p_bindings.add_argument(
        "--out",
        required=True,
        help="output directory (providers-eval.yaml + bindings-lambda-<tag>.yaml "
        "+ candidate-lambda-<tag>.yaml + bindings-lambda-<tag>.provenance.json)",
    )
    p_bindings.add_argument(
        "--lambdas",
        default=None,
        help="comma-separated cost-lambda values (default: the spec's "
        "bootstrap.lambdas, else 0 = pure accuracy)",
    )
    p_bindings.add_argument(
        "--exclude-decision",
        action="append",
        default=None,
        help="explicitly omit a named lane whose capability requirement no "
        "model satisfies (repeatable; passed to recipe builtin init)",
    )
    p_bindings.add_argument(
        "--vllm-sr-bin",
        default="vllm-sr",
        help="vllm-sr binary for recipe builtin list/init and config validate",
    )
    p_bindings.add_argument(
        "--catalog-version",
        default="latest",
        help="built-in bundle catalog version (recipe builtin list/init)",
    )
    p_bindings.add_argument(
        "--model-name",
        default=OBSERVE_MODEL_NAME,
        help="public entrypoint name of the generated config (recipe builtin "
        "init --model-name)",
    )
    p_bindings.add_argument(
        "--init-timeout",
        type=float,
        default=300.0,
        help="per-invocation timeout for recipe builtin list/init",
    )
    p_bindings.set_defaults(func=cmd_bindings)

    p_profile = sub.add_parser(
        "profile",
        help="per-model MCQ profiles grouped by the router's own labels "
        "(--labels, legacy domain-template path) or by template lanes "
        "(--lanes, template path; C3: rewrites task categories, runs the "
        "model_eval scripts direct per model, writes profile.json with the "
        "provenance; lane mode records unmeasured lanes instead of merging)",
    )
    p_profile.add_argument(
        "--spec", required=True, help="experiment spec YAML (models/generation/profile)"
    )
    p_profile.add_argument(
        "--labels",
        default=None,
        help="labels directory from observe-labels (labels-<split>.jsonl); "
        "exactly one of --labels/--lanes",
    )
    p_profile.add_argument(
        "--lanes",
        default=None,
        help="lanes directory from observe-lanes (lanes-<split>.jsonl); "
        "requires the spec's template section; exactly one of --labels/--lanes",
    )
    p_profile.add_argument(
        "--splits",
        default="calib,dev",
        help="comma-separated splits to profile (calib,dev; frozen_test is "
        "quarantined; calib feeds bootstrap, dev the answer bank)",
    )
    p_profile.add_argument(
        "--out",
        required=True,
        help="output directory (generation.json + <split>/{tasks,results,token_costs}"
        " + profile.json)",
    )
    p_profile.add_argument(
        "--workload",
        default=None,
        help="workload from spec.workloads (default: the first workload; "
        "transfer workloads are quarantined; M5)",
    )
    p_profile.add_argument(
        "--model-eval-dir",
        required=True,
        help="directory holding the invoked model_eval scripts "
        "(multi_benchmark_eval.py / mcq_judge.py / cost_aware_calib.py); "
        "the scripts run as subprocesses and their hashes land in the profile",
    )
    p_profile.add_argument(
        "--python-bin", default=sys.executable, help="python used to run the scripts"
    )
    p_profile.add_argument(
        "--concurrency", type=int, default=8, help="per-model request concurrency"
    )
    p_profile.add_argument(
        "--request-timeout",
        type=float,
        default=1800.0,
        help="per-request timeout in seconds (multi_benchmark_eval default)",
    )
    p_profile.add_argument(
        "--force",
        action="store_true",
        help="re-run all tasks instead of resuming completed records",
    )
    p_profile.set_defaults(func=cmd_profile)

    p_observe_lanes = sub.add_parser(
        "observe-lanes",
        help="observe which template lane each case hits (C2 template path: "
        "recipe builtin list reads the lanes, recipe builtin init binds the "
        "template to placeholder models, the bundle's pinned decision model "
        "decides; writes lanes-<split>.jsonl with case_id/lane/confidence)",
    )
    p_observe_lanes.add_argument(
        "--spec",
        required=True,
        help="experiment spec YAML (template section "
        "selects the built-in bundle/recipe; selects the split)",
    )
    p_observe_lanes.add_argument(
        "--split",
        default="dev",
        choices=["calib", "dev", "frozen_test"],
        help="workload split to observe (frozen_test is quarantined; B)",
    )
    p_observe_lanes.add_argument(
        "--workload",
        default=None,
        help="workload from spec.workloads (default: the first workload; "
        "transfer workloads are quarantined; M5)",
    )
    p_observe_lanes.add_argument(
        "--out",
        required=True,
        help="output directory for lanes-<split>.jsonl and its summary; "
        "config and router logs are retained under observe-<split>-<run>/",
    )
    p_observe_lanes.add_argument(
        "--vllm-sr-bin",
        default="vllm-sr",
        help="vllm-sr binary for recipe builtin list/init (its hash-held "
        "commands land in the summary)",
    )
    p_observe_lanes.add_argument(
        "--catalog-version",
        default="latest",
        help="built-in bundle catalog version (recipe builtin list/init)",
    )
    p_observe_lanes.add_argument(
        "--model-name",
        default=OBSERVE_MODEL_NAME,
        help="entrypoint model name of the observation config (recipe builtin "
        "init --model-name)",
    )
    p_observe_lanes.add_argument(
        "--router-bin", required=True, help="path to the semantic-router binary"
    )
    p_observe_lanes.add_argument(
        "--mock-backend",
        required=True,
        help="path to a mock/real OpenAI backend launcher script (serves the "
        "placeholder lane models; answers are irrelevant)",
    )
    p_observe_lanes.add_argument(
        "--python-bin", default=sys.executable, help="python used to launch the backend"
    )
    p_observe_lanes.add_argument("--backend-port", type=int, default=8000)
    p_observe_lanes.add_argument(
        "--listener-port",
        type=int,
        default=8899,
        help="listener port written into the observation config",
    )
    p_observe_lanes.add_argument(
        "--init-timeout",
        type=float,
        default=300.0,
        help="per-invocation timeout for recipe builtin list/init",
    )
    p_observe_lanes.add_argument(
        "--ready-timeout",
        type=int,
        default=300,
        help="model-runtime readiness timeout in seconds",
    )
    p_observe_lanes.add_argument(
        "--hf-home",
        default=None,
        help="HF_HOME for the router process (model artifacts)",
    )
    p_observe_lanes.add_argument(
        "--mock-extra-args",
        default="",
        help="extra args passed to the mock backend (split on spaces)",
    )
    p_observe_lanes.add_argument(
        "--request-timeout",
        type=float,
        default=180.0,
        help="per-request timeout in seconds",
    )
    p_observe_lanes.set_defaults(func=cmd_observe_lanes)

    p_observe = sub.add_parser(
        "observe-labels",
        help="observe the router's own domain labels for a split (C2: seed "
        "config + tagged probe -> labels-<split>.jsonl with case_id/"
        "router_label/confidence; unmatched decisions record other)",
    )
    p_observe.add_argument(
        "--spec", required=True, help="experiment spec YAML (selects the split)"
    )
    p_observe.add_argument(
        "--split",
        default="dev",
        choices=["calib", "dev", "frozen_test"],
        help="workload split to observe (frozen_test allowed: the C2 labels "
        "feed the model_eval verify driver's category rewrite, Y1)",
    )
    p_observe.add_argument(
        "--workload",
        default=None,
        help="workload from spec.workloads (default: the first workload; "
        "transfer workloads are quarantined; M5)",
    )
    p_observe.add_argument(
        "--out",
        required=True,
        help="output directory (seed-config.yaml + labels-<split>.jsonl + "
        "labels-<split>-summary.json + work/)",
    )
    p_observe.add_argument(
        "--classification-config",
        required=True,
        help="classification source config whose global.model_catalog.modules "
        "and .embeddings are injected into the seed config (Y2: the observe "
        "seed and every bootstrap candidate must share one classification "
        "source; cold-start passes the maintained config/config.yaml)",
    )
    p_observe.add_argument(
        "--router-bin", required=True, help="path to the semantic-router binary"
    )
    p_observe.add_argument(
        "--mock-backend",
        required=True,
        help="path to a mock/real OpenAI backend launcher script",
    )
    p_observe.add_argument(
        "--python-bin", default=sys.executable, help="python used to launch the backend"
    )
    p_observe.add_argument("--router-model", default="vllm-sr/auto")
    p_observe.add_argument("--backend-port", type=int, default=8000)
    p_observe.add_argument(
        "--ready-timeout",
        type=int,
        default=300,
        help="model-runtime readiness timeout in seconds",
    )
    p_observe.add_argument(
        "--hf-home",
        default=None,
        help="HF_HOME for the router process (model artifacts)",
    )
    p_observe.add_argument(
        "--mock-extra-args",
        default="",
        help="extra args passed to the mock backend (split on spaces)",
    )
    p_observe.add_argument(
        "--request-timeout",
        type=float,
        default=180.0,
        help="per-request timeout in seconds",
    )
    p_observe.set_defaults(func=cmd_observe_labels)

    p_toeval = cases_sub.add_parser(
        "to-model-eval",
        help="convert an MCQ source into model_eval tasks format (C1: "
        "task_id/prompt_user/category + answer letter, options rendered "
        "into the prompt; cases mirrors included so split works on it)",
    )
    p_toeval.add_argument(
        "--source",
        required=True,
        help="source JSONL: sr-bench normalized dataset rows "
        "(id/benchmark/messages/answer/metadata.stratum) or raw MCQ rows "
        "(id/source/category/question/options/answer)",
    )
    p_toeval.add_argument(
        "--out", required=True, help="output tasks JSONL path (model_eval format)"
    )
    p_toeval.add_argument(
        "--dataset",
        default=None,
        help="dataset label for raw rows without a source field",
    )
    p_toeval.set_defaults(func=cmd_cases_to_model_eval)

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

    p_attr_session = sub.add_parser(
        "attribute-session",
        help="session-aware agentic attribution (lane_flap / tool_loop_interrupted / selection_failed)",
    )
    p_attr_session.add_argument(
        "--records",
        required=True,
        help="path to the graded attribution records JSONL (with session fields)",
    )
    p_attr_session.add_argument(
        "--output-dir",
        required=True,
        help="directory for session-report.md and session-events.json",
    )
    p_attr_session.set_defaults(func=cmd_attribute_session)

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

    p_diag = sub.add_parser(
        "diagnose",
        help="decision coverage, signal distributions and threshold reachability over unlabeled rows (D)",
        description=(
            "Three no-ground-truth checks over replay rows: decisions that never fire (dead) "
            "or fire rarely, per-signal quantiles / histogram / range utilization, and rule "
            "predicate sites no traffic ever satisfies. Writes diagnosis.json "
            "(recipe-lab-diagnosis-v1, input-hashed, byte-stable) and diagnosis.md."
        ),
    )
    p_diag.add_argument(
        "--records",
        required=True,
        help="unlabeled diagnostic rows JSONL (collect-replays --unlabeled output)",
    )
    p_diag.add_argument("--config", required=True, help="base recipe config YAML")
    _add_recipe_selectors(p_diag)
    p_diag.add_argument(
        "--output-dir",
        required=True,
        help="directory for diagnosis.json and diagnosis.md",
    )
    p_diag.add_argument(
        "--expect-rare",
        default="",
        help="comma-separated decisions allowed to be rare (reported, never warned; e.g. security containment)",
    )
    p_diag.add_argument(
        "--rare-pct",
        type=float,
        default=DEFAULT_RARE_PCT,
        help="decisions above zero hits but under this traffic share %% are rare",
    )
    p_diag.add_argument(
        "--source",
        default="unlabeled replay rows",
        help="traffic source note recorded in the diagnosis",
    )
    p_diag.add_argument(
        "--min-sample",
        type=int,
        default=DEFAULT_MIN_SAMPLE,
        help="warn when the sample count is below this",
    )
    p_diag.add_argument(
        "--utilization-warn",
        type=float,
        default=DEFAULT_UTILIZATION_WARN,
        help="warn when a known-range signal uses less of its range than this",
    )
    p_diag.add_argument(
        "--agent-requests",
        default=None,
        help=(
            'agentic request manifest JSON ({"agent_requests": [...]}, Z9) that '
            "upgrades the AG4 separation check from record markers to a "
            "request-id join"
        ),
    )
    p_diag.set_defaults(func=cmd_diagnose)

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

    p_bimport = bank_sub.add_parser(
        "import",
        help="convert profile direct-model judged.jsonl into the answer bank (C6)",
    )
    p_bimport.add_argument(
        "--spec", required=True, help="experiment spec YAML (declares the models)"
    )
    p_bimport.add_argument(
        "--profile-dir",
        required=True,
        help="profile output directory (contains <split>/results/.../judged.jsonl)",
    )
    p_bimport.add_argument(
        "--split",
        default="dev",
        help="profile split to import (default: dev; the bank split)",
    )
    p_bimport.add_argument("--output", required=True, help="output bank JSONL path")
    p_bimport.set_defaults(func=cmd_bank)

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

    p_cal = sub.add_parser(
        "propose-calibrated",
        help="rule threshold candidates at observed distribution quantiles (D)",
        description=(
            "Reads one signal's observed distribution out of a diagnosis.json and proposes rule "
            "threshold candidates at chosen quantiles (--quantiles) or at a target traffic share "
            "(--target-share, equivalent to quantile 1 - share). Delegates to the rule-level "
            "proposer, so sites sharing (signal, op, value) move together. Each sidecar cites the "
            "diagnosis it came from; a diagnosis whose config hash does not match --base-config "
            "is rejected (exit 2)."
        ),
    )
    p_cal.add_argument(
        "--diagnosis", required=True, help="diagnosis.json from 'recipe-lab diagnose'"
    )
    p_cal.add_argument(
        "--base-config",
        required=True,
        help="base recipe config YAML the diagnosis was made over",
    )
    p_cal.add_argument(
        "--signal", required=True, help="decision signal name to calibrate"
    )
    p_cal.add_argument(
        "--from",
        dest="from_value",
        type=float,
        default=None,
        help="only move sites currently at this value (e.g. the 2.5 -> 1.75 BUG-3 calibration)",
    )
    p_cal.add_argument(
        "--quantiles",
        default=None,
        help="comma-separated quantiles in (0, 1], e.g. 0.5,0.75,0.9",
    )
    p_cal.add_argument(
        "--target-share",
        type=float,
        default=None,
        help="fraction of traffic meant to cross the threshold; equivalent to quantile 1 - share",
    )
    p_cal.add_argument(
        "--op",
        choices=["gte", "gt", "lte", "lt"],
        default=None,
        help="restrict to one predicate operator",
    )
    p_cal.add_argument(
        "--lo",
        type=float,
        default=None,
        help="threshold lower clamp (default: inferred from the signal)",
    )
    p_cal.add_argument(
        "--hi",
        type=float,
        default=None,
        help="threshold upper clamp (default: inferred from the signal)",
    )
    p_cal.add_argument(
        "--output-dir", default=None, help="directory for candidate config YAMLs"
    )
    p_cal.set_defaults(func=cmd_propose_calibrated)

    sub.add_parser(
        "suggest-depth",
        help="propose a Speed depth threshold from development evidence",
    )

    sub.add_parser(
        "draft",
        help="create a two-backend Speed recipe from declared cards (starter path)",
    )

    p_spp = sub.add_parser(
        "propose-session-policy",
        help="AG6: cause-keyed one-variable session-protection candidates "
        "(identity headers, stability weight, switch margin, agent identity "
        "header); drift releases and rejected selections come back as "
        "handoffs, not knobs",
    )
    p_spp.add_argument("--config", required=True, help="base config YAML to perturb")
    p_spp.add_argument(
        "--diagnosis",
        default=None,
        help="AG4 diagnosis JSON whose agentic.agent_separation triggers the "
        "agent-header candidate",
    )
    p_spp.add_argument(
        "--replay-summary",
        default=None,
        help="AG3 route-replay summary JSON whose cause-split switches / "
        "rejected selections trigger the tuning and identity candidates",
    )
    p_spp.add_argument(
        "--output-dir", required=True, help="directory for candidate config YAMLs"
    )
    p_spp.set_defaults(func=cmd_propose_session_policy)

    p_prof = sub.add_parser(
        "profile-agentic",
        help="AG5: aggregate per-model single-target sr-bench run reports "
        "into the agentic profile (recipe-lab-agentic-profile-v1)",
    )
    p_prof.add_argument(
        "--reports",
        action="append",
        required=True,
        help="run report JSON path; repeat once per model arm",
    )
    p_prof.add_argument("--output", required=True, help="output profile JSON path")
    p_prof.set_defaults(func=cmd_profile_agentic)

    p_boot = sub.add_parser(
        "bootstrap-agentic",
        help="AG5: generate an agentic candidate from the agent recipe "
        "template, the profile and the client identity contract (the shared "
        "unmeasured-lane selection + per-lane evidence in a sidecar)",
    )
    p_boot.add_argument(
        "--spec", required=True, help="experiment spec YAML (local_models)"
    )
    p_boot.add_argument(
        "--profile", required=True, help="agentic profile JSON (profile-agentic)"
    )
    p_boot.add_argument(
        "--template",
        default=None,
        help="agent recipe config.yaml to rewrite (default: config/recipes/agent/config.yaml "
        "next to the package)",
    )
    _add_recipe_selectors(p_boot)
    p_boot.add_argument(
        "--client",
        required=True,
        choices=list(CLIENTS),
        help="client whose identity headers the session protection reads",
    )
    p_boot.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=0.0,
        help="cost weight in the shared ranking (accuracy - lambda * "
        "normalized cost); 0 ranks by accuracy alone (default 0)",
    )
    p_boot.add_argument(
        "--local-model",
        action="append",
        default=None,
        help="local model for the privacy/security lanes (repeatable; defaults to "
        "the spec's local_models)",
    )
    p_boot.add_argument("--out", required=True, help="output candidate directory")
    p_boot.add_argument(
        "--validate",
        action="store_true",
        help="run the validator on the generated config before writing the candidate",
    )
    p_boot.add_argument(
        "--validate-bin",
        default="vllm-sr",
        help="validator invoked as '<bin> config validate --offline --config <path>'",
    )
    p_boot.set_defaults(func=cmd_bootstrap_agentic)

    p_run = sub.add_parser(
        "run", help="evaluate candidate configs in the replay tier (RANKED -> ledger)"
    )
    p_run.add_argument(
        "--tier",
        default="replay",
        choices=["replay", "route-replay"],
        help="replay tier (answer bank, RANKED records) or route-replay tier (AG3: recorded "
        "sessions replayed for routing evidence only; live tier uses the live-verdict command)",
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
    p_run.add_argument(
        "--bank",
        default=None,
        help="answer bank JSONL (replay tier; unused on the route-replay tier)",
    )
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
        help="provenance tag for the candidates: "
        + " | ".join((*_PROPOSER_TAGS, "agent:<id>")),
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
    p_run.add_argument(
        "--router-model",
        help="router entrypoint (defaults to vllm-sr/recipe-lab for bindings, vllm-sr/auto otherwise)",
    )
    p_run.add_argument(
        "--strong-model",
        default=None,
        help="route-replay tier: model name for the strong-model share (from the model_cards/cost_model naming)",
    )
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
        help="validator invoked as '<bin> config validate --offline --config <path>' (C)",
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
    # C7: model_eval driver (workload.driver == "model_eval") - every arm is
    # a multi_benchmark_eval + mcq_judge pass instead of an sr-bench run.
    p_verify.add_argument(
        "--tasks",
        default=None,
        help="model_eval driver: the contract-declared model_eval_tasks "
        "rendering for this split (from `cases to-model-eval`; hash-checked "
        "against the contract; required for that driver)",
    )
    p_verify.add_argument(
        "--labels",
        default=None,
        help="model_eval driver: the C2 labels-<split>.jsonl the driver "
        "rewrites the task categories from (legacy label mode) - exactly one "
        "of --labels/--lanes for that driver",
    )
    p_verify.add_argument(
        "--lanes",
        default=None,
        help="model_eval driver: the C2 lanes-<split>.jsonl the driver "
        "rewrites the task categories from (template lane mode; requires the "
        "spec's template section) - exactly one of --labels/--lanes",
    )
    p_verify.add_argument(
        "--candidate-config",
        default=None,
        help="model_eval driver: the candidate config file the candidate "
        "router must have loaded; its sha256 is checked against the router's "
        "GET /api/v1/config/hash and recorded in the ledger (required for "
        "that driver)",
    )
    p_verify.add_argument(
        "--baseline-router-endpoint",
        default=None,
        help="model_eval driver: OpenAI v1 base of the router serving the "
        "baseline config (required for that driver)",
    )
    p_verify.add_argument(
        "--candidate-router-endpoint",
        default=None,
        help="model_eval driver: OpenAI v1 base of the router serving the "
        "candidate config (required for that driver)",
    )
    p_verify.add_argument(
        "--candidate-management-endpoint",
        default=None,
        help="router management base URL of the router serving the candidate "
        "(e.g. http://127.0.0.1:8080); model_eval driver: its replay records "
        "provide the routing distribution behind the random mix R - without "
        "it the mix is not evaluable; sr-bench driver (agentic): its replay "
        "records judge the candidate arm's stability gates (Z4)",
    )
    p_verify.add_argument(
        "--baseline-management-endpoint",
        default=None,
        help="sr-bench driver (agentic): router management base URL of the "
        "router serving the baseline config; its replay records resolve the "
        "max_tool_loop_interrupted baseline pin (Z4)",
    )
    p_verify.add_argument(
        "--router-model",
        help="model_eval driver: model name of the router entrypoint both "
        "router arms call (defaults to vllm-sr/recipe-lab for bindings, vllm-sr/auto otherwise)",
    )
    p_verify.add_argument(
        "--model-eval-dir",
        default=None,
        help="model_eval driver: directory holding multi_benchmark_eval.py / "
        "mcq_judge.py (required for that driver)",
    )
    p_verify.add_argument(
        "--python-bin",
        default=sys.executable,
        help="model_eval driver: python used to run the scripts",
    )
    p_verify.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="model_eval driver: per-arm request concurrency",
    )
    p_verify.add_argument(
        "--request-timeout",
        type=float,
        default=1800.0,
        help="model_eval driver: per-request timeout in seconds",
    )
    p_verify.add_argument(
        "--force",
        action="store_true",
        help="model_eval driver: re-run all tasks instead of resuming completed records",
    )
    p_verify.add_argument("--round", type=int, default=1)
    p_verify.set_defaults(func=cmd_verify)

    p_run_agentic = sub.add_parser(
        "run-agentic",
        help="drive a real agent client over a task file through one target (AG2)",
    )
    p_run_agentic.add_argument(
        "--driver",
        choices=["claude-code"],
        default="claude-code",
        help="agent client driver (AG2 ships the headless claude-code driver)",
    )
    p_run_agentic.add_argument(
        "--tasks",
        required=True,
        help="task JSONL (SWE-bench rows: instance_id/repo/base_commit/"
        "problem_statement)",
    )
    p_run_agentic.add_argument(
        "--target",
        required=True,
        help="routing target URL: the Router's Anthropic endpoint, or a "
        "single-model endpoint",
    )
    p_run_agentic.add_argument(
        "--out",
        required=True,
        help="run directory: tasks/, results/, requests/, agent-requests.json, "
        "task-results.json",
    )
    p_run_agentic.add_argument(
        "--only",
        default=None,
        help="run a single instance_id (resume: existing results are kept)",
    )
    p_run_agentic.add_argument(
        "--retry-policy",
        help="optional JSON policy for one fresh retry on a client error or empty patch",
    )
    p_run_agentic.add_argument(
        "--claude-bin",
        default="claude",
        help="the claude executable (pilot default: ~/.local/bin/claude)",
    )
    p_run_agentic.add_argument(
        "--model",
        default=None,
        help="ANTHROPIC_MODEL alias (the target's entry-model name)",
    )
    p_run_agentic.add_argument(
        "--small-fast-model",
        default=None,
        help="ANTHROPIC_SMALL_FAST_MODEL alias",
    )
    p_run_agentic.add_argument(
        "--api-key",
        default="recipe-lab-dummy",
        help="ANTHROPIC_API_KEY value handed to the client",
    )
    p_run_agentic.add_argument(
        "--max-turns",
        type=int,
        default=60,
        help="headless client turn budget per task",
    )
    p_run_agentic.add_argument(
        "--timeout-s",
        type=int,
        default=1800,
        help="per-task wall-clock limit in seconds",
    )
    p_run_agentic.add_argument(
        "--max-output-tokens",
        type=int,
        default=8192,
        help="CLAUDE_CODE_MAX_OUTPUT_TOKENS (vLLM 400s when context+output "
        "exceeds max-model-len; AG0 F0c)",
    )
    p_run_agentic.add_argument(
        "--clone-url-template",
        default="https://github.com/{repo}.git",
        help="task repo clone URL template ({repo} substituted)",
    )
    p_run_agentic.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra client environment variable (repeatable)",
    )
    p_run_agentic.set_defaults(func=cmd_run_agentic)

    p_judge_agentic = sub.add_parser(
        "judge-agentic",
        help="judge a run-agentic directory with the SWE-bench official "
        "harness -> sr-bench-shaped per-task results (AG2)",
    )
    p_judge_agentic.add_argument(
        "--run-dir",
        required=True,
        help="a run-agentic output directory (results/ + requests/)",
    )
    p_judge_agentic.add_argument(
        "--judge-command",
        default=DEFAULT_JUDGE_COMMAND,
        help="judge command template; placeholders {python}, {predictions_path}, "
        "{dataset_name}, {run_id}, {max_workers}, {log_dir}, {report_path} "
        "(default: the swebench official harness)",
    )
    p_judge_agentic.add_argument(
        "--dataset-name",
        default="SWE-bench/SWE-bench_Verified",
        help="the harness dataset name or local dataset path",
    )
    p_judge_agentic.add_argument(
        "--run-id",
        default="recipe-lab-agentic",
        help="harness run id (also the report's directory name)",
    )
    p_judge_agentic.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="harness evaluation parallelism",
    )
    p_judge_agentic.add_argument(
        "--judge-timeout-s",
        type=int,
        default=14400,
        help="judge wall-clock limit in seconds",
    )
    p_judge_agentic.add_argument(
        "--predictions-path",
        default=None,
        help="existing harness predictions JSONL to use (default: build "
        "predictions.jsonl under <run-dir>/judge/ from results/)",
    )
    p_judge_agentic.add_argument(
        "--rebuild-predictions",
        action="store_true",
        help="rebuild predictions even when --predictions-path was given",
    )
    p_judge_agentic.add_argument(
        "--report-path",
        default=None,
        help="explicit report JSON to parse (default: this evaluation's "
        "aggregate under <run-dir>/judge/)",
    )
    p_judge_agentic.set_defaults(func=cmd_judge_agentic)
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
    p_show.add_argument(
        "--tier", choices=["replay", "route_replay", "preview", "live"], default=None
    )
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

    p_report = sub.add_parser(
        "report",
        help="render a report: replay a ledger (--ledger) or deliver explicitly "
        "selected receipts (--evidence), never both",
    )
    p_report.add_argument(
        "--ledger", default=None, help="ledger JSONL to replay the run report from"
    )
    p_report.add_argument(
        "--evidence",
        action="append",
        default=[],
        type=Path,
        help="one saved JSON manifest or receipt; repeat in workflow order",
    )
    p_report.add_argument(
        "--background", default=None, help="user-supplied context for this handoff"
    )
    p_report.add_argument("--output", required=True, help="output Markdown path")
    p_report.set_defaults(func=cmd_report)

    p_next = sub.add_parser(
        "next",
        help="agent context export (untrusted reference material header included)",
    )
    p_next.add_argument(
        "--spec",
        required=True,
        help="experiment spec YAML (objective.replay_metric ranks the best replay candidate, A2)",
    )
    p_next.add_argument("--ledger", required=True, help="ledger JSONL")
    p_next.add_argument("--output", required=True, help="output Markdown path")
    p_next.add_argument(
        "--symptom",
        action="append",
        default=[],
        help="observed symptom for the casebook's prior-evidence references; repeatable",
    )
    p_next.add_argument(
        "--tail", type=int, default=10, help="how many recent records to include"
    )
    p_next.add_argument(
        "--diagnosis",
        default=None,
        help="diagnosis.json artifact; its highlights join the context with a citable diag: ref (G3)",
    )
    p_next.set_defaults(func=cmd_next)

    p_prov = sub.add_parser(
        "provenance", help="emit the recipe-lab-run-v1 manifest for a ledger"
    )
    p_prov.add_argument("--ledger", required=True, help="ledger JSONL")
    p_prov.add_argument("--output", required=True, help="output manifest JSON path")
    p_prov.set_defaults(func=cmd_provenance)

    p_draft = sub.add_parser(
        "record-draft",
        help="draft a docs/records promotion entry from the latest PASS live record",
    )
    p_draft.add_argument("--ledger", required=True, help="ledger JSONL")
    p_draft.add_argument("--output", required=True, help="output Markdown draft path")
    p_draft.set_defaults(func=cmd_record_draft)

    p_bundle = sub.add_parser(
        "bundle",
        help="package final-verify evidence into a reviewable bundle (C9)",
    )
    p_bundle.add_argument("--spec", required=True, help="experiment spec YAML")
    p_bundle.add_argument("--ledger", required=True, help="evidence ledger JSONL")
    p_bundle.add_argument(
        "--record", required=True, help="ledger record id (rec-...) to package"
    )
    p_bundle.add_argument("--out", required=True, help="output bundle directory")
    p_bundle.add_argument(
        "--config", required=True, help="final candidate config YAML to package"
    )
    p_bundle.add_argument(
        "--skeleton",
        default=None,
        help="scaffold config the candidate was completed from (config.diff "
        "source; legacy bootstrap sidecars only - exactly one of "
        "--skeleton/--bindings)",
    )
    p_bundle.add_argument(
        "--bindings",
        default=None,
        help="bindings YAML the template candidate was generated from "
        "(bindings.yaml artifact; template bindings sidecars only - exactly "
        "one of --skeleton/--bindings)",
    )
    p_bundle.add_argument(
        "--profile", required=True, help="C3 profile.json to archive in the bundle"
    )
    p_bundle.add_argument(
        "--candidate-sidecar",
        required=True,
        help="candidate provenance JSON (bootstrap or bindings sidecar); its "
        "recorded hashes bind the packaged inputs to the evidence (Y4)",
    )
    p_bundle.add_argument(
        "--validate-bin",
        default="vllm-sr",
        help="config validator binary (same convention as bootstrap)",
    )
    p_bundle.add_argument(
        "--router-commit",
        default=None,
        help="router git commit the acceptance ran against (provenance)",
    )
    p_bundle.add_argument(
        "--model-revision",
        action="append",
        default=[],
        metavar="NAME=REVISION",
        help="model revision recorded in provenance (repeatable)",
    )
    p_bundle.add_argument(
        "--model-eval-script",
        default=None,
        help="path to the model_eval script; hashed into provenance",
    )
    p_bundle.add_argument(
        "--private-pattern",
        action="append",
        default=[],
        metavar="REGEX",
        help="extra private-content regex; a match fails the bundle (repeatable)",
    )
    p_bundle.set_defaults(func=cmd_bundle)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "suggest-depth":
        # the depth proposal owns its own argparse surface and receipt policy
        from .bench_depth import main as suggest_depth  # noqa: PLC0415

        return suggest_depth(argv[1:])
    if argv and argv[0] == "draft":
        # the starter draft owns its own argparse surface and receipt policy
        from .speed_draft import main as draft  # noqa: PLC0415

        return draft(argv[1:], prog="recipe-lab draft")
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
