"""recipe-lab CLI: plan / attribute / collect-replays.

This first slice covers Sub-issue 1 (misroute attribution) plus the contract
preview. The replay-tier loop, answer bank, proposers and the ledger arrive
with Sub-issue 2; the live tier with Sub-issue 3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .adapters.router_replay import (
    fetch_replays,
    load_cases,
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
from .contract import ContractError, ExperimentSpec


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _load_spec(path: str) -> ExperimentSpec:
    try:
        return ExperimentSpec.from_yaml(path)
    except ContractError as exc:
        raise SystemExit(f"contract error: {exc}") from exc
    except OSError as exc:
        raise SystemExit(f"cannot read spec: {exc}") from exc


def cmd_plan(args: argparse.Namespace) -> int:
    spec = _load_spec(args.spec)
    _print_json(spec.to_dict())
    return 0


def cmd_attribute(args: argparse.Namespace) -> int:
    spec = _load_spec(args.spec)
    try:
        cases = load_records(args.records)
    except (ContractError, OSError) as exc:
        raise SystemExit(f"cannot load attribution records: {exc}") from exc

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

    rows = attribute(
        cases, spec.oracle_policy, dataset_hash=spec.workload.split_hash("dev")
    )
    summary = summarize(rows)

    out_dir = Path(args.output_dir)
    jsonl_path = write_jsonl(rows, out_dir / "attribution.jsonl")
    report_path = out_dir / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_markdown(summary, spec.workload.split_hash("dev")), encoding="utf-8"
    )

    _print_json(
        {
            "records": len(cases),
            "attribution_jsonl": str(jsonl_path),
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
        raise SystemExit(
            "collect-replays: --unlabeled and --cases are mutually exclusive"
        )
    if not args.unlabeled and not args.cases:
        raise SystemExit(
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
        raise SystemExit(f"replay fetch error: {exc}") from exc

    output = Path(args.output)
    result: dict = {"replay_records": len(records), "output": str(output)}
    if args.unlabeled:
        rows = records_to_rows(records)
        result["rows"] = len(rows)
    else:
        try:
            cases = load_cases(args.cases)
        except ContractError as exc:
            raise SystemExit(f"cases error: {exc}") from exc
        rows, stats = records_to_cases(records, cases, family=args.family)
        write_cases(cases, output.with_suffix(".cases.jsonl"))
        result["rows"] = len(rows)
        result["join_stats"] = stats
    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    _print_json(result)
    return 0


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
