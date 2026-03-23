#!/usr/bin/env python3
"""Run a repo-native routing calibration loop against a live router apiserver."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from router_calibration_manifest import load_probe_manifest, resolve_manifest_assets
from router_calibration_report import render_markdown_summary
from router_calibration_support import (
    default_report_dir,
    deploy_config,
    evaluate_probes,
    fetch_router_snapshot,
    refresh_runtime_classification,
    run_validate,
    wait_for_router_ready,
    write_json,
)


def cmd_snapshot(args: argparse.Namespace) -> int:
    snapshot = fetch_router_snapshot(args.router_url)
    if args.output:
        write_json(Path(args.output), snapshot)
    else:
        print(json.dumps(snapshot, indent=2, ensure_ascii=False))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    manifest, probes = load_probe_manifest(Path(args.probes))
    report = {
        "manifest": manifest,
        "evaluation": evaluate_probes(args.router_url, probes, manifest),
    }
    if args.output:
        write_json(Path(args.output), report)
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def cmd_deploy(args: argparse.Namespace) -> int:
    response = deploy_config(
        args.router_url,
        Path(args.yaml),
        Path(args.dsl) if args.dsl else None,
    )
    ready_state = wait_for_router_ready(
        args.router_url,
        timeout_seconds=args.ready_timeout,
        interval_seconds=args.ready_interval,
    )
    refresh_state = None
    if not args.skip_runtime_refresh_barrier:
        refresh_state = refresh_runtime_classification(args.router_url)
        ready_state = wait_for_router_ready(
            args.router_url,
            timeout_seconds=args.ready_timeout,
            interval_seconds=args.ready_interval,
        )
    if args.output:
        write_json(
            Path(args.output),
            {
                "deploy": response,
                "runtime_refresh": refresh_state,
                "ready": ready_state,
            },
        )
    else:
        print(
            json.dumps(
                {
                    "deploy": response,
                    "runtime_refresh": refresh_state,
                    "ready": ready_state,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    result = run_validate(
        Path(args.dsl) if args.dsl else None,
        Path(args.yaml) if args.yaml else None,
    )
    if args.output:
        write_json(Path(args.output), result)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("valid", False) else 1


def cmd_run(args: argparse.Namespace) -> int:
    report_dir = Path(args.report_dir) if args.report_dir else default_report_dir()
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest, probes = load_probe_manifest(Path(args.probes))
    yaml_path, dsl_path = resolve_manifest_assets(
        manifest,
        Path(args.yaml) if args.yaml else None,
        Path(args.dsl) if args.dsl else None,
    )

    snapshot_before = fetch_router_snapshot(args.router_url)
    write_json(report_dir / "snapshot-before.json", snapshot_before)

    pre_eval = evaluate_probes(args.router_url, probes, manifest)
    write_json(report_dir / "eval-before.json", pre_eval)

    validate_result = None
    if not args.skip_validate:
        validate_result = run_validate(dsl_path, yaml_path)
        write_json(report_dir / "validate.json", validate_result)

    deploy_result = None
    refresh_result = None
    ready_result = None
    if yaml_path is not None:
        deploy_result = deploy_config(args.router_url, yaml_path, dsl_path)
        write_json(report_dir / "deploy.json", deploy_result)
        ready_result = wait_for_router_ready(
            args.router_url,
            timeout_seconds=args.ready_timeout,
            interval_seconds=args.ready_interval,
        )
        write_json(report_dir / "ready-after-deploy.json", ready_result)
        if not args.skip_runtime_refresh_barrier:
            refresh_result = refresh_runtime_classification(args.router_url)
            write_json(report_dir / "runtime-refresh.json", refresh_result)
            ready_result = wait_for_router_ready(
                args.router_url,
                timeout_seconds=args.ready_timeout,
                interval_seconds=args.ready_interval,
            )
            write_json(report_dir / "ready-after-runtime-refresh.json", ready_result)

    snapshot_after = fetch_router_snapshot(args.router_url)
    write_json(report_dir / "snapshot-after.json", snapshot_after)

    post_eval = evaluate_probes(args.router_url, probes, manifest)
    write_json(report_dir / "eval-after.json", post_eval)

    summary = render_markdown_summary(
        manifest,
        pre_eval,
        post_eval,
        validate_result,
        deploy_result,
    )
    (report_dir / "summary.md").write_text(summary, encoding="utf-8")

    print(f"Report written to {report_dir}")
    print(
        json.dumps(
            {
                "report_dir": str(report_dir),
                "pre_success_rate": pre_eval["success_rate"],
                "post_success_rate": post_eval["success_rate"],
                "pre_decision_success_rate": pre_eval["decision_success_rate"],
                "post_decision_success_rate": post_eval["decision_success_rate"],
                "deploy_version": (deploy_result or {}).get("version"),
                "runtime_refresh_applied": refresh_result is not None,
                "ready_phase": ((ready_result or {}).get("payload") or {}).get("phase"),
                "validate_ok": (validate_result or {}).get("valid"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if post_eval["passed"] else 1


def add_snapshot_subparser(subparsers: argparse._SubParsersAction) -> None:
    snapshot = subparsers.add_parser(
        "snapshot", help="Fetch router config and version state"
    )
    snapshot.add_argument(
        "--router-url",
        required=True,
        help="Router base URL, for example http://host:8080",
    )
    snapshot.add_argument("--output", help="Optional JSON output path")
    snapshot.set_defaults(func=cmd_snapshot)


def add_eval_subparser(subparsers: argparse._SubParsersAction) -> None:
    eval_parser = subparsers.add_parser(
        "eval", help="Run eval probes against the live router"
    )
    eval_parser.add_argument(
        "--router-url",
        required=True,
        help="Router base URL, for example http://host:8080",
    )
    eval_parser.add_argument("--probes", required=True, help="YAML probe manifest path")
    eval_parser.add_argument("--output", help="Optional JSON output path")
    eval_parser.set_defaults(func=cmd_eval)


def add_deploy_subparser(subparsers: argparse._SubParsersAction) -> None:
    deploy = subparsers.add_parser(
        "deploy", help="Deploy a YAML config to the live router"
    )
    deploy.add_argument(
        "--router-url",
        required=True,
        help="Router base URL, for example http://host:8080",
    )
    deploy.add_argument("--yaml", required=True, help="Canonical router YAML path")
    deploy.add_argument(
        "--dsl", help="Optional DSL source path to archive with the deploy"
    )
    deploy.add_argument(
        "--ready-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for GET /ready after deploy",
    )
    deploy.add_argument(
        "--ready-interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds for GET /ready after deploy",
    )
    deploy.add_argument(
        "--skip-runtime-refresh-barrier",
        action="store_true",
        help="Skip the /config/classification echo-back refresh barrier after deploy",
    )
    deploy.add_argument("--output", help="Optional JSON output path")
    deploy.set_defaults(func=cmd_deploy)


def add_validate_subparser(subparsers: argparse._SubParsersAction) -> None:
    validate = subparsers.add_parser(
        "validate", help="Run local sr-dsl validate for a DSL or YAML asset"
    )
    validate.add_argument("--dsl", help="DSL path to validate directly")
    validate.add_argument("--yaml", help="YAML path to decompile then validate")
    validate.add_argument("--output", help="Optional JSON output path")
    validate.set_defaults(func=cmd_validate)


def add_run_subparser(subparsers: argparse._SubParsersAction) -> None:
    run = subparsers.add_parser(
        "run", help="Run snapshot -> eval -> validate -> deploy -> eval"
    )
    run.add_argument(
        "--router-url",
        required=True,
        help="Router base URL, for example http://host:8080",
    )
    run.add_argument("--probes", required=True, help="YAML probe manifest path")
    run.add_argument(
        "--yaml",
        help="Canonical router YAML path to deploy. Defaults to manifest routing_assets.yaml when omitted.",
    )
    run.add_argument(
        "--dsl",
        help="Optional DSL source path for local validate and deploy archive. Defaults to manifest routing_assets.dsl when omitted.",
    )
    run.add_argument("--report-dir", help="Directory for JSON and Markdown reports")
    run.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip local validation before deploy",
    )
    run.add_argument(
        "--ready-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for GET /ready after deploy",
    )
    run.add_argument(
        "--ready-interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds for GET /ready after deploy",
    )
    run.add_argument(
        "--skip-runtime-refresh-barrier",
        action="store_true",
        help="Skip the /config/classification echo-back refresh barrier after deploy",
    )
    run.set_defaults(func=cmd_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run live routing calibration against a router apiserver."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_snapshot_subparser(subparsers)
    add_eval_subparser(subparsers)
    add_deploy_subparser(subparsers)
    add_validate_subparser(subparsers)
    add_run_subparser(subparsers)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
