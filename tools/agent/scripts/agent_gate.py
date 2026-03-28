#!/usr/bin/env python3
"""Agent-aware validation, skill resolution, and task reporting."""

from __future__ import annotations

import argparse
import sys

from agent_artifacts import (
    ReportArtifactWriteResult,
    display_artifact_path,
    write_default_report_artifacts,
    write_explicit_report_artifact,
)
from agent_resolution import (
    build_report,
    get_changed_files,
    resolve_context,
    resolve_environment,
    resolve_skill,
    run_local_e2e,
    unique_preserve_order,
)
from agent_scorecard import build_harness_scorecard
from agent_support import (
    run_go_lint,
    run_python_lint,
    run_reference_config_lint,
    run_rust_lint,
    run_test_commands,
)
from agent_validation import validate_manifests


def handle_changed_files(_args: argparse.Namespace, changed_files: list[str]) -> int:
    if changed_files:
        print("\n".join(changed_files))
    return 0


def handle_resolve(args: argparse.Namespace, changed_files: list[str]) -> int:
    context = resolve_context(changed_files)
    if args.format == "json":
        print(context.to_json())
    elif args.format == "env":
        print(context.to_env())
    else:
        print(context.to_summary())
    return 0


def handle_resolve_env(args: argparse.Namespace) -> int:
    env = resolve_environment(args.env)
    if args.field:
        value = getattr(env, args.field)
        if value is not None:
            print(value)
        return 0
    if args.format == "json":
        print(env.to_json())
    elif args.format == "env":
        print(env.to_env())
    else:
        print(env.to_summary())
    return 0


def handle_resolve_skill(args: argparse.Namespace, changed_files: list[str]) -> int:
    skill = resolve_skill(changed_files, getattr(args, "env", None))
    if args.format == "json":
        print(skill.to_json())
    elif args.format == "env":
        print(skill.to_env())
    else:
        print(skill.to_summary())
    return 0


def handle_report(args: argparse.Namespace, changed_files: list[str]) -> int:
    report = build_report(changed_files, args.env)
    report_json = report.to_json()
    if args.write:
        write_result = write_explicit_report_artifact(args.write, report_json)
        _emit_written_artifact_paths(write_result)
    elif args.write_default:
        write_result = write_default_report_artifacts(report_json)
        _emit_written_artifact_paths(write_result)
    if args.format == "json":
        print(report_json)
    else:
        print(report.to_summary(context_detail=args.context_detail))
    return 0


def handle_scorecard(args: argparse.Namespace) -> int:
    scorecard = build_harness_scorecard()
    if args.format == "json":
        print(scorecard.to_json())
    else:
        print(scorecard.to_summary())
    return 0


def handle_needs_smoke(_args: argparse.Namespace, changed_files: list[str]) -> int:
    print("true" if resolve_context(changed_files).requires_local_smoke else "false")
    return 0


def handle_run_tests(args: argparse.Namespace, changed_files: list[str]) -> int:
    context = resolve_context(changed_files)
    commands = context.fast_tests
    if args.mode == "feature":
        commands = unique_preserve_order([*context.fast_tests, *context.feature_tests])
    elif args.mode == "feature-only":
        commands = context.feature_tests
    return run_test_commands(commands, args.mode)


def _emit_written_artifact_paths(write_result: ReportArtifactWriteResult) -> None:
    for path in write_result.written_paths():
        print(
            f"Wrote agent-report artifact: {display_artifact_path(path)}",
            file=sys.stderr,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent gate helper")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_changed_files_subparser(subparsers)
    _add_resolve_subparser(subparsers)
    _add_resolve_skill_subparser(subparsers)
    _add_resolve_env_subparser(subparsers)
    _add_report_subparser(subparsers)
    _add_scorecard_subparser(subparsers)
    _add_needs_smoke_subparser(subparsers)
    _add_run_tests_subparser(subparsers)
    _add_simple_changed_file_subparser(subparsers, "run-e2e")
    _add_simple_changed_file_subparser(subparsers, "run-go-lint")
    _add_simple_changed_file_subparser(subparsers, "run-rust-lint")
    _add_simple_changed_file_subparser(subparsers, "run-python-lint")
    _add_simple_changed_file_subparser(subparsers, "run-config-contract-lint")
    subparsers.add_parser("validate")
    return parser


def _add_changed_file_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-ref", default=None)
    parser.add_argument("--changed-files", default=None)


def _add_format_arg(
    parser: argparse.ArgumentParser, *, choices: list[str], default: str
) -> None:
    parser.add_argument("--format", choices=choices, default=default)


def _add_changed_files_subparser(subparsers) -> None:
    changed = subparsers.add_parser("changed-files")
    _add_changed_file_args(changed)


def _add_resolve_subparser(subparsers) -> None:
    resolve = subparsers.add_parser("resolve")
    _add_changed_file_args(resolve)
    _add_format_arg(resolve, choices=["json", "env", "summary"], default="summary")


def _add_resolve_skill_subparser(subparsers) -> None:
    resolve_skill = subparsers.add_parser("resolve-skill")
    _add_changed_file_args(resolve_skill)
    resolve_skill.add_argument("--env", default=None)
    _add_format_arg(
        resolve_skill, choices=["json", "env", "summary"], default="summary"
    )


def _add_resolve_env_subparser(subparsers) -> None:
    resolve_env = subparsers.add_parser("resolve-env")
    resolve_env.add_argument("--env", required=True)
    resolve_env.add_argument(
        "--field",
        choices=["build_target", "serve_command", "smoke_config", "local_dev_fragment"],
        default=None,
    )
    _add_format_arg(resolve_env, choices=["json", "env", "summary"], default="summary")


def _add_report_subparser(subparsers) -> None:
    report = subparsers.add_parser("report")
    _add_changed_file_args(report)
    report.add_argument("--env", required=True)
    _add_format_arg(report, choices=["json", "summary"], default="summary")
    report.add_argument(
        "--context-detail",
        choices=["compact", "full"],
        default="compact",
        help="Control how much of the context pack is printed in summary mode.",
    )
    report_write = report.add_mutually_exclusive_group()
    report_write.add_argument(
        "--write",
        default=None,
        help="Write the JSON agent-report payload to a path (repo-root relative unless absolute).",
    )
    report_write.add_argument(
        "--write-default",
        action="store_true",
        help=(
            "Write the JSON agent-report payload to .agent-harness/reports/latest-report.json "
            "and a timestamped session copy."
        ),
    )


def _add_scorecard_subparser(subparsers) -> None:
    scorecard = subparsers.add_parser("scorecard")
    _add_format_arg(scorecard, choices=["json", "summary"], default="summary")


def _add_needs_smoke_subparser(subparsers) -> None:
    needs_smoke = subparsers.add_parser("needs-smoke")
    _add_changed_file_args(needs_smoke)


def _add_run_tests_subparser(subparsers) -> None:
    tests = subparsers.add_parser("run-tests")
    _add_changed_file_args(tests)
    tests.add_argument(
        "--mode", choices=["fast", "feature", "feature-only"], required=True
    )


def _add_simple_changed_file_subparser(subparsers, name: str) -> None:
    parser = subparsers.add_parser(name)
    _add_changed_file_args(parser)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "validate":
        return validate_manifests()
    if args.command == "resolve-env":
        return handle_resolve_env(args)
    if args.command == "scorecard":
        return handle_scorecard(args)

    changed_files = get_changed_files(
        getattr(args, "changed_files", None), getattr(args, "base_ref", None)
    )
    handlers = {
        "changed-files": handle_changed_files,
        "resolve": handle_resolve,
        "resolve-skill": handle_resolve_skill,
        "report": handle_report,
        "needs-smoke": handle_needs_smoke,
        "run-tests": handle_run_tests,
        "run-e2e": lambda _args, files: run_local_e2e(files),
        "run-go-lint": lambda cmd_args, files: run_go_lint(
            files, getattr(cmd_args, "base_ref", None)
        ),
        "run-rust-lint": lambda _args, files: run_rust_lint(files),
        "run-python-lint": lambda _args, files: run_python_lint(files),
        "run-config-contract-lint": lambda _args, _files: run_reference_config_lint(),
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.error(f"Unknown command: {args.command}")
        return 2
    return handler(args, changed_files)


if __name__ == "__main__":
    raise SystemExit(main())
