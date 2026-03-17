#!/usr/bin/env python3
"""Agent-aware validation, skill resolution, and task reporting."""

from __future__ import annotations

import argparse

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
    if args.format == "json":
        print(report.to_json())
    else:
        print(report.to_summary())
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent gate helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    changed = subparsers.add_parser("changed-files")
    changed.add_argument("--base-ref", default=None)
    changed.add_argument("--changed-files", default=None)

    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--base-ref", default=None)
    resolve.add_argument("--changed-files", default=None)
    resolve.add_argument(
        "--format", choices=["json", "env", "summary"], default="summary"
    )

    resolve_skill = subparsers.add_parser("resolve-skill")
    resolve_skill.add_argument("--base-ref", default=None)
    resolve_skill.add_argument("--changed-files", default=None)
    resolve_skill.add_argument("--env", default=None)
    resolve_skill.add_argument(
        "--format", choices=["json", "env", "summary"], default="summary"
    )

    resolve_env = subparsers.add_parser("resolve-env")
    resolve_env.add_argument("--env", required=True)
    resolve_env.add_argument(
        "--field",
        choices=["build_target", "serve_command", "smoke_config", "local_dev_fragment"],
        default=None,
    )
    resolve_env.add_argument(
        "--format", choices=["json", "env", "summary"], default="summary"
    )

    report = subparsers.add_parser("report")
    report.add_argument("--base-ref", default=None)
    report.add_argument("--changed-files", default=None)
    report.add_argument("--env", required=True)
    report.add_argument("--format", choices=["json", "summary"], default="summary")

    scorecard = subparsers.add_parser("scorecard")
    scorecard.add_argument("--format", choices=["json", "summary"], default="summary")

    needs_smoke = subparsers.add_parser("needs-smoke")
    needs_smoke.add_argument("--base-ref", default=None)
    needs_smoke.add_argument("--changed-files", default=None)

    tests = subparsers.add_parser("run-tests")
    tests.add_argument("--base-ref", default=None)
    tests.add_argument("--changed-files", default=None)
    tests.add_argument(
        "--mode", choices=["fast", "feature", "feature-only"], required=True
    )

    e2e = subparsers.add_parser("run-e2e")
    e2e.add_argument("--base-ref", default=None)
    e2e.add_argument("--changed-files", default=None)

    go_lint = subparsers.add_parser("run-go-lint")
    go_lint.add_argument("--base-ref", default=None)
    go_lint.add_argument("--changed-files", default=None)

    rust_lint = subparsers.add_parser("run-rust-lint")
    rust_lint.add_argument("--base-ref", default=None)
    rust_lint.add_argument("--changed-files", default=None)

    py_lint = subparsers.add_parser("run-python-lint")
    py_lint.add_argument("--base-ref", default=None)
    py_lint.add_argument("--changed-files", default=None)

    config_lint = subparsers.add_parser("run-config-contract-lint")
    config_lint.add_argument("--base-ref", default=None)
    config_lint.add_argument("--changed-files", default=None)

    subparsers.add_parser("validate")
    return parser


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
