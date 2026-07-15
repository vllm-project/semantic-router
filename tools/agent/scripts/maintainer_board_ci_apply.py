#!/usr/bin/env python3
"""Validate and apply maintainer board actions from reviewed CI artifacts."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import maintainer_board

REPO_ROOT = maintainer_board.REPO_ROOT
CI_ALLOWED_ACTIONS = frozenset({"label_issue", "label_pr"})
TARGET_PATTERN = re.compile(r"^#\d+$")
SOURCE_RUN_ID_PATTERN = re.compile(r"^\d+$")


@dataclass(frozen=True)
class ActionApplyResult:
    action: dict[str, Any]
    success: bool
    error: str | None = None


def ci_apply_allowed_labels(policy: dict[str, Any]) -> frozenset[str]:
    labels: set[str] = set()
    labels.update(policy["labels"]["lifecycle"].values())
    labels.update(policy["labels"]["pr_state"].values())
    return frozenset(labels)


def validate_ci_apply_actions(raw: Any, policy: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("proposed actions must be a JSON list")

    allowed_labels = ci_apply_allowed_labels(policy)
    validated: list[dict[str, Any]] = []
    for index, action in enumerate(raw):
        if not isinstance(action, dict):
            raise ValueError(f"action {index} must be an object")

        kind = action.get("action")
        if kind not in CI_ALLOWED_ACTIONS:
            raise ValueError(
                f"action {index} uses unsupported action {kind!r}; "
                f"CI apply allows only {sorted(CI_ALLOWED_ACTIONS)}"
            )

        target = action.get("target")
        if not isinstance(target, str) or not TARGET_PATTERN.match(target):
            raise ValueError(f"action {index} has invalid target {target!r}")

        labels = action.get("labels")
        if not isinstance(labels, list) or not labels:
            raise ValueError(f"action {index} must include a non-empty labels list")

        for label in labels:
            if not isinstance(label, str) or label not in allowed_labels:
                raise ValueError(f"action {index} uses disallowed label {label!r}")

        validated.append(action)
    return validated


def validate_source_run_id(value: str | None) -> str:
    if not value:
        raise ValueError("source_run_id is required for CI apply")
    if not SOURCE_RUN_ID_PATTERN.match(value):
        raise ValueError(f"source_run_id must be numeric, got {value!r}")
    return value


def build_label_command(action: dict[str, Any]) -> list[str]:
    command = ["gh", "issue", "edit", action["target"]]
    for label in action["labels"]:
        command.extend(["--add-label", label])
    return command


def apply_ci_actions(
    actions: list[dict[str, Any]],
    policy: dict[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[ActionApplyResult]:
    results: list[ActionApplyResult] = []
    for action in actions:
        try:
            payload = json.dumps(action, sort_keys=True)
            maintainer_board.sanitize_public_payload(payload, policy)
            runner(
                build_label_command(action),
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            results.append(ActionApplyResult(action, True))
        except subprocess.CalledProcessError as exc:
            error = (exc.stderr or exc.stdout or str(exc)).strip()
            results.append(
                ActionApplyResult(action, False, error or "gh command failed")
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            results.append(ActionApplyResult(action, False, str(exc)))
    return results


def render_apply_summary(
    results: list[ActionApplyResult],
    *,
    source_run_id: str,
    actions_file: Path,
) -> str:
    attempted = len(results)
    succeeded = sum(1 for result in results if result.success)
    failed = attempted - succeeded
    lines = [
        "## Maintainer Board Apply",
        "",
        f"Reviewed artifact from workflow run `{source_run_id}`.",
        f"Payload: `{actions_file}`.",
        "",
    ]
    if attempted == 0:
        lines.append("No proposed actions to apply.")
        lines.append("")
        return "\n".join(lines)

    lines.extend(
        [
            f"- Attempted: **{attempted}**",
            f"- Succeeded: **{succeeded}**",
            f"- Failed: **{failed}**",
            "",
            "### Results",
            "",
        ]
    )
    for index, result in enumerate(results, start=1):
        action = result.action
        labels = ", ".join(action.get("labels", []))
        status = "ok" if result.success else "failed"
        lines.append(
            f"{index}. `{action['action']}` {action['target']} "
            f"({labels}) — **{status}**"
        )
        if result.error:
            lines.append(f"   - {result.error}")
    lines.append("")
    return "\n".join(lines)


def write_summary(summary_path: Path | None, text: str) -> None:
    if summary_path is None:
        return
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def run_ci_apply(
    actions_file: Path,
    *,
    source_run_id: str,
    summary_path: Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int:
    policy = maintainer_board.load_policy()
    try:
        raw = json.loads(actions_file.read_text(encoding="utf-8"))
        actions = validate_ci_apply_actions(raw, policy)
    except (json.JSONDecodeError, ValueError) as exc:
        message = f"Invalid proposed actions payload: {exc}"
        print(message, file=sys.stderr)
        write_summary(
            summary_path,
            "\n".join(
                [
                    "## Maintainer Board Apply",
                    "",
                    message,
                    "",
                ]
            ),
        )
        return 2

    if not actions:
        write_summary(
            summary_path,
            render_apply_summary(
                [], source_run_id=source_run_id, actions_file=actions_file
            ),
        )
        return 0

    results = apply_ci_actions(actions, policy, runner=runner)
    write_summary(
        summary_path,
        render_apply_summary(
            results, source_run_id=source_run_id, actions_file=actions_file
        ),
    )
    return 0 if all(result.success for result in results) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--actions", required=True, help="Path to proposed-actions.json"
    )
    parser.add_argument(
        "--source-run-id",
        required=True,
        help="Reviewed Maintainer Board workflow run ID that produced the artifact",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional path to append a GitHub step summary (for example GITHUB_STEP_SUMMARY)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        source_run_id = validate_source_run_id(args.source_run_id)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    actions_path = Path(args.actions)
    if not actions_path.is_absolute():
        actions_path = REPO_ROOT / actions_path
    if not actions_path.is_file():
        print(f"Missing proposed actions file: {actions_path}", file=sys.stderr)
        return 2

    summary_path = Path(args.summary) if args.summary else None
    return run_ci_apply(
        actions_path, source_run_id=source_run_id, summary_path=summary_path
    )


if __name__ == "__main__":
    raise SystemExit(main())
