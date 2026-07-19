#!/usr/bin/env python3
"""Generate the local maintainer issue and PR operating board."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import maintainer_ga_readiness
import yaml
from maintainer_release_plan import (
    match_release_tasks_to_issues,
    open_release_tasks,
    release_plan_issue_context,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = REPO_ROOT / "tools" / "agent" / "maintainer-policy.yaml"
DEFAULT_ACTIONS = "proposed-actions.json"
TRANSIENT_GH_ERRORS = (
    "HTTP 502",
    "HTTP 503",
    "HTTP 504",
    "502 Bad Gateway",
    "503 Service Unavailable",
    "504 Gateway Timeout",
)


def load_policy() -> dict[str, Any]:
    with POLICY_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def output_dir(policy: dict[str, Any], override: str | None) -> Path:
    raw = override or policy["default_output_dir"]
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    (path / "snapshots").mkdir(parents=True, exist_ok=True)
    return path


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(microsecond=0)


def parse_time(value: str | None) -> dt.datetime:
    if not value:
        return dt.datetime.fromtimestamp(0, dt.UTC)
    normalized = value.replace("Z", "+00:00")
    return dt.datetime.fromisoformat(normalized).astimezone(dt.UTC)


def age_days(value: str | None, now: dt.datetime) -> int:
    return max(0, (now - parse_time(value)).days)


def is_transient_gh_error(result: subprocess.CompletedProcess[str]) -> bool:
    output = f"{result.stderr}\n{result.stdout}"
    return any(error in output for error in TRANSIENT_GH_ERRORS)


def gh_json(args: list[str], attempts: int = 3) -> Any:
    for attempt in range(1, attempts + 1):
        result = subprocess.run(
            ["gh", *args],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            if not result.stdout.strip():
                return None
            return json.loads(result.stdout)
        if attempt < attempts and is_transient_gh_error(result):
            delay = 2 ** (attempt - 1)
            sys.stderr.write(
                f"gh {' '.join(args)} failed with a transient GitHub error; "
                f"retrying in {delay}s ({attempt}/{attempts})\n"
            )
            time.sleep(delay)
            continue
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    raise SystemExit(1)


def attach_pr_status_rollups(prs: list[dict[str, Any]]) -> None:
    for pr in prs:
        details = gh_json(
            [
                "pr",
                "view",
                str(pr["number"]),
                "--json",
                "statusCheckRollup",
            ]
        )
        pr["statusCheckRollup"] = (details or {}).get("statusCheckRollup") or []


def fetch_github_state(
    policy: dict[str, Any],
    issue_limit_override: int | None = None,
    pr_limit_override: int | None = None,
) -> dict[str, Any]:
    issue_limit = str(issue_limit_override or policy.get("default_issue_limit", 100))
    pr_limit = str(pr_limit_override or policy.get("default_pr_limit", 50))
    issues = gh_json(
        [
            "issue",
            "list",
            "--state",
            "open",
            "--limit",
            issue_limit,
            "--json",
            "number,title,labels,milestone,assignees,createdAt,updatedAt,author,url",
        ]
    )
    prs = (
        gh_json(
            [
                "pr",
                "list",
                "--state",
                "open",
                "--limit",
                pr_limit,
                "--json",
                "number,title,labels,milestone,assignees,createdAt,updatedAt,author,url,isDraft,reviewDecision,mergeStateStatus,headRefName,baseRefName",
            ]
        )
        or []
    )
    attach_pr_status_rollups(prs)
    milestones = gh_json(["api", "repos/{owner}/{repo}/milestones?state=open"])
    labels = gh_json(["label", "list", "--json", "name,description,color"])
    return {
        "issues": issues or [],
        "pull_requests": prs or [],
        "milestones": milestones or [],
        "labels": labels or [],
    }


def label_names(item: dict[str, Any]) -> set[str]:
    return {label.get("name", "") for label in item.get("labels", [])}


def milestone_title(item: dict[str, Any]) -> str:
    milestone = item.get("milestone") or {}
    return milestone.get("title") or ""


def grouped_issue_state(
    issues: list[dict[str, Any]],
    policy: dict[str, Any],
    milestone: str | None,
    now: dt.datetime,
) -> dict[str, list[dict[str, Any]]]:
    lifecycle = policy["labels"]["lifecycle"]
    stale_days = int(policy["staleness"]["issue_stale_days"])
    groups: dict[str, list[dict[str, Any]]] = {
        "milestone-bound": [],
        "milestone-candidate": [],
        "incoming-triage": [],
        "backlog": [],
        "stale": [],
    }
    for issue in issues:
        labels = label_names(issue)
        item_milestone = milestone_title(issue)
        is_active_milestone = bool(item_milestone) and (
            milestone is None or item_milestone == milestone
        )
        if lifecycle["stale"] in labels or (
            not item_milestone and age_days(issue.get("updatedAt"), now) >= stale_days
        ):
            groups["stale"].append(issue)
        elif is_active_milestone:
            groups["milestone-bound"].append(issue)
        elif lifecycle["milestone_candidate"] in labels:
            groups["milestone-candidate"].append(issue)
        elif lifecycle["backlog"] in labels:
            groups["backlog"].append(issue)
        else:
            groups["incoming-triage"].append(issue)
    return groups


def check_state(pr: dict[str, Any]) -> str:
    rollup = pr.get("statusCheckRollup") or []
    if not rollup:
        return "unknown"
    conclusions = {
        item.get("conclusion") or item.get("state") or item.get("status")
        for item in rollup
    }
    normalized = {str(value).lower() for value in conclusions if value is not None}
    if any(
        value in {"failure", "failed", "error", "cancelled"} for value in normalized
    ):
        return "failing"
    if all(value in {"success", "completed", "passed"} for value in normalized):
        return "green"
    return "pending"


def grouped_pr_state(
    prs: list[dict[str, Any]], policy: dict[str, Any], now: dt.datetime
) -> dict[str, list[dict[str, Any]]]:
    pr_labels = policy["labels"]["pr_state"]
    close_days = int(policy["staleness"]["pr_close_candidate_days"])
    attention_days = int(policy["staleness"]["pr_needs_attention_days"])
    groups: dict[str, list[dict[str, Any]]] = {
        "merge-candidate": [],
        "review-now": [],
        "unblock": [],
        "needs-rebase": [],
        "close-candidate": [],
    }
    for pr in prs:
        labels = label_names(pr)
        stale = age_days(pr.get("updatedAt"), now)
        merge_state = str(pr.get("mergeStateStatus") or "").upper()
        checks = check_state(pr)
        review = str(pr.get("reviewDecision") or "").upper()
        if pr_labels["close_candidate"] in labels or stale >= close_days:
            groups["close-candidate"].append(pr)
        elif pr_labels["needs_rebase"] in labels or merge_state in {"DIRTY", "BEHIND"}:
            groups["needs-rebase"].append(pr)
        elif pr_labels["blocked"] in labels or checks == "failing":
            groups["unblock"].append(pr)
        elif (
            not pr.get("isDraft")
            and review == "APPROVED"
            and checks == "green"
            and merge_state not in {"DIRTY", "BLOCKED"}
        ):
            groups["merge-candidate"].append(pr)
        elif not pr.get("isDraft") and (
            checks in {"green", "unknown"} or stale >= attention_days
        ):
            groups["review-now"].append(pr)
        else:
            groups["unblock"].append(pr)
    return groups


def summarize_item(item: dict[str, Any]) -> str:
    number = item.get("number")
    title = item.get("title", "<untitled>")
    url = item.get("url", "")
    return f"#{number} {title} ({url})"


def render_today(snapshot: dict[str, Any]) -> str:
    lines = [
        "# Maintainer Today",
        "",
        f"Generated: {snapshot['generated_at']}",
        f"Milestone: {snapshot.get('active_milestone') or 'all'}",
        "",
    ]
    lines.extend(maintainer_ga_readiness.render(snapshot))
    lines.extend(["## PR Actions", ""])
    for group in (
        "merge-candidate",
        "review-now",
        "unblock",
        "needs-rebase",
        "close-candidate",
    ):
        items = snapshot["groups"]["pull_requests"].get(group, [])
        lines.append(f"### {group} ({len(items)})")
        lines.extend(f"- {summarize_item(item)}" for item in items[:20])
        if not items:
            lines.append("- none")
        lines.append("")

    lines.append("## Issue Actions")
    lines.append("")
    for group in (
        "milestone-bound",
        "milestone-candidate",
        "incoming-triage",
        "backlog",
        "stale",
    ):
        items = snapshot["groups"]["issues"].get(group, [])
        lines.append(f"### {group} ({len(items)})")
        lines.extend(f"- {summarize_item(item)}" for item in items[:20])
        if not items:
            lines.append("- none")
        lines.append("")
    lines.append("## Proposed Actions")
    lines.append("")
    actions = snapshot.get("proposed_actions", [])
    if actions:
        lines.extend(f"- {action['action']}: {action['target']}" for action in actions)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def render_milestone(snapshot: dict[str, Any]) -> str:
    milestone = snapshot.get("active_milestone") or "all"
    lines = [
        f"# Milestone {milestone}",
        "",
        f"Generated: {snapshot['generated_at']}",
        "",
        "## Bound Issues",
        "",
    ]
    for item in snapshot["groups"]["issues"]["milestone-bound"]:
        lines.append(f"- {summarize_item(item)}")
    if not snapshot["groups"]["issues"]["milestone-bound"]:
        lines.append("- none")
    lines.extend(["", "## Candidate Issues", ""])
    for item in snapshot["groups"]["issues"]["milestone-candidate"]:
        lines.append(f"- {summarize_item(item)}")
    if not snapshot["groups"]["issues"]["milestone-candidate"]:
        lines.append("- none")
    lines.extend(["", "## PRs Requiring Maintainer Action", ""])
    for group in ("review-now", "unblock", "needs-rebase", "close-candidate"):
        lines.append(f"### {group}")
        items = snapshot["groups"]["pull_requests"][group]
        lines.extend(f"- {summarize_item(item)}" for item in items)
        if not items:
            lines.append("- none")
        lines.append("")
    return "\n".join(lines)


def render_release_readiness(snapshot: dict[str, Any], release_plan: Path) -> str:
    milestone = snapshot.get("active_milestone") or "all"
    milestone_data = next(
        (
            item
            for item in snapshot.get("raw", {}).get("milestones", [])
            if item.get("title") == milestone
        ),
        {},
    )
    open_count = milestone_data.get("open_issues", "unknown")
    closed_count = milestone_data.get("closed_issues", "unknown")
    tasks = open_release_tasks(release_plan)
    issues = release_plan_issue_context(
        snapshot["groups"]["issues"]["milestone-bound"], release_plan
    )
    matches = match_release_tasks_to_issues(tasks, issues)
    missing = [item for item in matches if not item["matches"]]

    lines = [
        "# Release Readiness",
        "",
        f"Generated: {snapshot['generated_at']}",
        f"Milestone: {milestone}",
        f"Milestone issues: {open_count} open, {closed_count} closed",
        "",
    ]
    lines.extend(maintainer_ga_readiness.render(snapshot))
    lines.extend(
        [
            "## Open Plan Tasks",
            "",
        ]
    )
    if tasks:
        lines.extend(f"- {task}" for task in tasks)
    else:
        lines.append("- none")

    lines.extend(["", "## Plan Tasks Without Matching Milestone Issue", ""])
    if missing:
        for item in missing:
            lines.append(f"- {item['task']}")
    else:
        lines.append("- none")

    lines.extend(["", "## Plan Task Matches", ""])
    for item in matches:
        lines.append(f"### {item['task']}")
        if item["matches"]:
            for issue in item["matches"][:5]:
                lines.append(f"- {summarize_item(issue)}")
        else:
            lines.append("- no matching milestone issue")
        lines.append("")

    lines.extend(render_release_blockers(snapshot))

    lines.extend(["", "## PR Queue", ""])
    for group in ("merge-candidate", "review-now", "needs-rebase", "close-candidate"):
        items = snapshot["groups"]["pull_requests"].get(group, [])
        lines.append(f"- {group}: {len(items)}")
    lines.append("")
    return "\n".join(lines)


def render_release_blockers(snapshot: dict[str, Any]) -> list[str]:
    lines = ["## Release Blockers", ""]
    groups = release_blocker_groups(snapshot)
    if not groups:
        lines.append("- none")
        return lines
    for index, (heading, items) in enumerate(groups):
        if index:
            lines.append("")
        lines.append(f"### {heading}")
        lines.extend(items)
    return lines


def release_blocker_groups(snapshot: dict[str, Any]) -> list[tuple[str, list[str]]]:
    groups = []
    ga_readiness = snapshot.get("session_routing_ga_readiness") or {}
    ga_blockers = ga_readiness.get("blockers", [])
    if ga_blockers:
        groups.append(
            (
                "Session-Aware GA",
                [
                    f"- {item['status']}: {item['title']} (`{item['id']}`)"
                    for item in ga_blockers
                ],
            )
        )
    pr_blockers = snapshot["groups"]["pull_requests"]["unblock"]
    if pr_blockers:
        groups.append(
            (
                "PRs Requiring Unblock",
                [f"- {summarize_item(pr)}" for pr in pr_blockers],
            )
        )
    return groups


def proposed_actions(
    snapshot: dict[str, Any], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    lifecycle = policy["labels"]["lifecycle"]
    pr_state = policy["labels"]["pr_state"]
    for issue in snapshot["groups"]["issues"]["incoming-triage"]:
        actions.append(
            {
                "action": "label_issue",
                "target": f"#{issue['number']}",
                "labels": [lifecycle["needs_triage"]],
            }
        )
    for issue in snapshot["groups"]["issues"]["stale"]:
        actions.append(
            {
                "action": "label_issue",
                "target": f"#{issue['number']}",
                "labels": [lifecycle["stale"]],
            }
        )
    for pr in snapshot["groups"]["pull_requests"]["needs-rebase"]:
        actions.append(
            {
                "action": "label_pr",
                "target": f"#{pr['number']}",
                "labels": [pr_state["needs_rebase"]],
            }
        )
    for pr in snapshot["groups"]["pull_requests"]["close-candidate"]:
        actions.append(
            {
                "action": "label_pr",
                "target": f"#{pr['number']}",
                "labels": [pr_state["close_candidate"]],
            }
        )
    return actions


def build_snapshot(
    github_state: dict[str, Any], policy: dict[str, Any], milestone: str | None
) -> dict[str, Any]:
    now = utc_now()
    snapshot = {
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "active_milestone": milestone,
        "raw": github_state,
        "groups": {
            "issues": grouped_issue_state(
                github_state["issues"], policy, milestone, now
            ),
            "pull_requests": grouped_pr_state(
                github_state["pull_requests"], policy, now
            ),
        },
    }
    snapshot["proposed_actions"] = proposed_actions(snapshot, policy)
    maintainer_ga_readiness.attach_latest(snapshot)
    return snapshot


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "all"


def write_snapshot(
    snapshot: dict[str, Any], out_dir: Path, release_plan: Path | None = None
) -> None:
    today = utc_now().date().isoformat()
    (out_dir / "current.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "snapshots" / f"{today}.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "today.md").write_text(render_today(snapshot), encoding="utf-8")
    milestone_slug = slug(snapshot.get("active_milestone") or "all")
    (out_dir / f"milestone-{milestone_slug}.md").write_text(
        render_milestone(snapshot), encoding="utf-8"
    )
    (out_dir / DEFAULT_ACTIONS).write_text(
        json.dumps(snapshot["proposed_actions"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if release_plan:
        (out_dir / "release-readiness.md").write_text(
            render_release_readiness(snapshot, release_plan), encoding="utf-8"
        )


def load_current(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "current.json"
    if not path.exists():
        raise SystemExit(f"No maintainer snapshot found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize_public_payload(text: str, policy: dict[str, Any]) -> None:
    public_policy = policy.get("public_artifact_policy", {})
    for phrase in public_policy.get("forbidden_phrases", []):
        if phrase.lower() in text.lower():
            raise SystemExit(f"Public payload contains forbidden phrase: {phrase}")
    for pattern in public_policy.get("forbidden_patterns", []):
        if re.search(pattern, text):
            raise SystemExit(f"Public payload matches forbidden pattern: {pattern}")


def create_issue_actions(
    args: argparse.Namespace, policy: dict[str, Any]
) -> list[dict[str, Any]]:
    plan_path = Path(args.release_plan)
    if not plan_path.is_absolute():
        plan_path = REPO_ROOT / plan_path
    tasks = open_release_tasks(plan_path)
    if not args.include_matched:
        snapshot = load_current(output_dir(policy, args.output_dir))
        issues = release_plan_issue_context(
            snapshot["groups"]["issues"]["milestone-bound"], plan_path
        )
        tasks = [
            item["task"]
            for item in match_release_tasks_to_issues(tasks, issues)
            if not item["matches"]
        ]
    labels = [] if args.no_help_wanted else [policy["labels"]["help_wanted"]]
    track = policy.get("release_tracks", {}).get(args.track or "")
    if track and track.get("label"):
        labels.append(track["label"])
    if args.labels:
        labels.extend(args.labels.split(","))
    actions = []
    for task in tasks:
        title = f"{args.prefix} {task}" if args.prefix else task
        body = (
            "## Scope\n\n"
            f"{task}\n\n"
            "## Release Track\n\n"
            f"{args.track or 'unclassified'}\n\n"
            "## Validation\n\n"
            "Use the release plan and affected module gates to define the final test plan.\n"
        )
        sanitize_public_payload(title + "\n" + body, policy)
        actions.append(
            {
                "action": "create_issue",
                "target": "github",
                "title": title,
                "body": body,
                "labels": sorted({label.strip() for label in labels if label.strip()}),
                "milestone": args.milestone,
            }
        )
    return actions


def apply_actions(
    actions: list[dict[str, Any]], policy: dict[str, Any], confirm: bool
) -> None:
    if not confirm:
        raise SystemExit("Refusing to mutate GitHub without --confirm")
    for action in actions:
        payload = json.dumps(action, sort_keys=True)
        sanitize_public_payload(payload, policy)
        if action["action"] == "create_issue":
            command = [
                "gh",
                "issue",
                "create",
                "--title",
                action["title"],
                "--body",
                action["body"],
            ]
            for label in action.get("labels", []):
                command.extend(["--label", label])
            if action.get("milestone"):
                command.extend(["--milestone", action["milestone"]])
        elif action["action"] in {"label_issue", "label_pr"}:
            command = ["gh", "issue", "edit", action["target"]]
            for label in action.get("labels", []):
                command.extend(["--add-label", label])
        else:
            print(f"Skipping unsupported action: {action['action']}")
            continue
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def handle_sync(args: argparse.Namespace) -> int:
    policy = load_policy()
    out_dir = output_dir(policy, args.output_dir)
    release_plan = resolve_repo_path(args.release_plan) if args.release_plan else None
    snapshot = build_snapshot(
        fetch_github_state(policy, args.issue_limit, args.pr_limit),
        policy,
        args.milestone,
    )
    write_snapshot(snapshot, out_dir, release_plan)
    print(out_dir / "today.md")
    return 0


def handle_brief(args: argparse.Namespace) -> int:
    policy = load_policy()
    out_dir = output_dir(policy, args.output_dir)
    snapshot = load_current(out_dir)
    maintainer_ga_readiness.attach_latest(snapshot)
    print(render_today(snapshot))
    return 0


def handle_release_report(args: argparse.Namespace) -> int:
    policy = load_policy()
    out_dir = output_dir(policy, args.output_dir)
    snapshot = load_current(out_dir)
    maintainer_ga_readiness.attach_latest(snapshot)
    release_plan = resolve_repo_path(args.release_plan)
    report = render_release_readiness(snapshot, release_plan)
    if args.write:
        (out_dir / "release-readiness.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


def handle_create_issues(args: argparse.Namespace) -> int:
    policy = load_policy()
    actions = create_issue_actions(args, policy)
    text = json.dumps(actions, indent=2, sort_keys=True)
    if args.apply:
        apply_actions(actions, policy, args.confirm)
    else:
        print(text)
    return 0


def handle_apply(args: argparse.Namespace) -> int:
    policy = load_policy()
    actions_path = Path(args.actions)
    if not actions_path.is_absolute():
        actions_path = REPO_ROOT / actions_path
    actions = json.loads(actions_path.read_text(encoding="utf-8"))
    apply_actions(actions, policy, args.confirm)
    return 0


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser("sync")
    sync.add_argument("--milestone", default=None)
    sync.add_argument("--output-dir", default=None)
    sync.add_argument("--release-plan", default=None)
    sync.add_argument("--issue-limit", type=int, default=None)
    sync.add_argument("--pr-limit", type=int, default=None)
    sync.set_defaults(func=handle_sync)

    brief = subparsers.add_parser("brief")
    brief.add_argument("--output-dir", default=None)
    brief.set_defaults(func=handle_brief)

    release = subparsers.add_parser("release-report")
    release.add_argument("--release-plan", required=True)
    release.add_argument("--output-dir", default=None)
    release.add_argument("--write", action="store_true")
    release.set_defaults(func=handle_release_report)

    create = subparsers.add_parser("create-issues")
    create.add_argument("--release-plan", required=True)
    create.add_argument("--milestone", default=None)
    create.add_argument("--output-dir", default=None)
    create.add_argument("--track", default=None)
    create.add_argument("--prefix", default="[Release]")
    create.add_argument("--labels", default=None)
    create.add_argument("--include-matched", action="store_true")
    create.add_argument("--dry-run", action="store_true")
    create.add_argument("--no-help-wanted", action="store_true")
    create.add_argument("--apply", action="store_true")
    create.add_argument("--confirm", action="store_true")
    create.set_defaults(func=handle_create_issues)

    apply = subparsers.add_parser("apply")
    apply.add_argument("--actions", required=True)
    apply.add_argument("--confirm", action="store_true")
    apply.set_defaults(func=handle_apply)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
