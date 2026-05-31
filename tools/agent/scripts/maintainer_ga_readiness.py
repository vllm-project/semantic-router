"""Session-aware GA readiness helpers for the maintainer board."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
GA_READINESS_ROOT = REPO_ROOT / ".agent-harness" / "reports" / "session-routing-ga"


def latest_ga_readiness_report() -> dict[str, Any] | None:
    if not GA_READINESS_ROOT.exists():
        return None
    report_paths = sorted(
        GA_READINESS_ROOT.glob("*/ga-readiness.json"),
        key=lambda path: path.parent.name,
        reverse=True,
    )
    for path in report_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        return {
            "path": str(path.relative_to(REPO_ROOT)),
            "generated_at": data.get("generated_at", ""),
            "ga_ready": bool(data.get("ga_ready", False)),
            "blocker_count": int(data.get("blocker_count", 0)),
            "blockers": blocker_summaries(data),
        }
    return None


def blocker_summaries(report: dict[str, Any]) -> list[dict[str, str]]:
    blockers = []
    source = report.get("blockers")
    if not source:
        source = [
            item
            for item in report.get("requirements", [])
            if item.get("status") in {"blocked", "missing"}
        ]
    for item in source:
        blockers.append(
            {
                "id": str(item.get("id", "")),
                "title": str(item.get("title", "")),
                "status": str(item.get("status", "")),
            }
        )
    return blockers


def attach_latest(snapshot: dict[str, Any]) -> dict[str, Any]:
    latest = latest_ga_readiness_report()
    if latest is not None:
        snapshot["session_routing_ga_readiness"] = latest
    return snapshot


def render(snapshot: dict[str, Any]) -> list[str]:
    readiness = snapshot.get("session_routing_ga_readiness")
    if not readiness:
        return []
    lines = [
        "## Session-Aware GA Readiness",
        "",
        f"- report: `{readiness['path']}`",
        f"- generated_at: {readiness.get('generated_at') or 'unknown'}",
        f"- ga_ready: {str(readiness['ga_ready']).lower()}",
        f"- blockers: {readiness['blocker_count']}",
        "",
    ]
    blockers = readiness.get("blockers", [])
    if blockers:
        lines.extend(
            f"- {item['status']}: {item['title']} (`{item['id']}`)" for item in blockers
        )
    else:
        lines.append("- none")
    lines.append("")
    return lines
