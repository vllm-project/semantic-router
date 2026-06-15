#!/usr/bin/env python3
"""Helpers for local agent harness artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agent_support import REPO_ROOT

ARTIFACT_ROOT = REPO_ROOT / ".agent-harness"
REPORT_ARTIFACT_DIR = ARTIFACT_ROOT / "reports"
LATEST_REPORT_PATH = REPORT_ARTIFACT_DIR / "latest-report.json"
SESSION_REPORT_DIR = REPORT_ARTIFACT_DIR / "sessions"


@dataclass(frozen=True)
class ReportArtifactWriteResult:
    explicit_path: Path | None = None
    latest_path: Path | None = None
    session_path: Path | None = None

    def written_paths(self) -> list[Path]:
        return [
            path
            for path in (self.explicit_path, self.latest_path, self.session_path)
            if path is not None
        ]


def resolve_artifact_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def write_explicit_report_artifact(
    raw_path: str, payload: str
) -> ReportArtifactWriteResult:
    path = resolve_artifact_path(raw_path)
    _write_text_file(path, payload)
    return ReportArtifactWriteResult(explicit_path=path)


def write_default_report_artifacts(payload: str) -> ReportArtifactWriteResult:
    latest_path = LATEST_REPORT_PATH
    session_path = _build_session_report_path()
    _write_text_file(latest_path, payload)
    _write_text_file(session_path, payload)
    return ReportArtifactWriteResult(
        latest_path=latest_path,
        session_path=session_path,
    )


def display_artifact_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _build_session_report_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = SESSION_REPORT_DIR / f"{timestamp}-agent-report.json"
    suffix = 1
    while candidate.exists():
        candidate = SESSION_REPORT_DIR / f"{timestamp}-agent-report-{suffix}.json"
        suffix += 1
    return candidate


def _write_text_file(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_payload = payload if payload.endswith("\n") else f"{payload}\n"
    tmp_path = path.parent / f".{path.name}.tmp"
    tmp_path.write_text(normalized_payload, encoding="utf-8")
    tmp_path.replace(path)
