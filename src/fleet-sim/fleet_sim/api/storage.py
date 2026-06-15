"""Simple JSON-backed persistent storage for traces, fleets, and jobs."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

_STATE_ROOT = Path(os.environ.get("VLLM_SR_SIM_STATE_DIR", ".vllm-sr-sim"))
if not _STATE_ROOT.is_absolute():
    _STATE_ROOT = Path.cwd() / _STATE_ROOT

_BASE = _STATE_ROOT / "api_store"
_TRACES_DIR = _BASE / "traces"
_TRACE_META = _BASE / "traces_meta.json"
_FLEETS_FILE = _BASE / "fleets.json"
_JOBS_DIR = _BASE / "jobs"


def _ensure_dirs() -> None:
    _TRACES_DIR.mkdir(parents=True, exist_ok=True)
    _JOBS_DIR.mkdir(parents=True, exist_ok=True)
    if not _TRACE_META.exists():
        _TRACE_META.write_text("{}")
    if not _FLEETS_FILE.exists():
        _FLEETS_FILE.write_text("{}")


_ensure_dirs()


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Traces ────────────────────────────────────────────────────────────────────


def trace_upload_path(trace_id: str) -> Path:
    return _TRACES_DIR / trace_id


def save_trace_meta(trace_id: str, meta: dict) -> None:
    all_meta = json.loads(_TRACE_META.read_text())
    all_meta[trace_id] = meta
    _TRACE_META.write_text(json.dumps(all_meta, indent=2))


def get_trace_meta(trace_id: str) -> dict | None:
    all_meta = json.loads(_TRACE_META.read_text())
    return all_meta.get(trace_id)


def list_traces() -> list[dict]:
    return list(json.loads(_TRACE_META.read_text()).values())


def delete_trace(trace_id: str) -> bool:
    all_meta = json.loads(_TRACE_META.read_text())
    if trace_id not in all_meta:
        return False
    del all_meta[trace_id]
    _TRACE_META.write_text(json.dumps(all_meta, indent=2))
    p = trace_upload_path(trace_id)
    if p.exists():
        p.unlink()
    return True


def trace_seed_sentinel_path() -> Path:
    return _TRACE_META.parent / ".trace-seed-complete"


def trace_seed_completed() -> bool:
    return trace_seed_sentinel_path().exists()


def mark_trace_seed_completed() -> None:
    trace_seed_sentinel_path().write_text(now_iso())


# ── Fleets ────────────────────────────────────────────────────────────────────


def list_fleets() -> list[dict]:
    return list(json.loads(_FLEETS_FILE.read_text()).values())


def get_fleet(fleet_id: str) -> dict | None:
    return json.loads(_FLEETS_FILE.read_text()).get(fleet_id)


def save_fleet(fleet_id: str, data: dict) -> None:
    all_fleets = json.loads(_FLEETS_FILE.read_text())
    all_fleets[fleet_id] = data
    _FLEETS_FILE.write_text(json.dumps(all_fleets, indent=2))


def delete_fleet(fleet_id: str) -> bool:
    all_fleets = json.loads(_FLEETS_FILE.read_text())
    if fleet_id not in all_fleets:
        return False
    del all_fleets[fleet_id]
    _FLEETS_FILE.write_text(json.dumps(all_fleets, indent=2))
    return True


# ── Jobs ──────────────────────────────────────────────────────────────────────


def job_path(job_id: str) -> Path:
    return _JOBS_DIR / f"{job_id}.json"


def save_job(job_id: str, data: dict) -> None:
    job_path(job_id).write_text(json.dumps(data, indent=2))


def get_job(job_id: str) -> dict | None:
    p = job_path(job_id)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def list_jobs() -> list[dict]:
    jobs = []
    for p in sorted(
        _JOBS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True
    ):
        try:
            jobs.append(json.loads(p.read_text()))
        except Exception:
            pass
    return jobs


def delete_job(job_id: str) -> bool:
    p = job_path(job_id)
    if not p.exists():
        return False
    p.unlink()
    return True
