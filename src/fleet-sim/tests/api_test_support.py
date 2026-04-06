"""Shared helpers for fleet-sim API tests."""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

_DATA = Path(__file__).parent.parent / "data"


def _load_cdf(name: str = "azure") -> list:
    path = _DATA / f"{name}_cdf.json"
    raw = json.loads(path.read_text())
    return raw["cdf"] if isinstance(raw, dict) else raw


_MINIMAL_JSONL = (
    '{"prompt_tokens": 128, "generated_tokens": 64, "timestamp": 0.0}\n'
    '{"prompt_tokens": 256, "generated_tokens": 128, "timestamp": 1.0}\n'
    '{"prompt_tokens": 512, "generated_tokens": 64, "timestamp": 2.0}\n'
)

_MINIMAL_CSV = (
    "prompt_tokens,generated_tokens,timestamp\n"
    "128,64,0.0\n"
    "256,128,1.0\n"
    "512,64,2.0\n"
)


def _wait_job_done(client: TestClient, job_id: str, max_wait: float = 60.0) -> dict:
    """Poll until a job reaches terminal status or max_wait is exceeded."""
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        data = response.json()
        if data["status"] in ("done", "failed"):
            return data
        time.sleep(0.2)
    return client.get(f"/api/jobs/{job_id}").json()
