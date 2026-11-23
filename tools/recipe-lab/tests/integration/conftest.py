"""Fixtures for the real-component (V2) integration tests.

These tests drive the real semantic-router binary and, for the sr-bench
scenarios, a live sr-bench service. They skip unless the corresponding
environment is present:

- ``RECIPE_LAB_ROUTER_BIN``: path to the semantic-router binary (R1, R2)
- ``RECIPE_LAB_SRBENCH_URL``: base URL of a running sr-bench service (R2)
- ``RECIPE_LAB_SRBENCH_STORE``: sr-bench store directory (R2, target registry)
- ``RECIPE_LAB_SRBENCH_BIN``: vllm-sr CLI for target registration (R2)
- ``RECIPE_LAB_HF_HOME``: optional HF_HOME forwarded to router processes
- ``RECIPE_LAB_VLLM_URL``: optional real vLLM weak lane, OpenAI base URL
  without the /v1 suffix (X7); when set, the R2 single-model reference
  target points at the strong lane and the workload becomes the store's
  prepared MMLU-Pro smoke dataset instead of the 6-case mock set
- ``RECIPE_LAB_VLLM_STRONG_URL``: optional strong lane base URL (X7),
  defaulting to ``RECIPE_LAB_VLLM_URL``
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest
import yaml

RECIPE_LAB_ROOT = Path(__file__).resolve().parents[2]
SMOKE_DIR = RECIPE_LAB_ROOT / "examples" / "smoke"


def free_port() -> int:
    """Reserve an ephemeral TCP port (bind-and-release race is negligible)."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _probe(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2.0):
            return True
    except (urllib.error.URLError, OSError):
        return False


def wait_http_ready(url: str, timeout_s: float = 30.0) -> None:
    """Poll ``url`` until it answers, or raise."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _probe(url):
            return
        time.sleep(0.2)
    raise RuntimeError(f"{url} not ready within {timeout_s}s")


@pytest.fixture(scope="session")
def router_bin() -> str:
    """Path to the real semantic-router binary."""
    value = os.environ.get("RECIPE_LAB_ROUTER_BIN", "")
    if not value:
        pytest.skip("RECIPE_LAB_ROUTER_BIN is not set; real router required")
    return value


@pytest.fixture(scope="session")
def srbench_url() -> str:
    """Base URL of the live sr-bench service."""
    value = os.environ.get("RECIPE_LAB_SRBENCH_URL", "")
    if not value:
        pytest.skip("RECIPE_LAB_SRBENCH_URL is not set; live sr-bench required")
    return value


@pytest.fixture(scope="session")
def hf_home() -> str | None:
    """Optional HF_HOME forwarded to router processes (model artifacts)."""
    return os.environ.get("RECIPE_LAB_HF_HOME") or None


@pytest.fixture(scope="session")
def smoke_dir() -> Path:
    """The checked-in smoke assets this suite renders its stack from."""
    return SMOKE_DIR


@pytest.fixture(scope="session")
def smoke_cases(smoke_dir: Path) -> Path:
    return smoke_dir / "cases.jsonl"


@pytest.fixture(scope="session")
def mock_delay_ms() -> int:
    """The shared mock backend's per-completion latency (X8/1398).

    ``RECIPE_LAB_MOCK_DELAY_MS`` keeps sr-bench arms in flight so the
    fault-injection poll can observe them without duplicating the case
    set; the fault test widens its dataset only when the delay is off.
    """
    raw = os.environ.get("RECIPE_LAB_MOCK_DELAY_MS", "").strip()
    try:
        return int(float(raw)) if raw else 0
    except ValueError:
        return 0


@pytest.fixture(scope="session")
def mock_backend(smoke_cases: Path, mock_delay_ms: int) -> dict:
    """One shared mock OpenAI backend on an ephemeral port.

    Serves the answer-bank build (R1) and the sr-bench single-model
    reference targets (R2). Trial backends inside the runner use their own
    instances on separate ports.
    """
    port = free_port()
    command = [
        sys.executable,
        str(SMOKE_DIR / "mock_backend.py"),
        "--port",
        str(port),
        "--cases",
        str(smoke_cases),
    ]
    if mock_delay_ms:
        command += ["--delay-ms", str(mock_delay_ms)]
    proc = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_http_ready(f"http://127.0.0.1:{port}/v1/models")
        yield {"port": port, "endpoint": f"http://127.0.0.1:{port}"}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture()
def backend_port() -> int:
    """Ephemeral port for one test's runner-managed mock backend."""
    return free_port()


# ---- X7: optional real vLLM lanes for the R2 verify ----


@pytest.fixture(scope="session")
def vllm_lanes() -> dict | None:
    """The real vLLM lanes, or None when the suite runs mock-only (X7).

    ``RECIPE_LAB_VLLM_URL`` is the weak lane's OpenAI base URL (no /v1
    suffix) serving the smoke config's local model names;
    ``RECIPE_LAB_VLLM_STRONG_URL`` is the strong lane serving the frontier
    model name and defaults to the weak lane. Both lanes are optional:
    unset env keeps the fully-mocked default mode.
    """
    weak = os.environ.get("RECIPE_LAB_VLLM_URL", "").strip().rstrip("/")
    if not weak:
        return None
    strong = (os.environ.get("RECIPE_LAB_VLLM_STRONG_URL") or weak).strip().rstrip("/")
    return {"weak": weak, "strong": strong}


def _hostport(base_url: str) -> str:
    """host:port form of a lane base URL for the router's backend endpoints."""
    parsed = urllib.parse.urlparse(base_url)
    if not parsed.hostname:
        raise RuntimeError(f"not a usable lane base URL: {base_url!r}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return f"{parsed.hostname}:{port}"


@pytest.fixture(scope="session")
def backends(
    smoke_dir: Path, vllm_lanes: dict | None, mock_backend: dict
) -> dict[str, str]:
    """Model name -> host:port each smoke-config backend points at.

    Mock mode points every model at the shared mock backend. Real-lane
    mode (X7) sends the local models to the weak lane and the frontier
    model to the strong lane, so the routers under test dispatch to real
    inference and the depth gate moves real accuracy.
    """
    if vllm_lanes is None:
        return {
            "local/private-qwen": f"127.0.0.1:{mock_backend['port']}",
            "local/omni": f"127.0.0.1:{mock_backend['port']}",
            "cloud/frontier-reasoning": f"127.0.0.1:{mock_backend['port']}",
        }
    return {
        "local/private-qwen": _hostport(vllm_lanes["weak"]),
        "local/omni": _hostport(vllm_lanes["weak"]),
        "cloud/frontier-reasoning": _hostport(vllm_lanes["strong"]),
    }


# ---- R2: sr-bench registry and real standalone routers ----


def _http_get(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError):
        return None


def _wait_router_ready(mgmt_url: str, proc: subprocess.Popen, log_path: Path) -> None:
    """Poll the management API until the router's model runtime is ready."""
    deadline = time.monotonic() + 600.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"router exited early (see {log_path})")
        body = _http_get(f"{mgmt_url}/api/v1/inventory/model-runtime")
        if body and '"ready":true' in body:
            return
        time.sleep(1.0)
    raise RuntimeError(f"router not ready within 600s (see {log_path})")


def _router_config_hash(mgmt_url: str) -> str:
    """The router's active config document hash (64 hex) for sr-bench MoM."""
    body = _http_get(f"{mgmt_url}/api/v1/config/hash")
    if not body:
        raise RuntimeError("router /api/v1/config/hash unavailable")
    payload = json.loads(body)
    document_hash = str(payload.get("active_runtime_hash", ""))
    if len(document_hash) != 64:
        raise RuntimeError(
            f"router config hash is not 64 hex: {document_hash!r} "
            f"(config/hash keys: {sorted(payload)})"
        )
    return document_hash


def _render_router_config(
    smoke_dir: Path,
    backends: dict[str, str],
    listener_port: int,
    depth_threshold: str | None,
) -> str:
    """The smoke config rendered for one router instance (YAML-level).

    Backend endpoints are rewritten per model name (mock mode: everything
    on the mock backend; real-lane mode X7: local models on the weak
    lane, the frontier model on the strong lane), the listener moves to
    an ephemeral port, and the depth gate may be lowered for the
    candidate router.
    """
    config = yaml.safe_load(
        (smoke_dir / "smoke-config.yaml").read_text(encoding="utf-8")
    )
    for model in config["providers"]["models"]:
        endpoint = backends.get(model["name"])
        if endpoint is None:
            raise RuntimeError(
                f"smoke config model {model['name']!r} has no backend endpoint"
            )
        for ref in model["backend_refs"]:
            ref["endpoint"] = endpoint
    config["listeners"][0]["port"] = listener_port
    if depth_threshold is not None:
        config["routing"]["signals"]["decision"][0]["predicate"]["gte"] = float(
            depth_threshold
        )
    return yaml.safe_dump(config, sort_keys=False)


def _start_router(
    router_bin: str,
    smoke_dir: Path,
    backends: dict[str, str],
    hf_home: str | None,
    work_dir: Path,
    depth_threshold: str | None = None,
) -> dict:
    """Start one standalone router on the rendered smoke config."""
    listener_port = free_port()
    config_text = _render_router_config(
        smoke_dir, backends, listener_port, depth_threshold
    )
    config_path = work_dir / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    api_port = free_port()
    env = dict(os.environ)
    if hf_home:
        env["HF_HOME"] = hf_home
    log_path = work_dir / "router.log"
    proc = subprocess.Popen(
        [
            router_bin,
            "-config",
            str(config_path),
            "-gateway",
            "standalone",
            "-api-port",
            str(api_port),
        ],
        stdout=log_path.open("ab"),
        stderr=subprocess.STDOUT,
        env=env,
    )
    mgmt_url = f"http://127.0.0.1:{api_port}"
    try:
        _wait_router_ready(mgmt_url, proc, log_path)
        config_hash = _router_config_hash(mgmt_url)
    except BaseException:
        proc.terminate()
        raise
    return {
        "proc": proc,
        "listener_url": f"http://127.0.0.1:{listener_port}",
        "mgmt_url": mgmt_url,
        "config_hash": config_hash,
        "log_path": log_path,
    }


def _stop_router(stack: dict) -> None:
    proc = stack["proc"]
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def srbench_store() -> str:
    """sr-bench store directory backing the live service (target registry)."""
    value = os.environ.get("RECIPE_LAB_SRBENCH_STORE", "")
    if not value:
        pytest.skip("RECIPE_LAB_SRBENCH_STORE is not set; sr-bench store required")
    return value


@pytest.fixture(scope="session")
def srbench_bin() -> str:
    """vllm-sr CLI used to register targets and cancel runs."""
    value = os.environ.get("RECIPE_LAB_SRBENCH_BIN") or shutil.which("vllm-sr")
    if not value:
        pytest.skip("vllm-sr CLI not found; needed for sr-bench target registry")
    return value


@pytest.fixture(scope="session")
def router_a(
    router_bin: str,
    smoke_dir: Path,
    backends: dict[str, str],
    hf_home: str | None,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict:
    """Baseline router: the smoke config's own behavior (depth >= 0.6)."""
    stack = _start_router(
        router_bin, smoke_dir, backends, hf_home, tmp_path_factory.mktemp("router-a")
    )
    yield stack
    _stop_router(stack)


@pytest.fixture(scope="session")
def router_b(
    router_bin: str,
    smoke_dir: Path,
    backends: dict[str, str],
    hf_home: str | None,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict:
    """Candidate router: the depth gate lowered to 0.15 routes far more
    traffic to the frontier lane, so the AB comparison has real movement."""
    stack = _start_router(
        router_bin,
        smoke_dir,
        backends,
        hf_home,
        tmp_path_factory.mktemp("router-b"),
        depth_threshold="0.15",
    )
    yield stack
    _stop_router(stack)


# ---- X7: the R2 workload per mode ----


@pytest.fixture(scope="session")
def mmlu_smoke_dataset(srbench_store: str) -> dict:
    """The store's prepared MMLU-Pro smoke dataset (real-lane V2, X7).

    Scans the store's datasets for the newest smoke-profile dataset
    covering mmlu-pro. Returns its cases file, case count, and content
    hash so the spec and manifest can bind the real question set.
    """
    matches: list[tuple[float, Path]] = []
    for manifest_path in (Path(srbench_store) / "datasets").glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        benchmarks = {
            entry.get("id") if isinstance(entry, dict) else entry
            for entry in manifest.get("benchmarks") or []
        }
        if manifest.get("profile") == "smoke" and "mmlu-pro" in benchmarks:
            matches.append((manifest_path.stat().st_mtime, manifest_path.parent))
    if not matches:
        pytest.skip(
            "no prepared mmlu-pro smoke dataset in the sr-bench store "
            "(prepare one with: vllm-sr benchmark --store <store> dataset prepare)"
        )
    folder = max(matches)[1]
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    return {
        "path": folder / "cases.jsonl",
        "case_count": int(manifest["case_count"]),
        "sha256": str(manifest["sha256"]),
    }


@pytest.fixture(scope="session")
def workload(vllm_lanes: dict | None, smoke_cases: Path, request) -> dict:
    """The R2 verify workload for the active mode (X7).

    Mock mode keeps the 6-case answer-bank set with the dual-format
    contract binding. Real-lane mode swaps in the store's prepared
    MMLU-Pro smoke file as both the cases file and the sr-bench dataset
    (legacy binding: live_dataset_hash falls back to the cases content
    hash), so sr-bench scores real questions on real models.
    """
    if vllm_lanes is None:
        return {
            "mode": "mock",
            "cases": smoke_cases,
            "dataset": None,  # each test writes its own rendering file
            "min_cases": 3,
            "declare_rendering": True,
            "sampling": None,  # sr-bench defaults; the mock answers are short
            "limits": {
                "total_timeout_s": 1200,
                "idle_timeout_s": 300,
                "max_run_seconds": 3600,
            },
        }
    mmlu = request.getfixturevalue("mmlu_smoke_dataset")
    return {
        "mode": "vllm",
        "cases": mmlu["path"],
        "dataset": mmlu["path"],
        "min_cases": mmlu["case_count"],
        "declare_rendering": False,
        # Qwen3.5 thinking chains blew the budget even at 8192 tokens
        # (21-29% truncation on the second real-lane run); the runs disable
        # thinking through the chat template and keep a generous ceiling, so
        # completions land far under the cap on every arm (X7).
        "sampling": {
            "max_tokens": 8192,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "limits": {
            "concurrency": 4,
            "total_timeout_s": 7200,
            "idle_timeout_s": 600,
            "max_run_seconds": 14400,
            "case_timeout_s": 900,
            # sr-bench's default output cap is 4096; raising it here lets
            # sampling.max_tokens=8192 pass the request-params validation.
            "max_output_tokens": 8192,
        },
    }
