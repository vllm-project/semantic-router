#!/usr/bin/env python3
"""Qualify one GB10 host with the maintained Kai recipe.

The passing artifact is a digest-pinned ARM64 CUDA image and a receipt from
this host. A local ``latest`` tag can exercise ``vllm-sr serve --platform cuda``
and is not a qualification image. CPU and amd64 evidence cannot satisfy this
profile.

Example::

    VLLM_SR_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr-cuda@sha256:<digest> \
      python3 tools/qualification/gb10/qualification.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

RECEIPT_ID = "gb10-kai-cuda"
SCHEMA_VERSION = "gb10-qualification/v1"
MODEL_REPO = "vllm-sr/Decision-2.0-Kai-0.6B"
MODEL_REVISION = "cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764"
KAI_DEPLOYMENT = "decision-kai"
LOCAL_ALIAS = "local/generation"
EXTERNAL_ALIAS = "external/generation"
REASONING_GTE = 0.5
DIFFICULTY_GTE = 1.0
CATALOG_FLOOR_GIB = 4
LATENCY_SAMPLES = 20
STACK_NAME = "gb10-qualification"
PORT_OFFSET = 210
API_PORT = 8080 + PORT_OFFSET
LISTENER_PORT = 8899 + PORT_OFFSET
ROUTER_CONTAINER = f"{STACK_NAME}-vllm-sr-router-container"
READY_TIMEOUT_SECONDS = 1800
RECORDER_COMMAND = (
    "python3 src/model-runtime/tools/golden_answers.py "
    "vllm-sr/Decision-2.0-Kai-0.6B --device cuda:0"
)
FORBIDDEN_EVIDENCE = "linux-amd64-cpu"
_DIGEST = re.compile(r"@sha256:[0-9a-f]{64}\Z")
PROMPTS = (
    "Write a Python function that merges two sorted lists and explain its running time.",
    "Prove step by step why merging two sorted lists is linear, and write the Python function with its loop invariant.",
    "What is two plus two?",
    "Name one color.",
)
_RUNTIME_MODELS = r"""
import glob
import json
import socket
from http.client import HTTPConnection

class UnixHTTPConnection(HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost")
        self._unix_path = path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self._unix_path)
        self.sock = sock

cards = []
paths = glob.glob("/tmp/vllm-srun-*/*.sock") + glob.glob("/tmp/vsr-*/*.sock")
for path in paths:
    conn = UnixHTTPConnection(path)
    try:
        conn.request("GET", "/v1/models")
        response = conn.getresponse()
        body = response.read()
        if response.status != 200:
            continue
        cards.extend(json.loads(body).get("data") or [])
    finally:
        conn.close()
print(json.dumps({"data": cards}))
"""


class QualificationError(Exception):
    """The host, image, model, or route failed a qualification check."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def recipe_config_path() -> Path:
    return repo_root() / "config" / "recipes" / "gb10" / "config.yaml"


def golden_answers_path() -> Path:
    return (
        repo_root()
        / "src"
        / "model-runtime"
        / "vllm_srun"
        / "registry"
        / "golden_answers.json"
    )


def observed_architecture(system: str, machine: str) -> str:
    """Return linux/arm64 for a GB10 host."""
    if system.lower() == "linux" and machine.lower() in {"aarch64", "arm64"}:
        return "linux/arm64"
    raise QualificationError(
        f"this profile records linux/arm64; the host reported {system}/{machine}"
    )


def require_gb10_gpu(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the single GB10 at CUDA index 0."""
    matches = [row for row in rows if "gb10" in str(row.get("name", "")).lower()]
    if len(matches) != 1:
        names = [row.get("name") for row in rows]
        raise QualificationError(
            f"this profile requires exactly one GB10 GPU; nvidia-smi reported {names}"
        )
    gpu = matches[0]
    if gpu.get("index") != 0:
        raise QualificationError(
            "the recipe pins cuda:0; the GB10 is at index " f"{gpu.get('index')}"
        )
    return gpu


def require_digest_image(image: str) -> str:
    """Accept a digest pin and reject a latest tag."""
    pinned = image.strip()
    if ":latest" in pinned or not _DIGEST.search(pinned):
        raise QualificationError(
            "VLLM_SR_IMAGE must be a digest-pinned reference ending in "
            "@sha256:<64 hex>; a latest tag is a development image"
        )
    return pinned


def cuda_reference_key(compute_capability: str) -> str:
    if not str(compute_capability).startswith("sm_"):
        raise QualificationError(
            f"compute capability {compute_capability!r} is not an sm_ key"
        )
    return f"cuda:{compute_capability}"


def golden_reference_present(table: dict[str, Any], reference_key: str) -> bool:
    entry = table.get(MODEL_REPO) or {}
    answers = entry.get("answers") or {}
    return entry.get("revision") == MODEL_REVISION and reference_key in answers


def missing_reference_message(reference_key: str) -> str:
    return (
        f"golden reference {reference_key} for {MODEL_REPO} revision "
        f"{MODEL_REVISION} is absent. Record it on this host with: "
        f"{RECORDER_COMMAND}. Add the parity note under "
        "src/model-runtime/docs/records/ before this profile can pass."
    )


def select_alias(
    domain: str | None, reasoning: float | None, difficulty: float | None
) -> str:
    """Map Kai's three answers to the recipe's generation alias."""
    if domain is None or reasoning is None or difficulty is None:
        raise QualificationError("Kai did not answer domain, reasoning, and difficulty")
    if domain not in {"code", "math", "other"}:
        raise QualificationError(
            f"Kai domain {domain!r} is outside code, math, and other"
        )
    if domain == "code" and reasoning >= REASONING_GTE and difficulty >= DIFFICULTY_GTE:
        return EXTERNAL_ALIAS
    return LOCAL_ALIAS


def answers_from_preview(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("decision_result") or {}
    matched = (result.get("matched_signals") or {}).get("decision") or []
    domain = None
    for name in matched:
        if isinstance(name, str) and name.startswith("gb10_domain:"):
            domain = name.split(":", 1)[1]
    values = payload.get("signal_values") or {}
    return {
        "domain": domain,
        "reasoning": values.get("decision:gb10_reasoning"),
        "difficulty": values.get("decision:gb10_difficulty"),
        "selected_model": payload.get("selected_model") or "",
        "decision_name": result.get("decision_name") or "",
    }


def prove_route(answers: dict[str, Any], upstream_alias: str) -> dict[str, Any]:
    expected = select_alias(
        answers.get("domain"), answers.get("reasoning"), answers.get("difficulty")
    )
    if answers.get("selected_model") != expected or upstream_alias != expected:
        raise QualificationError(
            "route selected "
            f"{answers.get('selected_model')!r} via upstream {upstream_alias!r}; "
            f"Kai's answer selects {expected!r}"
        )
    return {
        "domain": answers["domain"],
        "reasoning_noul": answers["reasoning"],
        "difficulty_score": answers["difficulty"],
        "decision": answers["decision_name"],
        "alias": expected,
        "upstream_received": True,
    }


def require_both_cases(cases: list[dict[str, Any]]) -> None:
    aliases = {case["alias"] for case in cases}
    if aliases != {LOCAL_ALIAS, EXTERNAL_ALIAS}:
        raise QualificationError(
            "the bounded prompt set must produce both "
            f"{LOCAL_ALIAS} and {EXTERNAL_ALIAS}; observed {sorted(aliases)}"
        )


def parse_nvidia_smi(text: str) -> list[dict[str, Any]]:
    """Parse ``nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.used``."""
    rows = []
    for index, line in enumerate(line for line in text.splitlines() if line.strip()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise QualificationError(f"unexpected nvidia-smi row: {line}")
        name, compute_cap, driver, memory = parts
        major, minor = compute_cap.split(".", 1)
        rows.append(
            {
                "index": index,
                "name": name,
                "compute_capability": f"sm_{int(major)}{int(minor)}",
                "driver_version": driver,
                "memory_used_mib": int(float(memory)),
            }
        )
    if not rows:
        raise QualificationError("nvidia-smi reported no GPUs")
    return rows


def parse_image_platform(text: str) -> tuple[str, str]:
    parts = text.split()
    if len(parts) != 2:
        raise QualificationError(f"image inspect returned {text!r}")
    return parts[0], parts[1]


def require_arm64_image(architecture: str, operating_system: str) -> None:
    if architecture == "arm64" and operating_system == "linux":
        return
    raise QualificationError(
        "the qualification image must be linux/arm64; "
        f"inspect reported {operating_system}/{architecture}"
    )


def rewrite_endpoints(text: str, local_endpoint: str, external_endpoint: str) -> str:
    rewritten = text.replace(
        "endpoint: local-generation:8000", f"endpoint: {local_endpoint}", 1
    ).replace("endpoint: external-generation:8000", f"endpoint: {external_endpoint}", 1)
    if "local-generation:8000" in rewritten or "external-generation:8000" in rewritten:
        raise QualificationError("endpoint rewrite left a canonical backend in place")
    if local_endpoint not in rewritten or external_endpoint not in rewritten:
        raise QualificationError("endpoint rewrite missed a generation backend")
    return rewritten


def require_kai_card(cards: list[dict[str, Any]], reference_key: str) -> dict[str, Any]:
    matches = [
        card
        for card in cards
        if card.get("repo") == MODEL_REPO or card.get("id") == KAI_DEPLOYMENT
    ]
    if len(matches) != 1:
        raise QualificationError("GET /v1/models must list one Kai deployment")
    card = matches[0]
    device = str(card.get("device") or "")
    golden = card.get("golden") or {}
    if not device.startswith("cuda:"):
        raise QualificationError(
            f"Kai device is {device!r}; the profile requires a cuda device"
        )
    if golden.get("status") != "matched" or golden.get("reference") != reference_key:
        raise QualificationError(
            f"Kai golden status is {golden.get('status')!r} "
            f"reference {golden.get('reference')!r}; expected matched {reference_key}. "
            + missing_reference_message(reference_key)
        )
    if card.get("revision") != MODEL_REVISION:
        raise QualificationError("served Kai revision differs from the recipe pin")
    return card


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise QualificationError("latency distribution has no samples")
    rank = math.ceil(percent / 100 * len(ordered))
    index = min(len(ordered) - 1, max(0, rank - 1))
    return ordered[index]


def latency_distribution(samples_ms: list[float], prompt: str) -> dict[str, Any]:
    if len(samples_ms) != LATENCY_SAMPLES:
        raise QualificationError(
            f"latency sample count is {len(samples_ms)}; "
            f"the profile records {LATENCY_SAMPLES}"
        )
    return {
        "samples": len(samples_ms),
        "unit": "ms",
        "minimum": min(samples_ms),
        "p50": percentile(samples_ms, 50),
        "p95": percentile(samples_ms, 95),
        "maximum": max(samples_ms),
        "method": "nearest-rank",
        "conditions": {
            "state": "warm",
            "concurrency": 1,
            "prompt": prompt,
            "generation": "controlled stub returns immediately",
        },
    }


def build_receipt(
    *,
    image: str,
    gpu: dict[str, Any],
    reference_key: str,
    source_revision: str,
    cli_version: str,
    card: dict[str, Any],
    startup_seconds: float,
    memory_before_mib: int,
    memory_after_mib: int,
    cases: list[dict[str, Any]],
    latency: dict[str, Any],
    upstreams: dict[str, str],
) -> dict[str, Any]:
    require_both_cases(cases)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "id": RECEIPT_ID,
        "status": "passed",
        "architecture": "linux/arm64",
        "device": {
            "name": gpu["name"],
            "index": gpu["index"],
            "placement": "cuda:0",
            "accelerator": "cuda",
            "compute_capability": gpu["compute_capability"],
            "driver_version": gpu["driver_version"],
        },
        "image": image,
        "source_revision": source_revision,
        "cli_version": cli_version,
        "runtime": {
            "engine": card.get("engine"),
            "dtype": card.get("dtype"),
            "device": card.get("device"),
            "accelerator_validated_observed": bool(card.get("accelerator_validated")),
        },
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "golden_status": "matched",
            "golden_reference": reference_key,
        },
        "controlled_upstreams": {
            "aliases": upstreams,
            "service": "in-process OpenAI stub started by this profile",
        },
        "routing_cases": cases,
        "modeled_capacity": {
            "min_device_memory_gib": CATALOG_FLOOR_GIB,
            "role": "catalog placement input for Decision-2.0-Kai-0.6B",
        },
        "resources": {
            "startup_time_seconds": startup_seconds,
            "gpu_memory_used_mib_before_serve": memory_before_mib,
            "gpu_memory_used_mib_after_ready": memory_after_mib,
            "gpu_memory_footprint_mib": memory_after_mib - memory_before_mib,
            "gpu_memory_unit": "MiB",
            "gpu_memory_source": "nvidia-smi memory.used, device-wide",
            "latency": latency,
        },
        "qualifies_all_nvidia_devices": False,
    }
    validate_passing_receipt(receipt)
    return receipt


def validate_passing_receipt(receipt: dict[str, Any]) -> None:
    encoded = json.dumps(receipt)
    if FORBIDDEN_EVIDENCE in encoded:
        raise QualificationError(f"a passing receipt cannot carry {FORBIDDEN_EVIDENCE}")
    if receipt.get("id") != RECEIPT_ID or receipt.get("status") != "passed":
        raise QualificationError("receipt id or status is not a GB10 pass")
    if receipt.get("architecture") != "linux/arm64":
        raise QualificationError("receipt architecture must be linux/arm64")
    device = receipt.get("device") or {}
    if device.get("accelerator") != "cuda" or device.get("placement") != "cuda:0":
        raise QualificationError("receipt device must be cuda:0")
    if not _DIGEST.search(str(receipt.get("image") or "")):
        raise QualificationError("receipt image must be digest-pinned")
    reference = (receipt.get("model") or {}).get("golden_reference", "")
    if not str(reference).startswith("cuda:"):
        raise QualificationError("receipt golden reference must be a cuda key")
    require_both_cases(list(receipt.get("routing_cases") or []))
    resources = receipt.get("resources") or {}
    if "min_device_memory_gib" in resources:
        raise QualificationError(
            "measured resources must stay separate from the catalog memory floor"
        )
    latency = resources.get("latency") or {}
    if latency.get("samples") != LATENCY_SAMPLES or latency.get("unit") != "ms":
        raise QualificationError("receipt latency distribution is incomplete")
    if receipt.get("qualifies_all_nvidia_devices") is not False:
        raise QualificationError("this receipt qualifies one GB10 configuration")
    modeled = receipt.get("modeled_capacity") or {}
    if modeled.get("min_device_memory_gib") != CATALOG_FLOOR_GIB:
        raise QualificationError("catalog floor must stay on modeled_capacity")


class OwnedStack:
    """Names started by one qualification run."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def start(self, name: str) -> None:
        self.names.append(name)

    def stop_names(self) -> list[str]:
        names = list(self.names)
        self.names.clear()
        return names


class RecordingStub:
    """OpenAI-compatible stub bound for one generation alias."""

    def __init__(self, alias: str) -> None:
        self.alias = alias
        self.requests: list[str] = []
        self.port = 0
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> RecordingStub:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/v1/models"):
                    body = json.dumps(
                        {"object": "list", "data": [{"id": stub.alias}]}
                    ).encode()
                    self._send(200, body)
                    return
                self._send(404, b"")

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                stub.requests.append(self.path)
                body = json.dumps(
                    {
                        "id": "gb10-stub",
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                ).encode()
                self._send(200, body)

            def _send(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


def stack_env(image: str) -> dict[str, str]:
    env = os.environ.copy()
    env["VLLM_SR_IMAGE"] = image
    env["VLLM_SR_STACK_NAME"] = STACK_NAME
    env["VLLM_SR_PORT_OFFSET"] = str(PORT_OFFSET)
    return env


def _run_text(command: list[str], env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        raise QualificationError(
            f"{' '.join(command)} failed: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def _nvidia_smi() -> list[dict[str, Any]]:
    text = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    return parse_nvidia_smi(text)


def _http_json(
    method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 120
) -> tuple[int, dict[str, Any], dict[str, str]]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode())
            headers = {key.lower(): value for key, value in response.headers.items()}
            return response.status, body, headers
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"error": raw}
        return exc.code, body, {}


def _wait_ready(deadline: float) -> None:
    url = f"http://127.0.0.1:{API_PORT}/api/v1/inventory/model-runtime"
    while time.monotonic() < deadline:
        try:
            status, body, _headers = _http_json("GET", url, timeout=5)
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(2)
            continue
        if status == 200:
            for deployment in body.get("deployments") or []:
                device = str(deployment.get("device") or "")
                if (
                    deployment.get("name") == KAI_DEPLOYMENT
                    and deployment.get("ready")
                    and device.startswith("cuda:")
                ):
                    return
        time.sleep(2)
    raise QualificationError("Kai did not become ready on cuda before the timeout")


def _runtime_cards() -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["docker", "exec", ROUTER_CONTAINER, "python3", "-c", _RUNTIME_MODELS],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise QualificationError(
            "GET /v1/models inside the router container failed: "
            f"{completed.stderr.strip()}"
        )
    return list(json.loads(completed.stdout).get("data") or [])


def _preview(prompt: str) -> dict[str, Any]:
    status, body, _headers = _http_json(
        "POST",
        f"http://127.0.0.1:{API_PORT}/api/v1/routing/preview",
        {"text": prompt, "model": "vllm-sr/auto"},
    )
    if status != 200:
        raise QualificationError(f"routing preview failed with HTTP {status}: {body}")
    return answers_from_preview(body)


def _chat(prompt: str) -> tuple[str, float]:
    started = time.monotonic()
    status, _body, headers = _http_json(
        "POST",
        f"http://127.0.0.1:{LISTENER_PORT}/v1/chat/completions",
        {
            "model": "vllm-sr/auto",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
        },
    )
    elapsed_ms = (time.monotonic() - started) * 1000
    if status != 200:
        raise QualificationError(f"chat completion failed with HTTP {status}")
    return headers.get("x-vsr-selected-model", ""), elapsed_ms


def _stop_stack(env: dict[str, str], serve: subprocess.Popen[bytes] | None) -> None:
    subprocess.run(["vllm-sr", "stop"], check=False, env=env)
    if serve is not None and serve.poll() is None:
        serve.terminate()
        try:
            serve.wait(timeout=30)
        except subprocess.TimeoutExpired:
            serve.kill()


def qualify(image: str, receipt_path: Path) -> dict[str, Any]:
    """Serve the recipe on this host and write a passing receipt."""
    architecture = observed_architecture(platform.system(), platform.machine())
    pinned = require_digest_image(image)
    before_rows = _nvidia_smi()
    gpu = require_gb10_gpu(before_rows)
    reference_key = cuda_reference_key(str(gpu["compute_capability"]))
    golden_table = json.loads(golden_answers_path().read_text(encoding="utf-8"))
    if not golden_reference_present(golden_table, reference_key):
        raise QualificationError(missing_reference_message(reference_key))
    source_revision = _run_text(["git", "rev-parse", "HEAD"])
    cli_version = _run_text(["vllm-sr", "--version"])
    env = stack_env(pinned)
    serve: subprocess.Popen[bytes] | None = None
    config_dir = repo_root() / ".agent-harness" / "gb10-qualification"
    config_dir.mkdir(parents=True, exist_ok=True)
    try:
        with RecordingStub(LOCAL_ALIAS) as local_stub, RecordingStub(
            EXTERNAL_ALIAS
        ) as external_stub:
            rendered = rewrite_endpoints(
                recipe_config_path().read_text(encoding="utf-8"),
                f"host.docker.internal:{local_stub.port}",
                f"host.docker.internal:{external_stub.port}",
            )
            rendered_path = config_dir / "config.yaml"
            rendered_path.write_text(rendered, encoding="utf-8")
            started = time.monotonic()
            serve = subprocess.Popen(
                [
                    "vllm-sr",
                    "serve",
                    "--platform",
                    "cuda",
                    "--config",
                    str(rendered_path),
                ],
                env=env,
            )
            _wait_ready(started + READY_TIMEOUT_SECONDS)
            startup_seconds = time.monotonic() - started
            image_arch, image_os = parse_image_platform(
                _run_text(
                    [
                        "docker",
                        "image",
                        "inspect",
                        "--format",
                        "{{.Architecture}} {{.Os}}",
                        pinned,
                    ]
                )
            )
            require_arm64_image(image_arch, image_os)
            card = require_kai_card(_runtime_cards(), reference_key)
            after = require_gb10_gpu(_nvidia_smi())
            stubs = {
                LOCAL_ALIAS: local_stub,
                EXTERNAL_ALIAS: external_stub,
            }
            cases = []
            for prompt in PROMPTS:
                preview = _preview(prompt)
                before_counts = {
                    alias: len(stub.requests) for alias, stub in stubs.items()
                }
                header_model, _elapsed = _chat(prompt)
                if header_model and header_model != preview["selected_model"]:
                    raise QualificationError(
                        "chat header "
                        f"{header_model!r} disagrees with preview "
                        f"{preview['selected_model']!r}"
                    )
                received = [
                    alias
                    for alias, stub in stubs.items()
                    if len(stub.requests) > before_counts[alias]
                ]
                if len(received) != 1:
                    raise QualificationError(
                        f"expected one controlled upstream to receive the chat; got {received}"
                    )
                case = prove_route(preview, received[0])
                case["prompt"] = prompt
                cases.append(case)
            require_both_cases(cases)
            latency_prompt = PROMPTS[0]
            _chat(latency_prompt)
            samples = [_chat(latency_prompt)[1] for _ in range(LATENCY_SAMPLES)]
            receipt = build_receipt(
                image=pinned,
                gpu=gpu,
                reference_key=reference_key,
                source_revision=source_revision,
                cli_version=cli_version,
                card=card,
                startup_seconds=startup_seconds,
                memory_before_mib=int(gpu["memory_used_mib"]),
                memory_after_mib=int(after["memory_used_mib"]),
                cases=cases,
                latency=latency_distribution(samples, latency_prompt),
                upstreams={
                    LOCAL_ALIAS: f"host.docker.internal:{local_stub.port}",
                    EXTERNAL_ALIAS: f"host.docker.internal:{external_stub.port}",
                },
            )
            receipt["architecture"] = architecture
            validate_passing_receipt(receipt)
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
            return receipt
    finally:
        if serve is not None:
            _stop_stack(env, serve)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--receipt",
        type=Path,
        default=repo_root() / ".agent-harness" / "gb10-qualification" / "receipt.json",
    )
    args = parser.parse_args(argv)
    image = os.environ.get("VLLM_SR_IMAGE", "")
    try:
        receipt = qualify(image, args.receipt)
    except QualificationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"wrote {args.receipt} for {receipt['id']} on {receipt['architecture']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
