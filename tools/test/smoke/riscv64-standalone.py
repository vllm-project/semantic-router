#!/usr/bin/env python3
"""Smoke-test a native standalone Router image with the maintained provider fixture."""

import argparse
import json
import platform
import socket
import subprocess
import tempfile
import time
import uuid
import os
from http import HTTPStatus
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[3]
DIRECT = build_opener(ProxyHandler({}))


def check_response(raw, stream, expected):
    if stream:
        normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
        if not normalized.endswith("\n\n"):
            raise ValueError("SSE event is missing its terminating blank line")
        events = []
        for frame in normalized.split("\n\n"):
            data = [
                line[5:].removeprefix(" ")
                for line in frame.splitlines()
                if line.startswith("data:")
            ]
            if data:
                events.append("\n".join(data))
        if not events or events[-1] != "[DONE]":
            raise ValueError("SSE did not terminate with [DONE]")
        chunks = [json.loads(item) for item in events[:-1]]
        if not chunks or any(item.get("model") != expected for item in chunks):
            raise ValueError("SSE selected the wrong model")
        text = "".join(
            choice.get("delta", {}).get("content") or ""
            for item in chunks
            for choice in item.get("choices", [])
        )
    else:
        item = json.loads(raw)
        if item.get("model") != expected or not item.get("choices"):
            raise ValueError(
                "buffered response selected the wrong model or has no choices"
            )
        text = item["choices"][0].get("message", {}).get("content") or ""
    if text != "ok":
        raise ValueError("provider fixture response content was not preserved")
    return text


def check_media_type(content_type, stream):
    expected = "text/event-stream" if stream else "application/json"
    if content_type.split(";", 1)[0].strip().lower() != expected:
        raise ValueError("response has the wrong media type")


def finish_cleanup(actions, receipt, report_path):
    errors = []
    try:
        for name, action in actions:
            try:
                action()
            except Exception as error:
                errors.append(f"{name}: {type(error).__name__}")
    finally:
        receipt["cleanup_errors"] = errors
        if errors:
            receipt["passed"] = False
        report_path.write_text(json.dumps(receipt, indent=2) + "\n")


def configuration(backend, listener):
    models = []
    for name in ("general-model", "code-model"):
        models.append(
            {
                "name": name,
                "provider_model_id": name,
                "api_format": "openai",
                "backend_refs": [
                    {
                        "name": "public-smoke",
                        "endpoint": backend,
                        "protocol": "http",
                        "weight": 100,
                    }
                ],
            }
        )
    return {
        "version": "v0.3",
        "listeners": [
            {
                "name": "public-smoke",
                "address": "0.0.0.0",
                "port": listener,
                "timeout": "30s",
            }
        ],
        "providers": {"defaults": {"model": "general-model"}, "models": models},
        "routing": {
            "modelCards": [
                {"name": name, "modality": "ar"}
                for name in ("general-model", "code-model")
            ],
            "signals": {
                "keywords": [
                    {
                        "name": "code",
                        "operator": "OR",
                        "keywords": ["write code"],
                        "case_sensitive": False,
                    }
                ]
            },
            "decisions": [
                {
                    "name": "code-route",
                    "priority": 100,
                    "rules": {
                        "operator": "AND",
                        "conditions": [{"type": "keyword", "name": "code"}],
                    },
                    "modelRefs": [{"model": "code-model"}],
                },
                {
                    "name": "general-route",
                    "priority": 1,
                    "rules": {"operator": "AND", "conditions": []},
                    "modelRefs": [{"model": "general-model"}],
                },
            ],
        },
        "entrypoints": [{"model_names": ["vllm-sr/auto"], "recipe": "default"}],
        "global": {
            "router": {"model_selection": {"enabled": False}},
            "services": {
                "response_api": {"enabled": False},
                "router_replay": {"enabled": False},
                "observability": {"tracing": {"enabled": False}},
            },
            "stores": {"response_cache": {"enabled": False}},
            "integrations": {"tools": {"enabled": False}},
            "model_catalog": {"kbs": []},
        },
    }


def wait_ready(url, process=None):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError("provider fixture exited before readiness")
        try:
            with DIRECT.open(url, timeout=2) as response:
                if response.status == HTTPStatus.OK:
                    return
        except (OSError, URLError):
            pass
        time.sleep(1)
    raise RuntimeError(f"readiness timed out: {url}")


def remove_router(docker, identity, output):
    try:
        (output / "router.log").write_text(docker("logs", identity))
    finally:
        docker("rm", "--force", "--volumes", identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--provider-python", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "riscv64":
        parser.error("run on native riscv64 Linux")
    output = (args.output or Path(tempfile.mkdtemp(prefix="vsr-smoke-"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = output / "result.json"
    if report.exists():
        parser.error("choose a fresh output directory")
    identity = "router-smoke-" + uuid.uuid4().hex[:10]
    network = identity + "-backend"
    receipt = {"passed": False, "cases": [], "machine": platform.machine()}
    provider = None
    network_created = False
    container_attempted = False

    def docker(*args):
        return subprocess.check_output(
            ["docker", *args], text=True, stderr=subprocess.STDOUT, timeout=120
        )

    try:
        if (
            docker("info", "--format", "{{.OSType}}/{{.Architecture}}").strip()
            != "linux/riscv64"
        ):
            raise RuntimeError("Docker daemon must run on linux/riscv64")
        image = json.loads(docker("image", "inspect", args.image))[0]
        if image["Architecture"] != "riscv64" or image["Os"] != "linux":
            raise RuntimeError("Router image must be linux/riscv64")
        receipt["image"] = {"name": args.image, "id": image["Id"]}
        docker("network", "create", network)
        network_created = True
        gateway = json.loads(docker("network", "inspect", network))[0]["IPAM"][
            "Config"
        ][0]["Gateway"]
        with socket.socket() as probe:
            probe.bind((gateway, 0))
            backend_port = probe.getsockname()[1]
        with (output / "provider.log").open("w") as log:
            provider = subprocess.Popen(
                [
                    args.provider_python,
                    "-m",
                    "provider_mocker",
                    "--host",
                    gateway,
                    "--port",
                    str(backend_port),
                ],
                env=os.environ
                | {
                    "PROVIDER_MOCKER_SCENARIO": "cli",
                    "PYTHONPATH": str(ROOT / "tools/test/services/provider-mocker"),
                },
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        wait_ready(f"http://{gateway}:{backend_port}/health", provider)
        config = output / "config.json"
        config.write_text(
            json.dumps(configuration(f"{gateway}:{backend_port}", 19000), indent=2)
        )
        container_attempted = True
        docker(
            "run",
            "--detach",
            "--pull",
            "never",
            "--name",
            identity,
            "--network",
            network,
            "--publish",
            "127.0.0.1::19000",
            "--volume",
            f"{config}:/app/config/config.yaml:ro",
            image["Id"],
            "-gateway=standalone",
            "-listener-address=0.0.0.0",
        )
        initial_id = None
        for phase in ("initial", "restart"):
            if phase == "restart":
                docker("restart", "--time", "10", identity)
            container = json.loads(docker("inspect", identity))[0]
            if container["Image"] != image["Id"]:
                raise RuntimeError("unexpected Router image")
            if initial_id is not None and container["Id"] != initial_id:
                raise RuntimeError("restart did not retain the container")
            initial_id = container["Id"]
            port = container["NetworkSettings"]["Ports"]["19000/tcp"][0]["HostPort"]
            url = f"http://127.0.0.1:{port}"
            wait_ready(url + "/ready")
            for prompt, expected in (
                ("write code for binary search", "code-model"),
                ("Explain photosynthesis", "general-model"),
            ):
                for stream in (False, True):
                    request = Request(
                        url + "/v1/chat/completions",
                        data=json.dumps(
                            {
                                "model": "vllm-sr/auto",
                                "stream": stream,
                                "messages": [{"role": "user", "content": prompt}],
                            }
                        ).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with DIRECT.open(request, timeout=30) as response:
                        check_media_type(
                            response.headers.get("Content-Type", ""), stream
                        )
                        check_response(response.read().decode(), stream, expected)
                        receipt["cases"].append(
                            {
                                "phase": phase,
                                "stream": stream,
                                "model": expected,
                                "status": response.status,
                            }
                        )
        receipt["passed"] = True
    finally:
        actions = []
        if container_attempted:
            actions.append(("router", lambda: remove_router(docker, identity, output)))
        if provider is not None:

            def stop_provider():
                provider.terminate()
                try:
                    provider.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    provider.kill()
                    provider.wait(timeout=10)

            actions.append(("provider", stop_provider))
        if network_created:
            actions.append(("network", lambda: docker("network", "rm", network)))
        finish_cleanup(actions, receipt, report)
    if not receipt["passed"]:
        raise RuntimeError(f"smoke failed; see {report}")
    print(json.dumps({"passed": len(receipt["cases"]), "report": str(report)}))


if __name__ == "__main__":
    main()
