#!/usr/bin/env python3
"""Install vllm-sr[runtime] from the built wheels with CPU PyTorch and serve a tiny model."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import venv
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]
TORCH = "torch==2.10.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
READY_TIMEOUT_SECONDS = 300
HTTP_OK = 200
REQUEST = {
    "state": "Write a Python function that merges two sorted lists.",
    "questions": {
        "kind": {
            "type": "choice",
            "instructions": "What kind of work is this?",
            "criteria": {"code": "Writing or fixing code", "chat": "Anything else"},
        },
        "reasoning": {
            "type": "noul",
            "instructions": "Does answering this need multi-step reasoning?",
        },
    },
}


def declared_plugins() -> dict[str, list[str]]:
    """The built-in plugins src/model-runtime/pyproject.toml registers, by kind."""
    project = tomllib.loads(
        (ROOT / "src/model-runtime/pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    return {
        group.removeprefix("vllm_srun."): sorted(entries)
        for group, entries in project["entry-points"].items()
    }


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _call(base: str, path: str, body: dict | None = None) -> tuple[int, object]:
    request = urllib.request.Request(
        base + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={} if body is None else {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as failure:
        return failure.code, failure.read().decode()


def _wait_ready(base: str, server: subprocess.Popen, log: Path) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"vllm-sr serve exited early:\n{log.read_text()}")
        try:
            status, health = _call(base, "/health")
        except OSError:
            status, health = 0, None
        if (
            status == HTTP_OK
            and isinstance(health, dict)
            and health["status"] == "ready"
        ):
            return
        time.sleep(1)
    raise RuntimeError(f"vllm-sr serve was not ready in time:\n{log.read_text()}")


def check_runtime_wheel(
    runtime_wheel: Path,
    cli_wheel: Path,
    torch: str = TORCH,
    torch_index: str = TORCH_INDEX,
) -> None:
    runtime_wheel = runtime_wheel.resolve(strict=True)
    cli_wheel = cli_wheel.resolve(strict=True)
    version = runtime_wheel.name.split("-")[1]
    if cli_wheel.name.split("-")[1] != version:
        raise RuntimeError("vllm-sr and vllm-srun wheels carry different versions")
    with tempfile.TemporaryDirectory(prefix="vllm-srun-wheel-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        bin_dir = environment / "bin"
        python = str(bin_dir / "python")
        srun = str(bin_dir / "vllm-srun")
        # Nothing from the checkout, the caller's Hugging Face cache or pip
        # configuration reaches the installed packages; fixtures load offline.
        child_env = {
            "PATH": f"{bin_dir}{os.pathsep}{os.defpath}",
            "HOME": str(root),
            "TMPDIR": str(root),
            "PIP_CONFIG_FILE": os.devnull,
            "HF_HOME": str(root / "hf"),
            "HF_HUB_OFFLINE": "1",
        }

        def run(*arguments: str) -> str:
            result = subprocess.run(
                arguments,
                cwd=root,
                env=child_env,
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode:
                raise RuntimeError(
                    f"{' '.join(arguments)} failed:\n{result.stdout}{result.stderr}"
                )
            return result.stdout

        pip = (python, "-I", "-m", "pip", "install", "--disable-pip-version-check")
        run(*pip, torch, "--index-url", torch_index)
        # The extra must resolve to the runtime wheel built with it.
        run(*pip, f"{cli_wheel}[runtime]", str(runtime_wheel))
        installed = json.loads(
            run(
                python,
                "-I",
                "-c",
                "import json, torch; from importlib.metadata import version; "
                "print(json.dumps([version('vllm-sr'), version('vllm-srun'), "
                "torch.version.cuda, getattr(torch.version, 'hip', None)]))",
            )
        )
        if installed != [version, version, None, None]:
            raise RuntimeError(f"unexpected installed versions: {installed}")

        models = run(srun, "models").splitlines()
        if not any(
            line.startswith("vllm-sr/Decision-2.0-Kai-0.6B@") for line in models
        ):
            raise RuntimeError(f"vllm-srun models lists no built-in model: {models}")
        plugins = json.loads(run(srun, "plugins"))
        if plugins != declared_plugins():
            raise RuntimeError(f"installed plugins differ from pyproject: {plugins}")

        package = root / "decision"
        run(srun, "fixture", str(package), "--family", "decision2")
        port = _free_port()
        base = f"http://127.0.0.1:{port}"
        log = root / "serve.log"
        serve = [str(bin_dir / "vllm-sr"), "serve", str(package), "--device", "cpu"]
        with log.open("w") as output:
            server = subprocess.Popen(
                [*serve, "--port", str(port)],
                cwd=root,
                env=child_env,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            try:
                _wait_ready(base, server, log)
                status, response = _call(base, "/v1/decisions", REQUEST)
                if status != HTTP_OK or not isinstance(response, dict):
                    raise RuntimeError(f"/v1/decisions answered {status}: {response}")
                kind = response["answers"]["kind"]
                noul = response["answers"]["reasoning"]["noul"]
                if (
                    kind["choice"] not in REQUEST["questions"]["kind"]["criteria"]
                    or not math.isclose(
                        sum(kind["probabilities"].values()), 1.0, abs_tol=1e-6
                    )
                    or not 0.0 <= noul <= 1.0
                ):
                    raise RuntimeError(f"malformed decision answers: {response}")
            finally:
                server.send_signal(signal.SIGINT)
                try:
                    server.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
        print(
            f"vllm-sr[runtime] {version} with {torch} (CPU) lists "
            f"{len(models)} built-in models and served a fixture's decision."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime_wheel", type=Path)
    parser.add_argument("cli_wheel", type=Path)
    parser.add_argument("--torch", default=TORCH)
    parser.add_argument("--torch-index", default=TORCH_INDEX)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("the runtime wheel check runs on Linux")
    check_runtime_wheel(
        args.runtime_wheel, args.cli_wheel, args.torch, args.torch_index
    )


if __name__ == "__main__":
    main()
