#!/usr/bin/env python3
"""Run one pinned real model with the upstream llama.cpp CPU server image."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from http import HTTPStatus
from pathlib import Path

from smoke import run_smoke

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SPEC = json.loads((HERE / "model.json").read_text())
MAX_TCP_PORT = 65535


def verify_model(path):
    if path.stat().st_size != SPEC["size_bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == SPEC["sha256"]


def fetch_model(cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / SPEC["filename"]
    if path.exists():
        if not verify_model(path):
            raise RuntimeError(
                f"cached model checksum mismatch: {path}; remove it to retry"
            )
        return path
    url = (
        f"https://huggingface.co/{SPEC['repository']}/resolve/"
        f"{SPEC['revision']}/{SPEC['filename']}"
    )
    print(
        f"Downloading pinned {SPEC['repository']} ({SPEC['size_bytes']} bytes)",
        flush=True,
    )
    partial = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cache_dir, suffix=".partial", delete=False
        ) as out:
            partial = Path(out.name)
            with urllib.request.urlopen(url, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
        if not verify_model(partial):
            raise RuntimeError("downloaded model does not match its pinned SHA256")
        partial.replace(path)
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)
    return path


def container_args(runtime, name, model, port):
    return [
        runtime,
        "run",
        "--rm",
        "--name",
        name,
        "--cpus",
        "2",
        "--memory",
        "2g",
        "--pids-limit",
        "128",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "-p",
        f"127.0.0.1:{port}:8000",
        "--mount",
        f"type=bind,source={model.parent},target=/models,readonly",
        SPEC["server_image"],
        "--model",
        f"/models/{model.name}",
        "--alias",
        SPEC["model_id"],
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--ctx-size",
        "2048",
        "--parallel",
        "1",
        "--threads",
        "2",
        "--threads-batch",
        "2",
        "--n-gpu-layers",
        "0",
        "--n-predict",
        "64",
        "--jinja",
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
        "--timeout",
        "120",
    ]


def ready(base_url, process, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("llama.cpp exited before health readiness")
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
                if response.status == HTTPStatus.OK:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.5)
    raise RuntimeError("llama.cpp did not become healthy within 120 seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("serve", "smoke"))
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="localhost port; smoke accepts 0 for a random port",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / ".cache" / "tiny-model"
    )
    parser.add_argument(
        "--runtime", default=os.environ.get("CONTAINER_RUNTIME", "docker")
    )
    args = parser.parse_args()
    if args.command == "serve" and args.port == 0:
        parser.error("serve requires an explicit non-zero port")
    if not 0 <= args.port <= MAX_TCP_PORT:
        parser.error("port must be in 0..65535")
    model = fetch_model(args.cache_dir.resolve())
    name = "sr-tiny-model-" + uuid.uuid4().hex[:12]
    command = container_args(args.runtime, name, model, args.port)
    # Pinned upstream image: no repository-owned inference image is built.
    inspected = subprocess.run(
        [args.runtime, "image", "inspect", SPEC["server_image"]],
        capture_output=True,
        check=False,
    )
    if inspected.returncode:
        subprocess.run([args.runtime, "pull", SPEC["server_image"]], check=True)
    try:
        if args.command == "serve":
            print(
                f"Serving {SPEC['model_id']} on http://127.0.0.1:{args.port}",
                flush=True,
            )
            return subprocess.call(command)
        with tempfile.TemporaryFile(mode="w+") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                if args.port == 0:
                    deadline = time.monotonic() + 20
                    while True:
                        result = subprocess.run(
                            [args.runtime, "port", name, "8000/tcp"],
                            capture_output=True,
                            check=False,
                            text=True,
                        )
                        if result.returncode == 0 and result.stdout.strip():
                            args.port = int(result.stdout.strip().rsplit(":", 1)[1])
                            break
                        if process.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError(
                                "container did not publish its smoke port"
                            )
                        time.sleep(0.2)
                base_url = f"http://127.0.0.1:{args.port}"
                ready(base_url, process)
                print("PASS health readiness", flush=True)
                run_smoke(base_url, SPEC["model_id"])
            except BaseException:
                log.seek(0)
                print(log.read()[-16000:])
                raise
            finally:
                subprocess.run(
                    [args.runtime, "rm", "-f", name], capture_output=True, check=False
                )
                process.wait(timeout=15)
    finally:
        subprocess.run(
            [args.runtime, "rm", "-f", name], capture_output=True, check=False
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
