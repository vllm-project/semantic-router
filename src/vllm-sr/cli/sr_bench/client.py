"""CLI connection and detached service lifecycle; request submission is never retried."""

from __future__ import annotations

import os
import hashlib
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from . import VERSION
from .service import DEFAULT_STORE, DEFAULT_URL, PREFIX


class Client:
    def __init__(
        self, url=DEFAULT_URL, store=DEFAULT_STORE, autostart=True, verify_store=True
    ):
        self.url = url.rstrip("/")
        self.store = Path(store).expanduser().resolve()
        self.autostart = autostart
        self.verify_store = verify_store
        self.headers = {}
        if os.environ.get("SR_BENCH_TOKEN"):
            self.headers["Authorization"] = "Bearer " + os.environ["SR_BENCH_TOKEN"]

    def ready(self):
        try:
            response = requests.get(
                self.url + "/health", headers=self.headers, timeout=2
            )
            if response.status_code in {401, 403}:
                raise ValueError("Service authentication failed; set SR_BENCH_TOKEN")
            if (
                response.status_code == 200
                and response.json().get("version") == VERSION
            ):
                if (
                    self.verify_store
                    and urlparse(self.url).hostname in {"127.0.0.1", "localhost"}
                    and response.json().get("store_id")
                    != hashlib.sha256(str(self.store).encode()).hexdigest()
                ):
                    raise ValueError(
                        "This URL belongs to a different sr-bench store; select its --store or a different --url"
                    )
                return True
            return False
        except requests.ConnectionError:
            return False
        except requests.Timeout:
            return False

    def ensure(self):
        if self.ready():
            return
        parsed = urlparse(self.url)
        if not self.autostart or parsed.hostname not in {"localhost", "127.0.0.1"}:
            raise ValueError(
                "sr-bench service is unavailable; start vllm-sr benchmark serve"
            )
        self.store.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (self.store / "service.log").open("ab") as log:
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "cli.sr_bench.service",
                    "--store",
                    str(self.store),
                    "--port",
                    str(parsed.port or 8090),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        for _ in range(50):
            if self.ready():
                return
            time.sleep(0.1)
        raise ValueError("sr-bench service did not become ready; inspect service.log")

    def request(self, method, path, body=None):
        self.ensure()
        try:
            response = requests.request(
                method,
                self.url + PREFIX + path,
                headers=self.headers,
                json=body,
                timeout=30,
            )
        except requests.RequestException as exc:
            raise ValueError(
                "Service request failed; inspect runs before submitting again (requests are never retried)"
            ) from exc
        data = response.json()
        if response.status_code >= 400:
            raise ValueError(data.get("error", f"HTTP {response.status_code}"))
        return data
