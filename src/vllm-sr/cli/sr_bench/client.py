"""CLI connection and detached service lifecycle; request submission is never retried."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from . import VERSION
from .service import DEFAULT_STORE, DEFAULT_URL, PREFIX, service_credentials


class Client:
    def __init__(
        self, url=DEFAULT_URL, store=DEFAULT_STORE, autostart=True, verify_store=True
    ):
        self.url = url.rstrip("/")
        self.store = Path(store).expanduser().resolve()
        self.autostart = autostart
        self.verify_store = verify_store
        self.headers = {}
        self.token_env, token = service_credentials()
        if token:
            self.headers["Authorization"] = "Bearer " + token

    def ready(self):
        try:
            response = requests.get(
                self.url + "/health", headers=self.headers, timeout=2
            )
            if response.status_code in {401, 403}:
                raise ValueError(f"Service authentication failed; set {self.token_env}")
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
        markers = (
            "service.json",
            "service.lock",
            "service.log",
            "journal.sqlite3",
            "service-autostart.json",
        )
        if any((self.store / name).exists() for name in markers):
            raise ValueError(
                "sr-bench service is unavailable and this store has prior execution evidence; "
                "inspect its journal before explicitly starting vllm-sr benchmark serve "
                "(automatic restart is disabled)"
            )
        self.store.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Reserve first startup atomically so concurrent CLI readers cannot each
        # spawn a worker. Keep this receipt even when startup fails.
        try:
            receipt_fd = os.open(
                self.store / "service-autostart.json",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            raise ValueError(
                "sr-bench startup is already recorded; inspect the service before retrying"
            ) from None
        with os.fdopen(receipt_fd, "w") as receipt:
            json.dump({"url": self.url, "store": str(self.store)}, receipt)
            receipt.flush()
            os.fsync(receipt.fileno())
        environment = os.environ.copy()
        if token := os.environ.get(self.token_env):
            environment["SR_BENCH_TOKEN"] = token
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
                env=environment,
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
