"""The forward watchdog and the bounded stop, with forwards that do not return (CPU only)."""

import http.client
import json
import logging
import os
import subprocess
import sys
import threading
import time

import pytest
from starlette.testclient import TestClient
from vllm_srun.api import server
from vllm_srun.api.app import create_app
from vllm_srun.cli import build_parser, config_from_args
from vllm_srun.config import ServeConfig
from vllm_srun.scheduler.planner import cost
from vllm_srun.scheduler.scheduler import Scheduler

from .conftest import QUESTIONS, STATE, start_runtime
from .test_api_contract import check
from .test_scheduler import FakeModel, batcher, item, wait_until
from .test_server import SOURCE, free_port, request, wait_ready

REQUEST = {"state": STATE, "questions": QUESTIONS}


def hold(served):
    """Make the model's forwards wait for the returned event before they run."""
    gate = threading.Event()
    model = served.model
    for name in ("run", "run_shared", "run_approximate"):
        run = getattr(model, name, None)
        if run is None:
            continue

        def held(*args, _run=run, **kwargs):
            gate.wait()
            return _run(*args, **kwargs)

        setattr(model, name, held)
    return gate


def in_background(served):
    """Answer one request alone on another thread, as the golden check does."""
    thread = threading.Thread(
        target=served.answer_alone, args=("decisions", REQUEST), daemon=True
    )
    thread.start()
    return thread


def sample(metrics, name):
    for line in metrics.splitlines():
        if line.startswith(name + " "):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def test_the_scheduler_shows_the_forward_in_flight():
    gate = threading.Event()
    model = FakeModel(gate=gate)
    scheduler = Scheduler(model, batcher(model))
    scheduler.start()
    try:
        assert scheduler.forward is None
        rendered = item("a", 4)
        future = scheduler.submit([rendered], deadline=None)
        wait_until(lambda: scheduler.forward is not None)
        forward = scheduler.forward
        assert (forward.rows, forward.tokens, forward.requests) == (
            1,
            cost(rendered),
            1,
        )
        assert forward.started <= time.monotonic()
        assert forward.describe() == f"1 row, {cost(rendered)} tokens, 1 request"
        gate.set()
        future.result(timeout=5)
        assert scheduler.forward is None
    finally:
        gate.set()
        scheduler.stop()


def test_stop_gives_up_on_a_forward_that_does_not_return():
    gate = threading.Event()
    model = FakeModel(gate=gate)
    scheduler = Scheduler(model, batcher(model))
    scheduler.start()
    try:
        scheduler.submit([item("a", 4)], deadline=None)
        wait_until(lambda: scheduler.forward is not None)
        started = time.monotonic()
        assert scheduler.stop(0.1) is False
        assert time.monotonic() - started < 2
        assert scheduler.forward is not None
    finally:
        gate.set()
    assert scheduler.stop(5) is True


def test_a_stalled_forward_makes_the_process_neither_live_nor_ready_until_it_returns(
    qwen3_package, caplog
):
    runtime = start_runtime(qwen3_package, forward_timeout_seconds=0.5)
    served = runtime.served[0]
    client = TestClient(create_app(runtime))

    def metric(name):
        return sample(client.get("/metrics").text, name)

    stalls = f'vllm_srun_stalled_forwards_total{{model="{served.label}"}}'
    # A slow first forward of the golden check may already have counted.
    before = metric(stalls)
    gate = hold(served)
    try:
        assert client.get("/health/live").status_code == 200
        with caplog.at_level(logging.ERROR, logger="vllm_srun"):
            answer = in_background(served)
            wait_until(
                lambda: metric(stalls) == before + 1 and metric("vllm_srun_live") == 0
            )
        live = client.get("/health/live")
        assert live.status_code == 503
        check("Liveness", live.json())
        assert live.json()["status"] == "stalled"
        assert live.json()["reason"].startswith(
            f"{served.label}: a forward has run longer than "
            "--forward-timeout-seconds (0.5 s): "
        )
        health = client.get("/health")
        assert health.status_code == 503
        check("Health", health.json())
        assert health.json()["status"] == "degraded"
        refused = client.post("/v1/decisions", json=REQUEST)
        assert refused.status_code == 503
        assert refused.json()["error"]["code"] == "not_ready"
        assert metric("vllm_srun_ready") == 0
        assert any(
            "GET /health/live and GET /health answer 503 until it returns"
            in record.getMessage()
            for record in caplog.records
        )

        gate.set()
        answer.join(timeout=10)
        assert client.get("/health/live").json()["status"] == "alive"
        wait_until(lambda: client.get("/health").status_code == 200)
        assert metric("vllm_srun_ready") == 1 and metric("vllm_srun_live") == 1
        assert client.post("/v1/decisions", json=REQUEST).status_code == 200
    finally:
        gate.set()
        runtime.stop()


def test_a_forward_timeout_of_zero_turns_the_watchdog_off(qwen3_package):
    runtime = start_runtime(qwen3_package, forward_timeout_seconds=0)
    served = runtime.served[0]
    gate = hold(served)
    try:
        answer = in_background(served)
        wait_until(lambda: served.scheduler.forward is not None)
        time.sleep(0.3)
        assert runtime.stalls() == [] and runtime.health.ready
        gate.set()
        answer.join(timeout=10)
    finally:
        gate.set()
        runtime.stop()


def test_stop_returns_after_its_timeout_and_names_the_forward_it_abandoned(
    qwen3_package, caplog
):
    runtime = start_runtime(qwen3_package)
    served = runtime.served[0]
    gate = hold(served)
    closed = []
    served.model.close = lambda: closed.append(True)
    try:
        in_background(served)
        wait_until(lambda: served.scheduler.forward is not None)
        started = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="vllm_srun"):
            abandoned = runtime.stop(0.2)
        assert time.monotonic() - started < 2
        assert len(abandoned) == 1
        assert abandoned[0].startswith(f"{served.label}: a forward of ")
        assert " has run for " in abandoned[0]
        assert "the model stays open" in caplog.text
        assert closed == []
    finally:
        gate.set()


class StoppedRuntime:
    def __init__(self, abandoned):
        self.abandoned = abandoned
        self.timeouts = []

    def stop(self, timeout):
        self.timeouts.append(timeout)
        return self.abandoned


def test_the_server_exits_at_once_when_a_forward_outlives_the_shutdown_grace(
    monkeypatch, caplog
):
    exits = []
    monkeypatch.setattr(server.os, "_exit", exits.append)
    clean = StoppedRuntime([])
    server.stop(clean, 1.5)
    assert clean.timeouts == [1.5] and exits == []
    stuck = StoppedRuntime(
        ["m: a forward of 1 row, 8 tokens, 1 request has run for 9 s"]
    )
    with caplog.at_level(logging.ERROR, logger="vllm_srun"):
        server.stop(stuck, 0.0)
    assert exits == [server.ABANDONED_EXIT]
    assert "m: a forward of 1 row, 8 tokens, 1 request has run for 9 s" in caplog.text


@pytest.mark.parametrize(
    "flag", ["--forward-timeout-seconds", "--shutdown-grace-seconds"]
)
@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_negative_or_unbounded_seconds_are_refused(flag, value):
    args = build_parser().parse_args(["serve", "model", flag, value])
    with pytest.raises(SystemExit, match=flag):
        config_from_args(args)
    with pytest.raises(ValueError, match=flag[2:].replace("-", "_")):
        ServeConfig(**{flag[2:].replace("-", "_"): float(value)})


# A server whose forwards never return once its model passed the golden check, before it is ready.
HUNG_SERVER = """
import sys
import threading

from vllm_srun import cli, runtime

start = runtime.ServedModel._start


def start_then_hang(self, *args, **kwargs):
    start(self, *args, **kwargs)
    never = threading.Event()
    for name in ("run", "run_shared", "run_approximate"):
        setattr(self.model, name, lambda *args, **kwargs: never.wait())


runtime.ServedModel._start = start_then_hang
sys.exit(cli.main(sys.argv[1:]))
"""


def test_a_hung_forward_fails_liveness_and_sigterm_still_ends_the_process(
    qwen35_package,
):
    port = free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            HUNG_SERVER,
            "serve",
            str(qwen35_package),
            "--port",
            str(port),
            "--device",
            "cpu",
            "--log-level",
            "warning",
            "--forward-timeout-seconds",
            "1",
            "--shutdown-grace-seconds",
            "1",
        ],
        env=dict(
            os.environ,
            PYTHONPATH=str(SOURCE) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    def connect():
        return http.client.HTTPConnection("127.0.0.1", port, timeout=10)

    try:
        wait_ready(connect, process)
        hung = connect()
        hung.request(
            "POST",
            "/v1/decisions",
            body=json.dumps(REQUEST),
            headers={"content-type": "application/json"},
        )
        deadline = time.monotonic() + 30
        while request(connect(), "GET", "/health/live")[0] != 503:
            assert time.monotonic() < deadline, "liveness never failed"
            time.sleep(0.1)
        status, body = request(connect(), "GET", "/health/live")
        assert status == 503 and json.loads(body)["status"] == "stalled"
        assert request(connect(), "GET", "/health")[0] == 503
        stopping = time.monotonic()
        process.terminate()
        assert process.wait(timeout=30) == server.ABANDONED_EXIT
        assert time.monotonic() - stopping < 15
        output = process.stdout.read().decode()
        assert "exiting with a forward still running after the shutdown grace" in output
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=30)
