"""The router latency tool's connection-error tolerance (``tools/router_latency.py``)."""

import argparse
import http.server
import importlib.util
import json
import sys
import threading
from pathlib import Path
from typing import ClassVar

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "router_latency.py"
# preview() retries once, so the failing first input costs two attempts in
# the sequential pass and two more for the diagnostic first_response call.
MIN_FIRST_INPUT_ATTEMPTS = 4


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("router_latency", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _outcome(status, decision="chat", model="m1", ms=1.0):
    row = {"status": status, "decision": None, "model": None}
    if status == http.HTTPStatus.OK:
        row.update(decision=decision, model=model, ms=ms, signal_ms={})
    return row


def _run_file(path, outcomes):
    path.write_text(
        json.dumps({"passes": {"sequential": {}}, "outcomes": outcomes}),
        encoding="utf-8",
    )


def _round_file(path, outcomes):
    path.write_text(
        json.dumps(
            {
                "passes": {
                    "sequential": {
                        "p50_ms": 1.0,
                        "p95_ms": 1.0,
                        "p99_ms": 1.0,
                        "throughput_rps": 1.0,
                    }
                },
                "outcomes": outcomes,
            }
        ),
        encoding="utf-8",
    )


class _DroppingRouter(http.server.ThreadingHTTPServer):
    """Drops the connection for every request whose text starts with ``boom``."""

    lock = threading.Lock()
    attempts: ClassVar[dict[str, int]] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(length))
            text = body.get("text", "")
            with _DroppingRouter.lock:
                _DroppingRouter.attempts[text] = (
                    _DroppingRouter.attempts.get(text, 0) + 1
                )
            if text.startswith("boom"):
                self.connection.close()
                return
            payload = json.dumps(
                {"decision": {"decision_name": "stable"}, "recommended_model": "m1"}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass


def test_run_survives_first_input_disconnects(tool, tmp_path, monkeypatch):
    """A first input that keeps disconnecting must not discard completed passes."""
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(["boom-one", "ok-one"]), encoding="utf-8")
    out = tmp_path / "out.json"
    _DroppingRouter.attempts = {}
    server = _DroppingRouter(("127.0.0.1", 0), _DroppingRouter.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "router_latency.py",
            "run",
            "--url",
            f"http://127.0.0.1:{server.server_port}",
            "--corpus",
            str(corpus),
            "--out",
            str(out),
            "--label",
            "test",
            "--warmup",
            "0",
            "--repeat",
            "1",
            "--concurrency",
            "2",
        ],
    )
    try:
        assert tool.main() is None
    finally:
        server.shutdown()
    result = json.loads(out.read_text(encoding="utf-8"))
    assert _DroppingRouter.attempts["boom-one"] >= MIN_FIRST_INPUT_ATTEMPTS
    assert result["first_response"].get("error") == "first input request failed"
    assert result["passes"]["sequential"]["errors"] == 1
    assert result["passes"]["concurrency_2"]["errors"] == 1
    assert result["outcomes"][0] == {
        "status": "error",
        "decision": None,
        "model": None,
    }
    assert result["outcomes"][1]["status"] == http.HTTPStatus.OK
    assert result["outcomes"][1]["decision"] == "stable"


def test_compare_partitions_failures_from_parity(tool, tmp_path, capsys):
    """Double failures are not agreement and single-sided ones are not regressions.

    A 200 with a null decision still compares: the partition keys on status,
    not on the decision.
    """
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps(["ok", "changed", "boom-both", "boom-new", "err-empty", "empty"]),
        encoding="utf-8",
    )
    base, new = tmp_path / "base.json", tmp_path / "new.json"
    _run_file(
        base,
        [
            _outcome(200),
            _outcome(200),
            _outcome("error"),
            _outcome(200, decision=None, model=None),
            _outcome("error"),
            _outcome(200, decision=None, model=None),
        ],
    )
    _run_file(
        new,
        [
            _outcome(200, ms=1.5),
            _outcome(200, model="m2", ms=2.0),
            _outcome("error"),
            _outcome(500),
            _outcome(200, decision=None, model=None),
            _outcome(200, decision=None, model=None),
        ],
    )
    args = argparse.Namespace(base=str(base), new=str(new), corpus=str(corpus), show=5)
    tool.cmd_compare(args)
    out = capsys.readouterr().out
    assert (
        "decisions differing: 1 / 3 compared "
        "(1 failed on both sides, 2 failed on one side)" in out
    )
    assert "None -> None ms" not in out
    assert "'changed'" in out and "chat/m1 -> chat/m2" in out


def test_compare_reports_plain_counts_without_failures(tool, tmp_path, capsys):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(["ok", "same"]), encoding="utf-8")
    base, new = tmp_path / "base.json", tmp_path / "new.json"
    outcomes = [_outcome(200), _outcome(200, ms=2.0)]
    _run_file(base, outcomes)
    _run_file(new, outcomes)
    args = argparse.Namespace(base=str(base), new=str(new), corpus=str(corpus), show=5)
    tool.cmd_compare(args)
    out = capsys.readouterr().out
    assert "decisions differing: 0 / 2 compared" in out
    assert "failed on" not in out


def test_rounds_denominator_excludes_failures(tool, tmp_path, capsys):
    d = tmp_path
    for side in ("base", "new"):
        _round_file(
            d / f"r1-{side}.json",
            [
                _outcome(200),
                _outcome(200, decision=None, model=None),
                _outcome("error"),
                _outcome(500),
            ],
        )
    # round 2: input 3 fails only on the new side.
    _round_file(
        d / "r2-base.json",
        [
            _outcome(200),
            _outcome(200, decision=None, model=None),
            _outcome("error"),
            _outcome(200),
        ],
    )
    _round_file(
        d / "r2-new.json",
        [
            _outcome(200),
            _outcome(200, decision=None, model=None),
            _outcome("error"),
            _outcome(500),
        ],
    )
    args = argparse.Namespace(dir=str(d), base="base", new="new", out="")
    tool.cmd_rounds(args)
    out = capsys.readouterr().out
    assert (
        "same decision and model per round: [2, 2] of [2, 2] compared "
        "([2, 1] failed on both sides, [0, 1] failed on one side)" in out
    )
