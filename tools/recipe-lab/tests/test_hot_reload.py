"""Hot-reload replay tier tests (H).

Unit-level, no real stack: the router binary is a throwaway script that
mirrors the real router's process contract - it reports readiness next to
its -config file, binds the gateway port from the patched config's first
listener, serves the management API (config PUT + activation hash + replay
list) on its -api-port, and appends each chat request's record (carrying
the client x-request-id) to a JSON state file shared with the test. That
is the same data flow the real stack has (client x-request-id -> replay
record request_id), verified against the router source.

Covers the three contracted scenarios: hot success (+ pending activation
poll), PUT rejected -> restart fallback, record count mismatch ->
status=incomplete. The restart-mode tests stay in test_runner.py.
"""

from __future__ import annotations

import hashlib
import json
import socket
import sys
from pathlib import Path

import pytest
from recipe_lab.adapters.router_replay import CaseJoin, records_to_cases_by_request_id
from recipe_lab.bank import BankEntry
from recipe_lab.contract import OraclePolicy
from recipe_lab.runner import (
    HotReloadSession,
    RunnerEnv,
    candidate_tag,
    request_id_for_case,
)

POLICY = OraclePolicy.from_dict(
    {"math": "local/model-math", "code": "local/model-code"}
)

CASES = [
    {"case_id": "c1", "prompt": "prompt one", "truth": "code"},
    {"case_id": "c2", "prompt": "prompt two", "truth": "code"},
]

# The fake router routes everything to the code model: both decisions match
# the truth; c1 answers correctly, c2 fails on model capability (E3), so the
# accuracy is fixed at 0.5.
EXPECTED_ACCURACY = 0.5

# c1 is answered correctly by the routed model, c2 is not.
BANK = {
    ("c1", "local/model-code"): BankEntry("c1", "local/model-code", "ok", True),
    ("c2", "local/model-code"): BankEntry("c2", "local/model-code", "ok", False),
}

FAKE_ROUTER_SCRIPT = """#!/usr/bin/env python3
import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml

config_path = sys.argv[sys.argv.index("-config") + 1]
cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
GATE = int(cfg["listeners"][0]["port"])
API = int(sys.argv[sys.argv.index("-api-port") + 1])
STATE = os.environ["FAKE_STATE"]
lock = threading.Lock()


def _load():
    with open(STATE, encoding="utf-8") as fh:
        return json.load(fh)


def _save(state):
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh)
    os.replace(tmp, STATE)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        prompt = (body.get("messages") or [{}])[0].get("content", "")
        with lock:
            state = _load()
            state["records"].append(
                {
                    "request_id": self.headers.get("x-request-id", ""),
                    "prompt": prompt,
                    "decision": "code",
                    "selected_model": "local/model-code",
                }
            )
            _save(state)
        out = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_PUT(self):
        if not self.path.startswith("/api/v1/config"):
            self._send(404, {"error": "not found"})
            return
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8")
        with lock:
            state = _load()
            state["put_count"] += 1
            pending = "hot-pending" in raw
            if pending:
                state["pending_hash_polls"] = 1
            if "hot-drop" in raw:
                state["drop_next"] = True
            _save(state)
        if "hot-reject" in raw:
            self._send(409, {"error": {"code": "RESTART_REQUIRED", "message": "listeners changed"}})
            return
        if "hot-fail" in raw:
            self._send(200, {"status": "success", "activation_status": "failed"})
            return
        if pending:
            self._send(202, {"status": "success", "activation_status": "pending"})
            return
        self._send(200, {"status": "success", "activation_status": "active"})

    def do_GET(self):
        if self.path.startswith("/api/v1/config/hash"):
            with lock:
                state = _load()
                pending = state.get("pending_hash_polls", 0) > 0
                if pending:
                    state["pending_hash_polls"] -= 1
                    _save(state)
            self._send(
                200,
                {
                    "source_config_hash": "s",
                    "generated_runtime_hash": "g",
                    "activation_status": "pending" if pending else "active",
                },
            )
            return
        if self.path.startswith("/api/v1/observability/replays"):
            with lock:
                state = _load()
                records = list(state["records"])
                if state.get("drop_next"):
                    records = records[:-1]
                    state["drop_next"] = False
                    _save(state)
            self._send(200, {"data": records, "has_more": False})
            return
        self._send(404, {"error": "not found"})


# Bind both listeners before reporting readiness: the status file is the
# runner's primary readiness signal, and a request that arrives before
# serve_forever still queues in the listen backlog.
gateway = ThreadingHTTPServer(("127.0.0.1", GATE), Handler)
management = ThreadingHTTPServer(("127.0.0.1", API), Handler)
with open(os.path.join(os.path.dirname(config_path), "router-runtime.json"), "w", encoding="utf-8") as fh:
    json.dump({"ready": True}, fh)
threading.Thread(target=management.serve_forever, daemon=True).start()
gateway.serve_forever()
"""


def _seed_state(path: Path) -> None:
    """Write the fake router's initial shared state file."""
    path.write_text(
        json.dumps(
            {"records": [], "put_count": 0, "pending_hash_polls": 0, "drop_next": False}
        ),
        encoding="utf-8",
    )


class _StateView:
    """Read-only view of the fake router's JSON state file.

    The fake router is a separate process, so assertions read the same file
    the router's handlers mutate (put_count, records) instead of sharing an
    in-memory object.
    """

    def __init__(self, path: Path):
        self.path = path

    @property
    def put_count(self) -> int:
        return int(self._data["put_count"])

    @property
    def records(self) -> list[dict]:
        return list(self._data["records"])

    @property
    def _data(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.fixture()
def hot_stack(tmp_path):
    """A fake router binary (gateway + management API on the router's own
    per-boot ports, shared state via a JSON file) + a RunnerEnv."""
    state_path = tmp_path / "fake_state.json"
    _seed_state(state_path)
    router = tmp_path / "fake_router.py"
    router.write_text(FAKE_ROUTER_SCRIPT)
    router.chmod(0o755)
    mock = tmp_path / "mock_backend.py"
    mock.write_text("import time\ntime.sleep(300)\n")
    env = RunnerEnv(
        router_bin=str(router),
        mock_backend_script=str(mock),
        python_bin=sys.executable,
        work_dir=tmp_path / "work",
        backend_port=_free_port(),
        ready_timeout_s=30,
        reload_mode="hot",
        env_extra={"FAKE_STATE": str(state_path)},
    )
    yield _StateView(state_path), env


def _candidate(path: Path, marker: str = "") -> Path:
    """Write a minimal candidate config with one gateway listener.

    Each candidate carries its name as an x-candidate mapping key (and the
    scenario marker, if any, as x-hot-marker) so the keys survive the
    runner's yaml round-trip when the session patches the candidate onto
    the session's per-boot gateway ports: comment-only differences would
    collapse and wrongly hit the "already serving" fast path.
    """
    text = (
        "version: v0.2\n"
        "listeners:\n"
        "  - name: http\n"
        "    address: 0.0.0.0\n"
        "    port: 8899\n"
        f"x-candidate: {path.stem}\n"
    )
    if marker:
        text += f"x-hot-marker: {marker.lstrip('#').strip()}\n"
    path.write_text(text)
    return path


# ---- pure helpers ----


def test_candidate_tag_and_request_ids():
    assert candidate_tag("yaml") == hashlib.sha256(b"yaml").hexdigest()[:8]
    assert request_id_for_case("ab12", "c1") == "rl-ab12-c1"


def test_records_join_by_request_id():
    records = [
        {
            "request_id": "rl-aa11-c1",
            "decision": "code",
            "selected_model": "local/model-code",
        },
        {
            "request_id": "rl-aa11-c2",
            "decision": "math",
            "selected_model": "local/model-math",
        },
        {"request_id": "other-candidate", "decision": "x", "selected_model": "y"},
        {"decision": "no-id"},
    ]
    cases = {
        "rl-aa11-c1": CaseJoin("c1", "p1", "code"),
        "rl-aa11-c2": CaseJoin("c2", "p2", "math"),
    }
    rows, stats = records_to_cases_by_request_id(records, cases, family="d3")
    assert [row["case_id"] for row in rows] == ["c1", "c2"]
    assert stats == {"matched": 2, "unmatched": 1, "no_request_id": 1}
    assert rows[0]["request_id"] == "rl-aa11-c1"
    assert rows[0]["routed"] == "local/model-code"


# ---- session scenarios ----


def test_hot_reload_success_put_pending_and_request_ids(tmp_path, hot_stack):
    state, env = hot_stack
    cands = tmp_path / "cands"
    cands.mkdir()
    cand_a = _candidate(cands / "a.yaml")
    cand_b = _candidate(cands / "b.yaml", "# hot-pending")  # 202 -> hash poll

    session = HotReloadSession(
        env=env, boot_source=cand_a, router_model="vllm-sr/auto", family="d3"
    )
    try:
        session.ensure_started()
        r1 = session.evaluate(
            candidate_config=cand_a, cases=CASES, oracle_policy=POLICY, bank=BANK
        )
        assert r1["reload"] == "hot"
        assert r1["statuses"] == {"c1": 200, "c2": 200}
        assert r1["timings"]["switch_s"] == 0.0  # boot config: no PUT needed
        assert state.put_count == 0
        assert r1["summary"]["accuracy"] == EXPECTED_ACCURACY
        tag_a = candidate_tag(cand_a.read_text(encoding="utf-8"))
        # the tagged x-request-ids are what reached the replay store
        assert {rec["request_id"] for rec in state.records} == {
            request_id_for_case(tag_a, "c1"),
            request_id_for_case(tag_a, "c2"),
        }

        r2 = session.evaluate(
            candidate_config=cand_b, cases=CASES, oracle_policy=POLICY, bank=BANK
        )
        assert state.put_count == 1  # only the second candidate PUTs
        assert r2["timings"]["switch_s"] > 0
        assert r2["summary"]["accuracy"] == EXPECTED_ACCURACY
        tag_b = candidate_tag(cand_b.read_text(encoding="utf-8"))
        assert tag_b != tag_a
        assert any(
            rec["request_id"].startswith(f"rl-{tag_b}-") for rec in state.records
        )
        assert len(r2["rows"]) == 2  # candidate A's records never leak in
    finally:
        session.stop()


def test_hot_reload_rejected_put_falls_back_to_restart(tmp_path, hot_stack):
    state, env = hot_stack
    cands = tmp_path / "cands"
    cands.mkdir()
    cand_a = _candidate(cands / "a.yaml")
    cand_rej = _candidate(cands / "rej.yaml", "# hot-reject")
    cand_c = _candidate(cands / "c.yaml")

    session = HotReloadSession(
        env=env, boot_source=cand_a, router_model="vllm-sr/auto", family="d3"
    )
    try:
        session.ensure_started()
        session.evaluate(
            candidate_config=cand_a, cases=CASES, oracle_policy=POLICY, bank=BANK
        )
        result = session.evaluate_with_fallback(
            candidate_config=cand_rej,
            cases=CASES,
            oracle_policy=POLICY,
            bank=BANK,
        )
        assert result["reload"] == "restart-fallback"
        assert "409" in result["fallback_reason"]
        assert not session.running  # the fallback freed the ports
        assert result["statuses"] == {"c1": 200, "c2": 200}
        assert result["summary"]["accuracy"] == EXPECTED_ACCURACY

        # the next candidate re-boots the session hot from the boot source
        session.ensure_started()
        after = session.evaluate(
            candidate_config=cand_c, cases=CASES, oracle_policy=POLICY, bank=BANK
        )
        assert after["reload"] == "hot"
        assert state.put_count == 2  # rejected PUT + post-fallback PUT
        assert after["summary"]["accuracy"] == EXPECTED_ACCURACY
    finally:
        session.stop()


def test_hot_reload_count_mismatch_records_incomplete(tmp_path, hot_stack):
    _state, env = hot_stack
    cands = tmp_path / "cands"
    cands.mkdir()
    cand_a = _candidate(cands / "a.yaml")
    cand_drop = _candidate(cands / "drop.yaml", "# hot-drop")

    session = HotReloadSession(
        env=env, boot_source=cand_a, router_model="vllm-sr/auto", family="d3"
    )
    try:
        session.ensure_started()
        session.evaluate(
            candidate_config=cand_a, cases=CASES, oracle_policy=POLICY, bank=BANK
        )
        result = session.evaluate(
            candidate_config=cand_drop, cases=CASES, oracle_policy=POLICY, bank=BANK
        )
        assert result["incomplete"] is True
        assert "rows for 2 cases" in result["incomplete_reason"]
        assert "summary" not in result and "rows" not in result
    finally:
        session.stop()


def test_hot_reload_failed_activation_falls_back(tmp_path, hot_stack):
    _state, env = hot_stack
    cands = tmp_path / "cands"
    cands.mkdir()
    cand_a = _candidate(cands / "a.yaml")
    cand_fail = _candidate(cands / "fail.yaml", "# hot-fail")

    session = HotReloadSession(
        env=env, boot_source=cand_a, router_model="vllm-sr/auto", family="d3"
    )
    try:
        session.ensure_started()
        result = session.evaluate_with_fallback(
            candidate_config=cand_fail,
            cases=CASES,
            oracle_policy=POLICY,
            bank=BANK,
        )
        assert result["reload"] == "restart-fallback"
        assert "failed" in result["fallback_reason"]
        assert result["summary"]["accuracy"] == EXPECTED_ACCURACY
    finally:
        session.stop()
