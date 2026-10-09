"""Run the crew against a fake router, so the demo is checked without spending tokens."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _load(name: str):
    """Load a sibling script by path, so the tests run from any directory."""
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


crew = _load("crew")
report = _load("report")

COST = {"frontier": 0.001, "local": 0.0}


class FakeRouter(BaseHTTPRequestHandler):
    spent: dict[str, float] = {"frontier": 0.0, "local": 0.0}
    decisions = 0

    def log_message(self, *args) -> None:  # silence the default access log
        pass

    def do_GET(self) -> None:
        lines = [
            f'llm_model_cost_total{{currency="USD",model="{m}"}} {v}'
            for m, v in self.spent.items()
        ]
        lines += [
            f"llm_model_routing_latency_seconds_sum {self.decisions * 0.00025}",
            f"llm_model_routing_latency_seconds_count {self.decisions}",
        ]
        self.reply(200, "\n".join(lines).encode(), {"content-type": "text/plain"})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        last_user = next(
            m["content"] for m in reversed(body["messages"]) if m["role"] == "user"
        ).lower()
        selected, decision = body["model"], ""
        if selected == "MoM":
            deep = "find the defects" in last_user and "threading" in last_user
            selected, decision = (
                ("frontier", "deep-review") if deep else ("local", "default-local")
            )
        type(self).spent[selected] += COST[selected]
        type(self).decisions += 1
        names = [t["function"]["name"] for t in body.get("tools", [])]
        if names and not any(m["role"] == "tool" for m in body["messages"]):
            name = names[0]
            args = (
                "{}"
                if name == "list_changed_files"
                else json.dumps({"path": "../../defects.json"})
            )
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    }
                ],
            }
            finish = "tool_calls"
        else:
            message, finish = {
                "role": "assistant",
                "content": f"note from {selected}",
            }, "stop"
        payload = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": selected,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
            },
        }
        headers = {
            "content-type": "application/json",
            "x-vsr-selected-model": selected,
            "x-vsr-routing-latency-ms": "0.250",
            "x-vsr-cost": str(COST[selected]),
            "x-vsr-cost-currency": "USD",
        }
        if decision:
            headers["x-vsr-selected-decision"] = decision
        self.reply(200, json.dumps(payload).encode(), headers)

    def reply(self, status: int, data: bytes, headers: dict[str, str]) -> None:
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def start_router() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeRouter)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def run_arm(tmp_path: Path, arm: str) -> dict:
    server, url = start_router()
    try:
        [run_dir] = crew.main(
            [
                "--arm",
                arm,
                "--base-url",
                f"{url}/v1",
                "--router-metrics-url",
                f"{url}/metrics",
                "--out",
                str(tmp_path),
            ]
        )
    finally:
        server.shutdown()
    return json.loads((run_dir / "summary.json").read_text()) | {"dir": run_dir}


def test_per_call_routes_only_the_concurrency_review_to_frontier(
    tmp_path: Path,
) -> None:
    summary = run_arm(tmp_path, "per-call")
    # planner 2 + summarizer 4 + reviewer 3 files x 2 + writer 1
    assert summary["calls"] == 13 and summary["failed_calls"] == 0
    assert summary["by_agent"]["reviewer"]["selected_models"] == {
        "frontier": 2,
        "local": 4,
    }
    assert summary["selected_model_counts"] == {"local": 11, "frontier": 2}
    assert summary["decision_counts"] == {"default-local": 11, "deep-review": 2}
    assert summary["cost_from_headers"]["total"] == {"USD": 0.002}
    assert summary["router_metrics"]["cost_total"] == {"USD": 0.002}
    assert summary["router_metrics"]["routing_decisions"] == 13
    assert (
        summary["routing_latency_ms"]["samples"] == 13
        and summary["routing_latency_ms"]["p50"] == 0.25
    )


def test_per_agent_sends_every_review_call_to_frontier(tmp_path: Path) -> None:
    summary = run_arm(tmp_path, "per-agent")
    assert summary["by_agent"]["reviewer"]["selected_models"] == {"frontier": 6}
    assert summary["by_agent"]["summarizer"]["selected_models"] == {"local": 4}
    assert summary["frontier_calls"] == 6


def test_tools_cannot_read_outside_the_pull_request(tmp_path: Path) -> None:
    assert "not a file in this pull request" in crew.run_tool(
        "read_file", json.dumps({"path": "../../defects.json"})
    )
    assert "not a file" in crew.run_tool("read_file", json.dumps({"path": "PR.md"}))
    assert "def page(" in crew.run_tool(
        "read_file", json.dumps({"path": "pagination.py"})
    )
    assert crew.run_tool("read_file", "not json").startswith("error")
    summary = run_arm(tmp_path, "all-local")
    tool_results = [
        m["content"]
        for c in json.loads((summary["dir"] / "conversations.json").read_text())
        for m in c["messages"]
        if m["role"] == "tool"
    ]
    assert tool_results and not any('"D1"' in r for r in tool_results)


def test_report_combines_arms_and_grades(tmp_path: Path) -> None:
    for arm in ("per-call", "all-frontier"):
        summary = run_arm(tmp_path, arm)
        found = {f"D{i}": i <= 4 for i in range(1, 7)}
        (summary["dir"] / "grades.json").write_text(
            json.dumps({"found": found, "defects_found": 4, "defects_total": 6})
        )
    runs = report.load_runs(tmp_path)
    table = "\n".join(report.arm_table(runs))
    assert "| all-frontier | per-call |" in table
    assert "| Defects found (of 6) | 4 | 4 |" in table
    assert "| Cost source | /metrics | /metrics |" in table
    local = run_arm(tmp_path / "local", "all-local")
    assert local["router_metrics"]["cost_total"] == {}
    assert report.run_cost(local) == 0
    assert "| per-call | 13 | 0.250 | 0.250 | 0.250 |" in "\n".join(
        report.routing_table(runs)
    )
    assert "frontier 2, local 4" in "\n".join(report.agent_table(runs))
