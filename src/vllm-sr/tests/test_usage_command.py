"""`vllm-sr usage` against a stub Router management API.

The stub pages the replay list two records at a time, as the Router pages at
its own cap, keeps nothing for a since in the future, as the Router's records
are all older, and answers the aggregate with the totals a test sets.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest
from cli.main import main
from click.testing import CliRunner

REPLAYS = "/api/v1/observability/replays"


def _record(record_id, session, entrypoint, model, cost=None, currency="USD", **more):
    record = {
        "id": record_id,
        "session_id": session,
        "original_model": entrypoint,
        "selected_model": model,
        **more,
    }
    if cost is not None:
        record.update(actual_cost=cost, currency=currency)
    return record


# The second record again at the top of the second page, as when a record is
# stored between two pages of the newest-first list: it is counted once.
RECORDS = [
    _record("r1", "s1", "vllm-sr/auto", "qwen-msg", 0.00015),
    _record("r2", "s1", "vllm-sr/auto", "qwen-msg", 0.00015),
    _record("r2", "s1", "vllm-sr/auto", "qwen-msg", 0.00015),
    _record("r3", "s2", "vllm-sr/review", "qwen-chat", 0.00012),
]

AGGREGATE = {
    "record_count": 3,
    "lifecycle": {
        "completed": 2,
        "failed": 1,
        "aborted": 0,
        "in_progress": 0,
        "unknown": 0,
    },
    "summary": {
        "cost_record_count": 2,
        "by_currency": [
            {
                "currency": "USD",
                "total_saved": 0.00078,
                "gross_saved": 0.0009,
                "netted_router_call_spend": 0.00012,
                "baseline_spend": 0.001,
                "actual_spend": 0.0001,
                "router_call_spend": 0.00012,
                "total_spend": 0.00042,
                "cost_record_count": 2,
            }
        ],
    },
    "provider_cache": {"by_currency": []},
    "reviews": {"reviews": 1, "allowed": 1, "denied": 0, "approval_rate": 1.0},
}


@pytest.fixture()
def router():
    """A management API on a free port; .seen holds each request's path, query
    and Authorization header, .error answers every request with that status and
    body, and .ignores_since makes it list records whatever since says."""

    state = {
        "seen": [],
        "error": None,
        "records": RECORDS,
        "aggregate": AGGREGATE,
        "ignores_since": False,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            state["seen"].append((url.path, query, self.headers.get("Authorization")))
            if state["error"]:
                return self._send(*state["error"])
            if url.path == REPLAYS + "/aggregate":
                return self._send(200, state["aggregate"])
            records = state["records"]
            future = datetime.fromisoformat(query["since"]) > datetime.now(timezone.utc)
            if future and not state["ignores_since"]:
                records = []
            offset = int(query.get("offset", 0))
            data = records[offset : offset + min(2, int(query["limit"]))]
            page = {"data": data}
            if offset + len(data) < len(records):
                page["next_offset"] = offset + len(data)
            return self._send(200, page)

        def _send(self, status, body):
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()


def _usage(router, *args):
    return CliRunner().invoke(main, ["usage", "--endpoint", router["url"], *args])


def _since(router):
    return datetime.fromisoformat(router["seen"][0][1]["since"])


def _lines(result):
    return [" ".join(line.split()) for line in result.output.splitlines()]


def test_usage_is_a_top_level_command():
    assert "usage" in CliRunner().invoke(main, ["--help"]).output


def test_usage_summarizes_today_from_the_default_management_api(router, monkeypatch):
    # without --endpoint the client's own default, 8080 plus the port offset
    monkeypatch.setattr(
        "cli.router_management_client.default_management_base_url",
        lambda: router["url"],
    )
    monkeypatch.setenv("VSR_MGMT_TOKEN", "mgmt-token")

    result = CliRunner().invoke(main, ["usage"])

    assert result.exit_code == 0, result.output
    midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    assert _since(router) == midnight.astimezone()
    assert {auth for *_, auth in router["seen"]} == {"Bearer mgmt-token"}
    # every list page and aggregate reads the window, past one probe a day ahead
    windows = [query["since"] for _, query, _ in router["seen"]]
    assert set(windows[:1] + windows[2:]) == {midnight.astimezone().isoformat()}
    for expected in (
        "Requests 3: 2 completed, 1 failed, 0 aborted",
        "Sessions 2",
        "Spend 0.000420 USD (Router calls 0.000120 USD)",
        "Net savings 0.000780 USD (gross 0.000900 USD, Router calls netted"
        " 0.000120 USD, 2 requests compared)",
        "Reviews 1, approval 100% (1 allowed, 0 denied)",
        "qwen-msg 2 requests, 0.000300 USD",
        "qwen-chat 1 request, 0.000120 USD",
        "vllm-sr/auto 2 requests, 0.000300 USD",
        "vllm-sr/review 1 request, 0.000120 USD",
    ):
        assert expected in _lines(result), result.output
    assert "Prompt cache" not in result.output


def test_usage_rows_count_and_price_their_own_requests(router):
    # qwen-chat is the entrypoint of r1 and the selected model of r1 to r3;
    # each row adds its requests' own priced costs, Router calls included, per
    # currency, and a Looper hop's cost belongs to the request that made it
    hop = {"cost_accounting": {"unpriced_reason": "counted_in_parent"}}
    decision = {"kind": "decision_model", "calls": 1, "cost": 0.5, "currency": "usd"}
    router["records"] = [
        _record("r1", "s1", "qwen-chat", "qwen-chat", 1.0),
        _record(
            "r2",
            "s1",
            "vllm-sr/auto",
            "qwen-chat",
            2.0,
            cost_accounting={"router_calls": [decision]},
        ),
        _record("r3", "s1", "vllm-sr/auto", "qwen-chat", 4.0, "EUR"),
        _record("r4", "s2", "vllm-sr/auto", "qwen-msg"),
        _record("r5", "s2", "vllm-sr/auto", "qwen-msg", **hop),
    ]
    router["aggregate"] = {**AGGREGATE, "record_count": 5}

    result = _usage(router)

    assert result.exit_code == 0, result.output
    lines = _lines(result)
    models = lines[lines.index("Top models") + 1 : lines.index("Top entrypoints")]
    assert models == [
        "qwen-chat 3 requests, 4.000000 EUR, 3.500000 USD",
        "qwen-msg 2 requests, no price",
    ]
    assert lines[lines.index("Top entrypoints") + 1 :] == [
        "vllm-sr/auto 4 requests, 4.000000 EUR, 2.500000 USD, 1 with no price",
        "qwen-chat 1 request, 1.000000 USD",
    ]
    # no aggregate per row: the rows come from the window's records
    assert not [query for _, query, _ in router["seen"] if "model" in query]


def test_usage_json_reads_a_duration_window(router):
    before = datetime.now(timezone.utc)

    result = _usage(router, "--since", "1h", "--json")

    assert result.exit_code == 0, result.output
    assert abs(_since(router) - (before - timedelta(hours=1))) < timedelta(seconds=5)
    summary = json.loads(result.output)
    assert summary["requests"] == {
        "total": 3,
        "completed": 2,
        "failed": 1,
        "aborted": 0,
        "in_progress": 0,
        "unknown": 0,
    }
    assert summary["sessions"] == 2
    assert summary["cost"] == AGGREGATE["summary"]["by_currency"]
    assert summary["reviews"] == AGGREGATE["reviews"]
    assert summary["models"] == [
        {"name": "qwen-msg", "requests": 2, "spend": {"USD": 0.0003}, "unpriced": 0},
        {"name": "qwen-chat", "requests": 1, "spend": {"USD": 0.00012}, "unpriced": 0},
    ]
    assert [row["name"] for row in summary["entrypoints"]] == [
        "vllm-sr/auto",
        "vllm-sr/review",
    ]


def test_usage_shows_requests_in_an_unknown_state(router):
    lifecycle = {**AGGREGATE["lifecycle"], "completed": 1, "unknown": 1}
    router["aggregate"] = {**AGGREGATE, "lifecycle": lifecycle}

    result = _usage(router)

    assert "Requests 3: 1 completed, 1 failed, 0 aborted, 1 unknown" in _lines(result)


def test_usage_takes_days_and_rejects_other_windows(router):
    before = datetime.now(timezone.utc)
    result = _usage(router, "--since", "7d")
    assert result.exit_code == 0, result.output
    assert abs(_since(router) - (before - timedelta(days=7))) < timedelta(seconds=5)

    for window in ("soon", "99999999999d", "99999999999h"):
        result = _usage(router, "--since", window)
        assert result.exit_code == 2, (window, result.output)
        assert "--since" in result.output


@pytest.fixture()
def new_york(monkeypatch):
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_usage_counts_a_duration_in_elapsed_time_across_a_clock_change(
    router, new_york, monkeypatch
):
    # 01:30 EST, half an hour after New York set its clocks back from 02:00 EDT
    instant = datetime(2026, 11, 1, 6, 30, tzinfo=timezone.utc)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            local = instant.astimezone(tz)
            return local if tz else local.replace(tzinfo=None)

    monkeypatch.setattr("cli.commands.usage.datetime", Clock)
    # the probe's tomorrow is then in the stub's past: nothing to list
    router["records"] = []

    result = _usage(router, "--since", "1h")

    assert result.exit_code == 0, result.output
    assert _since(router) == instant - timedelta(hours=1)


def test_usage_without_replay_says_how_to_turn_it_on(router):
    router["error"] = (
        404,
        {"error": {"code": "REPLAY_UNAVAILABLE", "message": "replay is off"}},
    )

    result = _usage(router)

    assert result.exit_code == 1
    message = result.output.strip()
    assert len(message.splitlines()) == 1
    assert "Router Replay is off" in message
    assert "router_replay" in message


def test_usage_reports_another_not_found_as_it_is(router):
    # a 404 from something other than Replay, such as a wrong endpoint
    router["error"] = (404, {"error": {"code": "NOT_FOUND", "message": "no route"}})

    result = _usage(router)

    assert result.exit_code == 1
    assert "Router Replay is off" not in result.output
    assert "HTTP 404" in result.output


def test_usage_refuses_a_router_that_ignores_the_window(router):
    router["ignores_since"] = True

    result = _usage(router)

    assert result.exit_code == 1
    assert "since" in result.output
    # one probe of one record a day ahead tells
    probes = [q for _, q, _ in router["seen"] if q.get("limit") == "1"]
    assert len(probes) == 1
    ahead = datetime.fromisoformat(probes[0]["since"]) - datetime.now(timezone.utc)
    assert timedelta(hours=23) < ahead <= timedelta(days=1)


def test_usage_refuses_a_router_that_does_not_net_its_own_calls(router):
    # before gross_saved, total_saved left the Router's own calls in
    group = dict(AGGREGATE["summary"]["by_currency"][0])
    del group["gross_saved"], group["netted_router_call_spend"]
    router["aggregate"] = {**AGGREGATE, "summary": {"by_currency": [group]}}

    result = _usage(router)

    assert result.exit_code == 1
    assert "Upgrade the Router" in result.output
