import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"agent_kv_tier_{name}", HERE / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


placement_proxy = _load("placement_proxy")
prepare_traces = _load("prepare_traces")
report = _load("report")


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_sticky_places_a_new_session_by_recent_prompt_bytes_and_keeps_it():
    clock = Clock()
    p = placement_proxy.Placement("sticky", 2, clock=clock)
    assert p.pick("a", 9000) == 0
    assert p.pick("b", 100) == 1
    # replica 1 carries less recent prompt load, so the next new session goes there
    assert p.pick("c", 50) == 1
    # a known session stays on its replica whatever the load is now
    assert p.pick("a", 10) == 0
    assert p.pick("b", 99999) == 1


def test_sticky_ignores_sessions_idle_past_the_window():
    clock = Clock()
    p = placement_proxy.Placement("sticky", 2, sticky_idle=300, clock=clock)
    assert p.pick("a", 9000) == 0
    clock.now = 301
    assert p.pick("b", 10) == 0
    clock.now = 302
    p.touch("b")
    assert p.pick("c", 10) == 1


def test_session_hash_matches_the_report_and_is_stable():
    sid = "c002c6bbe260bc87ee6ff7cea1b366cee2ed"
    p = placement_proxy.Placement("session", 2)
    expected = int(hashlib.sha256(sid.encode()).hexdigest(), 16) % 2
    assert p.pick(sid, 1) == p.pick(sid, 2) == expected


def test_least_request_and_round_robin():
    p = placement_proxy.Placement("least_request", 3)
    p.inflight = [2, 0, 1]
    assert p.pick("x", 1) == 1
    p.inflight = [0, 0, 0]
    assert p.pick("x", 1) == 0
    rr = placement_proxy.Placement("round_robin", 2)
    assert [rr.pick("same", 1) for _ in range(4)] == [0, 1, 0, 1]


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        placement_proxy.Placement("random", 2)


def test_summarize_requests_counts_successes_with_nearest_rank_quantiles():
    rows = [
        {
            "success": "True",
            "ttft": str(t),
            "input_tokens": "100",
            "trace_id": f"s{t % 2}",
        }
        for t in range(1, 11)
    ] + [{"success": "False", "ttft": "999", "input_tokens": "5000", "trace_id": "s9"}]
    assert report.summarize_requests(rows) == {
        "requests": 10,
        "input_tokens": 1000,
        "sessions": 2,
        "ttft_p50": 6.0,
        "ttft_p90": 10.0,
        "ttft_max": 10.0,
    }


def test_parse_server_log_reads_pool_last_hits_and_mean_waiting():
    lines = [
        "INFO 10-20 13:34:09 [kv_cache_utils.py:2404] GPU KV cache size: 801,376 tokens, Maximum concurrency",
        "Engine 000: Avg prompt throughput: 1.0 tokens/s, Running: 2 reqs, Waiting: 3 reqs, GPU KV cache usage: "
        "4.0%, Prefix cache hit rate: 20.0%, External prefix cache hit rate: 1.5%",
        "Engine 000: Avg prompt throughput: 1.0 tokens/s, Running: 4 reqs, Waiting: 1 reqs, GPU KV cache usage: "
        "13.8%, Prefix cache hit rate: 35.0%, External prefix cache hit rate: 10.0%",
    ]
    assert report.parse_server_log(lines) == {
        "kv_pool_tokens": 801376,
        "last_prefix_hit": 35.0,
        "last_external_hit": 10.0,
        "mean_waiting": 2.0,
    }
    plain = report.parse_server_log([lines[1].split(", External")[0]])
    assert (plain["last_prefix_hit"], plain["last_external_hit"]) == (20.0, None)


def test_parse_client_log_reads_the_tester_summary():
    lines = [
        "INFO -   TTFT avg/p50/p95/max: 20.81s / 4.16s / 75.15s / 342.37s",
        "INFO -   Avg Workload Cache Hit Rate: 72.4%",
        "INFO -   Peak Working Set: 1,027,968 tokens",
    ]
    assert report.parse_client_log(lines) == {
        "tester_ttft_avg_p50_p95_max": [20.81, 4.16, 75.15, 342.37],
        "tester_cache_hit": 72.4,
        "peak_working_set_tokens": 1027968,
    }


def test_prefix_medians_group_by_arm_and_length():
    rows = [
        {"arm": "gpu", "tokens": 4096, "ttft_first": f, "ttft_second": s}
        for f, s in [(3, 0.1), (1, 0.3), (2, 0.2)]
    ]
    assert report.prefix_medians(rows) == {("gpu", 4096): (2, 0.2)}


def test_split_traces_writes_one_file_per_session(tmp_path):
    ids = ["a", "b"]
    lines = [json.dumps({"id": ids[0]}) + "\n", "\n", json.dumps({"id": ids[1]}) + "\n"]
    assert prepare_traces.split_traces(lines, tmp_path) == len(ids)
    assert json.loads((tmp_path / "trace_0001.json").read_text()) == {"id": "b"}
