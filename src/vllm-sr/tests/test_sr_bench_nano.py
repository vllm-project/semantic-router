"""EXPERIMENTAL sr-bench-nano: frozen ids, one generation, no output cap, timeouts."""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from cli.commands.benchmark_nano import nano_group
from cli.sr_bench import nano
from cli.sr_bench.contracts import digest, plan
from cli.sr_bench.engine import Engine
from cli.sr_bench.harness_worker import _last_fence_code
from cli.sr_bench.nano_prepare import freeze
from cli.sr_bench.report import make_report
from cli.sr_bench.store import TERMINAL, Store
from click.testing import CliRunner

SIZES = {
    "mmlu-pro": (150, 500),
    "simpleqa-verified": (100, 300),
    "gpqa-diamond": (60, 138),
    "livecodebench": (60, 150),
    "hle": (60, 150),
}


def test_frozen_lists_have_sizes_disjoint_splits_and_verified_hashes():
    document = nano.frozen_ids()
    assert document["version"] == nano.IDS_VERSION
    assert document["seed"] == nano.SEED == 20260918
    assert nano.body_sha256(document) == document["sha256"]
    assert set(document["benchmarks"]) == set(nano.BENCHMARKS)
    for benchmark, (dev, holdout) in SIZES.items():
        splits = document["benchmarks"][benchmark]["splits"]
        dev_ids = [task["id"] for task in splits["nano"]["tasks"]]
        holdout_ids = [task["id"] for task in splits["nano-holdout"]["tasks"]]
        assert (len(dev_ids), len(holdout_ids)) == (dev, holdout)
        assert len(set(dev_ids)) == dev and len(set(holdout_ids)) == holdout
        assert not set(dev_ids) & set(holdout_ids)
        assert splits["nano"]["ids_sha256"] == digest(dev_ids)
        assert splits["nano-holdout"]["ids_sha256"] == digest(holdout_ids)
        for task in splits["nano"]["tasks"] + splits["nano-holdout"]["tasks"]:
            assert task["id"].startswith(benchmark + "/")
            assert all(len(v) == 64 for k, v in task.items() if k.endswith("sha256"))
            assert "prompt" not in task and "answer" not in task
    assert document["benchmarks"]["gpqa-diamond"]["population"]["count"] == 198
    assert [
        f["name"] for f in document["benchmarks"]["livecodebench"]["source"]["files"]
    ] == [
        "test5.jsonl",
        "test6.jsonl",
    ]


def test_mmlu_windows_stay_inside_sr_bench_quick_and_standard_pools():
    assert nano.WINDOWS["mmlu-pro"]["nano"] == (0, 150)
    offset, count = nano.WINDOWS["mmlu-pro"]["nano-holdout"]
    assert offset >= 500
    assert offset + count <= 2500


def test_equal_weights_only_for_nano():
    assert dict.fromkeys(SIZES, 0.2) == nano.WEIGHTS
    assert sum(nano.WEIGHTS.values()) == pytest.approx(1)


@pytest.mark.skipif(
    not os.environ.get("SR_BENCH_NANO_SOURCE_DIR"),
    reason="set SR_BENCH_NANO_SOURCE_DIR to regenerate from pinned sources",
)
def test_freeze_regenerates_the_committed_list(tmp_path):
    source = os.environ["SR_BENCH_NANO_SOURCE_DIR"]
    first = freeze(tmp_path, source)
    assert first == freeze(tmp_path, source) == nano.frozen_ids()


@pytest.mark.parametrize(
    ("reference", "reply", "correct"),
    [
        ("42", "Explanation: x\nExact Answer: 42\nConfidence: 90%", True),
        ("0.25", "Exact Answer: 0.249", True),
        ("0.25", "Exact Answer: 0.26", False),
        ("3/4", "Exact Answer: \\frac{3}{4}", True),
        ("Paris", "Exact Answer: **paris.**", True),
        ("Paris", "Exact Answer: Lyon", False),
        ("7", "The result is \\boxed{7}", True),
        ("7", "seven", False),
    ],
)
def test_hle_exact_grader_is_deterministic(reference, reply, correct):
    result = nano.grade_hle({"answer": reference}, reply)
    assert result["correct"] is correct
    assert result["details"]["grader_version"] == nano.HLE_GRADER


def test_hle_subset_keeps_only_judge_free_exact_answers():
    rows = [
        {"answer": "12", "metadata": {"answer_type": "exactMatch"}},
        {"answer": "x^2 + 1", "metadata": {"answer_type": "exactMatch"}},
        {"answer": "B", "metadata": {"answer_type": "multipleChoice"}},
        {"answer": "Kyoto", "metadata": {"answer_type": "exactMatch"}},
    ]
    cases = [{**row, "messages": [{"role": "user", "content": "q"}]} for row in rows]
    kept = nano.render_hle(cases)
    assert [case["answer"] for case in kept] == ["12", "Kyoto"]
    assert kept[0]["messages"][0] == {"role": "system", "content": nano.HLE_SYSTEM}


def test_simpleqa_official_template_is_pinned_and_letters_fail_closed():
    template = nano.simpleqa_template()
    assert "{predicted_answer}" in template and "A: CORRECT" in template
    messages = nano.simpleqa_messages(
        {"messages": [{"role": "user", "content": "Q?"}], "answer": "Gold"}, "Pred"
    )
    assert "Question: Q?" in messages[0]["content"]
    assert nano.simpleqa_verdict("A") == "correct"
    assert nano.simpleqa_verdict("C") == "not_attempted"
    with pytest.raises(ValueError, match="no A/B/C"):
        nano.simpleqa_verdict("")


def test_lcb_extraction_takes_only_the_last_fenced_block():
    reply = "```\nx ⇔ y\n```\nthen\n```python\nprint(1)\nprint(2)\n```\n"
    assert _last_fence_code(reply) == "print(1)\nprint(2)"
    assert _last_fence_code("no code") == ""


class Target(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        server = self.server
        server.requests.append({"body": body, "headers": dict(self.headers)})
        prompt = body["messages"][-1]["content"]
        if "slow" in prompt:
            time.sleep(server.slow_s)
        answer = server.answers.get(prompt, "A")
        usage = {"prompt_tokens": 5, "completion_tokens": server.completion_tokens}
        self.send_response(200)
        if not body["stream"]:
            payload = json.dumps(
                {
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"content": answer},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                }
            ).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        events = [
            {
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": answer, "tool_calls": None},
                        "finish_reason": "stop",
                    }
                ],
            },
            {"model": body["model"], "choices": [], "usage": usage},
        ]
        try:
            for event in events:
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


@pytest.fixture
def target():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Target)
    server.requests, server.answers = [], {}
    server.slow_s, server.completion_tokens = 0, 3
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _case(benchmark, identity, prompt, answer):
    return {
        "id": f"{benchmark}/{identity}",
        "benchmark": benchmark,
        "messages": [{"role": "user", "content": prompt}],
        "answer": answer,
        "metadata": {"stratum": "s", "split": "dev"},
    }


@pytest.fixture
def frozen(monkeypatch):
    """Synthetic frozen list built with the real task hashing."""
    cases = [
        _case("mmlu-pro", "1", "Return A", "A"),
        _case("mmlu-pro", "2", "slow Return A", "A"),
        _case("hle", "3", "Exact?", "42"),
        _case("simpleqa-verified", "4", "Who?", "Ada"),
    ]
    tasks = {name: [] for name in nano.BENCHMARKS}
    for case in cases:
        tasks[case["benchmark"]].append(nano.task_record(case))
    document = {
        "sha256": "f" * 64,
        "benchmarks": {
            name: {"splits": {"nano": {"tasks": rows}, "nano-holdout": {"tasks": []}}}
            for name, rows in tasks.items()
        },
    }
    monkeypatch.setattr(nano, "frozen_ids", lambda: document)
    return cases


def _manifest(target, cases, **updates):
    base = f"http://127.0.0.1:{target.server_port}/v1"
    payload = {
        "version": "sr-bench-1.0",
        "profile": "nano",
        "cost_policy": "capability_only",
        "targets": [{"id": "mine", "kind": "single", "model": "m", "base_url": base}],
        "cases": cases,
    }
    if any(case["benchmark"] == "simpleqa-verified" for case in cases):
        payload["auxiliary_targets"] = {
            "simpleqa-grader": {
                "id": "simpleqa-grader",
                "kind": "single",
                "model": "grader-model",
                "base_url": base,
                "request_params": {"temperature": 0},
            }
        }
        payload["benchmark_options"] = {
            "simpleqa-verified": {
                "judge": "simpleqa-grader",
                "grader_version": nano.SIMPLEQA_GRADER,
            }
        }
    payload.update(updates)
    return payload


def _wait(store, run_id, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        run = store.get(run_id)
        if run["status"] in TERMINAL:
            return run
        time.sleep(0.02)
    raise AssertionError("run did not terminate")


def test_nano_plan_freezes_uncapped_protocol_and_equal_weights(target, frozen):
    frozen_plan = plan(_manifest(target, frozen[:1]))
    assert frozen_plan["output_policy"] == "uncapped"
    assert "max_tokens" not in frozen_plan["sampling"]
    assert frozen_plan["limits"]["total_timeout_s"] == 1800
    assert frozen_plan["limits"]["idle_timeout_s"] == 1800
    assert frozen_plan["benchmark_weights"] == nano.WEIGHTS


def test_nano_plan_rejects_unfrozen_cases_caps_and_missing_grader(target, frozen):
    tampered = [{**frozen[0], "messages": [{"role": "user", "content": "Return B"}]}]
    with pytest.raises(ValueError, match="frozen"):
        plan(_manifest(target, tampered))
    with pytest.raises(ValueError, match="uncapped"):
        plan(_manifest(target, frozen[:1], output_policy="bounded"))
    with pytest.raises(ValueError, match="capability_only"):
        plan(_manifest(target, frozen[:1], cost_policy="require_priced"))
    no_grader = _manifest(target, frozen[3:4])
    del no_grader["auxiliary_targets"], no_grader["benchmark_options"]
    with pytest.raises(ValueError, match="SimpleQA grader is not configured"):
        plan(no_grader)
    hot = _manifest(target, frozen[3:4])
    hot["auxiliary_targets"]["simpleqa-grader"]["request_params"] = {"temperature": 1}
    with pytest.raises(ValueError, match="temperature=0"):
        plan(hot)


def test_uncapped_output_is_reserved_for_nano(target):
    document = {
        "version": "sr-bench-1.0",
        "cost_policy": "capability_only",
        "output_policy": "uncapped",
        "targets": [
            {
                "id": "t",
                "kind": "single",
                "model": "m",
                "base_url": f"http://127.0.0.1:{target.server_port}/v1",
            }
        ],
        "cases": [
            {
                "id": "q",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "x"}],
                "answer": "A",
            }
        ],
    }
    with pytest.raises(ValueError, match="reserved for nano"):
        plan(document)


def test_nano_run_one_generation_no_cap_timeout_status_and_grader_record(
    tmp_path, target, frozen, monkeypatch
):
    target.slow_s = 1.5
    target.answers = {"Exact?": "Exact Answer: 42"}
    grader_prompt = nano.simpleqa_messages(frozen[3], "A")[0]["content"]
    target.answers[grader_prompt] = "A"
    monkeypatch.setenv("NANO_GRADER_HEADER", "secret-value")
    document = _manifest(
        target,
        frozen,
        limits={"total_timeout_s": 1, "idle_timeout_s": 1, "case_timeout_s": 5},
    )
    document["auxiliary_targets"]["simpleqa-grader"]["header_env"] = {
        "X-Grader-Tenant": "NANO_GRADER_HEADER"
    }
    store = Store(tmp_path)
    run = _wait(store, Engine(store).start(document)["id"])
    assert run["status"] == "completed"
    assert run["progress"]["timeout"] == 1 and run["progress"]["failed"] == 0
    statuses = {r["case_id"]: r["status"] for r in store.results(run["id"])}
    assert statuses == {
        "mmlu-pro/1": "completed",
        "mmlu-pro/2": "timeout",
        "hle/3": "completed",
        "simpleqa-verified/4": "completed",
    }
    subject = [r for r in target.requests if r["body"]["model"] == "m"]
    assert len(subject) == len(frozen)
    assert all("max_tokens" not in r["body"] for r in subject)
    grader = [r for r in target.requests if r["body"]["model"] == "grader-model"]
    assert len(grader) == 1 and grader[0]["body"]["temperature"] == 0
    assert grader[0]["headers"]["X-Grader-Tenant"] == "secret-value"
    report = make_report(store, run["id"])
    score = report["summary"]["targets"][0]
    assert score["timeouts"] == 1 and score["failed"] == 0
    assert score["accuracy"] == pytest.approx(3 / 4)
    section = report["nano"]
    assert section["targets"][0]["max_tokens_sent"] is False
    assert section["targets"][0]["timeouts"] == 1
    simpleqa = section["graders"]["simpleqa-verified"]
    assert simpleqa["model"] == "grader-model"
    assert simpleqa["template_sha256"] == nano.SIMPLEQA_TEMPLATE_SHA256
    assert simpleqa["header_names"] == ["X-Grader-Tenant"]
    assert "secret-value" not in json.dumps(report)


def test_explicit_max_tokens_non_streaming_and_round_cap_flag(tmp_path, target, frozen):
    target.completion_tokens = 1024
    document = _manifest(target, frozen[:1])
    document["targets"][0].update(
        {"stream": False, "request_params": {"max_tokens": 120000}}
    )
    store = Store(tmp_path)
    run = _wait(store, Engine(store).start(document)["id"])
    assert run["status"] == "completed"
    body = target.requests[0]["body"]
    assert body["stream"] is False and "stream_options" not in body
    assert body["max_tokens"] == 120000
    row = make_report(store, run["id"])["nano"]["targets"][0]
    assert row["max_tokens_sent"] is True and row["max_tokens"] == 120000
    assert row["stream"] is False
    assert row["suspected_output_cap_cases"] == ["mmlu-pro/1"]


def test_header_env_rejects_reserved_names_and_raw_values(target, frozen):
    for header_env in (
        {"Authorization": "ENV"},
        {"X-SR-Bench-Expected-Config-Hash": "ENV"},
        {"X-Tenant": "not an env"},
    ):
        document = _manifest(target, frozen[:1])
        document["targets"][0]["header_env"] = header_env
        with pytest.raises(ValueError, match="header_env"):
            plan(document)


def test_nano_cli_manifest_fails_fast_without_grader(tmp_path, frozen):
    dataset = tmp_path / "manifest.json"
    dataset.write_text(
        json.dumps(
            {
                "profile": "nano",
                "path": str(tmp_path / "cases.jsonl"),
                "sha256": "0" * 64,
                "benchmarks": ["simpleqa-verified"],
            }
        )
    )
    targets = tmp_path / "targets.json"
    targets.write_text("[]")
    result = CliRunner().invoke(
        nano_group,
        [
            "manifest",
            "--dataset",
            str(dataset),
            "--targets",
            str(targets),
            "--output",
            str(tmp_path / "run.json"),
        ],
        env={"SR_BENCH_NANO_GRADER_BASE_URL": "", "SR_BENCH_NANO_GRADER_MODEL": ""},
    )
    assert result.exit_code != 0
    assert "SimpleQA grader is not configured" in result.output
    assert not Path(tmp_path / "run.json").exists()
