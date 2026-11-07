from __future__ import annotations

import json
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from cli.sr_bench.engine import Engine
from cli.sr_bench.report import make_report
from cli.sr_bench.store import Store
from cli.sr_bench.transport import CallFailure, mom_usage, normalize_usage
from test_sr_bench import Target, manifest, target, wait_run

__all__: list[str] = ["target"]

UNKNOWN_RECEIPT: dict[str, object] = {
    "version": 1,
    "complete": False,
    "calls": [],
}


class UnknownUsageTarget(Target):
    def end_headers(self) -> None:
        self.send_header("x-vsr-model-usage", json.dumps(UNKNOWN_RECEIPT))
        self.send_header("x-vsr-inference-call-count", "1")
        self.send_header("x-vsr-selected-model", "model")
        super().end_headers()


def test_incomplete_empty_receipt_is_unknown_and_never_a_direct_call() -> None:
    result = mom_usage(
        response_headers={
            "x-vsr-model-usage": json.dumps(UNKNOWN_RECEIPT),
            "x-vsr-inference-call-count": "1",
            "x-vsr-selected-model": "model",
        },
        response_usage=normalize_usage({"prompt_tokens": 10, "completion_tokens": 3}),
        prices={
            "model": {"input": 1, "cached_input": 1, "cache_write": 1, "output": 1}
        },
        cache_read_reported=True,
        cache_write_reported=True,
    )
    assert result == {
        "usage": None,
        "cost_usd": None,
        "model_usage": [],
        "inference_call_count": None,
        "usage_complete": False,
        "cache_read_reported": None,
        "cache_write_reported": None,
    }


@pytest.mark.parametrize(
    "receipt",
    [
        {**UNKNOWN_RECEIPT, "complete": True},
        {**UNKNOWN_RECEIPT, "complete": 0},
        {**UNKNOWN_RECEIPT, "complete": "false"},
        {**UNKNOWN_RECEIPT, "version": True},
        {**UNKNOWN_RECEIPT, "version": 1.0},
        {**UNKNOWN_RECEIPT, "version": "1"},
        {**UNKNOWN_RECEIPT, "calls": None},
        {**UNKNOWN_RECEIPT, "calls": {}},
        {**UNKNOWN_RECEIPT, "calls": [{}]},
        [],
        None,
    ],
)
def test_unknown_receipt_does_not_admit_invalid_contracts(receipt: object) -> None:
    with pytest.raises(CallFailure):
        mom_usage(
            response_headers={"x-vsr-model-usage": json.dumps(receipt)},
            response_usage=None,
            prices={},
        )


@pytest.mark.parametrize(
    ("policy", "bound", "status"),
    [
        ("capability_only", None, "completed"),
        ("capability_only", 1, "failed"),
        ("require_priced", 1, "failed"),
    ],
)
def test_unknown_accounting_respects_frozen_bounds(
    tmp_path: Path,
    target: ThreadingHTTPServer,
    policy: str,
    bound: int | None,
    status: str,
) -> None:
    target.RequestHandlerClass = UnknownUsageTarget
    target.ack = "a" * 64
    document = manifest(target)
    document["cost_policy"] = policy
    document["targets"][0].update(kind="mom", config_hash=target.ack)
    if bound is not None:
        document["targets"][0]["max_inference_calls"] = bound
    store = Store(tmp_path)
    engine = Engine(store)
    run = engine.start(document)
    engine.threads[run["id"]].join(timeout=10)
    assert not engine.threads[run["id"]].is_alive()
    assert wait_run(store, run["id"])["status"] == status
    assert len(target.requests) == 1
    call = store.calls(run["id"])[0]
    assert call["model_usage"] == []
    assert call["usage"] is call["cost_usd"] is call["inference_call_count"] is None
    if status == "completed":
        assert store.results(run["id"])[0]["correct"] is True
        metrics = make_report(store, run["id"])["summary"]["targets"][0]
        assert metrics["tokens"] is metrics["cost_usd"] is metrics["call_count"] is None
        assert metrics["request_count"] == 1
    else:
        assert store.results(run["id"])[0].get("score") is None
        assert "unknown" in store.results(run["id"])[0]["error"].lower()
