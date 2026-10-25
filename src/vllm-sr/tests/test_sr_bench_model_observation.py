from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from cli.sr_bench.engine import Engine
from cli.sr_bench.report import _observed_model, make_report
from cli.sr_bench.store import Store


@pytest.mark.parametrize(
    ("selected_header", "stream_model", "expected"),
    [
        (None, None, None),
        ("header-model", None, "header-model"),
        (None, "stream-model", "stream-model"),
        ("header-model", "stream-model", "header-model"),
    ],
)
def test_successful_http_response_keeps_observation_separate_from_request(
    tmp_path: Path,
    selected_header: str | None,
    stream_model: str | None,
    expected: str | None,
) -> None:
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def do_POST(self) -> None:
            requests.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            if selected_header:
                self.send_header("x-vsr-selected-model", selected_header)
            self.end_headers()
            event = {
                "choices": [
                    {"index": 0, "delta": {"content": "A"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                **({"model": stream_model} if stream_model else {}),
            }
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    store = Store(tmp_path)
    engine = Engine(store)
    try:
        run = engine.start(
            {
                "version": "sr-bench-1.0",
                "targets": [
                    {
                        "id": "subject",
                        "kind": "single",
                        "model": "requested-entrypoint",
                        "base_url": f"http://127.0.0.1:{provider.server_port}/v1",
                        "prices": {
                            name: {
                                "input": 1,
                                "cached_input": 1,
                                "cache_write": 1,
                                "output": 1,
                            }
                            for name in ("requested-entrypoint", "stream-model")
                        },
                    }
                ],
                "cases": [
                    {
                        "id": "one",
                        "benchmark": "mmlu-pro",
                        "messages": [{"role": "user", "content": "Return A"}],
                        "answer": "A",
                    }
                ],
                "limits": {
                    "total_timeout_s": 3,
                    "idle_timeout_s": 2,
                    "max_run_seconds": 10,
                },
            }
        )
        engine.threads[run["id"]].join(timeout=10)
        assert not engine.threads[run["id"]].is_alive()
        assert store.get(run["id"])["status"] == "completed"
        call = store.calls(run["id"])[0]
        target = make_report(store=store, run_id=run["id"])["summary"]["targets"][0]
        print(
            json.dumps(
                {
                    "requested_model": requests[0]["model"],
                    "selected_header": selected_header,
                    "stream_model": stream_model,
                    "receipt_model": call["model"],
                    "selected_model": call.get("selected_model"),
                    "provider_model_observed": call.get("provider_model_observed"),
                    "report_selected_models": target["selected_models"],
                    "cost_usd": target["cost_usd"],
                },
                sort_keys=True,
            )
        )
        assert requests[0]["model"] == "requested-entrypoint"
        assert call["model"] == (stream_model or "requested-entrypoint")
        assert call["selected_model"] == expected
        assert call["provider_model_observed"] is bool(stream_model)
        assert target["selected_models"] == ({expected: 1} if expected else {})
        assert target["cost_usd"] == pytest.approx(13 / 1_000_000)
    finally:
        engine.close()
        provider.shutdown()
        provider.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    ("receipt", "expected"),
    [
        ({"status": "completed", "model": "legacy-model"}, "legacy-model"),
        ({"status": "failed", "model": "requested-entrypoint"}, None),
        (
            {
                "status": "failed",
                "model": "requested-entrypoint",
                "selected_model": "header-model",
            },
            "header-model",
        ),
    ],
)
def test_legacy_and_failed_receipt_observations_stay_compatible(
    receipt: dict[str, Any], expected: str | None
) -> None:
    assert _observed_model(receipt) == expected
