"""Native capacities are proved by real local render/stream HTTP fixtures."""

import copy
import json
import threading
import time
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from cli.sr_bench.candidate_plans import candidate_manifest, validate_candidate_protocol
from cli.sr_bench.contracts import digest, plan
from cli.sr_bench.engine import Context, Engine
from cli.sr_bench.native_output import NativeOutputError, validate_recipes
from cli.sr_bench.replay_validation import ReplayValidator
from cli.sr_bench.report import _comparison_protocol
from cli.sr_bench.store import Store
from cli.sr_bench.transport import CallFailure, chat

MODEL_LIMITS = {"physical": {"context_window": 32, "max_output_tokens": 24}}


class NativeTarget(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        self.server.requests.append((self.path, body))
        if self.path.endswith("/routing/preview"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "config_hash": "a" * 64,
                        "selection_status": "selected",
                        "selected_model": "physical",
                    }
                ).encode()
            )
            return
        if self.path.endswith("/render"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.flush()
            self.server.render_started.set()
            self.server.render_release.wait(2)
            data = self.server.render
            encoded = data if isinstance(data, bytes) else json.dumps(data).encode()
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(encoded)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        for key, value in self.server.response_headers.items():
            self.send_header(key, value)
        self.end_headers()
        event = {
            "model": self.server.response_model,
            "choices": [
                {"index": 0, "delta": {"content": "A"}, "finish_reason": "stop"}
            ],
            "usage": self.server.usage,
        }
        if self.server.response_model is None:
            del event["model"]
        with suppress(BrokenPipeError, ConnectionResetError):
            if self.server.role_only_initial:
                first = {"choices": [{"index": 0, "delta": {"role": "assistant"}}]}
                self.wfile.write(("data: " + json.dumps(first) + "\n\n").encode())
                self.wfile.flush()
            self.wfile.write(
                ("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode()
            )
            self.wfile.flush()


@pytest.fixture
def native_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), NativeTarget)
    server.requests = []
    server.render = {
        "model": "physical",
        "token_ids": list(range(12)),
        "sampling_params": {"max_tokens": 20},
    }
    server.render_started = threading.Event()
    server.render_release = threading.Event()
    server.render_release.set()
    server.response_model = "physical"
    server.role_only_initial = False
    server.response_headers = {}
    server.usage = {
        "prompt_tokens": 12,
        "completion_tokens": 1,
        "prompt_tokens_details": {"cached_tokens": 3, "cache_creation_tokens": 2},
    }
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.render_release.set()
    server.shutdown()
    server.server_close()


def document(server, mom=False):
    target = {
        "id": "subject",
        "kind": "mom" if mom else "single",
        "model": "balance" if mom else "physical",
        "base_url": f"http://127.0.0.1:{server.server_port}/v1",
        "native_limits": copy.deepcopy(MODEL_LIMITS),
        "prices": {
            "physical": {"input": 1, "cached_input": 1, "cache_write": 1, "output": 1}
        },
        "request_params": {"chat_template_kwargs": {"reasoning_effort": "max"}},
    }
    if mom:
        target.update(
            config_hash="a" * 64,
            capture_recipe=True,
            max_inference_calls=1,
            preview_url=f"http://127.0.0.1:{server.server_port}/api/v1/routing/preview",
        )
    return {
        "version": "sr-bench-1.0",
        "output_policy": "native",
        "targets": [target],
        "cases": [
            {
                "id": "case",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "A"}],
                "answer": "A",
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "tool", "parameters": {"type": "object"}},
                    }
                ],
            }
        ],
        "limits": {"total_timeout_s": 3, "idle_timeout_s": 2, "max_run_seconds": 10},
    }


def request(frozen, **kwargs):
    return chat(
        frozen["targets"][0],
        frozen["cases"][0]["messages"],
        frozen["sampling"],
        frozen["limits"],
        kwargs.pop("cancelled", lambda: False),
        output_policy="native",
        **kwargs,
    )


def recipe_provenance(frozen):
    target = frozen["targets"][0]
    return {
        "recipe_snapshots": {
            target["id"]: {
                "config_hash": target["config_hash"],
                "entrypoints": [{"model_names": [target["model"]], "recipe": "chosen"}],
                "recipes": [
                    {
                        "name": "chosen",
                        "routing": {
                            "decisions": [
                                {
                                    "modelRefs": [{"model": "physical"}],
                                    "plugins": [
                                        {
                                            "type": "request_params",
                                            "configuration": {
                                                "default_max_tokens": "auto"
                                            },
                                        }
                                    ],
                                }
                            ]
                        },
                    }
                ],
            }
        }
    }


def test_native_single_render_resolves_actual_context_and_preserves_effort(
    native_server, tmp_path
):
    frozen = plan(document(native_server))
    assert frozen["limits"]["max_output_tokens"] == 24
    assert "max_tokens" not in frozen["sampling"]
    assert plan(frozen)["plan_sha256"] == frozen["plan_sha256"]
    path = tmp_path / "call.sse"
    result = request(
        frozen, extra_body={"tools": frozen["cases"][0]["tools"]}, stream_path=path
    )
    rendered, generated = native_server.requests
    assert rendered[0] == "/v1/chat/completions/render"
    assert generated[0] == "/v1/chat/completions"
    assert "max_tokens" not in rendered[1]
    assert generated[1] == {**rendered[1], "max_tokens": 20}
    assert generated[1]["chat_template_kwargs"]["reasoning_effort"] == "max"
    assert generated[1]["tools"] == frozen["cases"][0]["tools"]
    assert result["native_output"]["input_tokens"] == 12
    assert result["native_output"]["max_output_tokens"] == 20
    assert result["native_output"]["configured_max_output_tokens"] == 24
    reconstructed = {
        **rendered[1],
        "max_tokens": result["native_output"]["max_output_tokens"],
    }
    assert digest(reconstructed) == result["native_output"]["effective_request_sha256"]
    assert digest(generated[1]) == result["native_output"]["effective_request_sha256"]
    assert result["inference_call_count"] == 1 and result["output_complete"] is True
    assert path.read_bytes().endswith(b"data: [DONE]\n\n")


@pytest.mark.parametrize(
    "change",
    [
        "sampling_cap",
        "target_cap",
        "ceiling",
        "unknown_limits",
        "boolean_limits",
        "wrong_model",
    ],
)
def test_native_plan_rejects_artificial_or_unproved_capacity(native_server, change):
    source = document(native_server)
    if change == "sampling_cap":
        source["sampling"] = {"max_tokens": 4}
    elif change == "target_cap":
        source["targets"][0]["request_params"]["max_tokens"] = 4
    elif change == "ceiling":
        source["limits"]["max_output_tokens"] = 4
    elif change == "unknown_limits":
        del source["targets"][0]["native_limits"]
    elif change == "boolean_limits":
        source["targets"][0]["native_limits"]["physical"]["context_window"] = True
    else:
        source["targets"][0]["native_limits"] = {"other": MODEL_LIMITS["physical"]}
    with pytest.raises(ValueError):
        plan(source)
    assert native_server.requests == []


@pytest.mark.parametrize(
    "change", ["lower_cap", "wrong_model", "overflow", "multimodal", "invalid_json"]
)
def test_native_bad_render_never_dispatches_generation(native_server, change):
    if change == "lower_cap":
        native_server.render["sampling_params"]["max_tokens"] = 4
    elif change == "wrong_model":
        native_server.render["model"] = "other"
    elif change == "overflow":
        native_server.render["token_ids"] = list(range(33))
    elif change == "multimodal":
        native_server.render["features"] = {"private": "not supported"}
    else:
        native_server.render = b"secret provider body is not json"
    with pytest.raises(CallFailure) as failure:
        request(plan(document(native_server)))
    assert "secret" not in str(failure.value)
    assert len(native_server.requests) == 1
    assert native_server.requests[0][0].endswith("/render")


def test_native_render_cancellation_interrupts_stalled_first_body_byte(native_server):
    native_server.render_release.clear()
    cancel = threading.Event()
    result = []

    def call():
        try:
            request(plan(document(native_server)), cancelled=cancel.is_set)
        except CallFailure as exc:
            result.append(exc)

    thread = threading.Thread(target=call)
    thread.start()
    assert native_server.render_started.wait(1)
    cancel.set()
    thread.join(0.8)
    assert not thread.is_alive()
    assert "cancelled" in str(result[0])
    assert len(native_server.requests) == 1


def set_native_headers(server):
    server.response_headers = {
        "X-SR-Bench-Config-Hash": "a" * 64,
        "X-VSR-Selected-Model": "physical",
        "X-VSR-Inference-Call-Count": "1",
        "X-VSR-Effective-Input-Tokens": "12",
        "X-VSR-Effective-Max-Output-Tokens": "20",
    }


def test_native_mom_omits_cap_and_uses_actual_dispatch_ack(native_server):
    set_native_headers(native_server)
    result = request(plan(document(native_server, mom=True)))
    assert len(native_server.requests) == 1
    assert "max_tokens" not in native_server.requests[0][1]
    assert result["native_output"]["source"] == "router_dispatch"
    assert result["native_output"]["max_output_tokens"] == 20


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "smaller",
        "wrong_config",
        "logical_alias",
        "input_usage",
        "output_usage",
    ],
)
def test_native_mom_unproved_capacity_fails_once(native_server, change):
    set_native_headers(native_server)
    if change == "missing":
        del native_server.response_headers["X-VSR-Effective-Input-Tokens"]
    elif change == "smaller":
        native_server.response_headers["X-VSR-Effective-Max-Output-Tokens"] = "4"
    elif change == "wrong_config":
        native_server.response_headers["X-SR-Bench-Config-Hash"] = "b" * 64
    elif change == "logical_alias":
        native_server.response_model = "upstream-alias"
    elif change == "input_usage":
        native_server.usage["prompt_tokens"] = 13
    else:
        native_server.usage["completion_tokens"] = 21
    with pytest.raises(CallFailure):
        request(plan(document(native_server, mom=True)))
    assert len(native_server.requests) == 1


def test_native_candidate_and_comparison_freeze_policy_and_model_limits(native_server):
    baseline = {
        "id": "baseline",
        "status": "completed",
        "manifest": plan(document(native_server)),
    }
    candidate = plan(
        candidate_manifest(baseline, document(native_server, mom=True)["targets"])
    )
    validate_candidate_protocol(baseline, candidate)
    _comparison_protocol(baseline["manifest"], candidate)
    for field in ("output_policy", "native_limits"):
        changed = copy.deepcopy(candidate)
        if field == "output_policy":
            changed[field] = "bounded"
        else:
            changed["targets"][0][field]["physical"]["context_window"] += 1
        with pytest.raises(ValueError):
            validate_candidate_protocol(baseline, changed)
        with pytest.raises(ValueError):
            _comparison_protocol(baseline["manifest"], changed)


def test_native_recipe_gate_rejects_clip_and_unregistered_candidates(native_server):
    frozen = plan(document(native_server, mom=True))
    original = recipe_provenance(frozen)
    validate_recipes(frozen, original)
    for change in ("fixed", "clip", "unregistered"):
        provenance = copy.deepcopy(original)
        decision = provenance["recipe_snapshots"]["subject"]["recipes"][0]["routing"][
            "decisions"
        ][0]
        if change == "fixed":
            decision["plugins"][0]["configuration"]["default_max_tokens"] = 4
        elif change == "clip":
            decision["plugins"][0]["configuration"]["max_tokens_limit"] = 4
        else:
            decision["modelRefs"][0]["model"] = "other"
        with pytest.raises(NativeOutputError):
            validate_recipes(frozen, provenance)


def test_native_engine_records_resolved_budget_and_reserves_native_capacity(
    native_server, tmp_path
):
    store = Store(tmp_path)
    engine = Engine(store)
    source = document(native_server)
    run = engine.start(source)
    deadline = time.monotonic() + 3
    while store.get(run["id"])["status"] not in {"completed", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert store.get(run["id"])["status"] == "completed"
    call = store.calls(run["id"])[0]
    assert call["native_output"]["max_output_tokens"] == 20
    reconstructed = {
        **call["request"]["effective_body"],
        "max_tokens": call["native_output"]["max_output_tokens"],
    }
    assert digest(reconstructed) == call["native_output"]["effective_request_sha256"]
    assert reconstructed == native_server.requests[-1][1]
    assert call["request"]["effective_body"]["tools"] == source["cases"][0]["tools"]
    assert store.results(run["id"])[0]["correct"] is True
    preview_doc = document(native_server, mom=True)
    preview_doc["mode"] = "preview"
    preview, _ = store.create(plan(preview_doc))
    store.status(preview["id"], "completed")
    eligibility = ReplayValidator(store, store.get(run["id"])).validate(
        store.get(preview["id"])
    )
    assert eligibility["eligible"] is False
    assert "native_output_not_replayable" in {r["code"] for r in eligibility["reasons"]}


def wait_run(store, run_id):
    deadline = time.monotonic() + 3
    while store.get(run_id)["status"] not in {"completed", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    return store.get(run_id)


def test_native_preview_omits_cap_and_has_no_generation_calls(
    native_server, tmp_path, monkeypatch
):
    source = document(native_server, mom=True)
    source["mode"] = "preview"
    monkeypatch.setattr("cli.sr_bench.engine.capture_runner", recipe_provenance)
    store = Store(tmp_path)
    run = Engine(store).start(source)
    assert wait_run(store, run["id"])["status"] == "completed"
    assert len(native_server.requests) == 1
    path, body = native_server.requests[0]
    assert path == "/api/v1/routing/preview"
    assert "max_tokens" not in body
    assert body["tools"] == source["cases"][0]["tools"]
    assert store.calls(run["id"]) == []


def test_native_recipe_rejection_precedes_store_creation(
    native_server, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "cli.sr_bench.engine.capture_runner", lambda _frozen: {"recipe_snapshots": {}}
    )
    store = Store(tmp_path)
    with pytest.raises(NativeOutputError, match="captured entrypoint"):
        Engine(store).start(document(native_server, mom=True))
    assert store.list() == []
    assert native_server.requests == []


def test_native_reservation_uses_frozen_capacity_before_render(native_server, tmp_path):
    source = document(native_server)
    source["targets"][0]["prices"]["physical"] = {
        "input": 0,
        "cached_input": 0,
        "cache_write": 0,
        "output": 1_000_000,
    }
    source["limits"]["max_cost_usd"] = 23
    store = Store(tmp_path)
    run = Engine(store).start(source)
    assert wait_run(store, run["id"])["status"] == "failed"
    assert "Insufficient cost budget" in store.results(run["id"])[0]["error"]
    assert native_server.requests == []
    assert store.calls(run["id"]) == []


def test_native_auxiliary_uses_the_same_render_contract(native_server, tmp_path):
    source = document(native_server)
    source["auxiliary_targets"] = {"judge": {**source["targets"][0], "id": "judge"}}
    frozen = plan(source)
    store = Store(tmp_path)
    engine = Engine(store)
    run, _ = store.create(frozen)
    context = Context(
        engine,
        run["id"],
        frozen,
        frozen["cases"][0],
        frozen["targets"][0],
        threading.Event(),
        time.monotonic() + 3,
    )
    response = context.call(
        [{"role": "user", "content": "Grade A"}], role="judge", target="judge"
    )
    assert response["native_output"]["max_output_tokens"] == 20
    assert len(native_server.requests) == 2
    assert store.calls(run["id"])[0]["role"] == "judge"


def test_native_missing_render_capacity_is_not_an_assumed_default(native_server):
    native_server.render["sampling_params"] = {}
    source = document(native_server)
    source["targets"][0]["native_limits"]["physical"]["max_output_tokens"] = 8
    with pytest.raises(CallFailure, match="provider capacity"):
        request(plan(source))
    assert len(native_server.requests) == 1


def test_native_mom_accepts_role_only_initial_chunk_without_model(native_server):
    set_native_headers(native_server)
    native_server.role_only_initial = True
    result = request(plan(document(native_server, mom=True)))
    assert result["native_output"]["provider_model_observed"] is True
    assert result["model"] == "physical"
    assert len(native_server.requests) == 1


@pytest.mark.parametrize("mom", [False, True])
def test_native_requires_observed_response_model_not_request_default(
    native_server, mom
):
    if mom:
        set_native_headers(native_server)
    native_server.role_only_initial = True
    native_server.response_model = None
    with pytest.raises(
        CallFailure, match="response model acknowledgement missing"
    ) as failure:
        request(plan(document(native_server, mom=mom)))
    assert failure.value.partial["native_output"]["provider_model_observed"] is False
    assert len(native_server.requests) == (1 if mom else 2)
