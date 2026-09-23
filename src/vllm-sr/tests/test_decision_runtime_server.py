"""Pinned Decision startup, family adaptation, and backend ownership."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_runtime.backend import BackendInputTooLargeError  # noqa: E402
from decision_runtime.contracts import SystemOneRequest  # noqa: E402
from decision_runtime.entrypoint import parse_launch_args  # noqa: E402
from decision_runtime.physical_batching import DecisionRow  # noqa: E402
from decision_runtime.row_executor import TorchDecisionRowExecutor  # noqa: E402
from decision_runtime.runtime_factory import (  # noqa: E402
    RuntimeAssemblyError,
    RuntimeLaunchConfig,
    assemble_runtime,
)
from decision_runtime.runtime_profile import load_runtime_profile  # noqa: E402

MODEL = "llm-semantic-router/Decision-1.0-Kai-0.6B"
REVISION = "7185f514f54b8f93c55998b1e8f9c5cc67f0d029"
CONTENT_ID = "a" * 64


def _config(root: Path, **changes) -> RuntimeLaunchConfig:
    values = {
        "model": MODEL,
        "revision": REVISION,
        "backend": "rocm",
        "artifact_root": root,
        "artifact_content_id": CONTENT_ID,
        "host": "0.0.0.0",
        "port": 8000,
        "max_batch": 4,
        "max_concurrency": 3,
        "max_queue": 2,
    }
    values.update(changes)
    return RuntimeLaunchConfig(**values)


def _request(state: str) -> SystemOneRequest:
    return SystemOneRequest.model_validate(
        {
            "model": MODEL,
            "state": state,
            "questions": {
                "yes": {"type": "noul", "instructions": "Is this urgent?"},
                "owner": {
                    "type": "choice",
                    "instructions": "Choose owner.",
                    "criteria": {"platform": None, "billing": "Billing team"},
                },
                "rating": {
                    "type": "score",
                    "instructions": "Rate impact.",
                    "criteria": ["low", "medium", "high"],
                },
            },
        }
    )


def test_entrypoint_parses_exact_drun_command(tmp_path: Path):
    args = [
        "--model",
        MODEL,
        "--revision",
        REVISION,
        "--backend",
        "rocm",
        "--artifact-root",
        str(tmp_path),
        "--artifact-content-id",
        CONTENT_ID,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--max-batch",
        "4",
        "--max-concurrency",
        "3",
        "--max-queue",
        "2",
    ]
    parsed = parse_launch_args(args)
    assert parsed == _config(tmp_path)
    with pytest.raises(SystemExit):
        parse_launch_args([*args, "--unexpected"])
    with pytest.raises(SystemExit):
        parse_launch_args([*args[:-1], "-1"])


def test_assembly_reopens_artifact_before_loading(monkeypatch, tmp_path: Path):
    from decision_runtime import runtime_factory

    profile = load_runtime_profile(REVISION)
    model = SimpleNamespace(
        catalog=SimpleNamespace(model_id=MODEL, revision=REVISION),
        profile=profile,
    )
    events = []
    resident = SimpleNamespace(max_length=profile.max_input_tokens, tokenizer=object())

    def resolve(model_id, *, backend):
        events.append(("resolve", model_id, backend))
        return model

    def reopen(root, resolved, *, expected_content_id):
        events.append(("verify", root, resolved, expected_content_id))
        return SimpleNamespace(data_root=root / "native")

    def load(resolved, artifact, backend):
        events.append(("load", resolved, artifact.data_root, backend))
        return resident

    monkeypatch.setattr(runtime_factory, "resolve_decision_runtime_model", resolve)
    monkeypatch.setattr(runtime_factory, "open_verified_artifact", reopen)
    monkeypatch.setattr(runtime_factory, "_device_target", lambda backend: "gfx942")
    monkeypatch.setattr(runtime_factory, "_load_family", load)
    assembled = assemble_runtime(_config(tmp_path))
    assert [event[0] for event in events] == ["resolve", "verify", "load"]
    assert events[1][1:] == (tmp_path, model, CONTENT_ID)
    assert events[2][2:] == (tmp_path / "native", "rocm")
    assert assembled.backend.physical_batch_size == 4
    assert assembled.scheduler.max_concurrency == 3
    assert assembled.scheduler.max_queue == 2
    assert assembled.backend.models()[0].name == MODEL

    events.clear()
    with pytest.raises(RuntimeAssemblyError, match="revision"):
        assemble_runtime(_config(tmp_path, revision="b" * 40))
    assert [event[0] for event in events] == ["resolve"]


@pytest.mark.parametrize(
    ("revision", "expected_manifest"),
    [
        ("3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd", True),
        ("0665a41108e8f0b33a9515c98311c45947b99399", False),
    ],
)
def test_qwen_loader_uses_verified_manifest_layout(
    monkeypatch, tmp_path: Path, revision: str, expected_manifest: bool
):
    from decision_runtime import qwen35_torch, runtime_factory

    profile = load_runtime_profile(revision)
    model = SimpleNamespace(profile=profile)
    artifact = SimpleNamespace(data_root=tmp_path)
    calls = []
    sentinel = object()

    def load(root, **options):
        calls.append((root, options))
        return sentinel

    monkeypatch.setattr(qwen35_torch.Qwen35TorchRuntime, "load", load)
    assert runtime_factory._load_family(model, artifact, "cpu") is sentinel
    assert calls[0][0] == tmp_path
    assert calls[0][1]["temperature"] == profile.temperature
    assert calls[0][1]["max_length"] == profile.max_input_tokens
    assert calls[0][1]["expected_manifest_sha256"] == (
        profile.artifact.manifest.sha256 if expected_manifest else None
    )


def test_vela_loader_receives_native_data_root_and_manifest_digest(
    monkeypatch, tmp_path: Path
):
    from decision_runtime import runtime_factory, vela_torch

    profile = load_runtime_profile(REVISION)
    model = SimpleNamespace(profile=profile)
    artifact = SimpleNamespace(data_root=tmp_path / "native")
    calls = []

    def load(root, **options):
        calls.append((root, options))
        return object()

    monkeypatch.setattr(vela_torch.VelaTorchRuntime, "load", load)
    runtime_factory._load_family(model, artifact, "cpu")
    assert calls[0][0] == tmp_path / "native"
    assert calls[0][1] == {
        "max_length": profile.max_input_tokens,
        "backend": "cpu",
        "expected_manifest_sha256": profile.artifact.manifest.sha256,
    }


class _VelaTokenizer:
    cls_token_id = 1
    bos_token_id = 1
    sep_token_id = 2
    eos_token_id = 2
    pad_token_id = 0
    mask_token_id = 3

    def __call__(self, text, *, add_special_tokens, truncation):
        assert not add_special_tokens and not truncation
        return {"input_ids": [ord(char) % 100 + 4 for char in text]}


class _RecordingVela:
    def __init__(self, max_length: int) -> None:
        self.max_length = max_length
        self.tokenizer = _VelaTokenizer()
        self.calls = []

    def predict_encoded(self, rows):
        self.calls.append(rows)
        assert len({row.type for row in rows}) == 1
        return tuple(
            SimpleNamespace(
                question_id=row.question_id,
                type=row.type,
                probabilities=tuple(
                    [1.0 / len(row.marker_positions)] * len(row.marker_positions)
                ),
                input_tokens=row.input_tokens,
            )
            for row in rows
        )


def test_vela_executor_preserves_question_order_and_type_batch_keys():
    profile = load_runtime_profile(REVISION)
    resident = _RecordingVela(profile.max_input_tokens)
    executor = TorchDecisionRowExecutor(resident, profile)
    request = _request("An outage was reported.")
    rows = tuple(
        DecisionRow(MODEL, request.state, question_id, question)
        for question_id, question in request.questions.items()
    )

    async def scenario():
        prepared = await executor.prepare_rows(rows)
        assert [item.batch_key for item in prepared] == [
            "vela:noul",
            "vela:choice",
            "vela:score",
        ]
        return tuple([await executor.predict_rows((item,)) for item in prepared])

    predictions = asyncio.run(scenario())
    assert [item[0].question_id for item in predictions] == list(request.questions)
    assert [len(item[0].probabilities) for item in predictions] == [2, 2, 3]
    assert len(resident.calls) == 3


def test_vela_executor_rejects_complete_oversized_row():
    profile = load_runtime_profile(REVISION)
    resident = _RecordingVela(profile.max_input_tokens)
    executor = TorchDecisionRowExecutor(resident, profile)
    request = _request("x" * (profile.max_input_tokens + 1))
    row = DecisionRow(MODEL, request.state, "yes", request.questions["yes"])
    with pytest.raises(BackendInputTooLargeError):
        asyncio.run(executor.prepare_rows((row,)))
    assert resident.calls == []


class _QwenTokenizer:
    def __init__(self) -> None:
        self.segments = []

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        self.segments.append(text)
        return [ord(char) % 100 + 1 for char in text]

    def __call__(self, texts, **options):
        assert options["truncation"] is False
        self.segments.extend(texts)
        return {"input_ids": [[ord(char) % 100 + 1 for char in text] for text in texts]}


def test_qwen_executor_can_score_mixed_types_in_one_forward():
    profile = load_runtime_profile("3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd")
    calls = []
    tokenizer = _QwenTokenizer()

    def predict(rows):
        calls.append(rows)
        return tuple(
            SimpleNamespace(
                question_id=row.question_id,
                type=row.type,
                probabilities=(1.0 / len(row.candidate_positions),)
                * len(row.candidate_positions),
                input_tokens=row.input_tokens,
            )
            for row in rows
        )

    resident = SimpleNamespace(
        max_length=profile.max_input_tokens,
        tokenizer=tokenizer,
        predict_encoded=predict,
    )
    executor = TorchDecisionRowExecutor(resident, profile)
    request = _request("An outage was reported.")
    rows = tuple(
        DecisionRow(MODEL, request.state, question_id, question)
        for question_id, question in request.questions.items()
    )

    async def scenario():
        prepared = await executor.prepare_rows(rows)
        assert {item.batch_key for item in prepared} == {"qwen3.5"}
        return await executor.predict_rows(prepared)

    results = asyncio.run(scenario())
    assert [result.question_id for result in results] == list(request.questions)
    assert [len(result.probabilities) for result in results] == [2, 2, 3]
    assert len(calls) == 1
    assert any('"description":null' in segment for segment in tokenizer.segments)


def test_server_lifespan_closes_physical_backend(monkeypatch, tmp_path: Path):
    from decision_runtime import server

    class Backend:
        def __init__(self):
            self.closed = False

        def models(self):
            return (
                SimpleNamespace(
                    name=MODEL, description="model", release_date="unknown"
                ),
            )

        async def ready(self):
            return not self.closed

        async def aclose(self):
            self.closed = True

    backend = Backend()
    scheduler = SimpleNamespace()
    monkeypatch.setattr(
        server,
        "assemble_runtime",
        lambda config: SimpleNamespace(backend=backend, scheduler=scheduler),
    )
    app = server.create_runtime_app(_config(tmp_path))

    async def scenario():
        async with app.router.lifespan_context(app):
            assert await backend.ready()
        assert backend.closed
        assert app.state.decision_backend_closed

    asyncio.run(scenario())


def test_server_closes_backend_when_uvicorn_fails_before_lifespan(
    monkeypatch, tmp_path: Path
):
    from decision_runtime import server

    closed = []

    class Backend:
        async def aclose(self):
            closed.append(True)

    app = SimpleNamespace(
        state=SimpleNamespace(decision_backend=Backend(), decision_backend_closed=False)
    )
    monkeypatch.setattr(server, "create_runtime_app", lambda config: app)

    def fail(*args, **kwargs):
        raise RuntimeError("server startup failed")

    monkeypatch.setattr("uvicorn.run", fail)
    with pytest.raises(RuntimeError, match="server startup failed"):
        server.run_server(_config(tmp_path))
    assert closed == [True]
