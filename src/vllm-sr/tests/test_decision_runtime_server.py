"""Pinned Decision startup, family adaptation, and backend ownership."""

import asyncio
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_runtime.artifacts import ArtifactFile, VerifiedArtifact  # noqa: E402
from decision_runtime.backend import (  # noqa: E402
    BackendInputTooLargeError,
    BackendUnavailableError,
    ModelDescriptor,
)
from decision_runtime.contracts import SystemOneRequest  # noqa: E402
from decision_runtime.entrypoint import parse_launch_args  # noqa: E402
from decision_runtime.metrics import RuntimeMetrics  # noqa: E402
from decision_runtime.physical_batching import (  # noqa: E402
    DecisionRow,
    PhysicalBatchBackend,
    PreparedDecisionRow,
)
from decision_runtime.row_executor import TorchDecisionRowExecutor  # noqa: E402
from decision_runtime.runtime_factory import (  # noqa: E402
    RuntimeAssemblyError,
    RuntimeLaunchConfig,
    assemble_runtime,
)
from decision_runtime.runtime_profile import (  # noqa: E402
    ArtifactManifestIdentity,
    load_runtime_profile,
)

MODEL = "llm-semantic-router/Decision-1.0-Kai-0.6B"
PROFILE_ID = "Decision-1.0-Kai-0.6B"
REVISION = "7185f514f54b8f93c55998b1e8f9c5cc67f0d029"
CONTENT_ID = "a" * 64


def _observed_manifest(profile):
    return ArtifactManifestIdentity(profile.artifact.manifest_path, "d" * 64, 100)


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


def test_entrypoint_parses_exact_decision_serve_command(tmp_path: Path):
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


def test_entrypoint_rejects_removed_experimental_graph_flag(tmp_path: Path):
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
        "--max-batch",
        "8",
        "--max-concurrency",
        "1",
        "--max-queue",
        "0",
    ]
    assert parse_launch_args(args).max_batch == 8
    with pytest.raises(SystemExit):
        parse_launch_args([*args, "--experimental-qwen-rocm-graph-b8"])


def test_assembly_reopens_artifact_before_loading(monkeypatch, tmp_path: Path):
    from decision_runtime import runtime_factory  # noqa: PLC0415

    profile = load_runtime_profile(PROFILE_ID, revision=REVISION)
    model = SimpleNamespace(
        catalog=SimpleNamespace(
            model_id=MODEL, revision=REVISION, parameter_size="0.6B"
        ),
        profile=profile,
    )
    events = []
    resident = SimpleNamespace(max_length=profile.max_input_tokens, tokenizer=object())

    def resolve(model_id, *, revision, backend):
        events.append(("resolve", model_id, revision, backend))
        if revision != REVISION:
            raise runtime_factory.RuntimeModelResolutionError("unavailable revision")
        return model

    def reopen(root, resolved, *, expected_content_id):
        events.append(("verify", root, resolved, expected_content_id))
        return SimpleNamespace(data_root=root / "native")

    def load(resolved, artifact, backend, *, physical_batch_size):
        events.append(("load", resolved, artifact.data_root, backend))
        assert physical_batch_size == 4
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
    assert assembled.scheduler.max_active_rows == assembled.backend.max_pending_rows
    assert assembled.backend.models()[0].name == MODEL

    events.clear()
    with pytest.raises(RuntimeAssemblyError, match="revision"):
        assemble_runtime(_config(tmp_path, revision="b" * 40))
    assert [event[0] for event in events] == ["resolve"]


def test_assembly_rejects_eos_rocm_batch_before_artifact_reopen(
    monkeypatch, tmp_path: Path
):
    from decision_runtime import runtime_factory  # noqa: PLC0415

    eos_model = "llm-semantic-router/Decision-1.0-Eos-0.8B"
    eos_revision = "3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd"
    profile = load_runtime_profile("Decision-1.0-Eos-0.8B", revision=eos_revision)
    model = SimpleNamespace(profile=profile)
    artifact_calls = []
    monkeypatch.setattr(
        runtime_factory, "resolve_decision_runtime_model", lambda *a, **k: model
    )
    monkeypatch.setattr(
        runtime_factory,
        "open_verified_artifact",
        lambda *a, **k: artifact_calls.append((a, k)),
    )

    with pytest.raises(RuntimeAssemblyError, match="profile maximum 8"):
        assemble_runtime(
            _config(tmp_path, model=eos_model, revision=eos_revision, max_batch=9)
        )

    assert artifact_calls == []


@pytest.mark.parametrize(
    ("profile_id", "revision"),
    (
        ("Decision-1.0-Sol-2B", "0665a41108e8f0b33a9515c98311c45947b99399"),
        ("Decision-1.0-Nox-4B", "0bb833504965c0eabdb9630b7bbd385cb2fe5cd4"),
    ),
)
def test_graph_assembly_wires_events_to_the_server_metrics(
    monkeypatch, tmp_path: Path, profile_id: str, revision: str
):
    from decision_runtime import runtime_factory  # noqa: PLC0415

    model_id = f"llm-semantic-router/{profile_id}"
    profile = load_runtime_profile(profile_id, revision=revision)
    model = SimpleNamespace(
        catalog=SimpleNamespace(model_id=model_id),
        profile=profile,
    )
    artifact = SimpleNamespace(
        data_root=tmp_path,
        revision=revision,
        content_id=CONTENT_ID,
        manifest=_observed_manifest(profile),
    )
    resident = SimpleNamespace(max_length=profile.max_input_tokens, tokenizer=object())

    def load(_model, _artifact, _backend, **options):
        options["graph_event_recorder"]("replay")
        return resident

    monkeypatch.setattr(
        runtime_factory, "resolve_decision_runtime_model", lambda *a, **k: model
    )
    monkeypatch.setattr(
        runtime_factory, "open_verified_artifact", lambda *a, **k: artifact
    )
    monkeypatch.setattr(runtime_factory, "_device_target", lambda backend: "gfx942")
    monkeypatch.setattr(
        runtime_factory, "require_runtime_backend", lambda *a, **k: None
    )
    monkeypatch.setattr(runtime_factory, "_load_family", load)
    metrics = RuntimeMetrics()
    assemble_runtime(
        _config(
            tmp_path,
            model=model_id,
            revision=revision,
            max_batch=8,
        ),
        metrics=metrics,
    )
    assert (
        f'decision_runtime_qwen_rocm_graph_events_total{{model="{model_id}",'
        'event="replay"} 1'
    ) in metrics.render(())


@pytest.mark.parametrize(
    ("profile_id", "revision", "expected_manifest"),
    [
        (
            "Decision-1.0-Eos-0.8B",
            "3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd",
            True,
        ),
        (
            "Decision-1.0-Sol-2B",
            "0665a41108e8f0b33a9515c98311c45947b99399",
            False,
        ),
    ],
)
def test_qwen_loader_uses_verified_manifest_layout(
    monkeypatch,
    tmp_path: Path,
    profile_id: str,
    revision: str,
    expected_manifest: bool,
):
    from decision_runtime import qwen35_torch, runtime_factory  # noqa: PLC0415

    profile = load_runtime_profile(profile_id, revision=revision)
    model = SimpleNamespace(profile=profile)
    calibration_path = "temperature.json" if expected_manifest else "runtime.json"
    (tmp_path / calibration_path).write_text(
        json.dumps({"temperature": profile.temperature}), encoding="utf-8"
    )
    artifact = SimpleNamespace(
        data_root=tmp_path,
        files=(ArtifactFile(calibration_path, calibration_path, "b" * 64, 1),),
        manifest=_observed_manifest(profile),
    )
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
        artifact.manifest.sha256 if expected_manifest else None
    )
    # Native Eos is a ROCm-specific execution policy. CPU always uses the
    # reference kernel inside the family loader, regardless of this option.
    assert calls[0][1]["gated_delta_kernel_policy"] == "accelerated"
    assert calls[0][1]["native_rocm_max_physical_batch_size"] is None


def test_eos_rocm_execution_policy_reaches_family_loader(monkeypatch, tmp_path: Path):
    from decision_runtime import (  # noqa: PLC0415
        qwen35_rocm_binder,
        qwen35_torch,
        runtime_factory,
    )

    profile = load_runtime_profile(
        "Decision-1.0-Eos-0.8B",
        revision="3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd",
    )
    (tmp_path / "temperature.json").write_text(
        json.dumps({"temperature": profile.temperature}), encoding="utf-8"
    )
    artifact = SimpleNamespace(
        data_root=tmp_path,
        files=(ArtifactFile("temperature.json", "temperature.json", "b" * 64, 1),),
        manifest=_observed_manifest(profile),
        content_id="c" * 64,
    )
    binder = object()
    options = {}
    sentinel = object()
    monkeypatch.setattr(
        qwen35_rocm_binder,
        "create_qwen_rocm_profile_binder",
        lambda: binder,
    )

    def load(_root, **kwargs):
        options.update(kwargs)
        return sentinel

    monkeypatch.setattr(qwen35_torch.Qwen35TorchRuntime, "load", load)
    assert (
        runtime_factory._load_family(SimpleNamespace(profile=profile), artifact, "rocm")
        is sentinel
    )
    assert options["gated_delta_kernel_policy"] == "native_torch"
    assert options["native_rocm_max_physical_batch_size"] == 8
    assert options["rocm_profile_binder"] is binder
    assert "enable_rocm_graph" not in options


@pytest.mark.parametrize(
    ("profile_id", "revision"),
    (
        ("Decision-1.0-Sol-2B", "0665a41108e8f0b33a9515c98311c45947b99399"),
        ("Decision-1.0-Nox-4B", "0bb833504965c0eabdb9630b7bbd385cb2fe5cd4"),
    ),
)
def test_profiled_qwen_graph_receives_verified_artifact_identity(
    monkeypatch, tmp_path: Path, profile_id: str, revision: str
):
    from decision_runtime import qwen35_torch, runtime_factory  # noqa: PLC0415

    profile = load_runtime_profile(profile_id, revision=revision)
    model = SimpleNamespace(
        profile=profile,
        catalog=SimpleNamespace(model_id=f"llm-semantic-router/{profile_id}"),
    )
    artifact = SimpleNamespace(
        data_root=tmp_path,
        content_id=CONTENT_ID,
        manifest=_observed_manifest(profile),
    )
    calls = []
    monkeypatch.setattr(runtime_factory, "_qwen_temperature", lambda *a, **k: 1.0)
    monkeypatch.setattr(
        qwen35_torch.Qwen35TorchRuntime,
        "load",
        lambda root, **options: calls.append((root, options)),
    )

    def recorder(_event):
        return None

    runtime_factory._load_family(
        model,
        artifact,
        "rocm",
        physical_batch_size=8,
        graph_event_recorder=recorder,
    )
    assert calls[0][1]["enable_rocm_graph"] is True
    assert calls[0][1]["artifact_content_id"] == CONTENT_ID
    assert calls[0][1]["physical_batch_size"] == 8
    assert calls[0][1]["graph_event_recorder"] is recorder
    assert calls[0][1]["graph_prewarm_padded_tokens"] == (
        (128,) if profile_id.endswith("Nox-4B") else ()
    )


@pytest.mark.parametrize(
    ("profile_id", "revision", "backend", "batch"),
    (
        ("Decision-1.0-Nox-4B", "0bb833504965c0eabdb9630b7bbd385cb2fe5cd4", "rocm", 16),
        ("Decision-1.0-Lux-9B", "bd45a30aee8c84032791c245c70f86dee5389cc8", "rocm", 8),
        ("Decision-1.0-Sol-2B", "0665a41108e8f0b33a9515c98311c45947b99399", "rocm", 16),
        ("Decision-1.0-Sol-2B", "0665a41108e8f0b33a9515c98311c45947b99399", "cpu", 8),
    ),
)
def test_non_profiled_qwen_path_does_not_enable_graph(
    profile_id: str,
    revision: str,
    backend: str,
    batch: int,
    tmp_path: Path,
    monkeypatch,
):
    from decision_runtime import qwen35_torch, runtime_factory  # noqa: PLC0415

    model = SimpleNamespace(
        catalog=SimpleNamespace(model_id=f"llm-semantic-router/{profile_id}"),
        profile=load_runtime_profile(profile_id, revision=revision),
    )
    artifact = SimpleNamespace(
        data_root=tmp_path,
        content_id=CONTENT_ID,
        manifest=SimpleNamespace(path="bundle-manifest.json", sha256="a" * 64),
    )
    loads = []
    monkeypatch.setattr(runtime_factory, "_qwen_temperature", lambda *a, **k: 1.0)
    monkeypatch.setattr(
        qwen35_torch.Qwen35TorchRuntime,
        "load",
        lambda *args, **kwargs: loads.append(kwargs),
    )
    runtime_factory._load_family(model, artifact, backend, physical_batch_size=batch)
    assert len(loads) == 1
    assert "enable_rocm_graph" not in loads[0]
    assert "artifact_content_id" not in loads[0]


def test_qwen_calibration_is_read_from_verified_snapshot(tmp_path: Path):
    from decision_runtime.runtime_factory import _qwen_temperature  # noqa: PLC0415

    (tmp_path / "temperature.json").write_text(
        json.dumps({"temperature": 1.75}), encoding="utf-8"
    )
    artifact = VerifiedArtifact(
        root=tmp_path,
        data_root=tmp_path,
        content_id="a" * 64,
        repository_id="llm-semantic-router/Decision-1.0-Sol-2B",
        revision="c" * 40,
        files=(ArtifactFile("temperature.json", "temperature.json", "b" * 64, 1),),
        manifest=ArtifactManifestIdentity("bundle-manifest.json", "d" * 64, 10),
    )
    assert _qwen_temperature(artifact, fallback=9.0) == 1.75
    (tmp_path / "temperature.json").write_text(
        json.dumps({"temperature": -1}), encoding="utf-8"
    )
    with pytest.raises(RuntimeAssemblyError, match="temperature"):
        _qwen_temperature(artifact, fallback=9.0)


def test_vela_loader_receives_native_data_root_and_manifest_digest(
    monkeypatch, tmp_path: Path
):
    from decision_runtime import runtime_factory, vela_torch  # noqa: PLC0415

    profile = load_runtime_profile(PROFILE_ID, revision=REVISION)
    model = SimpleNamespace(profile=profile)
    artifact = SimpleNamespace(
        data_root=tmp_path / "native", manifest=_observed_manifest(profile)
    )
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
        "expected_manifest_sha256": artifact.manifest.sha256,
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
    profile = load_runtime_profile(PROFILE_ID, revision=REVISION)
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
    profile = load_runtime_profile(PROFILE_ID, revision=REVISION)
    resident = _RecordingVela(profile.max_input_tokens)
    executor = TorchDecisionRowExecutor(resident, profile)
    request = _request("x" * (profile.max_input_tokens + 1))
    row = DecisionRow(MODEL, request.state, "yes", request.questions["yes"])
    with pytest.raises(BackendInputTooLargeError):
        asyncio.run(executor.prepare_rows((row,)))
    assert resident.calls == []


def test_model_forward_is_not_queued_behind_default_pool_preparation():
    profile = load_runtime_profile(
        "Decision-1.0-Eos-0.8B",
        revision="3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd",
    )
    resident = _RecordingQwen(profile.max_input_tokens)
    executor = TorchDecisionRowExecutor(resident, profile)
    request = _request("A billing question.")
    row = DecisionRow(MODEL, request.state, "yes", request.questions["yes"])
    prepared = (
        PreparedDecisionRow(
            row=row,
            batch_key="qwen3.5",
            payload=executor._encode_rows((row,))[0],
        ),
    )
    started = threading.Event()
    release = threading.Event()

    def occupy_default_pool():
        started.set()
        assert release.wait(timeout=5)

    async def scenario():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as default_pool:
            loop.set_default_executor(default_pool)
            blocker = loop.run_in_executor(None, occupy_default_pool)
            try:
                assert started.wait(timeout=1)
                result = await asyncio.wait_for(executor.predict_rows(prepared), 1)
                assert result[0].question_id == "yes"
                assert len(resident.calls) == 1
            finally:
                release.set()
                await blocker
                backend = PhysicalBatchBackend(
                    ModelDescriptor(MODEL, "test", "unknown"), executor
                )
                await backend.aclose()
            assert not await executor.ready()
            with pytest.raises(BackendUnavailableError):
                await executor.predict_rows(prepared)

    asyncio.run(scenario())


def test_backend_shutdown_waits_for_active_inference_thread():
    profile = load_runtime_profile(
        "Decision-1.0-Eos-0.8B",
        revision="3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd",
    )
    resident = _RecordingQwen(profile.max_input_tokens)
    original_predict = resident.predict_encoded
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_predict(rows):
        started.set()
        try:
            assert release.wait(timeout=5)
            return original_predict(rows)
        finally:
            finished.set()

    resident.predict_encoded = blocking_predict
    executor = TorchDecisionRowExecutor(resident, profile)
    backend = PhysicalBatchBackend(
        ModelDescriptor(MODEL, "test", "unknown"),
        executor,
        physical_batch_size=1,
    )
    request = SystemOneRequest.model_validate(
        {
            "model": MODEL,
            "state": "A billing question.",
            "questions": {"yes": {"type": "noul", "instructions": "Is this urgent?"}},
        }
    )

    async def scenario():
        caller = asyncio.create_task(backend.infer(request))
        try:
            assert await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=2)
            closing = asyncio.create_task(backend.aclose())
            await asyncio.sleep(0)
            assert not closing.done()
        finally:
            release.set()
        await closing
        with pytest.raises(BackendUnavailableError):
            await caller
        assert finished.is_set()
        assert not await executor.ready()

    asyncio.run(scenario())


def test_cancelled_row_preparation_waits_for_tokenizer_before_releasing_credit():
    from decision_runtime.scheduler import ModelScheduler  # noqa: PLC0415

    profile = load_runtime_profile(PROFILE_ID, revision=REVISION)
    executor = TorchDecisionRowExecutor(
        _RecordingVela(profile.max_input_tokens), profile
    )
    request = _request("A billing question.")
    row = DecisionRow(MODEL, request.state, "yes", request.questions["yes"])
    original_encode = executor._encode_rows
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_encode(rows):
        started.set()
        try:
            assert release.wait(timeout=5)
            return original_encode(rows)
        finally:
            finished.set()

    executor._encode_rows = blocking_encode

    async def scenario():
        scheduler = ModelScheduler(
            [MODEL], max_concurrency=1, max_queue=0, max_active_rows=1
        )
        task = asyncio.create_task(
            scheduler.run(MODEL, lambda: executor.prepare_rows((row,)), row_cost=1)
        )
        try:
            assert await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            snapshot = (await scheduler.snapshots())[0]
            assert (snapshot.running, snapshot.active_rows) == (1, 1)
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        snapshot = (await scheduler.snapshots())[0]
        assert (snapshot.running, snapshot.active_rows) == (0, 0)

    asyncio.run(scenario())


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


class _RecordingQwen:
    def __init__(self, max_length: int) -> None:
        self.max_length = max_length
        self.tokenizer = _QwenTokenizer()
        self.calls = []

    def predict_encoded(self, rows):
        self.calls.append(rows)
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


def test_qwen_executor_can_score_mixed_types_in_one_forward():
    profile = load_runtime_profile(
        "Decision-1.0-Eos-0.8B",
        revision="3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd",
    )
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
    from decision_runtime import server  # noqa: PLC0415

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
        lambda config, *, metrics: SimpleNamespace(
            backend=backend, scheduler=scheduler
        ),
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
    from decision_runtime import server  # noqa: PLC0415

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
