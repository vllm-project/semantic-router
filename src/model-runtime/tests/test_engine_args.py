"""Engine arguments: how a served model's arguments reach its batcher, its engine and its card."""

from __future__ import annotations

import asyncio

import pytest
from starlette.testclient import TestClient
from vllm_srun import dtypes
from vllm_srun.accel import onednn
from vllm_srun.api.app import create_app
from vllm_srun.config import DEFAULT_MAX_NUM_SEQS, EngineArgs, ModelConfig, ServeConfig
from vllm_srun.engines.native.engine import NativeEngine
from vllm_srun.errors import VerificationError
from vllm_srun.families.task_heads import family as task_heads
from vllm_srun.plugins.base import BackboneSpec, DtypePolicy, ModelSpec
from vllm_srun.runtime import Runtime
from vllm_srun.testing.fixtures import write_fixture

from .conftest import QUESTIONS, STATE

DECISION2_GPU = DtypePolicy(approximate_kernels=True)
DECISION3 = DtypePolicy(
    autocast=None, bf16_resident=False, gpu_weights="bfloat16", cpu_weights="bfloat16"
)
TASK_HEADS = DtypePolicy(
    weights="float32", autocast=None, head="float32", bf16_resident=False
)


def spec(policy, variants=None):
    return ModelSpec(
        "tiny",
        BackboneSpec("qwen3", {}, ()),
        policy,
        max_input_tokens=512,
        kernel_variants=variants or {},
    )


@pytest.mark.parametrize(
    "policy, gpu, published",
    [
        (DECISION2_GPU, True, "bfloat16"),
        (DECISION2_GPU, False, "float32"),
        (DECISION3, True, "bfloat16"),
        (DECISION3, False, "bfloat16"),
        (TASK_HEADS, True, "float32"),
        (TASK_HEADS, False, "float32"),
    ],
)
def test_auto_and_the_published_precision_keep_the_familys_policy(
    policy, gpu, published
):
    assert dtypes.published(policy, gpu) == published
    for dtype in ("auto", published):
        assert dtypes.resolve(dtype, policy, gpu) == "auto"
        kept = spec(policy, {"linear": onednn.PACKED})
        assert dtypes.apply(kept, dtype, gpu) is kept


def test_another_dtype_holds_every_weight_in_it_with_the_reference_kernels():
    consented = DtypePolicy(
        autocast=None,
        bf16_resident=False,
        reduced_cpu="float32-packed",
        approximate_kernels=True,
    )
    applied = dtypes.apply(
        spec(consented, {"linear": onednn.PACKED}), "bfloat16", False
    )
    assert applied.dtype == DtypePolicy(
        autocast=None,
        bf16_resident=False,
        gpu_weights="bfloat16",
        cpu_weights="bfloat16",
    )
    assert applied.kernel_variants == {}
    assert dtypes.apply(spec(DECISION3), "float32", True).dtype == DtypePolicy(
        autocast=None, bf16_resident=False
    )


def serve(package, **args):
    runtime = Runtime(
        ServeConfig(
            models=(ModelConfig(model=str(package), device="cpu", **args),),
            result_cache_entries=0,
        )
    )
    runtime.start(background=False)
    return runtime


def recorded_loads(monkeypatch):
    loads = []
    read = NativeEngine.read

    def recording(self, spec, accelerator, device, options):
        loads.append((spec, options))
        return read(self, spec, accelerator, device, options)

    monkeypatch.setattr(NativeEngine, "read", recording)
    return loads


def test_the_published_precision_runs_the_consented_kernels_and_copies(
    qwen3_package, monkeypatch
):
    loads = recorded_loads(monkeypatch)
    runtime = serve(qwen3_package)
    try:
        ((loaded, options),) = loads
        assert not options.exact_kernels_only and options.reduced_precision
        assert loaded.dtype == DECISION2_GPU
        served = runtime.served[0]
        assert served.dtype == "auto" and served.precision == "float32"
        assert served.health.golden.detail is None
    finally:
        runtime.stop()


def test_another_dtype_is_served_and_named_without_golden_references(
    qwen3_package, monkeypatch
):
    loads = recorded_loads(monkeypatch)
    runtime = serve(qwen3_package, dtype="bfloat16")
    try:
        ((loaded, options),) = loads
        assert options.exact_kernels_only and not options.reduced_precision
        assert (
            loaded.dtype.cpu_weights == "bfloat16"
            and not loaded.dtype.approximate_kernels
        )
        served = runtime.served[0]
        card = served.card([])
        assert (
            card["dtype"] == "bfloat16" and card["engine_args"]["dtype"] == "bfloat16"
        )
        assert card["numerics"] == "approximate"
        golden = served.health.golden
        assert golden.status == "unverified"
        assert (
            golden.detail
            == "golden references are recorded at --dtype auto, not bfloat16"
        )
        status, body = asyncio.run(
            runtime.call(
                "decisions",
                {
                    "state": STATE,
                    "questions": QUESTIONS,
                    "options": {"return_meta": True},
                },
            )
        )
        assert status == 200 and body["meta"]["dtype"] == "bfloat16"
        assert list(body["answers"]) == list(QUESTIONS)
    finally:
        runtime.stop()


@pytest.mark.parametrize("override", ["none", "model", "process"])
def test_placement_estimates_the_dtype_a_model_or_the_process_asks_for(
    qwen3_package, monkeypatch, override
):
    import vllm_srun.runtime as runtime_module
    from vllm_srun.accel.cuda import CUDAAccelerator
    from vllm_srun.accel.rocm import ROCmAccelerator
    from vllm_srun.errors import PlacementError
    from vllm_srun.placement import ACTIVATION_RESERVE, place
    from vllm_srun.plugins.base import DeviceInfo

    gpu = DeviceInfo("rocm", 0, "AMD Instinct MI300X", 192 << 30, 180 << 30, True)
    monkeypatch.setattr(ROCmAccelerator, "available", lambda self: True)
    monkeypatch.setattr(ROCmAccelerator, "devices", lambda self: [gpu])
    monkeypatch.setattr(CUDAAccelerator, "available", lambda self: False)
    placed = []

    def placing(spec, device, parameters, *args):
        placed.append((parameters, place(spec, device, parameters, *args)))
        raise PlacementError("placed")  # this host has no GPU to load the weights on

    monkeypatch.setattr(runtime_module, "place", placing)
    model = ModelConfig(
        model=str(qwen3_package),
        device="rocm:0",
        dtype="float32" if override == "model" else None,
    )
    process = EngineArgs(dtype="float32" if override == "process" else "auto")
    runtime = Runtime(ServeConfig(models=(model,), engine_args=process))
    with pytest.raises(PlacementError, match="placed"):
        runtime.load()
    ((parameters, placement),) = placed
    fp32 = 4 * parameters + ACTIVATION_RESERVE
    if override == "none":
        assert placement.estimated_bytes < fp32
    else:
        assert placement.estimated_bytes == fp32


def test_float16_needs_a_gpu(qwen3_package):
    runtime = Runtime(
        ServeConfig(
            models=(
                ModelConfig(model=str(qwen3_package), device="cpu", dtype="float16"),
            )
        )
    )
    with pytest.raises(
        RuntimeError, match="float16 runs on GPUs; use bfloat16 or float32 on a CPU"
    ):
        runtime.start(background=False)
    runtime.stop()


def test_a_models_own_engine_arguments_reach_its_batcher_and_scheduler(qwen3_package):
    runtime = Runtime(
        ServeConfig(
            models=(
                ModelConfig(
                    model=str(qwen3_package),
                    name="own",
                    device="cpu",
                    max_num_seqs=3,
                    batch_window_ms=0.0,
                ),
                ModelConfig(model=str(qwen3_package), name="process", device="cpu"),
            ),
            engine_args=EngineArgs(max_num_batched_tokens=1024, batch_window_ms=5.0),
        )
    )
    runtime.start(background=False)
    try:
        own, process = runtime.served
        assert own.scheduler.batcher.max_num_seqs == 3
        assert own.scheduler.batcher.max_num_batched_tokens == 1024
        assert own.scheduler.limits.batch_window_ms == 0.0
        assert process.scheduler.batcher.max_num_seqs == DEFAULT_MAX_NUM_SEQS
        assert process.scheduler.limits.batch_window_ms == 5.0
        assert own.card([])["engine_args"] == {
            "max_num_seqs": 3,
            "max_num_batched_tokens": 1024,
            "batch_window_ms": 0.0,
            "dtype": "auto",
            "enable_shared_context": True,
            "max_model_len": None,
        }
    finally:
        runtime.stop()


def test_shared_context_defaults_on_for_decision2_and_can_be_switched_off(
    qwen3_package,
):
    runtime = serve(qwen3_package, enable_shared_context=False)
    try:
        served = runtime.served[0]
        assert not served.shared and served.scheduler.batcher.shared is None
    finally:
        runtime.stop()
    runtime = serve(qwen3_package)
    try:
        assert runtime.served[0].shared
    finally:
        runtime.stop()


def test_task_heads_keep_shared_context_off_and_report_exact_numerics_where_batch_invariant(
    tmp_path,
):
    package = write_fixture(tmp_path / "heads", family="task_heads")
    runtime = serve(package)
    try:
        served = runtime.served[0]
        assert not served.shared
        if served.model.batch_invariant:
            assert served.numerics == "exact"
        else:
            # no packed linear on this host, or the load-time probes found the
            # packed products or the whole forward moving with the batch
            assert served.numerics == "approximate"
    finally:
        runtime.stop()


def test_a_task_model_is_exact_where_its_packed_linears_and_forward_are_invariant(
    tmp_path, monkeypatch
):
    if not onednn.available():
        pytest.skip("oneDNN's packed linear needs an x86 CPU")
    monkeypatch.setattr(onednn, "invariant", lambda layers: True)
    monkeypatch.setattr(task_heads, "batch_invariant", lambda model, vocab: True)
    package = write_fixture(tmp_path / "heads", family="task_heads")
    runtime = serve(package)
    try:
        served = runtime.served[0]
        assert served.model.engine_model.batch_invariant
        assert served.model.batch_invariant
        assert served.numerics == "exact"
    finally:
        runtime.stop()


def test_a_task_model_is_approximate_where_its_packed_linears_are_not_invariant(
    tmp_path, monkeypatch
):
    if not onednn.available():
        pytest.skip("oneDNN's packed linear needs an x86 CPU")
    monkeypatch.setattr(onednn, "invariant", lambda layers: False)
    package = write_fixture(tmp_path / "heads", family="task_heads")
    runtime = serve(package)
    try:
        served = runtime.served[0]
        assert not served.model.engine_model.batch_invariant
        assert not served.model.batch_invariant
        assert served.numerics == "approximate"
    finally:
        runtime.stop()


def test_max_model_len_lowers_a_decision_models_input_limit(qwen3_package):
    runtime = serve(qwen3_package, max_model_len=1000)
    try:
        card = runtime.served[0].card([])
        assert card["limits"]["max_input_tokens"] == 1000
        assert card["engine_args"]["max_model_len"] == 1000
    finally:
        runtime.stop()


def test_without_max_model_len_the_package_limit_stands(qwen3_package):
    runtime = serve(qwen3_package)
    try:
        card = runtime.served[0].card([])
        assert card["limits"]["max_input_tokens"] == 2048
        assert card["engine_args"]["max_model_len"] is None
    finally:
        runtime.stop()


def refused(package, max_model_len):
    runtime = Runtime(
        ServeConfig(
            models=(
                ModelConfig(
                    model=str(package), device="cpu", max_model_len=max_model_len
                ),
            )
        )
    )
    try:
        with pytest.raises(RuntimeError) as error:
            runtime.start(background=False)
    finally:
        runtime.stop()
    return str(error.value)


def test_max_model_len_may_equal_but_not_exceed_the_package_limit(qwen3_package):
    runtime = serve(qwen3_package, max_model_len=2048)
    try:
        assert runtime.served[0].card([])["limits"]["max_input_tokens"] == 2048
    finally:
        runtime.stop()
    assert refused(qwen3_package, 2049) == (
        "max_model_len 2049 is above Decision-2.0-Tiny-Qwen3's limit of 2048 tokens"
    )


def test_a_failed_golden_check_names_a_max_model_len_below_its_questions(
    qwen3_package,
):
    runtime = Runtime(
        ServeConfig(
            models=(
                ModelConfig(model=str(qwen3_package), device="cpu", max_model_len=128),
            )
        )
    )
    try:
        with pytest.raises(
            VerificationError,
            match="golden answers are malformed; max_model_len 128 may be below "
            "the length of the model's golden questions",
        ):
            runtime.start(background=False)
    finally:
        runtime.stop()


def test_max_model_len_is_refused_for_a_model_that_is_not_a_decision_model(tmp_path):
    from vllm_srun.testing.vela2 import write_encoder_package

    package = write_encoder_package(tmp_path / "vela2")
    assert refused(package, 128) == (
        "max_model_len applies to Decision models only, not vela2"
    )


# A question prompt of 482 tokens on the Decision 3 text fixture.
LONG_STATE = "x " * 150


def test_a_question_over_max_model_len_fails_with_max_length_exceeded(
    d3_text_package,
):
    runtime = serve(d3_text_package, max_model_len=400)
    try:
        model = runtime.served[0].model
        assert model.length_error(400) is None
        assert model.length_error(401).code == "max_length_exceeded"
        question = {"q": QUESTIONS["reasoning"]}
        status, body = asyncio.run(
            runtime.call("decisions", {"state": LONG_STATE, "questions": question})
        )
        assert status == 200
        assert body["answers"]["q"]["error"] == "max_length_exceeded"
        status, body = asyncio.run(
            runtime.call("decisions", {"state": STATE, "questions": question})
        )
        assert status == 200 and "error" not in body["answers"]["q"]
    finally:
        runtime.stop()


def test_system_one_refuses_a_question_over_the_process_max_model_len(
    d3_text_package,
):
    runtime = Runtime(
        ServeConfig(
            models=(ModelConfig(model=str(d3_text_package), device="cpu"),),
            engine_args=EngineArgs(max_model_len=400),
            result_cache_entries=0,
        )
    )
    runtime.start(background=False)
    try:
        client = TestClient(create_app(runtime))
        question = {"q": QUESTIONS["reasoning"]}
        response = client.post(
            "/v1/systemone", json={"state": LONG_STATE, "questions": question}
        )
        assert response.status_code == 400
        assert response.json()["error"]["question_error"] == "max_length_exceeded"
        response = client.post(
            "/v1/systemone", json={"state": STATE, "questions": question}
        )
        assert response.status_code == 200
    finally:
        runtime.stop()
