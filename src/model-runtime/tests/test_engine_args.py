"""Engine arguments: how a served model's arguments reach its batcher, its engine and its card."""

from __future__ import annotations

import asyncio

import pytest
from vllm_srun import dtypes
from vllm_srun.accel import onednn
from vllm_srun.config import DEFAULT_MAX_NUM_SEQS, EngineArgs, ModelConfig, ServeConfig
from vllm_srun.engines.native.engine import NativeEngine
from vllm_srun.plugins.base import BackboneSpec, DtypePolicy, ModelSpec
from vllm_srun.runtime import Runtime
from vllm_srun.testing.fixtures import write_fixture

from .conftest import QUESTIONS, STATE

# The qwen3 fixture stores its Linear weights (73,728 elements) BF16; the
# family reports that share so placement can price BF16 residency.
DECISION2_GPU = DtypePolicy(approximate_kernels=True, bf16_parameters=73_728)
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
        elif onednn.available():
            pytest.fail(
                "a native CPU task model with oneDNN's packed linear is batch-invariant"
            )
    finally:
        runtime.stop()
