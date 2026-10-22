"""GPU: merged forwards pack their rows on the device's variable-length kernels, and the token budget is calibrated.

The Decision 3.0 vision fixture served on a GPU: merged text and image rows run packed (variable-length
attention, the chunked gated-delta kernel over ``cu_seqlens``, one packed tower call) and answer as each
question alone; the batch limits the engine arguments leave unset come from the device class and the startup
calibration, which a state-once package's branched forwards pass too.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllm_srun.accel.gpu import _wrap_varlen  # noqa: E402
from vllm_srun.accel.kernels import varlen_attention_ref  # noqa: E402
from vllm_srun.config import ModelConfig, ServeConfig  # noqa: E402
from vllm_srun.engines.native.models.packed import Packed  # noqa: E402
from vllm_srun.runtime import Runtime  # noqa: E402
from vllm_srun.scheduler import calibration  # noqa: E402
from vllm_srun.testing.decision3 import write_fixture  # noqa: E402

from .test_decision3_family import QUESTIONS  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("groups", [1, 2])
def test_installed_varlen_api_matches_per_row_attention(groups):
    """Exercise the installed torch interface and its real GPU GQA kernel."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA or ROCm GPU")
    from torch.nn.attention.varlen import varlen_attn

    torch.manual_seed(4)
    packed = Packed([3, 7, 1], torch.device("cuda"))
    q = torch.randn(11, 4, 64, device="cuda", dtype=torch.bfloat16)
    k, v = (
        torch.randn(11, 4 // groups, 64, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    )
    out = _wrap_varlen(varlen_attn)(q, k, v, packed, scale=64**-0.5, groups=groups)
    want = varlen_attention_ref(q, k, v, packed, scale=64**-0.5, groups=groups)
    torch.testing.assert_close(out, want, rtol=0.02, atol=0.02)


def _serve(path, variant, seed):
    """The fixture ``variant`` served on the GPU with default engine arguments."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA or ROCm GPU")
    device = "rocm:0" if torch.version.hip else "cuda:0"
    package = write_fixture(path, variant, seed=seed)
    runtime = Runtime(
        ServeConfig(models=(ModelConfig(model=str(package), device=device),))
    )
    runtime.start(background=False)
    return runtime


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    runtime = _serve(tmp_path_factory.mktemp("d3-gpu") / "d3-vision", "vision", 11)
    yield runtime.served[0]
    runtime.stop()


@pytest.fixture(scope="module")
def state_once_served(tmp_path_factory):
    runtime = _serve(
        tmp_path_factory.mktemp("d3-gpu") / "prefix_bidir", "prefix_bidir", 31
    )
    yield runtime.served[0]
    runtime.stop()


def _gap(got, want):
    return max(
        abs(a - b)
        for got_row, want_row in zip(got, want, strict=True)
        for a, b in zip(got_row, want_row, strict=True)
    )


def test_merged_rows_pack_where_the_kernels_allow(served):
    engine = served.model.engine_model
    if not engine.supports_packed:
        pytest.skip(engine.receipt()["packed"]["skipped"])
    assert served.model.packs_merged_rows
    assert engine.receipt()["packed"] == {"attention": "torch-varlen-attention"}


def test_packed_rows_answer_as_each_question_alone(served, png_url):
    model = served.model
    if not model.packs_merged_rows:
        pytest.skip("this device's kernels keep merged forwards padded")
    text = [
        item
        for state in (
            "Merge two sorted lists.",
            "Prove that there are infinitely many primes. " * 6,
            {"ticket": "refund", "lines": ["a", "b"]},
        )
        for item in model.plan(state, QUESTIONS).items
    ]
    images = [
        item
        for urls in ([png_url(80, 60, 61)], [png_url(64, 96, 62), png_url(64, 64, 63)])
        for item in model.plan_images("Shown?", QUESTIONS, urls).items
    ]
    for rows in (text, images):
        merged = model.run_approximate(rows)
        alone = [model.run([row])[0] for row in rows]
        assert _gap(merged, alone) < 0.02


@pytest.mark.parametrize("which", ["served", "state_once_served"])
def test_the_token_budget_is_calibrated_at_startup(which, request):
    served = request.getfixturevalue(which)
    entry = calibration.device_class(served.placement.device)
    assert entry is not None
    assert served.args.max_num_seqs == entry.max_num_seqs
    assert served.limits_source == f"the {entry.name} device class, calibrated"
    assert (
        calibration.MIN_CALIBRATED_TOKENS // 4
        <= served.args.max_num_batched_tokens
        <= entry.max_num_batched_tokens
    )
    assert (
        served.scheduler.batcher.max_num_batched_tokens
        == served.args.max_num_batched_tokens
    )
