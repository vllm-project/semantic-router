"""GPU: merged forwards pack their rows on the device's variable-length kernels, and the token budget is calibrated.

The Decision 3.0 vision fixture served on a GPU: merged text and image rows run packed (variable-length
attention, the chunked gated-delta kernel over ``cu_seqlens``, one packed tower call) and answer as each
question alone; the batch limits the engine arguments leave unset come from the device class and the startup
calibration.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllm_srun.config import ModelConfig, ServeConfig  # noqa: E402
from vllm_srun.runtime import Runtime  # noqa: E402
from vllm_srun.scheduler import calibration  # noqa: E402
from vllm_srun.testing.decision3 import write_fixture  # noqa: E402

from .test_decision3_family import QUESTIONS  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA or ROCm GPU")
    device = "rocm:0" if torch.version.hip else "cuda:0"
    package = write_fixture(
        tmp_path_factory.mktemp("d3-gpu") / "d3-vision", "vision", seed=11
    )
    runtime = Runtime(
        ServeConfig(models=(ModelConfig(model=str(package), device=device),))
    )
    runtime.start(background=False)
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


def test_the_token_budget_is_calibrated_at_startup(served):
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
