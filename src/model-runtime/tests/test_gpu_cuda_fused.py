"""CUDA: Qwen3.5 fused layers and graph replay work without causal-conv1d."""

from __future__ import annotations

import pytest
import torch
from vllm_srun.accel import gpu
from vllm_srun.accel.cuda import CUDAAccelerator
from vllm_srun.engines.native import fast

from .test_gpu_fast_path import batch, build, qwen3_5

pytestmark = pytest.mark.gpu


def test_qwen35_fused_layers_and_graphs_without_causal_conv1d(monkeypatch):
    accelerator = CUDAAccelerator()
    if not accelerator.available():
        pytest.skip("needs a CUDA GPU")
    device = accelerator.devices()[0]
    if int((device.arch or "sm_0").removeprefix("sm_")) < 80:
        pytest.skip("fused kernels need Ampere or newer")
    optional = gpu._optional
    monkeypatch.setattr(
        gpu,
        "_optional",
        lambda module, attribute: (
            None if module == "causal_conv1d" else optional(module, attribute)
        ),
    )
    module = build(qwen3_5(4, 256, 16, 2), 7, False, accelerator, device)
    kernels = module.kernels
    kernels.allow_approximate = True
    if kernels.select("chunk_gated_delta_rule").source != "fla":
        pytest.skip("needs FLA")
    if not kernels.has("add_rmsnorm"):
        pytest.skip("needs Triton")
    assert kernels.select("causal_conv1d").source == "torch-reference"
    assert kernels.has("gdn_prep")
    assert fast.fused_unavailable(module, kernels) is None
    assert fast.install_fused(module) == 4
    graphs = fast.Graphs(module, fast.Masks())
    generator = torch.Generator().manual_seed(11)
    for lengths in ([13], [5, 17, 33]):
        ids, mask = batch(lengths, generator, accelerator.torch_device(device))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            eager = graphs(ids, mask, lengths).clone()
            assert torch.isfinite(eager).all()
            for _ in range(2):
                assert torch.equal(eager, graphs(ids, mask, lengths))
    receipt = graphs.receipt()
    assert receipt["failed"] == 0
    assert receipt["replays"] > 0
