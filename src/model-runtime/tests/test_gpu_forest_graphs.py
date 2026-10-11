"""GPU (CUDA): a replayed forest graph equals the eager forest forward bit for bit.

Vela 2.0's decoders run their trees as forests (``models/forest.py``). On CUDA the engine replays a graph per
exact forest shape; this checks the replay against the eager forward at a random Qwen3.5 backbone's Eos-0.8B
dimensions, with eager forwards of other shapes between replays (the order that faults on ROCm, which keeps
forests eager).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllm_srun.accel.cuda import CUDAAccelerator  # noqa: E402
from vllm_srun.engines.native import fast  # noqa: E402
from vllm_srun.engines.native.models.forest import ForestShape  # noqa: E402

from .test_gpu_fast_path import build, qwen3_5  # noqa: E402

pytestmark = pytest.mark.gpu

# (prefix lengths, block lengths, block owners)
SHAPES = [
    ([37], [115], [0]),
    ([12], [336, 121, 125, 187, 135, 203, 480], [0] * 7),
    ([60, 41], [30, 64, 17], [0, 1, 1]),
]


def forest(prefixes, blocks, owners, generator, device):
    width = max(prefixes)
    prefix_ids = torch.zeros((len(prefixes), width), dtype=torch.long)
    prefix_mask = torch.zeros_like(prefix_ids)
    for row, length in enumerate(prefixes):
        prefix_ids[row, width - length :] = torch.randint(
            0, 1024, (length,), generator=generator
        )
        prefix_mask[row, width - length :] = 1
    block_width = max(blocks)
    block_ids = torch.zeros((len(blocks), block_width), dtype=torch.long)
    block_mask = torch.zeros_like(block_ids)
    for row, length in enumerate(blocks):
        block_ids[row, :length] = torch.randint(0, 1024, (length,), generator=generator)
        block_mask[row, :length] = 1
    shape = ForestShape(
        tuple(width - n for n in prefixes), tuple(blocks), tuple(owners)
    )
    tensors = (prefix_ids, prefix_mask, block_ids, block_mask, torch.tensor(owners))
    return tuple(t.to(device) for t in tensors), shape


def test_forest_graph_replays_equal_the_eager_forest():
    accelerator = CUDAAccelerator()
    if not accelerator.available():
        pytest.skip("needs a CUDA GPU")
    device = accelerator.devices()[0]
    backbone = build(qwen3_5(4, 1024, 16, 8), 7, False, accelerator, device)
    graphs = fast.Graphs(backbone, fast.Masks())
    generator = torch.Generator().manual_seed(5)
    cases = [
        forest(*case, generator, accelerator.torch_device(device)) for case in SHAPES
    ]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(3):
            for inputs, shape in cases:

                def body(*tensors, shape=shape):
                    return backbone.forward_forest(*tensors, shape)[1]

                want = body(*inputs).clone()
                got = graphs.run(
                    ("forest", inputs[0].shape, inputs[2].shape, shape),
                    inputs,
                    body,
                    inputs[0].numel() + inputs[2].numel(),
                ).clone()
                assert torch.equal(want, got), shape
    stats = graphs.receipt()
    assert (
        stats["captures"] == len(SHAPES)
        and stats["replays"] > 0
        and stats["failed"] == 0
    )
