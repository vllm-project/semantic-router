"""GPU (CUDA): the native engine's graph replays equal its eager forwards bit for bit.

Vela 2.0's decoders run their trees as forests (``models/forest.py``). On CUDA the engine replays a graph per
exact forest shape and per exact padded shape of the ordinary forward. This loads one small decoder package
twice, with graphs and without, and runs forests and ordinary batches of several shapes through
``NativeEngineModel`` three times (first use eager, second captured, third replayed), interleaved so that
eager forwards of other shapes run between replays (the order that faults on ROCm, which keeps forests eager).
Two inputs share each shape, so a result kept from one must survive the other's replay.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllm_srun.accel.cuda import CUDAAccelerator  # noqa: E402
from vllm_srun.engines.native.engine import NativeEngine  # noqa: E402
from vllm_srun.families.vela2.family import Vela2Family  # noqa: E402
from vllm_srun.plugins.base import (  # noqa: E402
    EngineOptions,
    ForwardBatch,
    PackageRef,
    TreeBatch,
)
from vllm_srun.testing.vela2 import write_decoder_package  # noqa: E402

pytestmark = pytest.mark.gpu

# (prefix lengths, block lengths, block owners)
FORESTS = [
    ([37], [115], [0]),
    ([12], [33, 121, 125, 87, 135, 203, 48], [0] * 7),
    ([60, 41], [30, 64, 17], [0, 1, 1]),
    # the same tensor shapes as the last forest, with other lengths and owners
    ([41, 60], [64, 17, 30], [1, 0, 0]),
]
# row lengths of an ordinary right-padded batch
ROWS = [[40], [70, 52, 9]]


def forest(prefixes, blocks, owners, generator):
    def ids(n):
        return torch.randint(5, 380, (n,), generator=generator).tolist()

    return TreeBatch(
        [ids(n) for n in prefixes], [ids(n) for n in blocks], owners, "rows"
    )


def rows(lengths, generator):
    width = max(lengths)
    input_ids = torch.zeros((len(lengths), width), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row, n in enumerate(lengths):
        input_ids[row, :n] = torch.randint(5, 380, (n,), generator=generator)
        attention_mask[row, :n] = 1
    return ForwardBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        gather=torch.tensor([[0, n - 1] for n in lengths]),
        query=torch.tensor([n - 1 for n in lengths]),
        lengths=list(lengths),
    )


def outputs(model, batch):
    if isinstance(batch, TreeBatch):
        return (model.tree(batch).hidden,)
    out = model.forward(batch)
    return out.gathered, out.query


def load(tmp_path, graphs):
    accelerator = CUDAAccelerator()
    if not accelerator.available():
        pytest.skip("needs a CUDA GPU")
    family = Vela2Family()
    verified = family.verify(PackageRef(write_decoder_package(tmp_path / "decoder")))
    spec = family.describe(verified)
    device = accelerator.devices()[0]
    return [
        NativeEngine().load(spec, accelerator, device, EngineOptions(graphs=on))
        for on in graphs
    ]


def test_engine_graph_replays_equal_the_eager_engine(tmp_path):
    graphed, eager = load(tmp_path, (True, False))
    generator = torch.Generator().manual_seed(5)
    pairs = [
        (forest(*case, generator), forest(*case, generator)) for case in FORESTS
    ] + [(rows(case, generator), rows(case, generator)) for case in ROWS]
    for _ in range(3):
        for first, second in pairs:
            kept = outputs(graphed, first)
            again = outputs(graphed, second)
            for got, want in zip(
                kept + again,
                outputs(eager, first) + outputs(eager, second),
                strict=True,
            ):
                assert torch.equal(got, want)
    stats = graphed.receipt()["graphs"]
    assert (
        stats["captures"] == len(pairs)
        and stats["replays"] > 0
        and stats["failed"] == 0
    ), stats


def test_no_graph_is_captured_past_the_memory_budget(tmp_path):
    (model,) = load(tmp_path, (True,))
    model.graphs.max_bytes = 0
    batch = forest(*FORESTS[0], torch.Generator().manual_seed(5))
    for _ in range(3):
        model.tree(batch)
    stats = model.receipt()["graphs"]
    assert stats["captures"] == 0 and stats["full"] > 0, stats


def test_eager_memory_does_not_spend_the_graph_budget(tmp_path):
    """Memory an eager forward holds between captures is not charged to the graphs."""
    (model,) = load(tmp_path, (True,))
    model.graphs.max_bytes = 256 << 20
    generator = torch.Generator().manual_seed(5)
    first, second = (forest(*case, generator) for case in FORESTS[:2])
    for _ in range(3):
        model.tree(first)
    held = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    for _ in range(3):
        model.tree(second)
    del held
    stats = model.receipt()["graphs"]
    assert stats["captures"] == 2 and stats["full"] == 0, stats
