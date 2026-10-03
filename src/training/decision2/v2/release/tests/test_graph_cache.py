"""Graph cache policy of the release runtime's fast path (``runtime/fast.py`` ``Graphs``).

A shape is captured on its second use while the cache has room (``max_graphs``
graphs and ``max_output_bytes`` of outputs); once it is full, new shapes run the
eager forward and the captured graphs stay and replay. No graph is ever
destroyed: on ROCm, evicting large graphs from the shared memory pool crashed
the GPU process. A stand-in for torch records captures and replays, so this runs
without a GPU; ``gpu_fast_path`` checks the same policy bit for bit on a GPU.
"""

from __future__ import annotations

import contextlib
import types
import unittest

from v2.release.runtime import fast

BYTES_PER_TOKEN = 4


class Tensor:
    """A right-padded (rows, length) batch on the GPU, or an output of that shape."""

    is_cuda = True

    def __init__(self, rows: int, length: int):
        self.shape = (rows, length)

    def dim(self) -> int:
        return 2

    def clone(self) -> Tensor:
        return Tensor(*self.shape)

    def copy_(self, other: Tensor) -> Tensor:
        assert other.shape == self.shape
        return self

    def numel(self) -> int:
        return self.shape[0] * self.shape[1]

    def element_size(self) -> int:
        return BYTES_PER_TOKEN


class Graph:
    def __init__(self, made: list):
        self.replays = 0
        made.append(self)

    def replay(self) -> None:
        self.replays += 1


class Torch:
    bfloat16 = "bfloat16"

    def __init__(self):
        self.graphs_made: list[Graph] = []
        made = self.graphs_made

        class Stream:
            def wait_stream(self, other) -> None:
                pass

        current = Stream()
        self.cuda = types.SimpleNamespace(
            graph_pool_handle=object,
            Stream=Stream,
            current_stream=lambda: current,
            stream=lambda stream: contextlib.nullcontext(),
            synchronize=lambda: None,
            CUDAGraph=lambda: Graph(made),
            graph=lambda graph, pool=None: contextlib.nullcontext(),
        )

    def is_tensor(self, value) -> bool:
        return isinstance(value, Tensor)

    def is_inference_mode_enabled(self) -> bool:
        return True

    def is_autocast_enabled(self, device: str) -> bool:
        return True

    def get_autocast_dtype(self, device: str) -> str:
        return self.bfloat16

    def autocast(self, *args, **kwargs):
        return contextlib.nullcontext()


class Backbone:
    """Counts eager forwards (calls outside capture) and returns an output of the input's shape."""

    def __init__(self):
        self.capturing = False
        self.eager = 0

    def forward(self, input_ids, attention_mask, **kwargs) -> Tensor:
        if not self.capturing:
            self.eager += 1
        return Tensor(*input_ids.shape)


class Masks:
    def build(self, attention_mask, padded: bool):
        return attention_mask


def cache(**limits) -> tuple[fast.Graphs, Backbone, Torch]:
    torch, backbone = Torch(), Backbone()
    graphs = fast.Graphs(backbone, torch, Masks(), **limits)
    capture = graphs._capture

    def capturing(*args, **kwargs):
        backbone.capturing = True
        try:
            return capture(*args, **kwargs)
        finally:
            backbone.capturing = False

    graphs._capture = capturing
    return graphs, backbone, torch


def run(graphs: fast.Graphs, backbone: Backbone, rows: int, length: int) -> Tensor:
    ids, mask = Tensor(rows, length), Tensor(rows, length)
    graphs.lengths = [length] * rows
    return backbone.forward(input_ids=ids, attention_mask=mask, use_cache=False)


SHAPES = [(1, 8), (1, 16), (2, 32), (8, 64)]


class GraphCacheTest(unittest.TestCase):
    def test_full_cache_runs_new_shapes_eagerly(self):
        graphs, backbone, torch = cache(max_graphs=2)
        for _ in range(3):
            for rows, length in SHAPES:
                self.assertEqual(
                    run(graphs, backbone, rows, length).shape, (rows, length)
                )
        first, second = ((rows, length, False, False) for rows, length in SHAPES[:2])
        self.assertEqual(list(graphs.graphs), [first, second])
        self.assertEqual(
            graphs.stats,
            {"captures": 2, "replays": 4, "eager": 8, "full": 4, "failed": 0},
        )
        self.assertEqual(backbone.eager, 8)
        self.assertEqual(
            [entry["graph"] for entry in graphs.graphs.values()], torch.graphs_made
        )
        self.assertEqual([g.replays for g in torch.graphs_made], [2, 2])

    def test_output_budget_stops_capture(self):
        rows, length = SHAPES[-1]
        graphs, backbone, torch = cache(
            max_output_bytes=rows * length * BYTES_PER_TOKEN
        )
        for _ in range(3):
            for shape in reversed(SHAPES):
                run(graphs, backbone, *shape)
        self.assertEqual(list(graphs.graphs), [(rows, length, False, False)])
        self.assertEqual(graphs.output_bytes, rows * length * BYTES_PER_TOKEN)
        self.assertEqual(graphs.stats["captures"], 1)
        self.assertEqual(graphs.stats["full"], 6)
        self.assertEqual(len(torch.graphs_made), 1)
        self.assertEqual(torch.graphs_made[0].replays, 2)

    def test_captured_graphs_are_never_evicted(self):
        graphs, backbone, torch = cache(max_graphs=3)
        shapes = [(rows, 8 * n) for n in range(1, 41) for rows in (1, 3)]
        for _ in range(4):
            for shape in shapes:
                run(graphs, backbone, *shape)
        kept = [(rows, length, False, False) for rows, length in shapes[:3]]
        self.assertEqual(list(graphs.graphs), kept)
        self.assertEqual(len(torch.graphs_made), 3)
        self.assertEqual([g.replays for g in torch.graphs_made], [3, 3, 3])
        self.assertNotIn("evicted", graphs.stats)
        self.assertEqual(graphs.stats["full"], 3 * (len(shapes) - 3))

    def test_large_shapes_stay_eager(self):
        graphs, backbone, _ = cache(max_tokens=64)
        for _ in range(3):
            run(graphs, backbone, 2, 64)
        self.assertEqual(graphs.graphs, {})
        self.assertEqual(graphs.stats["eager"], 3)
        self.assertEqual(graphs.stats["full"], 0)


if __name__ == "__main__":
    unittest.main()
