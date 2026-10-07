"""The native engine's encoder path on GPUs: length buckets and a graph per bucket.

A short encoder forward is hundreds of tiny kernels, so on a GPU its latency
is launch overhead. Rows are padded to a bucket (rows and width from
``ROW_BUCKETS`` / ``WIDTH_BUCKETS``), the key-padding mask is built from the
host-known lengths inside the forward, and each bucket's forward is captured
as a HIP / CUDA graph on its second use and replayed afterwards. Eager runs of
a bucket shape compute exactly what its replay computes, so an answer never
depends on whether its bucket was captured yet. Padded keys get exactly zero
attention weight; padded queries' rows are dropped. Rows wider than the last
width bucket, batches above ``MAX_GRAPH_TOKENS`` padded tokens and buckets
that pad too much (``pads_little``) are GPU-bound: they run eagerly on the
packed layout (on MI325X a 512-token row is faster unpadded, 5.4 ms, than as a
masked bucket replay, 5.8 ms).
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .models import EncoderBackbone

ROW_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
WIDTH_BUCKETS = (16, 32, 48, 64, 96, 128, 192, 256, 384)
MAX_GRAPH_TOKENS = 8192
MAX_GRAPHS = 256
CAPTURE_AFTER = 2
# A padded grid pays while it pads at most this share of its real tokens, or
# while it is at most LAUNCH_BOUND_TOKENS (launch overhead outweighs the padding).
MAX_PADDING = 0.25
LAUNCH_BOUND_TOKENS = 1024
# A graph's bucket and outputs: (rows, width, exits, normalize).
GraphKey = tuple[int, int, tuple[int, ...], bool]


def pads_little(padded: int, real: int) -> bool:
    """Whether ``padded`` grid positions for ``real`` tokens are worth one padded kernel."""
    return padded <= LAUNCH_BOUND_TOKENS or padded <= (1 + MAX_PADDING) * real


def bucket(value: int, buckets: Sequence[int]) -> int | None:
    """The smallest bucket holding ``value``; None above the largest."""
    index = bisect.bisect_left(buckets, value)
    return buckets[index] if index < len(buckets) else None


class EncoderGraphs:
    """Bucketed padded forwards of an encoder backbone, captured as graphs."""

    def __init__(
        self,
        backbone: EncoderBackbone,
        device: torch.device,
        *,
        capture_after: int = CAPTURE_AFTER,
        max_graphs: int = MAX_GRAPHS,
        max_tokens: int = MAX_GRAPH_TOKENS,
    ):
        self.backbone = backbone
        self.device = device
        self.capture_after = capture_after
        self.max_graphs = max_graphs
        self.max_tokens = max_tokens
        self.graphs: dict[GraphKey, dict[str, Any]] = {}
        self.seen: dict[GraphKey, int] = {}
        self.failed: set[GraphKey] = set()
        self.pool: Any = None
        self.stats = {"captures": 0, "replays": 0, "eager": 0, "packed": 0, "failed": 0}

    def shape(self, lengths: Sequence[int]) -> tuple[int, int] | None:
        """The ``(rows, width)`` bucket of a batch; None when it runs packed."""
        rows = bucket(len(lengths), ROW_BUCKETS)
        width = bucket(max(lengths), WIDTH_BUCKETS)
        if rows is None or width is None or rows * width > self.max_tokens:
            return None
        if not pads_little(rows * width, sum(lengths)):
            return None
        return rows, width

    def __call__(
        self,
        input_ids: torch.Tensor,
        lengths: list[int],
        exits: tuple[int, ...],
        normalize: bool,
    ) -> dict[int, torch.Tensor]:
        """Hidden states by exit for packed ``input_ids`` ``[N]``, packed ``[N, H]`` on return."""
        shape = self.shape(lengths)
        if shape is None:
            self.stats["packed"] += 1
            layout = self.backbone.packed(lengths, self.device)
            return self.backbone.encode(
                input_ids.to(self.device), layout, exits, normalize
            )
        rows, width = shape
        ids, valid, index = self._pad(input_ids, lengths, rows, width)
        key = (rows, width, exits, normalize)
        entry = self.graphs.get(key)
        if entry is None:
            self.seen[key] = self.seen.get(key, 0) + 1
            if (
                key in self.failed
                or self.seen[key] < self.capture_after
                or len(self.graphs) >= self.max_graphs
                or not torch.is_inference_mode_enabled()
            ):
                self.stats["eager"] += 1
                hidden = self._forward(ids, valid, exits, normalize)
                return {
                    layer: value.view(rows * width, -1)[index]
                    for layer, value in hidden.items()
                }
            entry = self._capture(key, ids, valid, exits, normalize)
            if entry is None:
                self.stats["eager"] += 1
                hidden = self._forward(ids, valid, exits, normalize)
                return {
                    layer: value.view(rows * width, -1)[index]
                    for layer, value in hidden.items()
                }
        entry["ids"].copy_(ids)
        entry["valid"].copy_(valid)
        entry["graph"].replay()
        self.stats["replays"] += 1
        return {
            layer: value.view(rows * width, -1)[index]
            for layer, value in entry["output"].items()
        }

    def _pad(
        self, input_ids: torch.Tensor, lengths: Sequence[int], rows: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Packed IDs as ``[rows, width]`` padded rows, their key mask and the packed positions."""
        positions = torch.cat(
            [
                torch.arange(n, dtype=torch.long) + row * width
                for row, n in enumerate(lengths)
            ]
        )
        valid = torch.zeros(rows * width, dtype=torch.bool)
        valid[positions] = True
        ids = torch.zeros(rows * width, dtype=input_ids.dtype, device=self.device)
        index = positions.to(self.device, non_blocking=True)
        ids.index_copy_(0, index, input_ids.to(self.device))
        return (
            ids.view(rows, width),
            valid.view(rows, width).to(self.device, non_blocking=True),
            index,
        )

    def _forward(
        self,
        ids: torch.Tensor,
        valid: torch.Tensor,
        exits: tuple[int, ...],
        normalize: bool,
    ) -> dict[int, torch.Tensor]:
        rows, width = ids.shape
        layout = self.backbone.masked(valid, rows, width, self.device)
        return self.backbone.encode(ids, layout, exits, normalize)

    def _capture(
        self,
        key: GraphKey,
        ids: torch.Tensor,
        valid: torch.Tensor,
        exits: tuple[int, ...],
        normalize: bool,
    ) -> dict[str, Any] | None:
        static = {"ids": ids.clone(), "valid": valid.clone()}

        def body() -> dict[int, torch.Tensor]:
            return self._forward(static["ids"], static["valid"], exits, normalize)

        try:
            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]  # torch leaves Stream's constructor unannotated
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    body()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            # Device work is serialized, but other threads may still query the device
            # meanwhile (placement); global capture mode would fail them and the capture.
            with torch.cuda.graph(
                graph, pool=self.pool, capture_error_mode="thread_local"
            ):
                output = body()
            torch.cuda.synchronize()
        except Exception:
            torch.cuda.synchronize()
            self.failed.add(key)
            self.stats["failed"] += 1
            return None
        entry = {**static, "graph": graph, "output": output}
        self.graphs[key] = entry
        self.stats["captures"] += 1
        return entry

    def receipt(self) -> dict[str, Any]:
        return {**self.stats, "cached": len(self.graphs)}
