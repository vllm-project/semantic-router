"""The five family sizes: bases, size classes and Stage T settings.

Size classes follow the board's own parameter counts (vision encoder included): an entrant belongs to the
member nearest to it in log(total parameters), so the boundaries are geometric midpoints between our
models' totals. Learning rates scale about 1/sqrt(size) from d3's 2e-6 at 27B.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Size:
    name: str
    repo: str
    base: str
    revision: str
    total_params: int
    lr: float
    gpus: int
    token_budget: int

    @property
    def short(self) -> str:
        return self.name.removeprefix("d3-")


SIZES = {
    s.short: s
    for s in (
        Size(
            "d3-flash",
            "vllm-sr/d3-flash",
            "Qwen/Qwen3.5-9B",
            "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            9_653_104_368,
            3e-6,
            8,
            65_536,
        ),
        Size(
            "d3-mini",
            "vllm-sr/d3-mini",
            "Qwen/Qwen3.5-4B",
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            4_659_865_088,
            5e-6,
            4,
            98_304,
        ),
        Size(
            "d3-nano",
            "vllm-sr/d3-nano",
            "Qwen/Qwen3.5-2B",
            "15852e8c16360a2fea060d615a32b45270f8a8fc",
            2_274_069_824,
            7e-6,
            2,
            196_608,
        ),
        Size(
            "d3-lite",
            "vllm-sr/d3-lite",
            "Qwen/Qwen3.5-0.8B",
            "2fc06364715b967f1860aea9cf38778875588b17",
            873_438_784,
            1e-5,
            1,
            262_144,
        ),
        Size(
            "d3-edge",
            "vllm-sr/d3-edge",
            "Qwen/Qwen3-0.6B-Base",
            "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
            596_049_920 + 100_592_896,
            1e-5,
            1,
            131_072,
        ),
    )
}
D3_TOTAL_PARAMS = 27_800_000_000


def boundaries() -> list[tuple[str, float]]:
    """Upper bound of each class (total parameters), smallest first; the last class is unbounded."""
    ordered = sorted(SIZES.values(), key=lambda s: s.total_params)
    totals = [s.total_params for s in ordered] + [D3_TOTAL_PARAMS]
    return [
        (s.short, math.sqrt(totals[i] * totals[i + 1])) for i, s in enumerate(ordered)
    ]


def size_class(total_params: float) -> str:
    for name, upper in boundaries():
        if total_params <= upper:
            return name
    return "d3"
