"""The batch limits a model's engine arguments leave to the runtime.

A GPU takes the limits of its memory class (``DEVICE_CLASSES``); the CPU keeps
``config.DEFAULT_MAX_NUM_SEQS`` and ``config.DEFAULT_MAX_NUM_BATCHED_TOKENS``.
On a GPU the runtime then calibrates the token budget once the golden check
has warmed the model's kernels (``calibrate``): it times merged forwards of the
model's calibration rows (``LoadedModel.calibration_items``) at doubling
budgets up to the class's and keeps the smallest budget whose cost per token is
within ``KNEE_TOLERANCE`` of the cheapest measured (the efficiency knee), no
larger than what one forward does in ``MAX_FORWARD_SECONDS`` and than what fits
the device's free memory with ``MEMORY_MARGIN`` to spare. A limit the engine
arguments set is never changed, and a model without calibration rows keeps its
class's limits.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

from ..config import DEFAULT_MAX_NUM_BATCHED_TOKENS, DEFAULT_MAX_NUM_SEQS, EngineArgs
from ..plugins.base import DeviceInfo, LoadedModel

log = logging.getLogger("vllm_srun")

GIB = 1 << 30
# Tokens of one calibration row: about a question row of a decision request.
ROW_TOKENS = 256
# The smallest budget calibration times, and the smallest it chooses.
MIN_CALIBRATED_TOKENS = 1024
# Cost per token within this fraction of the cheapest measured counts as the plateau.
# Serving adds a fixed cost per forward that these timings leave out, so the knee
# stays close to the cheapest budget.
KNEE_TOLERANCE = 0.02
# Longest one merged forward should hold the device, so a request that arrives
# during it waits about this long at most.
MAX_FORWARD_SECONDS = 0.3
# Free device memory a calibrated forward leaves unused (caches, other workers).
MEMORY_MARGIN = 0.15
# Timed forwards per budget, after one untimed warm-up; the fastest counts.
TIMED_FORWARDS = 2


@dataclass(frozen=True)
class DeviceClass:
    """The batch limits of GPUs with at least ``min_memory_gib`` of device memory."""

    name: str
    min_memory_gib: float
    max_num_batched_tokens: int
    max_num_seqs: int


DEVICE_CLASSES = (
    DeviceClass("160 GB and more", 150, 32_768, 256),
    DeviceClass("70 to 160 GB", 60, 16_384, 128),
    DeviceClass("35 to 70 GB", 30, 8_192, 64),
    DeviceClass("under 35 GB", 0, 4_096, 32),
)


def device_class(device: DeviceInfo) -> DeviceClass | None:
    """The class of a GPU by its memory (the smallest when unknown); None for the CPU."""
    if device.accelerator == "cpu":
        return None
    gib = (device.total_memory or 0) / GIB
    return next(entry for entry in DEVICE_CLASSES if gib >= entry.min_memory_gib)


def resolve(args: EngineArgs, device: DeviceInfo) -> tuple[EngineArgs, str]:
    """``args`` with the limits it leaves unset filled in for ``device``, and where they came from."""
    if args.max_num_seqs is not None and args.max_num_batched_tokens is not None:
        return args, "the engine arguments"
    entry = device_class(device)
    seqs, tokens = (
        (DEFAULT_MAX_NUM_SEQS, DEFAULT_MAX_NUM_BATCHED_TOKENS)
        if entry is None
        else (entry.max_num_seqs, entry.max_num_batched_tokens)
    )
    source = "the CPU defaults" if entry is None else f"the {entry.name} device class"
    return (
        replace(
            args,
            max_num_seqs=args.max_num_seqs if args.max_num_seqs is not None else seqs,
            max_num_batched_tokens=(
                args.max_num_batched_tokens
                if args.max_num_batched_tokens is not None
                else tokens
            ),
        ),
        source,
    )


@dataclass(frozen=True)
class Forward:
    tokens: int
    seconds: float
    peak_bytes: int | None


@dataclass(frozen=True)
class Calibration:
    """The chosen token budget and the forwards that chose it."""

    max_num_batched_tokens: int
    knee: int
    latency_cap: int
    memory_cap: int | None
    forwards: tuple[Forward, ...]
    seconds: float

    def describe(self) -> str:
        timed = ", ".join(
            f"{forward.tokens} tokens {forward.seconds * 1000:.1f} ms"
            + (
                ""
                if forward.peak_bytes is None
                else f" {forward.peak_bytes / GIB:.2f} GiB"
            )
            for forward in self.forwards
        )
        memory = "none" if self.memory_cap is None else str(self.memory_cap)
        return (
            f"max_num_batched_tokens {self.max_num_batched_tokens} "
            f"(knee {self.knee}, {MAX_FORWARD_SECONDS:g} s cap {self.latency_cap}, "
            f"memory cap {memory}) from {timed} in {self.seconds:.1f} s"
        )


def calibrate(
    model: LoadedModel[Any, Any],
    execute: Callable[[Callable[[], Any]], Any],
    budget: int,
    memory: Callable[[], tuple[int, Callable[[], int | None]]] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> Calibration | None:
    """The token budget at most ``budget`` that ``model``'s merged forwards do best at, or None.

    ``execute`` runs work on the model's device. ``memory``, when the device
    reports it, returns the bytes a forward may still allocate and a probe
    that returns the peak bytes allocated since it was taken. None when the
    model has no calibration rows.
    """
    started = clock()
    sizes = []
    size = max(MIN_CALIBRATED_TOKENS, budget // 8)
    while size < budget:
        sizes.append(size)
        size *= 2
    sizes.append(max(budget, MIN_CALIBRATED_TOKENS))
    forwards: list[Forward] = []
    memory_cap: int | None = None
    for size in sizes:
        rows = max(1, size // ROW_TOKENS)
        items = model.calibration_items(rows, size // rows)
        if items is None:
            return None
        execute(partial(model.run_approximate, items))
        available, probe = memory() if memory is not None else (None, None)
        seconds = float("inf")
        peak: int | None = None
        for _ in range(TIMED_FORWARDS):
            begun = clock()
            execute(partial(model.run_approximate, items))
            seconds = min(seconds, clock() - begun)
            if probe is not None:
                measured = probe()
                peak = (
                    measured
                    if peak is None or measured is None
                    else max(peak, measured)
                )
        forwards.append(Forward(size, seconds, peak))
        if (
            available is not None
            and peak is not None
            and peak > (1 - MEMORY_MARGIN) * available
        ):
            memory_cap = (
                forwards[-2].tokens if len(forwards) > 1 else MIN_CALIBRATED_TOKENS
            )
            break
        if seconds > MAX_FORWARD_SECONDS:
            break
    fitting = [
        forward
        for forward in forwards
        if forward.seconds <= MAX_FORWARD_SECONDS
        and (memory_cap is None or forward.tokens <= memory_cap)
    ]
    if fitting:
        latency_cap = fitting[-1].tokens
    else:
        first = forwards[0]
        latency_cap = max(
            MIN_CALIBRATED_TOKENS // 4,
            int(first.tokens * MAX_FORWARD_SECONDS / first.seconds)
            // ROW_TOKENS
            * ROW_TOKENS,
        )
    cheapest = min(
        (forward.seconds / forward.tokens for forward in fitting), default=None
    )
    knee = next(
        (
            forward.tokens
            for forward in fitting
            if cheapest is not None
            and forward.seconds / forward.tokens <= (1 + KNEE_TOLERANCE) * cheapest
        ),
        latency_cap,
    )
    chosen = min(knee, latency_cap, budget)
    if memory_cap is not None:
        chosen = min(chosen, memory_cap)
    return Calibration(
        max_num_batched_tokens=chosen,
        knee=knee,
        latency_cap=latency_cap,
        memory_cap=memory_cap,
        forwards=tuple(forwards),
        seconds=clock() - started,
    )


def gpu_memory(
    torch_device: Any,
) -> Callable[[], tuple[int, Callable[[], int | None]]] | None:
    """``calibrate``'s memory probe for a CUDA or HIP device, or None for other devices."""
    import torch

    if torch_device.type != "cuda":
        return None

    def take() -> tuple[int, Callable[[], int | None]]:
        torch.cuda.synchronize(torch_device)
        free, _ = torch.cuda.mem_get_info(torch_device)
        allocated = torch.cuda.memory_allocated(torch_device)
        cached = torch.cuda.memory_reserved(torch_device) - allocated
        torch.cuda.reset_peak_memory_stats(torch_device)

        def peak() -> int | None:
            torch.cuda.synchronize(torch_device)
            return torch.cuda.max_memory_allocated(torch_device) - allocated

        return free + cached, peak

    return take
