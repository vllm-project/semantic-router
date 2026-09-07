"""Single-process and torchrun distributed runtime helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    distributed: bool
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistributedContext:
    """Initialize NCCL/RCCL when launched with torchrun."""
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not distributed:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DistributedContext(0, 1, 0, False, device)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", init_method="env://", device_id=device)
    return DistributedContext(rank, world_size, local_rank, True, device)


def wrap_distributed(model: Any, context: DistributedContext) -> Any:
    """Wrap a training module for the active device topology."""
    if context.distributed:
        return DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            find_unused_parameters=True,
        )
    if torch.cuda.device_count() > 1:
        return torch.nn.DataParallel(model)
    return model


def barrier(context: DistributedContext) -> None:
    if context.distributed:
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def rank0_print(context: DistributedContext, *args: Any, **kwargs: Any) -> None:
    if context.is_main:
        print(*args, **kwargs)
