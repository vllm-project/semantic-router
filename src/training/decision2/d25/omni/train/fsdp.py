"""FSDP2 plumbing: process group, sharding, rank-0 weight broadcast, full-state gather, resume.

This is the seam to Vega's FSDP2 trainer: once ``d25.vega.train`` is committed, these functions
delegate to its equivalents where they match (process-group setup, resume format), and only the
Omni-specific unit layout below stays here.

Unit layout (one ``fully_shard`` each): every ViT block, the ViT stem (patch embedding and position
table), the merger, every decoder layer, the readout (FP32 compute) and the root (embeddings, final
norm). Parameters are FP32 masters; compute is BF16 with FP32 gradient reduction. The model is
built on the meta device, sharded, materialised on the device and filled from rank 0's full state
dict, so host memory holds one copy of the checkpoint instead of one per rank.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from d25.omni.model import checkpoint


@dataclass(frozen=True)
class Distributed:
    rank: int = 0
    world: int = 1
    local_rank: int = 0
    device: torch.device = torch.device("cpu")

    @property
    def primary(self) -> bool:
        return self.rank == 0

    @property
    def sharded(self) -> bool:
        return self.world > 1

    def barrier(self) -> None:
        if self.sharded:
            dist.barrier()

    def sum(self, values: list[float]) -> list[float]:
        if not self.sharded:
            return list(values)
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor)
        return tensor.tolist()

    def gather_objects(self, value: Any) -> list[Any]:
        if not self.sharded:
            return [value]
        values: list[Any] = [None] * self.world
        dist.all_gather_object(values, value)
        return values

    def broadcast_object(self, value: Any) -> Any:
        if not self.sharded:
            return value
        box = [value if self.primary else None]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def close(self) -> None:
        if self.sharded and dist.is_initialized():
            dist.destroy_process_group()


def initialize() -> Distributed:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    cuda = torch.cuda.is_available()
    if world == 1:
        return Distributed(device=torch.device("cuda:0" if cuda else "cpu"))
    if int(os.environ.get("LOCAL_WORLD_SIZE", world)) != world:
        raise ValueError("single-node training only (RDMA is unavailable)")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if cuda:
        torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl" if cuda else "gloo", timeout=timedelta(hours=2))
    return Distributed(
        rank, world, local_rank, torch.device(f"cuda:{local_rank}" if cuda else "cpu")
    )


def shard(
    model, mesh_device: str, world: int, reshard_after_forward: bool = True
) -> None:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mesh = init_device_mesh(mesh_device, (world,))
    bf16 = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fp32 = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
    visual = model.backbone.visual
    for block in visual.blocks:
        fully_shard(
            block,
            mesh=mesh,
            mp_policy=bf16,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(
        visual.merger,
        mesh=mesh,
        mp_policy=bf16,
        reshard_after_forward=reshard_after_forward,
    )
    fully_shard(
        visual, mesh=mesh, mp_policy=bf16, reshard_after_forward=reshard_after_forward
    )
    for layer in model.backbone.language_model.layers:
        fully_shard(
            layer,
            mesh=mesh,
            mp_policy=bf16,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(
        model.readout,
        mesh=mesh,
        mp_policy=fp32,
        reshard_after_forward=reshard_after_forward,
    )
    fully_shard(model, mesh=mesh, mp_policy=bf16)


def full_state_from_checkpoint(directory: str | Path) -> dict[str, torch.Tensor]:
    """``backbone.*`` and ``readout.weight`` tensors of a code-readout v1 checkpoint, FP32 on CPU."""
    state = {
        f"backbone.{key}": tensor.float()
        for key, tensor in checkpoint.iter_tensors(directory)
    }
    state["readout.weight"] = checkpoint.load_readout(directory).float()
    return state


def build(
    init: str | Path,
    distributed: Distributed,
    attention_mode: str,
    scales: dict[str, float],
    gradient_checkpointing: bool = True,
):
    """Sharded (or, on one process, plain) FP32 model loaded from ``init`` with the arm applied."""
    from transformers import Qwen3_5Config

    from d25.omni.train import arms
    from d25.omni.train.model import OmniDecisionModel, reset_nonpersistent_buffers

    if not distributed.sharded:
        model = OmniDecisionModel.from_checkpoint(init, attention_mode=attention_mode)
        model.to(distributed.device)
    else:
        config = Qwen3_5Config.from_pretrained(str(init))
        with torch.device("meta"):
            model = OmniDecisionModel.from_config(config, attention_mode)
    counts = arms.apply(model, scales)
    if gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if distributed.sharded:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        shard(model, distributed.device.type, distributed.world)
        model.to_empty(device=distributed.device)
        state = full_state_from_checkpoint(init) if distributed.primary else {}
        set_model_state_dict(
            model,
            state,
            options=StateDictOptions(
                full_state_dict=True, broadcast_from_rank0=True, strict=True
            ),
        )
        del state
        reset_nonpersistent_buffers(model, distributed.device)
    return model, counts


def full_state(model, distributed: Distributed) -> dict[str, torch.Tensor]:
    """Full FP32 state dict on rank 0 (CPU); empty on other ranks."""
    if not distributed.sharded:
        return {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        }
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    return get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )


def save_resume(
    model,
    optimizer,
    directory: Path,
    trainer_state: dict[str, Any],
    distributed: Distributed,
) -> None:
    """Sharded model and optimizer state (torch.distributed.checkpoint) plus ``trainer.pt``; atomic."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        get_optimizer_state_dict,
    )

    tmp = directory.with_name(directory.name + ".partial")
    if distributed.primary and tmp.exists():
        shutil.rmtree(tmp)
    distributed.barrier()
    state = {
        "model": get_model_state_dict(model),
        "optim": get_optimizer_state_dict(model, optimizer),
    }
    dcp.save(state, checkpoint_id=str(tmp))
    if distributed.primary:
        torch.save(trainer_state, tmp / "trainer.pt")
        if directory.exists():
            shutil.rmtree(directory)
        tmp.replace(directory)
        for old in directory.parent.glob("step-*"):
            if old != directory and not old.name.endswith(".partial"):
                shutil.rmtree(old)
    distributed.barrier()


def load_resume(model, optimizer, directory: Path) -> dict[str, Any]:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        get_optimizer_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
    )

    state = {
        "model": get_model_state_dict(model),
        "optim": get_optimizer_state_dict(model, optimizer),
    }
    dcp.load(state, checkpoint_id=str(directory))
    set_model_state_dict(model, state["model"])
    set_optimizer_state_dict(model, optimizer, state["optim"])
    return torch.load(directory / "trainer.pt", map_location="cpu", weights_only=False)


def latest_resume(root: Path) -> Path | None:
    candidates = sorted(
        p for p in root.glob("step-*") if p.is_dir() and not p.name.endswith(".partial")
    )
    return candidates[-1] if candidates else None
