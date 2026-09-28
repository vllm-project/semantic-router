"""Checkpoint persistence shared by the compact training loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def unwrap_training_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying multimodal embedder."""
    unwrapped = model.module if hasattr(model, "module") else model
    return unwrapped.model if hasattr(unwrapped, "model") else unwrapped


def save_model(model: torch.nn.Module, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    unwrap_training_model(model).save_pretrained(str(destination))


def save_epoch_checkpoint(
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    output_dir: str | Path,
    state: dict[str, Any],
) -> None:
    destination = Path(output_dir)
    save_model(model, destination)
    torch.save(
        {
            **state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        },
        destination / "training_state.pt",
    )


def save_batch_checkpoint(
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    output_dir: str | Path,
    state: dict[str, Any],
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **state,
            "model_state_dict": unwrap_training_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        },
        destination / "batch_checkpoint.pt",
    )


def write_metrics(output_dir: str | Path, metrics: dict[str, Any]) -> None:
    path = Path(output_dir) / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
