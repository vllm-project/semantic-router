#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as handle:
        if config_path.endswith((".yaml", ".yml")):
            import yaml  # noqa: PLC0415 - keep CLI help import-light

            return yaml.safe_load(handle)
        return json.load(handle)


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.config:
        return load_config(args.config)

    if args.final_dir:
        config_path = os.path.join(args.final_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"No config.json found under final dir {args.final_dir}. Pass --config explicitly for checkpoint eval."
            )
        return load_config(config_path)

    raise ValueError("--config is required when evaluating a checkpoint directory.")


def load_final_weights(model: torch.nn.Module, final_dir: str) -> None:
    import torch  # noqa: PLC0415 - optional runtime dependency

    model_path = os.path.join(final_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Final model weights not found: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)


def load_checkpoint_weights(model: torch.nn.Module, checkpoint_dir: str) -> None:
    import torch  # noqa: PLC0415 - optional runtime dependency
    from safetensors.torch import (  # noqa: PLC0415 - optional runtime dependency
        load_file as safetensors_load_file,
    )

    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    pytorch_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        state_dict = safetensors_load_file(safetensors_path, device="cpu")
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No model weights found under {checkpoint_dir}. Expected model.safetensors or pytorch_model.bin."
        )
    model.load_state_dict(state_dict)


def build_eval_loader(
    cfg: dict[str, Any], max_samples: int | None = None
) -> DataLoader:
    from torch.utils.data import (  # noqa: PLC0415 - optional runtime dependency
        DataLoader,
        Subset,
    )

    from .data import (  # noqa: PLC0415 - optional runtime dependency
        CachedShardDataset,
        collate_records,
    )
    from .tri_encoder import (  # noqa: PLC0415 - optional runtime dependency
        _build_cached_loader_kwargs,
    )

    validation_cfg = cfg.get("validation", {})
    cache_dir = validation_cfg.get("cache_dir")
    if not cache_dir:
        raise ValueError("validation.cache_dir is required for tri-encoder evaluation.")

    eval_dataset = CachedShardDataset(
        cache_dir,
        shard_cache_limit=int(validation_cfg.get("shard_cache_limit", 2)),
        prefetch_shards=int(validation_cfg.get("shard_prefetch", 1)),
    )
    if max_samples is not None:
        eval_dataset = Subset(eval_dataset, range(min(max_samples, len(eval_dataset))))
    training_cfg = cfg.get("training", {})
    return DataLoader(
        eval_dataset,
        batch_size=int(
            validation_cfg.get("batch_size", training_cfg.get("batch_size", 1))
        ),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_records,
        **_build_cached_loader_kwargs(
            int(
                validation_cfg.get(
                    "num_workers", max(1, int(training_cfg.get("num_workers", 4)) // 2)
                )
            ),
            int(training_cfg.get("prefetch_factor", 4)),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate datacenter tri-encoder checkpoint or final model"
    )
    parser.add_argument(
        "--config", default=None, help="Path to training config YAML or JSON"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Accelerate checkpoint directory, e.g. checkpoint-2000",
    )
    parser.add_argument(
        "--final-dir",
        default=None,
        help="Final export directory containing model.pt and config.json",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optionally evaluate only the first N validation samples",
    )
    args = parser.parse_args()

    if bool(args.checkpoint_dir) == bool(args.final_dir):
        raise ValueError("Provide exactly one of --checkpoint-dir or --final-dir.")

    cfg = resolve_config(args)
    from accelerate import Accelerator  # noqa: PLC0415 - optional runtime dependency

    from .runtime import (  # noqa: PLC0415 - optional runtime dependency
        normalize_mixed_precision,
    )
    from .tri_encoder import (  # noqa: PLC0415 - optional runtime dependency
        build_datacenter_tri_encoder_model,
        evaluate_tri_encoder_model,
    )

    model = build_datacenter_tri_encoder_model(cfg)

    if args.final_dir:
        load_final_weights(model, args.final_dir)
    else:
        load_checkpoint_weights(model, args.checkpoint_dir)

    accelerator = Accelerator(
        mixed_precision=normalize_mixed_precision(
            cfg.get("training", {}).get("mixed_precision", "bf16")
        )
    )
    eval_loader = build_eval_loader(cfg, max_samples=args.max_samples)
    model, eval_loader = accelerator.prepare(model, eval_loader)

    metrics = evaluate_tri_encoder_model(
        model,
        eval_loader,
        accelerator,
        float(cfg.get("loss", {}).get("scale", 20.0)),
    )
    if accelerator.is_main_process:
        payload = {
            "mode": "final" if args.final_dir else "checkpoint",
            "path": args.final_dir or args.checkpoint_dir,
            **metrics,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
