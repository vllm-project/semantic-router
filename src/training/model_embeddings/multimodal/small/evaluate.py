"""Evaluate a compact multimodal checkpoint on cached paired tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()

    import torch  # noqa: PLC0415 - optional runtime dependency
    from torch.utils.data import DataLoader  # noqa: PLC0415 - optional dependency

    from .data import (  # noqa: PLC0415 - optional runtime dependency
        CachedAudioDataset,
        CachedTensorDataset,
        audio_cached_collate_fn,
        cached_collate_fn,
    )
    from .stages import (  # noqa: PLC0415 - optional runtime dependency
        create_model,
        load_weights,
    )
    from .training import evaluate  # noqa: PLC0415 - optional runtime dependency
    from .wrappers import (  # noqa: PLC0415 - optional runtime dependency
        AudioContrastiveWrapper,
        ContrastiveTrainingWrapper,
    )

    config = load_config(args.config)
    feature_key = config["data"].get("feature_key", "pixel_values")
    audio = feature_key == "input_features"
    dataset_type = CachedAudioDataset if audio else CachedTensorDataset
    collate = audio_cached_collate_fn if audio else cached_collate_fn
    dataset = dataset_type(
        args.cache,
        shuffle_shards=False,
        dynamic_discovery=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        collate_fn=collate,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(config["model"])
    load_weights(model, args.checkpoint)
    wrapper_type = AudioContrastiveWrapper if audio else ContrastiveTrainingWrapper
    wrapped = wrapper_type(model).to(device)
    metrics = evaluate(wrapped, loader, device, max_batches=args.max_batches)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
