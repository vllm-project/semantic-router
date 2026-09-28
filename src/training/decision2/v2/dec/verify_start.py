"""Load candidate start weights through the shared loaders and count parameters.

For a Decision 1.0 package the native output is its typed candidate head; for
an official Qwen Base or general checkpoint the text backbone is loaded and a
new, untrained candidate head is attached (the official native output is text
generation, not a typed decision). Writes one JSON receipt per start.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from training.model.data import canonical
from training.model.source import source_fingerprint
from training.model.train import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument(
        "--kind", choices=("decision1", "base", "posttrained"), required=True
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from training.model.decision_model import DecisionModel

    if args.kind == "decision1":
        model, _ = DecisionModel.from_decision1(args.path, 256)
        native = json.loads((args.path / "decision_config.json").read_text())
        native_output = "typed candidate head (Choice/Noul/Score probabilities)"
    else:
        model, _ = DecisionModel.from_base(
            args.path, args.revision, 256, source_stage=args.kind
        )
        native = json.loads((args.path / "config.json").read_text())
        native_output = "generative text/vision model; decision head newly attached"
    text = sum(p.numel() for p in model.backbone.parameters())
    head = sum(p.numel() for p in model.head.parameters())
    config = model.backbone.config
    source = source_fingerprint(args.path)
    atomic_json(
        args.output,
        {
            "repo": args.repo,
            "revision": args.revision,
            "kind": args.kind,
            "source_sha256": hashlib.sha256(
                canonical(source["files_sha256"]).encode("utf-8")
            ).hexdigest(),
            "text_parameters": text,
            "head_parameters": head,
            "loaded_decision_parameters": text + head,
            "head_trained": args.kind == "decision1",
            "hidden_size": config.hidden_size,
            "layers": config.num_hidden_layers,
            "native_output": native_output,
            "native_architectures": native.get("architectures")
            or native.get("architecture"),
        },
    )
    print(json.dumps({"repo": args.repo, "text": text, "head": head}), flush=True)


if __name__ == "__main__":
    main()
