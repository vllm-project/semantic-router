#!/usr/bin/env python3
"""
Export 2D Matryoshka ONNX models for mmBERT.

Creates separate models for each target layer to enable true early-exit speedup.
Unlike PyTorch, ONNX graphs are static - separate models are needed for speedup.

Usage:
    python export_2d_matryoshka.py --output ./mmbert-2d-matryoshka

Output structure:
    ./mmbert-2d-matryoshka/
    ├── layer-6/
    │   ├── model.onnx          # Fast model (3.3x speedup, 56% quality)
    │   ├── tokenizer.json
    │   └── config.json
    ├── layer-22/
    │   ├── model.onnx          # Full model (100% quality)
    │   ├── tokenizer.json
    │   └── config.json
    └── model_config.json       # Metadata about available layers
"""

import json
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from pathlib import Path
import argparse


class LayerExitModel(nn.Module):
    """Model that outputs hidden state at a specific layer."""

    def __init__(self, base_model, target_layer: int):
        super().__init__()
        self.model = base_model
        self.target_layer = target_layer

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # hidden_states[0] = embeddings, [1] = after layer 1, etc.
        return outputs.hidden_states[self.target_layer]


def export_layer_model(
    base_model,
    tokenizer,
    target_layer: int,
    output_dir: Path,
    opset_version: int = 18,
):
    """Export ONNX model for a specific layer."""

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Exporting layer {target_layer}...")

    # Create layer-specific model
    model = LayerExitModel(base_model, target_layer)
    model.eval()

    # Create dummy input
    test_input = tokenizer("Hello world", return_tensors="pt", padding=True)

    # Test forward pass
    with torch.no_grad():
        output = model(test_input["input_ids"], test_input["attention_mask"])

    # Export to ONNX
    onnx_path = output_dir / "model.onnx"

    torch.onnx.export(
        model,
        (test_input["input_ids"], test_input["attention_mask"]),
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        },
    )

    # Save tokenizer and config
    tokenizer.save_pretrained(str(output_dir))

    # Save layer config
    layer_config = {
        "layer": target_layer,
        "hidden_size": base_model.config.hidden_size,
        "total_layers": base_model.config.num_hidden_layers,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(layer_config, f, indent=2)

    # Get model size
    total_size = sum(f.stat().st_size for f in output_dir.glob("model.onnx*"))
    size_mb = total_size / 1e6

    print(f"    ✓ Exported: {size_mb:.1f} MB")
    return size_mb


def main():
    parser = argparse.ArgumentParser(description="Export 2D Matryoshka ONNX models")
    parser.add_argument(
        "--model",
        default="llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[6, 22],
        help="Layers to export (default: 6 22)",
    )
    parser.add_argument(
        "--output", default="./mmbert-2d-matryoshka", help="Output directory"
    )
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    args = parser.parse_args()

    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("2D Matryoshka ONNX Model Export")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Layers: {args.layers}")
    print(f"Output: {output_base}")

    # Load base model once
    print("\nLoading base model...")
    base_model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model.eval()

    total_layers = base_model.config.num_hidden_layers
    hidden_size = base_model.config.hidden_size

    print(f"  Total layers: {total_layers}")
    print(f"  Hidden size: {hidden_size}")

    # Export each layer
    print("\nExporting models:")
    results = {}

    for layer in args.layers:
        if layer > total_layers:
            print(f"  ✗ Layer {layer} > total layers {total_layers}, skipping")
            continue

        output_dir = output_base / f"layer-{layer}"
        size_mb = export_layer_model(
            base_model,
            tokenizer,
            layer,
            output_dir,
            args.opset,
        )
        results[layer] = {
            "path": str(output_dir / "model.onnx"),
            "size_mb": size_mb,
        }

    # Create master config
    model_config = {
        "model_name": args.model,
        "total_layers": total_layers,
        "hidden_size": hidden_size,
        "available_layers": args.layers,
        "models": results,
        "usage": {
            "layer_6": "Fast inference (3.3x speedup, ~56% quality) - good for routing/classification",
            "layer_22": "Full quality (baseline) - good for search/RAG",
        },
        "expected_speedup": {
            "layer_6": "3.3x faster than layer 22",
            "layer_22": "baseline (1.0x)",
        },
    }

    config_path = output_base / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_base}")
    print("\nModels exported:")
    for layer, info in results.items():
        print(f"  layer-{layer}/model.onnx ({info['size_mb']:.1f} MB)")

    print(f"\nConfig: {config_path}")

    print("\n" + "=" * 60)
    print("Usage in Rust/Go:")
    print("=" * 60)
    print(
        """
// Load models at startup
let fast_model = Session::builder()?
    .with_execution_providers([ROCmExecutionProvider::default().build()])?
    .commit_from_file("mmbert-2d-matryoshka/layer-6/model.onnx")?;

let full_model = Session::builder()?
    .with_execution_providers([ROCmExecutionProvider::default().build()])?
    .commit_from_file("mmbert-2d-matryoshka/layer-22/model.onnx")?;

// At runtime, choose based on latency/quality needs:
let embedding = if need_fast_response {
    fast_model.run(inputs)?   // ~2.6ms, 56% quality
} else {
    full_model.run(inputs)?   // ~11ms, 100% quality
};
"""
    )


if __name__ == "__main__":
    main()
