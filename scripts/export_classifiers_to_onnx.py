#!/usr/bin/env python3
"""
Export merged LoRA classifier models to ONNX format.

Models:
- mmbert32k-intent-classifier: 14-class sequence classification
- mmbert32k-jailbreak-detector: 2-class sequence classification
- mmbert32k-pii-detector: 35-label token classification
- mmbert32k-factcheck-classifier: binary fact-check routing
- mmbert32k-feedback-detector: 4-class satisfaction (user feedback)

Uses optimum for ONNX export with proper handling of ModernBERT architecture.
"""

import os
import argparse
import shutil
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    PreTrainedTokenizerFast,
)
from optimum.onnxruntime import (
    ORTModelForSequenceClassification,
    ORTModelForTokenClassification,
)


def export_sequence_classifier(model_path: str, output_path: str, opset: int = 14):
    """Export a sequence classification model to ONNX."""
    print(f"\n{'='*60}")
    print(f"Exporting: {model_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

    # Load tokenizer (fallback for TokenizersBackend or incompatible tokenizer_config)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except (ValueError, OSError, AttributeError) as e:
        if (
            "TokenizersBackend" in str(e)
            or "does not exist" in str(e)
            or "has no attribute" in str(e)
        ):
            # Load from tokenizer.json only to avoid tokenizer_config issues
            tokenizer_file = Path(model_path) / "tokenizer.json"
            if tokenizer_file.exists():
                tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
            else:
                raise
        else:
            raise

    # Load model
    print("Loading PyTorch model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    )
    model.eval()

    # Get model info
    config = model.config
    print(f"  Architecture: {config.architectures}")
    print(f"  Num labels: {config.num_labels}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Export to ONNX using optimum
    print("Exporting to ONNX...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        model_path,
        export=True,
    )

    # Save ONNX model
    ort_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Copy label mappings if they exist
    for mapping_file in [
        "label_mapping.json",
        "category_mapping.json",
        "jailbreak_type_mapping.json",
        "fact_check_mapping.json",
    ]:
        src = Path(model_path) / mapping_file
        if src.exists():
            shutil.copy(src, Path(output_path) / mapping_file)
            print(f"  Copied {mapping_file}")

    # Verify the exported model
    print("Verifying ONNX model...")
    ort_model_loaded = ORTModelForSequenceClassification.from_pretrained(output_path)

    # Test inference
    test_text = "This is a test sentence for verification."
    inputs = tokenizer(
        test_text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    with torch.no_grad():
        pt_outputs = model(**inputs)

    ort_outputs = ort_model_loaded(**inputs)

    # Compare outputs
    pt_logits = pt_outputs.logits.numpy()
    ort_logits = ort_outputs.logits.numpy()

    diff = abs(pt_logits - ort_logits).max()
    print(f"  Max logit difference: {diff:.6f}")

    if diff < 1e-4:
        print("  ✓ ONNX model verified successfully!")
    else:
        print(f"  ⚠ Warning: Logit difference {diff} is larger than expected")

    # Print ONNX file size
    onnx_path = Path(output_path) / "model.onnx"
    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"  ONNX model size: {size_mb:.1f} MB")

    return output_path


def export_token_classifier(model_path: str, output_path: str, opset: int = 14):
    """Export a token classification model to ONNX."""
    print(f"\n{'='*60}")
    print(f"Exporting: {model_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model
    print("Loading PyTorch model...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    )
    model.eval()

    # Get model info
    config = model.config
    print(f"  Architecture: {config.architectures}")
    print(f"  Num labels: {config.num_labels}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Export to ONNX using optimum
    print("Exporting to ONNX...")
    ort_model = ORTModelForTokenClassification.from_pretrained(
        model_path,
        export=True,
    )

    # Save ONNX model
    ort_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Copy label mappings if they exist
    for mapping_file in ["label_mapping.json"]:
        src = Path(model_path) / mapping_file
        if src.exists():
            shutil.copy(src, Path(output_path) / mapping_file)
            print(f"  Copied {mapping_file}")

    # Verify the exported model
    print("Verifying ONNX model...")
    ort_model_loaded = ORTModelForTokenClassification.from_pretrained(output_path)

    # Test inference
    test_text = "John Smith's email is john@example.com and SSN is 123-45-6789."
    inputs = tokenizer(
        test_text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    with torch.no_grad():
        pt_outputs = model(**inputs)

    ort_outputs = ort_model_loaded(**inputs)

    # Compare outputs
    pt_logits = pt_outputs.logits.numpy()
    ort_logits = ort_outputs.logits.numpy()

    diff = abs(pt_logits - ort_logits).max()
    print(f"  Max logit difference: {diff:.6f}")

    if diff < 1e-4:
        print("  ✓ ONNX model verified successfully!")
    else:
        print(f"  ⚠ Warning: Logit difference {diff} is larger than expected")

    # Print ONNX file size
    onnx_path = Path(output_path) / "model.onnx"
    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"  ONNX model size: {size_mb:.1f} MB")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export classifier models to ONNX")
    parser.add_argument(
        "--model",
        choices=["intent", "jailbreak", "pii", "factcheck", "feedback", "all"],
        default="all",
        help="Which model to export",
    )
    parser.add_argument("--output-dir", default=".", help="Base output directory")
    args = parser.parse_args()

    base_dir = Path(args.output_dir)

    models = {
        "intent": {
            "input": "mmbert32k-intent-classifier-merged-r32",
            "output": "mmbert32k-intent-classifier-onnx",
            "type": "sequence",
        },
        "jailbreak": {
            "input": "mmbert32k-jailbreak-detector-merged-r32",
            "output": "mmbert32k-jailbreak-detector-onnx",
            "type": "sequence",
        },
        "pii": {
            "input": "mmbert32k-pii-detector-merged-r32",
            "output": "mmbert32k-pii-detector-onnx",
            "type": "token",
        },
        "factcheck": {
            "input": "mmbert32k-factcheck-classifier-merged",
            "output": "mmbert32k-factcheck-classifier-merged-onnx",
            "type": "sequence",
        },
        "feedback": {
            "input": "mmbert32k-feedback-detector-merged",
            "output": "mmbert32k-feedback-detector-merged-onnx",
            "type": "sequence",
        },
    }

    to_export = (
        [args.model]
        if args.model != "all"
        else ["intent", "jailbreak", "pii", "factcheck", "feedback"]
    )

    for model_name in to_export:
        model_info = models[model_name]
        input_path = base_dir / model_info["input"]
        output_path = base_dir / model_info["output"]

        if not input_path.exists():
            print(f"⚠ Skipping {model_name}: {input_path} not found")
            continue

        if model_info["type"] == "sequence":
            export_sequence_classifier(str(input_path), str(output_path))
        else:
            export_token_classifier(str(input_path), str(output_path))

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
