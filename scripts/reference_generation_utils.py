#!/usr/bin/env python3
"""
Common utilities for generating reference embeddings.

This module provides shared functionality for generating reference embeddings
from different model architectures to validate Rust implementations.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def save_reference_file(
    output_path: Path,
    model_name: str,
    model_id: str,
    results: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Save reference embeddings to JSON file with consistent formatting.

    Args:
        output_path: Path to output JSON file
        model_name: Human-readable model name
        model_id: Model identifier (e.g., Hugging Face model ID)
        results: List of embedding results
        metadata: Additional metadata to include
    """
    output = {
        "model_name": model_name,
        "model_id": model_id,
        "num_test_cases": len(results),
        "test_cases": results,
    }

    if metadata:
        output["metadata"] = metadata

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"✅ Reference file saved: {output_path}")
    print(f"{'=' * 80}")
    print(f"  Model: {model_name}")
    print(f"  Model ID: {model_id}")
    print(f"  Test cases: {len(results)}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")


def prepare_device(force_cpu: bool = False) -> torch.device:
    """
    Prepare computation device (GPU if available, otherwise CPU).

    Args:
        force_cpu: Force CPU usage even if GPU is available

    Returns:
        torch.device for computation
    """
    if force_cpu:
        device = torch.device("cpu")
        print("  Device: CPU (forced)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Device: CUDA ({torch.cuda.get_device_name(0)})")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("  Device: CPU (no CUDA available)")

    return device


def convert_tensors_to_lists(
    input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> tuple:
    """
    Convert input tensors to Python lists for JSON serialization.

    Args:
        input_ids: Input token IDs tensor
        attention_mask: Attention mask tensor

    Returns:
        Tuple of (input_ids_list, attention_mask_list, seq_len)
    """
    input_ids_list = input_ids[0].cpu().numpy().tolist()
    attention_mask_list = attention_mask[0].cpu().numpy().tolist()
    seq_len = int(attention_mask.sum().item())

    return input_ids_list, attention_mask_list, seq_len


def format_test_case_result(
    case_name: str,
    text: str,
    input_ids_list: List[int],
    attention_mask_list: List[int],
    seq_len: int,
    embedding: np.ndarray,
    additional_fields: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Format a test case result into standard structure.

    Args:
        case_name: Name/identifier for the test case
        text: Input text
        input_ids_list: Token IDs as list
        attention_mask_list: Attention mask as list
        seq_len: Sequence length
        embedding: Embedding vector
        additional_fields: Any additional fields to include

    Returns:
        Formatted result dictionary
    """
    result = {
        "name": case_name,
        "input": {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "full_text_length": len(text),
        },
        "tokenization": {
            "seq_len": seq_len,
            "input_shape": [1, len(input_ids_list)],
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
        },
        "embedding": embedding.tolist(),
        "embedding_shape": list(embedding.shape),
        "embedding_dim": embedding.shape[-1],
    }

    if additional_fields:
        result.update(additional_fields)

    return result


def print_embedding_stats(embedding: torch.Tensor, step: int, total: int, name: str):
    """
    Print statistics about generated embeddings.

    Args:
        embedding: Generated embedding tensor
        step: Current step number
        total: Total number of steps
        name: Name of the test case
    """
    print(f"\n[{step}/{total}] Processing: {name}")
    print(f"  Embedding shape: {list(embedding.shape)}")
    print(f"  Embedding norm: {embedding.norm().item():.6f} (should be ~1.0)")


def setup_output_directory(output_file: str) -> Path:
    """
    Set up output directory and return output path.

    Args:
        output_file: Output filename

    Returns:
        Path object for output file
    """
    # Create output directory
    output_dir = Path("candle-binding/reference")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / output_file
    print(f"  Output file: {output_path}")

    return output_path
