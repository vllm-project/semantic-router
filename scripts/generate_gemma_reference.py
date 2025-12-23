#!/usr/bin/env python3
"""
Generate EmbeddingGemma official reference embeddings for validating Rust implementation

This script uses sentence-transformers to generate reference embeddings
from the EmbeddingGemma-300M model, which includes the complete pipeline:
  1. Gemma3 Transformer
  2. Mean Pooling
  3. Dense Bottleneck (768 → 3072 → 768)
  4. L2 Normalization

Key differences from Qwen3:
- Uses Mean Pooling (not Last Token Pooling)
- Has Dense Bottleneck (768 → 3072 → 768)
- Supports Matryoshka Representation (768/512/256/128)

Note: We use sentence-transformers to ensure we get the complete model
with Dense Bottleneck, and also extract tokenization details for Rust testing.

Usage:
    python scripts/generate_gemma_reference.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


def mean_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Official Mean Pooling implementation for EmbeddingGemma

    Reference: https://huggingface.co/google/embeddinggemma-300m

    Args:
        last_hidden_states: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]

    Returns:
        pooled: [batch_size, hidden_size]
    """
    # Expand attention mask to match hidden states dimensions
    # attention_mask: [batch, seq_len] -> [batch, seq_len, hidden_size]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    )

    # Sum embeddings weighted by attention mask
    sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, dim=1)

    # Sum attention mask to get actual token counts
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    # Mean = sum / count
    return sum_embeddings / sum_mask


def truncate_and_renormalize(embeddings: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Matryoshka Representation: Truncate embeddings and re-normalize

    Args:
        embeddings: [batch_size, 768]
        target_dim: 768, 512, 256, or 128

    Returns:
        truncated: [batch_size, target_dim] with L2 norm = 1.0
    """
    # Truncate to target dimension
    truncated = embeddings[:, :target_dim]

    # Re-normalize to L2 norm = 1.0
    norm = np.linalg.norm(truncated, axis=1, keepdims=True)
    normalized = truncated / norm

    return normalized


def main():
    print("=" * 80)
    print("EmbeddingGemma Reference Generation Script")
    print("=" * 80)

    # Model path (relative to project root)
    # Script should be run from project root: python scripts/generate_gemma_reference.py
    model_path = Path("models/mom-embedding-flash")

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("\nPlease ensure:")
        print("  1. The model has been downloaded:")
        print("     cd models")
        print(
            "     huggingface-cli download google/embeddinggemma-300m --local-dir embeddinggemma-300m"
        )
        print("  2. Run this script from the project root directory:")
        print("     python scripts/generate_gemma_reference.py")
        sys.exit(1)

    print(f"Model path: {model_path.absolute()}")

    # Test cases
    test_cases = [
        {
            "name": "short_text",
            "text": "What is deep learning?",
        },
        {
            "name": "medium_text",
            "text": "Artificial intelligence is a field of computer science that aims to create intelligent machines that work and react like humans. "
            * 5,
        },
        {
            "name": "long_text",
            "text": "Deep learning is a subset of machine learning that uses neural networks with multiple layers. "
            * 20,
        },
        {
            "name": "batch_test_1",
            "text": "The quick brown fox jumps over the lazy dog.",
        },
        {
            "name": "batch_test_2",
            "text": "Machine learning models can learn patterns from data.",
        },
    ]

    print(f"\nTest cases defined: {len(test_cases)}")

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    # Load tokenizer (for extracting input_ids and attention_mask)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    print("  Tokenizer loaded successfully")

    # Load SentenceTransformer model (includes Transformer + Pooling + Dense + Normalize)
    # CRITICAL: Use EAGER attention to match Rust implementation!
    model = SentenceTransformer(
        str(model_path),
        device=str(device),
        model_kwargs={"attn_implementation": "eager"},
    )
    print("  Model loaded successfully")
    print(f"  Model type: {type(model)}")
    print(f"  Model modules: {[type(m).__name__ for m in model._modules.values()]}")

    # Get config from the underlying transformer
    transformer_model = model._modules["0"].auto_model
    print(
        f"  Max position embeddings: {transformer_model.config.max_position_embeddings}"
    )
    print(
        f"  Attention implementation: {transformer_model.config._attn_implementation} (should be 'eager')"
    )

    # Generate embeddings
    print("\n" + "=" * 80)
    print("Generating reference embeddings...")
    print("=" * 80)

    results = []

    # Matryoshka dimensions to test
    matryoshka_dims = [768, 512, 256, 128]

    for i, case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Processing: {case['name']}")
        print(f"  Original text length: {len(case['text'])} chars")

        # Tokenize (for extracting input_ids and attention_mask)
        tokenized = tokenizer(
            [case["text"]],
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=2048,  # EmbeddingGemma max length
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        seq_len = attention_mask.sum().item()

        print(f"  Tokenized length: {seq_len} tokens")
        print(f"  Input shape: {list(input_ids.shape)}")

        # Forward pass using SentenceTransformer
        # This applies the complete pipeline:
        # 1. Gemma3 Transformer (with embedding scaling)
        # 2. Mean Pooling
        # 3. Dense Bottleneck (768 → 3072 → 768)
        # 4. L2 Normalization
        with torch.no_grad():
            embeddings = model.encode(
                [case["text"]],
                convert_to_tensor=True,
                normalize_embeddings=True,  # Ensure L2 normalization
                batch_size=1,
            )

        print(f"  Embedding shape: {list(embeddings.shape)}")
        print(f"  Embedding norm: {embeddings.norm().item():.6f} (should be ~1.0)")

        # Convert to numpy for processing
        embeddings_np = embeddings[0].cpu().float().numpy()

        # Generate Matryoshka variants
        matryoshka_embeddings = {}
        for dim in matryoshka_dims:
            if dim == 768:
                # Full dimension, no truncation
                matryoshka_embeddings[dim] = embeddings_np.tolist()
            else:
                # Truncate and re-normalize
                truncated = truncate_and_renormalize(embeddings_np.reshape(1, -1), dim)
                matryoshka_embeddings[dim] = truncated[0].tolist()
                print(
                    f"  Matryoshka {dim}-dim norm: {np.linalg.norm(truncated[0]):.6f}"
                )

        # Convert input_ids and attention_mask to lists for Rust consumption
        input_ids_list = input_ids[0].cpu().numpy().tolist()
        attention_mask_list = attention_mask[0].cpu().numpy().tolist()

        # Store result
        result = {
            "name": case["name"],
            "input": {
                "text": (
                    case["text"][:100] + "..."
                    if len(case["text"]) > 100
                    else case["text"]
                ),
                "full_text_length": len(case["text"]),
            },
            "tokenization": {
                "seq_len": int(seq_len),
                "input_shape": list(input_ids.shape),
                "input_ids": input_ids_list,
                "attention_mask": attention_mask_list,
            },
            "embedding_full": matryoshka_embeddings[768],
            "embedding_shape": [1, 768],
            "embedding_dim": 768,
            "matryoshka": {
                str(dim): matryoshka_embeddings[dim] for dim in matryoshka_dims
            },
        }

        results.append(result)
        print(f"  Result stored with {len(matryoshka_dims)} Matryoshka variants")

    # Batch processing test
    print("\n" + "=" * 80)
    print("Testing batch processing...")
    print("=" * 80)

    batch_texts = [case["text"] for case in test_cases[:2]]  # Use first 2 cases
    print(f"  Batch size: {len(batch_texts)}")

    try:
        # Tokenize batch (for extracting input_ids and attention_mask)
        batch_tokenized = tokenizer(
            batch_texts,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        print(f"  Batch input shape: {list(batch_tokenized['input_ids'].shape)}")

        # Forward pass using SentenceTransformer
        with torch.no_grad():
            batch_embeddings = model.encode(
                batch_texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                batch_size=len(batch_texts),
            )

        if batch_embeddings is not None:
            print(f"  Batch embeddings shape: {list(batch_embeddings.shape)}")

            # Convert to lists
            batch_input_ids = batch_tokenized["input_ids"].cpu().numpy().tolist()
            batch_attention_mask = (
                batch_tokenized["attention_mask"].cpu().numpy().tolist()
            )
            batch_embeddings_list = batch_embeddings.cpu().float().numpy().tolist()

            # Store batch result
            batch_result = {
                "name": "batch_processing_test",
                "input": {
                    "texts": [
                        t[:50] + "..." if len(t) > 50 else t for t in batch_texts
                    ],
                    "batch_size": len(batch_texts),
                },
                "tokenization": {
                    "input_ids": batch_input_ids,
                    "attention_mask": batch_attention_mask,
                },
                "embeddings": batch_embeddings_list,
                "embedding_shape": list(batch_embeddings.shape),
            }
            results.append(batch_result)
            print("  Batch result stored")
    except Exception as e:
        print(f"  Batch processing failed: {e}")
        import traceback

        traceback.print_exc()

    # Save results
    output_path = Path("candle-binding/test_data/gemma_reference_outputs.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    print("\n" + "=" * 80)
    print(f"Saving results to: {output_path}")
    print("=" * 80)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} reference embeddings")
    print(f"File size: {output_path.stat().st_size / 1024:.2f} KB")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for result in results:
        if result["name"] == "batch_processing_test":
            print(
                f"  {result['name']:<30} | Batch: {result['input']['batch_size']} | Dim: 768"
            )
        else:
            print(
                f"  {result['name']:<30} | Chars: {result['input']['full_text_length']:>5} | Matryoshka: 4 dims"
            )

    print("\n" + "=" * 80)
    print("Reference generation completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Implement Rust validation test: gemma_validation_test.rs")
    print("  2. Compare Rust output with these reference embeddings")
    print("  3. Verify cosine similarity > 0.99")


if __name__ == "__main__":
    main()
