#!/usr/bin/env python3
"""
Generate Qwen3 official reference embeddings for validating Rust implementation

This script uses the official Transformers library to generate reference embeddings
from the Qwen3-Embedding-0.6B model, which will be compared against our Rust
implementation to ensure numerical consistency.

Usage:
    python scripts/generate_qwen3_reference.py
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Official Last Token Pooling implementation from Qwen3-Embedding

    Reference: https://github.com/qwenlm/qwen3-embedding

    Args:
        last_hidden_states: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]

    Returns:
        pooled: [batch_size, hidden_size]
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        # For left padding, the last token is always at position -1
        return last_hidden_states[:, -1]
    else:
        # For right padding, find the actual last token position
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    """
    Official instruction template for task-specific embeddings

    Reference: https://github.com/qwenlm/qwen3-embedding

    Args:
        task_description: The task instruction
        query: The query text

    Returns:
        formatted_text: The formatted instruction + query
    """
    return f"Instruct: {task_description}\nQuery: {query}"


def main():
    print("=" * 80)
    print("Qwen3-Embedding Reference Generation Script")
    print("=" * 80)

    # Model path (relative to project root)
    # Script should be run from project root: python scripts/generate_qwen3_reference.py
    model_path = Path("models/mom-embedding-pro")

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("\nPlease ensure:")
        print("  1. The model has been downloaded:")
        print("     cd models")
        print(
            "     huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir Qwen3-Embedding-0.6B"
        )
        print("  2. Run this script from the project root directory:")
        print("     python scripts/generate_qwen3_reference.py")
        sys.exit(1)

    print(f"Model path: {model_path.absolute()}")

    # Test cases
    test_cases = [
        {
            "name": "short_text_no_instruction",
            "text": "What is deep learning?",
            "instruction": None,
        },
        {
            "name": "short_text_with_instruction",
            "text": "What is the capital of China?",
            "instruction": "Given a web search query, retrieve relevant passages that answer the query",
        },
        {
            "name": "medium_text",
            "text": "Artificial intelligence is a field of computer science that aims to create intelligent machines that work and react like humans. "
            * 10,
            "instruction": None,
        },
        {
            "name": "long_text",
            "text": "A" * 5000,  # ~5000 characters, should result in ~1000+ tokens
            "instruction": None,
        },
    ]

    print(f"\nTest cases defined: {len(test_cases)}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        padding_side="left",  # CRITICAL: must be left for Last Token Pooling
        trust_remote_code=True,
    )
    print(f"  Tokenizer loaded. Padding side: {tokenizer.padding_side}")

    # Load model
    print("\nLoading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    if device.type == "cuda":
        print("  Note: Using GPU with Flash Attention 2 (if available)")
        model = AutoModel.from_pretrained(
            str(model_path),
            attn_implementation="flash_attention_2",  # Official recommendation
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(device)
    else:
        print("  Note: Using CPU (slower, no Flash Attention)")
        model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True).to(
            device
        )

    model.eval()
    print("  Model loaded successfully")

    # Generate embeddings
    print("\n" + "=" * 80)
    print("Generating reference embeddings...")
    print("=" * 80)

    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Processing: {case['name']}")

        # Prepare text
        text = case["text"]
        if case["instruction"]:
            text = get_detailed_instruct(case["instruction"], text)
            print(f"  Instruction applied: {case['instruction'][:50]}...")

        # Tokenize
        print(f"  Original text length: {len(case['text'])} chars")
        inputs = tokenizer(
            [text],
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=32768,  # Qwen3 max length
        ).to(device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        seq_len = attention_mask.sum().item()

        print(f"  Tokenized length: {seq_len} tokens")
        print(f"  Input shape: {list(input_ids.shape)}")

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state

            # Apply Last Token Pooling
            embedding = last_token_pool(last_hidden_state, attention_mask)

            # L2 Normalization (official implementation does this)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        print(f"  Embedding shape: {list(embedding.shape)}")
        print(f"  Embedding norm: {embedding.norm().item():.6f} (should be ~1.0)")

        # Convert to list
        embedding_list = embedding[0].cpu().float().numpy().tolist()

        # Convert input_ids and attention_mask to lists for Rust consumption
        input_ids_list = input_ids[0].cpu().numpy().tolist()
        attention_mask_list = attention_mask[0].cpu().numpy().tolist()

        # Store result
        results.append(
            {
                "name": case["name"],
                "input": {
                    "text": (
                        case["text"][:100] + "..."
                        if len(case["text"]) > 100
                        else case["text"]
                    ),
                    "full_text_length": len(case["text"]),
                    "instruction": case["instruction"],
                },
                "tokenization": {
                    "seq_len": int(seq_len),
                    "input_shape": list(input_ids.shape),
                    "input_ids": input_ids_list,
                    "attention_mask": attention_mask_list,
                },
                "embedding": embedding_list,
                "embedding_shape": list(embedding.shape),
                "embedding_dim": embedding.shape[1],
            }
        )

        print(f"  Result stored. Embedding dimension: {embedding.shape[1]}")

    # Save results
    output_path = Path("candle-binding/test_data/qwen3_reference_outputs.json")
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
        print(
            f"  {result['name']:<30} | Tokens: {result['tokenization']['seq_len']:>5} | Dim: {result['embedding_dim']}"
        )

    print("\n" + "=" * 80)
    print("Reference generation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
