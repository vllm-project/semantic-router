#!/usr/bin/env python3
"""
Prepare medical domain dataset from MedQuAD.

Dataset: MedQuAD - Medical Question Answering Dataset
Source: https://github.com/abachaa/MedQuAD
Size: 47,457 medical question-answer pairs
License: Public domain (NIH/NLM)
Quality: HIGH - Curated from trusted sources (NIH, CDC, WHO, etc.)

Sources included:
- NIH Senior Health
- GARD (Genetic and Rare Diseases)
- CDC
- FDA
- NIDDK
- NINDS
- And more trusted medical institutions

This script downloads and prepares the questions for cache embedding training.
"""

from datasets import load_dataset
from tqdm import tqdm
import json
from pathlib import Path


def prepare_medquad_dataset(
    output_path: str = "data/cache_embeddings/medical/unlabeled_queries.jsonl",
    max_queries: int = None,
    min_length: int = 10,
    max_length: int = 200,
):
    """
    Prepare MedQuAD dataset for cache embedding training.

    Args:
        output_path: Where to save the queries
        max_queries: Limit number of queries (None = all)
        min_length: Minimum query length in characters
        max_length: Maximum query length in characters
    """
    print("Loading MedQuAD dataset from HuggingFace...")
    print("This dataset contains medical Q&A from NIH, CDC, FDA, and other trusted sources")

    # Load the MedQuAD dataset
    dataset = load_dataset("keivalya/MedQuAD-MedicalQnADataset", split="train")

    print(f"Loaded {len(dataset)} question-answer pairs")

    # Extract unique questions
    queries = []
    seen = set()

    for item in tqdm(dataset, desc="Extracting queries"):
        # Get the question
        question = item.get("Question", "").strip()

        if not question:
            continue

        # Filter by length
        if len(question) < min_length or len(question) > max_length:
            continue

        # Remove duplicates
        if question.lower() in seen:
            continue
        seen.add(question.lower())

        queries.append(question)

        if max_queries and len(queries) >= max_queries:
            break

    print(f"Extracted {len(queries)} unique medical questions")

    # Create output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save queries in JSONL format
    with open(output_file, "w") as f:
        for query in tqdm(queries, desc="Saving"):
            f.write(json.dumps({"query": query}) + "\n")

    print(f"\nSaved {len(queries)} queries to {output_path}")
    print(f"\nDataset statistics:")
    print(f"  Source: MedQuAD (NIH, CDC, FDA, etc.)")
    print(f"  Total queries: {len(queries)}")
    print(f"  Avg length: {sum(len(q) for q in queries) / len(queries):.1f} characters")
    print(f"\nExample queries:")
    for i, query in enumerate(queries[:5], 1):
        print(f"  {i}. {query}")

    print(f"\nNext steps:")
    print(f"  1. Generate training triplets:")
    print(f"     python3 src/training/cache_embeddings/generate_training_data.py \\")
    print(f"       --input {output_path} \\")
    print(f"       --output data/cache_embeddings/medical/triplets.jsonl \\")
    print(f"       --domain medical")
    print(f"\n  2. Train LoRA adapter:")
    print(f"     python3 src/training/cache_embeddings/lora_trainer.py \\")
    print(f"       --train-data data/cache_embeddings/medical/triplets.jsonl \\")
    print(f"       --output models/medical-cache-lora")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare MedQuAD dataset for cache embedding training")
    parser.add_argument(
        "--output",
        default="data/cache_embeddings/medical/unlabeled_queries.jsonl",
        help="Output file path for queries (JSONL format)"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum number of queries to extract (default: all)"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum query length in characters (default: 10)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=200,
        help="Maximum query length in characters (default: 200)"
    )

    args = parser.parse_args()

    prepare_medquad_dataset(
        output_path=args.output,
        max_queries=args.max_queries,
        min_length=args.min_length,
        max_length=args.max_length,
    )
