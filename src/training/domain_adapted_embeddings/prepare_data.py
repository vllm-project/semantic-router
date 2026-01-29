#!/usr/bin/env python3
"""
Prepare Training Data for Domain-Adapted Embedding Fine-Tuning

Supports two input formats:
1. JSON/JSONL files with Q&A pairs
2. HuggingFace datasets

Output format:
- corpus_chunks.pkl: List of document chunks
- train_queries.pkl: Training queries with ground-truth chunk IDs
- test_queries.pkl: Test queries with ground-truth chunk IDs

Usage:
    # From HuggingFace dataset
    python prepare_data.py --source huggingface --dataset keivalya/MedQuad-MedicalQnADataset

    # From JSON file
    python prepare_data.py --source json --input-file data/qa_pairs.json

    # From JSONL file
    python prepare_data.py --source jsonl --input-file data/qa_pairs.jsonl
"""

import argparse
import json
import os
import pickle
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_from_huggingface(dataset_name: str, split: str = "train") -> List[Dict]:
    """
    Load Q&A pairs from a HuggingFace dataset.

    Expected format: Each item should have 'Question' and 'Answer' fields
    (or 'question' and 'answer').
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)

    # Get the appropriate split
    if split in dataset:
        data_split = dataset[split]
    elif "train" in dataset:
        data_split = dataset["train"]
    else:
        # Use first available split
        first_split = list(dataset.keys())[0]
        print(f"Using split: {first_split}")
        data_split = dataset[first_split]

    qa_pairs = []
    for idx, item in enumerate(tqdm(data_split, desc="Loading")):
        # Handle different key names
        question = item.get("Question") or item.get("question") or ""
        answer = item.get("Answer") or item.get("answer") or ""

        if question and answer:
            qa_pairs.append(
                {
                    "id": idx,
                    "question": question,
                    "answer": answer,
                }
            )

    print(f"Loaded {len(qa_pairs)} Q&A pairs")
    return qa_pairs


def load_from_json(input_file: str) -> List[Dict]:
    """
    Load Q&A pairs from a JSON file.

    Expected format:
    [
        {"question": "...", "answer": "..."},
        ...
    ]
    """
    print(f"Loading from JSON: {input_file}")
    with open(input_file, "r") as f:
        data = json.load(f)

    qa_pairs = []
    for idx, item in enumerate(data):
        question = item.get("question") or item.get("Question") or ""
        answer = item.get("answer") or item.get("Answer") or ""

        if question and answer:
            qa_pairs.append(
                {
                    "id": idx,
                    "question": question,
                    "answer": answer,
                }
            )

    print(f"Loaded {len(qa_pairs)} Q&A pairs")
    return qa_pairs


def load_from_jsonl(input_file: str) -> List[Dict]:
    """
    Load Q&A pairs from a JSONL file (one JSON object per line).

    Expected format:
    {"question": "...", "answer": "..."}
    {"question": "...", "answer": "..."}
    """
    print(f"Loading from JSONL: {input_file}")
    qa_pairs = []

    with open(input_file, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            question = item.get("question") or item.get("Question") or ""
            answer = item.get("answer") or item.get("Answer") or ""

            if question and answer:
                qa_pairs.append(
                    {
                        "id": idx,
                        "question": question,
                        "answer": answer,
                    }
                )

    print(f"Loaded {len(qa_pairs)} Q&A pairs")
    return qa_pairs


def chunk_text(text: str, max_length: int = 1000) -> List[str]:
    """
    Split long text into chunks respecting sentence boundaries.

    Args:
        text: Text to chunk
        max_length: Maximum characters per chunk

    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]

    # Simple sentence splitting
    sentences = (
        text.replace("! ", "!|").replace("? ", "?|").replace(". ", ".|").split("|")
    )

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


def prepare_data(
    qa_pairs: List[Dict],
    output_dir: str,
    test_size: float = 0.2,
    chunk_size: int = 1000,
) -> Tuple[int, int, int]:
    """
    Prepare training data from Q&A pairs.

    Args:
        qa_pairs: List of Q&A dicts with 'question' and 'answer' keys
        output_dir: Directory to save output files
        test_size: Fraction of data to use for testing
        chunk_size: Maximum characters per chunk

    Returns:
        Tuple of (num_chunks, num_train, num_test)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create corpus chunks from answers
    print("\nCreating corpus chunks...")
    corpus_chunks = []
    qa_id_to_chunk_ids = {}

    for qa in tqdm(qa_pairs, desc="Chunking"):
        answer = qa["answer"]
        qa_id = qa["id"]

        # Chunk long answers
        chunks = chunk_text(answer, max_length=chunk_size)

        qa_id_to_chunk_ids[qa_id] = []
        for chunk_text_str in chunks:
            chunk_id = len(corpus_chunks)
            corpus_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text_str,
                    "qa_id": qa_id,
                }
            )
            qa_id_to_chunk_ids[qa_id].append(chunk_id)

    print(f"Created {len(corpus_chunks)} corpus chunks")

    # Split into train/test
    print("\nSplitting data...")
    train_qas, test_qas = train_test_split(
        qa_pairs, test_size=test_size, random_state=42
    )
    print(f"Train: {len(train_qas)} Q&A pairs")
    print(f"Test: {len(test_qas)} Q&A pairs")

    # Create queries with ground-truth chunk IDs
    def create_queries(qas: List[Dict]) -> List[Dict]:
        queries = []
        for qa in qas:
            chunk_ids = qa_id_to_chunk_ids.get(qa["id"], [])
            if chunk_ids:
                queries.append(
                    {
                        "query": qa["question"],
                        "ground_truth_chunk_ids": chunk_ids,
                    }
                )
        return queries

    train_queries = create_queries(train_qas)
    test_queries = create_queries(test_qas)

    print(f"Train queries: {len(train_queries)}")
    print(f"Test queries: {len(test_queries)}")

    # Save data
    print("\nSaving data...")

    with open(f"{output_dir}/corpus_chunks.pkl", "wb") as f:
        pickle.dump(corpus_chunks, f)

    with open(f"{output_dir}/train_queries.pkl", "wb") as f:
        pickle.dump(train_queries, f)

    with open(f"{output_dir}/test_queries.pkl", "wb") as f:
        pickle.dump(test_queries, f)

    # Save sample as JSON for inspection
    with open(f"{output_dir}/corpus_sample.json", "w") as f:
        json.dump(corpus_chunks[:10], f, indent=2)

    with open(f"{output_dir}/queries_sample.json", "w") as f:
        json.dump(train_queries[:10], f, indent=2)

    print(f"\nData saved to {output_dir}/")
    print(f"  - corpus_chunks.pkl ({len(corpus_chunks)} chunks)")
    print(f"  - train_queries.pkl ({len(train_queries)} queries)")
    print(f"  - test_queries.pkl ({len(test_queries)} queries)")
    print(f"  - corpus_sample.json (first 10 for inspection)")
    print(f"  - queries_sample.json (first 10 for inspection)")

    return len(corpus_chunks), len(train_queries), len(test_queries)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for domain-adapted embedding fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From HuggingFace dataset (MedQuAD)
  python prepare_data.py --source huggingface --dataset keivalya/MedQuad-MedicalQnADataset

  # From custom HuggingFace dataset
  python prepare_data.py --source huggingface --dataset your-org/your-dataset

  # From JSON file
  python prepare_data.py --source json --input-file data/qa_pairs.json

  # From JSONL file with custom output directory
  python prepare_data.py --source jsonl --input-file data/qa.jsonl --output-dir my_data

Input JSON/JSONL format:
  [{"question": "What is X?", "answer": "X is..."}]
        """,
    )

    parser.add_argument(
        "--source",
        required=True,
        choices=["huggingface", "json", "jsonl"],
        help="Data source type",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="HuggingFace dataset name (required for --source huggingface)",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Input file path (required for --source json/jsonl)",
    )
    parser.add_argument(
        "--output-dir", default="data", help="Output directory (default: data)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing (default: 0.2)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size in characters (default: 1000)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="HuggingFace dataset split to use (default: train)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.source == "huggingface" and not args.dataset:
        parser.error("--dataset is required when using --source huggingface")
    if args.source in ["json", "jsonl"] and not args.input_file:
        parser.error("--input-file is required when using --source json/jsonl")

    # Load data
    print("=" * 70)
    print("PREPARING TRAINING DATA")
    print("=" * 70)

    if args.source == "huggingface":
        qa_pairs = load_from_huggingface(args.dataset, args.split)
    elif args.source == "json":
        qa_pairs = load_from_json(args.input_file)
    else:  # jsonl
        qa_pairs = load_from_jsonl(args.input_file)

    if not qa_pairs:
        print("ERROR: No Q&A pairs loaded. Check your input data.")
        return

    # Prepare data
    num_chunks, num_train, num_test = prepare_data(
        qa_pairs,
        output_dir=args.output_dir,
        test_size=args.test_size,
        chunk_size=args.chunk_size,
    )

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)
    print(f"\nNext step: Run training")
    print(f"  python train.py --data-dir {args.output_dir} --output-dir models/trained")


if __name__ == "__main__":
    main()
