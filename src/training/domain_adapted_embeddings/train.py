#!/usr/bin/env python3
"""
Domain-Adapted Embedding Fine-Tuning via Iterative Hard-Negative Mining

Based on: "Distilling an LLM's Wisdom: A Framework for Creating Domain Adapted
Financial Embedding Models" (arXiv:2512.08088)

This script fine-tunes an embedding model for improved retrieval in a specific domain
using iterative hard-negative mining with ground-truth labels.

Usage:
    # Basic training
    python train.py --data-dir data --output-dir models/trained

    # With custom hyperparameters
    python train.py --data-dir data --output-dir models/trained \
        --iterations 3 --learning-rate 1e-6

Results on MedQuAD dataset: +71.18% MRR@5 improvement over baseline (lr=5e-5).
"""

import argparse
import gc
import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader
from tqdm import tqdm

# ============================================================
# DEFAULT CONFIGURATION
# These are the proven hyperparameters from our experiments
# ============================================================
DEFAULT_MODEL = "llm-semantic-router/mmbert-embed-32k-2d-matryoshka"
DEFAULT_ITERATIONS = 2
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_EPOCHS = 2
DEFAULT_BATCH_SIZE = 8
DEFAULT_MARGIN = (
    0.1  # Important: SentenceTransformers default is 5.0 which performs poorly
)
DEFAULT_EASY_TO_HARD_RATIO = 2
DEFAULT_TOP_K = 100
DEFAULT_HARD_NEG_RANK = 15


def load_data(
    data_dir: str, num_queries: int = None
) -> Tuple[List, List, List, List, Dict]:
    """Load prepared data files."""
    with open(f"{data_dir}/corpus_chunks.pkl", "rb") as f:
        corpus_list = pickle.load(f)
    with open(f"{data_dir}/train_queries.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(f"{data_dir}/test_queries.pkl", "rb") as f:
        test_data = pickle.load(f)

    if num_queries and num_queries < len(train_data):
        train_data = train_data[:num_queries]

    chunk_ids = [c["chunk_id"] for c in corpus_list]
    chunk_texts = [c["text"] for c in corpus_list]
    chunk_id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}

    return corpus_list, train_data, test_data, chunk_texts, chunk_id_to_idx


def evaluate(
    model: SentenceTransformer,
    test_data: List,
    chunk_texts: List,
    chunk_id_to_idx: Dict,
    k: int = 5,
) -> Dict[str, float]:
    """
    Evaluate model on test set.

    Returns:
        Dict with MRR@k, Recall@k metrics
    """
    model.eval()
    query_texts = [q["query"] for q in test_data]

    print("  Encoding queries...")
    query_embs = model.encode(query_texts, batch_size=32, show_progress_bar=True)
    print("  Encoding corpus...")
    chunk_embs = model.encode(chunk_texts, batch_size=32, show_progress_bar=True)

    mrr_sum = 0
    recall_sum = 0
    valid_queries = 0

    for i, q in enumerate(test_data):
        gt_indices = {
            chunk_id_to_idx[cid]
            for cid in q["ground_truth_chunk_ids"]
            if cid in chunk_id_to_idx
        }
        if not gt_indices:
            continue

        valid_queries += 1
        scores = np.dot(chunk_embs, query_embs[i])
        ranked = np.argsort(-scores)[:k]

        # MRR@k
        for rank, idx in enumerate(ranked, 1):
            if idx in gt_indices:
                mrr_sum += 1.0 / rank
                break

        # Recall@k
        hits = len(set(ranked) & gt_indices)
        recall_sum += hits / len(gt_indices)

    return {
        f"MRR@{k}": mrr_sum / valid_queries if valid_queries > 0 else 0,
        f"Recall@{k}": recall_sum / valid_queries if valid_queries > 0 else 0,
    }


def mine_triplets(
    model: SentenceTransformer,
    train_data: List,
    chunk_texts: List,
    chunk_id_to_idx: Dict,
    top_k: int = 100,
    hard_neg_rank: int = 15,
) -> Tuple[List, List]:
    """
    Mine both HARD and EASY triplets using ground-truth labels.

    Hard triplets: Push apart hard negatives (ranked high but wrong)
    Easy triplets: Reinforce correct behavior (anti-forgetting)

    Returns:
        Tuple of (hard_triplets, easy_triplets)
    """
    model.eval()
    query_texts = [q["query"] for q in train_data]

    print("  Encoding queries...")
    query_embs = model.encode(query_texts, batch_size=32, show_progress_bar=True)
    print("  Encoding corpus...")
    chunk_embs = model.encode(chunk_texts, batch_size=32, show_progress_bar=True)

    hard_triplets = []
    easy_triplets = []

    print("  Mining triplets...")
    for i, q in enumerate(tqdm(train_data, desc="Mining")):
        gt_indices = {
            chunk_id_to_idx[cid]
            for cid in q["ground_truth_chunk_ids"]
            if cid in chunk_id_to_idx
        }
        if not gt_indices:
            continue

        scores = np.dot(chunk_embs, query_embs[i])
        ranked = np.argsort(-scores)[:top_k]

        # Categorize by position in ranking
        hard_negs = [idx for idx in ranked[:hard_neg_rank] if idx not in gt_indices]
        easy_negs = [
            idx for idx in ranked[hard_neg_rank:top_k] if idx not in gt_indices
        ]
        hard_pos = [
            idx for idx in ranked[5:top_k] if idx in gt_indices
        ]  # GT ranked low
        easy_pos = [idx for idx in ranked[:5] if idx in gt_indices]  # GT ranked high

        # HARD triplet: hard positive + hard negative
        if hard_negs and hard_pos:
            hard_triplets.append(
                (q["query"], chunk_texts[hard_pos[0]], chunk_texts[hard_negs[0]])
            )

        # EASY triplet: easy positive + negative (anti-forgetting)
        if easy_pos:
            neg_idx = (
                easy_negs[0] if easy_negs else (hard_negs[0] if hard_negs else None)
            )
            if neg_idx is not None:
                easy_triplets.append(
                    (q["query"], chunk_texts[easy_pos[0]], chunk_texts[neg_idx])
                )

    return hard_triplets, easy_triplets


def train_iteration(
    model: SentenceTransformer,
    triplets: List[Tuple],
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
    margin: float,
) -> SentenceTransformer:
    """Train model on triplets for one iteration."""
    examples = [InputExample(texts=[a, p, n]) for a, p, n in triplets]
    loader = DataLoader(examples, batch_size=batch_size, shuffle=True)

    # CRITICAL: Use margin=0.1, not default
    loss_fn = losses.TripletLoss(
        model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=margin,
    )

    model.fit(
        train_objectives=[(loader, loss_fn)],
        epochs=num_epochs,
        warmup_steps=100,
        optimizer_params={"lr": learning_rate},
        weight_decay=0.01,
        show_progress_bar=True,
    )

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune embedding model using iterative hard-negative mining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train.py --data-dir data --output-dir models/my_model

  # Use a different base model
  python train.py --data-dir data --base-model sentence-transformers/all-MiniLM-L6-v2

  # Adjust hyperparameters
  python train.py --data-dir data --iterations 3 --learning-rate 1e-6
        """,
    )

    # Required arguments
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing prepared data (corpus_chunks.pkl, train_queries.pkl, test_queries.pkl)",
    )
    parser.add_argument(
        "--output-dir",
        default="models/trained",
        help="Directory to save trained model (default: models/trained)",
    )

    # Model configuration
    parser.add_argument(
        "--base-model",
        default=DEFAULT_MODEL,
        help=f"Base SentenceTransformer model (default: {DEFAULT_MODEL})",
    )

    # Training hyperparameters
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Number of training iterations (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Limit number of training queries (default: use all)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Epochs per iteration (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help=f"Triplet loss margin (default 0.1, not SentenceTransformers default of 5.0) (default: {DEFAULT_MARGIN})",
    )
    parser.add_argument(
        "--easy-to-hard-ratio",
        type=int,
        default=DEFAULT_EASY_TO_HARD_RATIO,
        help=f"Ratio of easy to hard triplets for anti-forgetting (default: {DEFAULT_EASY_TO_HARD_RATIO})",
    )

    # Mining parameters
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Top K chunks to consider for mining (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--hard-neg-rank",
        type=int,
        default=DEFAULT_HARD_NEG_RANK,
        help=f"Negatives ranked within this are 'hard' (default: {DEFAULT_HARD_NEG_RANK})",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("DOMAIN-ADAPTED EMBEDDING FINE-TUNING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Base model: {args.base_model}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Margin: {args.margin}")
    print(f"  Easy:Hard ratio: {args.easy_to_hard_ratio}")

    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    corpus_list, train_data, test_data, chunk_texts, chunk_id_to_idx = load_data(
        args.data_dir, args.num_queries
    )
    print(f"Corpus: {len(chunk_texts)} chunks")
    print(f"Train queries: {len(train_data)}")
    print(f"Test queries: {len(test_data)}")

    # Load model
    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = SentenceTransformer(args.base_model, trust_remote_code=True)
    model = model.to(device)

    # Baseline evaluation
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION")
    print("=" * 70)
    baseline_metrics = evaluate(model, test_data, chunk_texts, chunk_id_to_idx)
    baseline_mrr = baseline_metrics["MRR@5"]
    print(f"\nBaseline MRR@5: {baseline_mrr:.4f}")
    print(f"Baseline Recall@5: {baseline_metrics['Recall@5']:.4f}")

    best_mrr = baseline_mrr
    best_iteration = 0
    all_triplets = []  # Cumulative across iterations

    # Training loop
    for iteration in range(1, args.iterations + 1):
        print("\n" + "=" * 70)
        print(f"ITERATION {iteration}/{args.iterations}")
        print("=" * 70)

        # Mine triplets
        print("\nStep 1: Mining triplets...")
        hard_triplets, easy_triplets = mine_triplets(
            model,
            train_data,
            chunk_texts,
            chunk_id_to_idx,
            top_k=args.top_k,
            hard_neg_rank=args.hard_neg_rank,
        )
        print(f"  Mined: {len(hard_triplets)} hard, {len(easy_triplets)} easy triplets")

        # Balance easy:hard ratio
        n_easy = min(len(easy_triplets), len(hard_triplets) * args.easy_to_hard_ratio)
        iteration_triplets = hard_triplets + easy_triplets[:n_easy]
        print(
            f"  Using: {len(hard_triplets)} hard + {n_easy} easy = {len(iteration_triplets)}"
        )

        if len(iteration_triplets) == 0:
            print("  WARNING: No triplets mined. Skipping iteration.")
            continue

        # Accumulate (don't replace) - key for anti-forgetting
        all_triplets.extend(iteration_triplets)
        print(f"  Cumulative training set: {len(all_triplets)} triplets")

        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()

        # Train
        print("\nStep 2: Training...")
        model = train_iteration(
            model,
            all_triplets,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            margin=args.margin,
        )

        # Evaluate
        print("\nStep 3: Evaluating...")
        metrics = evaluate(model, test_data, chunk_texts, chunk_id_to_idx)
        mrr = metrics["MRR@5"]
        change = (mrr - baseline_mrr) / baseline_mrr * 100
        print(f"\n  MRR@5: {mrr:.4f} ({change:+.2f}% vs baseline)")
        print(f"  Recall@5: {metrics['Recall@5']:.4f}")

        # Save if best
        if mrr > best_mrr:
            best_mrr = mrr
            best_iteration = iteration
            model.save(f"{args.output_dir}/best")
            print(f"\n  *** New best! Saved to {args.output_dir}/best ***")

        # Always save iteration checkpoint
        model.save(f"{args.output_dir}/iteration_{iteration}")

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    final_change = (best_mrr - baseline_mrr) / baseline_mrr * 100
    print(f"\nBaseline MRR@5: {baseline_mrr:.4f}")
    print(f"Best MRR@5:     {best_mrr:.4f} (iteration {best_iteration})")
    print(f"Improvement:    {final_change:+.2f}%")
    print(f"\nBest model saved to: {args.output_dir}/best")

    # Save training summary
    summary = {
        "base_model": args.base_model,
        "baseline_mrr": baseline_mrr,
        "best_mrr": best_mrr,
        "best_iteration": best_iteration,
        "improvement_percent": final_change,
        "hyperparameters": {
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "margin": args.margin,
            "easy_to_hard_ratio": args.easy_to_hard_ratio,
        },
    }

    with open(f"{args.output_dir}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved to: {args.output_dir}/training_summary.json")


if __name__ == "__main__":
    main()
