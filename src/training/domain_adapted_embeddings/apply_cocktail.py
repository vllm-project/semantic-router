#!/usr/bin/env python3
"""
Standalone LM-Cocktail: Apply weight merging to an already-trained model.

This script merges a fine-tuned model with its base model to preserve
general knowledge while keeping domain specialization.

Usage:
    python apply_cocktail.py \
        --base-model llm-semantic-router/mmbert-embed-32k-2d-matryoshka \
        --finetuned-model models/best \
        --output-dir models/cocktail \
        --alpha 0.5

    # Test multiple alpha values
    python apply_cocktail.py \
        --base-model llm-semantic-router/mmbert-embed-32k-2d-matryoshka \
        --finetuned-model models/best \
        --output-dir models/cocktail \
        --alpha 0.3 0.5 0.7 \
        --eval-data-dir data
"""

import argparse
import json
import os
import pickle
from typing import Dict, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def apply_lm_cocktail(
    base_model_path: str,
    finetuned_model_path: str,
    output_path: str,
    alpha: float = 0.5,
) -> SentenceTransformer:
    """
    LM-Cocktail: Merge fine-tuned weights with base model.

    Args:
        base_model_path: Path or HF name of the original base model
        finetuned_model_path: Path to the fine-tuned model
        output_path: Where to save the merged model
        alpha: Merge ratio (0.0 = pure base, 1.0 = pure fine-tuned)

    Returns:
        Merged SentenceTransformer model
    """
    print(f"\n  Loading base model: {base_model_path}")
    base_model = SentenceTransformer(
        base_model_path, trust_remote_code=True, device="cpu"
    )

    print(f"  Loading fine-tuned model: {finetuned_model_path}")
    finetuned_model = SentenceTransformer(
        finetuned_model_path, trust_remote_code=True, device="cpu"
    )

    # Get the underlying transformer model state dicts
    base_sd = base_model[0].auto_model.state_dict()
    fine_sd = finetuned_model[0].auto_model.state_dict()

    # Merge weights: merged = alpha * finetuned + (1 - alpha) * base
    print(
        f"  Merging weights with alpha={alpha} (fine-tuned) + {1-alpha:.1f} (base)..."
    )
    merged_sd = {}
    for key in base_sd:
        if key in fine_sd:
            merged_sd[key] = alpha * fine_sd[key] + (1 - alpha) * base_sd[key]
        else:
            merged_sd[key] = base_sd[key]

    # Apply merged weights to fine-tuned model (preserves tokenizer, config, etc.)
    finetuned_model[0].auto_model.load_state_dict(merged_sd)

    # Free base model memory
    del base_model
    del base_sd
    del fine_sd
    del merged_sd
    torch.cuda.empty_cache()

    # Save merged model
    os.makedirs(output_path, exist_ok=True)
    finetuned_model.save(output_path)
    print(f"  Merged model saved to: {output_path}")

    return finetuned_model


def evaluate(
    model: SentenceTransformer,
    test_data: List,
    chunk_texts: List,
    chunk_id_to_idx: Dict,
    k: int = 5,
) -> Dict[str, float]:
    """Evaluate model on test set."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Apply LM-Cocktail weight merging to a trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--base-model",
        required=True,
        help="Base model path or HuggingFace name",
    )
    parser.add_argument(
        "--finetuned-model",
        required=True,
        help="Path to the fine-tuned model",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save merged model(s)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[0.5],
        help="Alpha value(s) for merging. Can specify multiple: --alpha 0.3 0.5 0.7",
    )
    parser.add_argument(
        "--eval-data-dir",
        help="Optional: data directory for evaluation (must contain test_queries.pkl, corpus_chunks.pkl)",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load evaluation data if provided
    test_data = None
    chunk_texts = None
    chunk_id_to_idx = None
    baseline_mrr = None

    if args.eval_data_dir:
        print("=" * 70)
        print("LOADING EVALUATION DATA")
        print("=" * 70)
        with open(f"{args.eval_data_dir}/corpus_chunks.pkl", "rb") as f:
            corpus_list = pickle.load(f)
        with open(f"{args.eval_data_dir}/test_queries.pkl", "rb") as f:
            test_data = pickle.load(f)

        chunk_ids = [c["chunk_id"] for c in corpus_list]
        chunk_texts = [c["text"] for c in corpus_list]
        chunk_id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}
        print(f"  Test queries: {len(test_data)}")
        print(f"  Corpus chunks: {len(chunk_texts)}")

        # Evaluate base model for baseline
        print("\n" + "=" * 70)
        print("BASELINE EVALUATION (Base Model)")
        print("=" * 70)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model = SentenceTransformer(
            args.base_model, trust_remote_code=True, device="cpu"
        )
        base_model = base_model.to(device)
        baseline_metrics = evaluate(base_model, test_data, chunk_texts, chunk_id_to_idx)
        baseline_mrr = baseline_metrics["MRR@5"]
        print(f"  Base model MRR@5: {baseline_mrr:.4f}")
        del base_model
        torch.cuda.empty_cache()

        # Evaluate fine-tuned model
        print("\n" + "=" * 70)
        print("FINE-TUNED MODEL EVALUATION")
        print("=" * 70)
        finetuned_model = SentenceTransformer(
            args.finetuned_model, trust_remote_code=True, device="cpu"
        )
        finetuned_model = finetuned_model.to(device)
        finetuned_metrics = evaluate(
            finetuned_model, test_data, chunk_texts, chunk_id_to_idx
        )
        finetuned_mrr = finetuned_metrics["MRR@5"]
        finetuned_change = (finetuned_mrr - baseline_mrr) / baseline_mrr * 100
        print(
            f"  Fine-tuned MRR@5: {finetuned_mrr:.4f} ({finetuned_change:+.2f}% vs base)"
        )
        del finetuned_model
        torch.cuda.empty_cache()

    results = {}

    for alpha in args.alpha:
        print("\n" + "=" * 70)
        print(f"APPLYING LM-COCKTAIL (alpha={alpha})")
        print("=" * 70)

        output_path = f"{args.output_dir}/alpha_{alpha}"
        merged_model = apply_lm_cocktail(
            base_model_path=args.base_model,
            finetuned_model_path=args.finetuned_model,
            output_path=output_path,
            alpha=alpha,
        )

        if test_data is not None:
            print("\n  Evaluating merged model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            merged_model = merged_model.to(device)
            metrics = evaluate(merged_model, test_data, chunk_texts, chunk_id_to_idx)
            mrr = metrics["MRR@5"]
            change = (mrr - baseline_mrr) / baseline_mrr * 100

            print(f"\n  Results (alpha={alpha}):")
            print(f"    MRR@5: {mrr:.4f} ({change:+.2f}% vs base)")
            print(f"    Recall@5: {metrics['Recall@5']:.4f}")

            results[alpha] = {
                "mrr": mrr,
                "recall": metrics["Recall@5"],
                "improvement_vs_base": change,
                "output_path": output_path,
            }

        del merged_model
        torch.cuda.empty_cache()

    # Summary
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\n{'Alpha':<10} {'MRR@5':<10} {'vs Base':<15} {'Path'}")
        print("-" * 70)
        print(f"{'base':<10} {baseline_mrr:<10.4f} {'+0.00%':<15}")
        print(f"{'finetuned':<10} {finetuned_mrr:<10.4f} {finetuned_change:+.2f}%")
        for alpha, r in sorted(results.items()):
            print(
                f"{alpha:<10} {r['mrr']:<10.4f} {r['improvement_vs_base']:+.2f}%{'':>6} {r['output_path']}"
            )

        # Save results
        summary = {
            "base_model": args.base_model,
            "finetuned_model": args.finetuned_model,
            "baseline_mrr": baseline_mrr,
            "finetuned_mrr": finetuned_mrr,
            "cocktail_results": results,
        }
        with open(f"{args.output_dir}/cocktail_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {args.output_dir}/cocktail_summary.json")


if __name__ == "__main__":
    main()
