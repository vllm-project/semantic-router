#!/usr/bin/env python3
"""
Evaluate multi-domain LoRA model on separate domain test sets.

Computes margin-based metrics: avg(pos_sim) - avg(neg_sim)
for medical, law, and programming domains.
"""

import argparse
import json
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from peft import PeftModel
from typing import List, Dict, Tuple
from tqdm import tqdm

try:
    from huggingface_hub import hf_hub_download

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def load_baseline_model():
    """Load baseline embedding model."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")


def load_lora_model(lora_path: str):
    """Load LoRA-trained embedding model."""
    base_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    base_model[0].auto_model = PeftModel.from_pretrained(
        base_model[0].auto_model, lora_path
    )
    return base_model


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def download_test_set(domain: str) -> str:
    """Download test set from HuggingFace if not available locally."""
    if not HF_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required to download test sets. "
            "Install with: pip install huggingface_hub"
        )

    # Map domain to filename
    filenames = {
        "medical": "medical/test_set.jsonl",
        "law": "law/test_triplets.jsonl",
        "programming": "programming/test_triplets.jsonl",
    }

    if domain.lower() not in filenames:
        raise ValueError(
            f"Unknown domain: {domain}. Choose from: medical, law, programming"
        )

    filename = filenames[domain.lower()]

    print(f"  Downloading {domain} test set from HuggingFace...")
    return hf_hub_download(
        repo_id="llm-semantic-router/cache-embedding-test-sets",
        filename=filename,
        repo_type="dataset",
    )


def load_test_triplets(test_file: str, sample_size: int = 2000) -> List[Dict]:
    """Load test triplets from JSONL file or download from HuggingFace."""
    # If test_file doesn't exist locally, try to download from HF
    if not os.path.exists(test_file):
        # Try to infer domain from filename
        if "medical" in test_file:
            test_file = download_test_set("medical")
        elif "law" in test_file:
            test_file = download_test_set("law")
        elif "programming" in test_file:
            test_file = download_test_set("programming")
        else:
            raise FileNotFoundError(
                f"Test file not found: {test_file}\n"
                "Specify --download-from-hf to download from HuggingFace"
            )

    triplets = []
    with open(test_file) as f:
        for line in f:
            triplets.append(json.loads(line))

    # Sample if needed
    if len(triplets) > sample_size:
        import random

        random.seed(42)
        triplets = random.sample(triplets, sample_size)

    return triplets


def evaluate_model(
    model, triplets: List[Dict], model_name: str
) -> Tuple[float, float, float]:
    """
    Evaluate model on test triplets.

    Returns:
        (avg_pos_sim, avg_neg_sim, margin)
    """
    pos_sims = []
    neg_sims = []

    print(f"\nEvaluating {model_name} on {len(triplets)} triplets...")

    for triplet in tqdm(triplets, desc=f"{model_name}"):
        anchor = triplet["anchor"]
        positive = triplet["positive"]
        negative = triplet["negative"]

        # Encode
        anchor_emb = model.encode(anchor, convert_to_numpy=True)
        pos_emb = model.encode(positive, convert_to_numpy=True)
        neg_emb = model.encode(negative, convert_to_numpy=True)

        # Compute similarities
        pos_sim = cosine_similarity(anchor_emb, pos_emb)
        neg_sim = cosine_similarity(anchor_emb, neg_emb)

        pos_sims.append(pos_sim)
        neg_sims.append(neg_sim)

    avg_pos = np.mean(pos_sims)
    avg_neg = np.mean(neg_sims)
    margin = avg_pos - avg_neg

    return avg_pos, avg_neg, margin


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-domain LoRA model")
    parser.add_argument(
        "--lora-path", required=True, help="Path to multi-domain LoRA model"
    )
    parser.add_argument(
        "--medical-test",
        default="data/cache_embeddings/medical/test_set.jsonl",
        help="Medical test set path (downloads from HF if not found)",
    )
    parser.add_argument(
        "--law-test",
        default="data/cache_embeddings/law/test_triplets.jsonl",
        help="Law test set path (downloads from HF if not found)",
    )
    parser.add_argument(
        "--programming-test",
        default="data/cache_embeddings/programming/test_triplets.jsonl",
        help="Programming test set path (downloads from HF if not found)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Number of triplets to sample per domain",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Multi-Domain LoRA Evaluation")
    print("=" * 80)

    # Load models
    print("\nLoading models...")
    baseline_model = load_baseline_model()
    print("✓ Baseline model loaded")

    lora_model = load_lora_model(args.lora_path)
    print("✓ Multi-domain LoRA model loaded")

    # Test on each domain
    domains = [
        ("Medical", args.medical_test, 35.6),
        ("Law", args.law_test, 23.2),
        ("Programming", args.programming_test, 15.6),
    ]

    results = []

    for domain_name, test_file, baseline_improvement in domains:
        print(f"\n{'=' * 80}")
        print(f"{domain_name.upper()} DOMAIN EVALUATION")
        print(f"{'=' * 80}")

        # Load test triplets
        triplets = load_test_triplets(test_file, args.sample_size)
        print(f"Loaded {len(triplets)} test triplets")

        # Evaluate baseline
        base_pos, base_neg, base_margin = evaluate_model(
            baseline_model, triplets, "Baseline"
        )

        # Evaluate LoRA
        lora_pos, lora_neg, lora_margin = evaluate_model(
            lora_model, triplets, "Multi-Domain LoRA"
        )

        # Compute improvement
        absolute_improvement = lora_margin - base_margin
        relative_improvement = (absolute_improvement / base_margin) * 100

        print(f"\n{domain_name} Results:")
        print(f"  Baseline margin:        {base_margin:.4f}")
        print(f"  Multi-domain margin:    {lora_margin:.4f}")
        print(f"  Absolute improvement:   {absolute_improvement:+.4f}")
        print(f"  Relative improvement:   {relative_improvement:+.1f}%")
        print(f"  Domain-specific baseline: +{baseline_improvement}%")

        results.append(
            {
                "domain": domain_name,
                "baseline_margin": float(base_margin),
                "lora_margin": float(lora_margin),
                "absolute_improvement": float(absolute_improvement),
                "relative_improvement": float(relative_improvement),
                "domain_specific_baseline": float(baseline_improvement),
            }
        )

    # Overall summary
    print(f"\n{'=' * 80}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 80}")

    avg_improvement = np.mean([r["relative_improvement"] for r in results])
    min_improvement = min([r["relative_improvement"] for r in results])
    max_improvement = max([r["relative_improvement"] for r in results])

    print(f"\nMulti-Domain LoRA Performance:")
    print(f"  Average improvement: {avg_improvement:+.1f}%")
    print(f"  Min improvement:     {min_improvement:+.1f}%")
    print(f"  Max improvement:     {max_improvement:+.1f}%")

    print(f"\nDomain-Specific LoRA Baselines:")
    for r in results:
        print(f"  {r['domain']:12s}: +{r['domain_specific_baseline']}%")

    print(f"\nDecision:")
    if min_improvement >= 15.0:
        print("  ✅ SUCCESS! Multi-domain LoRA achieves ≥15% on all domains.")
        print("  → Use multi-domain approach (594MB total)")
    elif min_improvement >= 10.0:
        print("  ⚠️  MARGINAL. Multi-domain LoRA achieves 10-15% improvement.")
        print("  → Consider trade-off: memory vs performance")
    else:
        print("  ❌ INSUFFICIENT. Multi-domain LoRA achieves <10% on some domains.")
        print("  → Use model pool with LoRA switching (714MB total)")

    print("=" * 80)

    # Save results
    output_file = f"{args.lora_path}/evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
