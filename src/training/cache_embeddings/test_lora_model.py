#!/usr/bin/env python3
"""
Test the LoRA adapter (trained on proper triplets) vs baseline model.

Compares:
1. Baseline: sentence-transformers/all-MiniLM-L12-v2 (no fine-tuning)
2. LoRA: medical-cache-lora (trained on domain-specific triplets)
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from peft import PeftModel
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path


def load_baseline_model():
    """Load the baseline embedding model (no LoRA)."""
    print("Loading baseline model: sentence-transformers/all-MiniLM-L12-v2")
    return SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")


def load_lora_model(lora_path):
    """Load the base model with LoRA adapter applied."""
    print(f"Loading LoRA adapter from: {lora_path}")
    base_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

    # Apply LoRA adapter to the base model's first module (BERT)
    base_model[0].auto_model = PeftModel.from_pretrained(
        base_model[0].auto_model, lora_path
    )
    return base_model


def create_test_queries():
    """Create test queries with paraphrases and negatives."""
    return [
        {
            "original": "What are the symptoms of diabetes?",
            "paraphrase": "What are the signs and symptoms of diabetes mellitus?",
            "negative": "How is diabetes treated?",
        },
        {
            "original": "How to diagnose hypertension?",
            "paraphrase": "What are the diagnostic methods for high blood pressure?",
            "negative": "What are the risk factors for hypertension?",
        },
        {
            "original": "What causes heart disease?",
            "paraphrase": "What are the underlying causes of cardiovascular disease?",
            "negative": "How can heart disease be prevented?",
        },
        {
            "original": "What are the side effects of chemotherapy?",
            "paraphrase": "What adverse effects can occur from cancer chemotherapy?",
            "negative": "What types of chemotherapy drugs are available?",
        },
        {
            "original": "How is COVID-19 transmitted?",
            "paraphrase": "What are the transmission routes of the coronavirus?",
            "negative": "What are the symptoms of COVID-19?",
        },
    ]


def compute_similarities(model, queries):
    """Compute cosine similarities for each test case."""
    results = []

    for q in queries:
        # Encode all texts
        original_emb = model.encode([q["original"]])[0]
        paraphrase_emb = model.encode([q["paraphrase"]])[0]
        negative_emb = model.encode([q["negative"]])[0]

        # Compute similarities
        positive_sim = cosine_similarity([original_emb], [paraphrase_emb])[0][0]
        negative_sim = cosine_similarity([original_emb], [negative_emb])[0][0]

        results.append(
            {
                "original": q["original"],
                "paraphrase": q["paraphrase"],
                "negative": q["negative"],
                "positive_similarity": float(positive_sim),
                "negative_similarity": float(negative_sim),
                "margin": float(positive_sim - negative_sim),
            }
        )

    return results


def evaluate_model(model, model_name, queries):
    """Evaluate a model on test queries."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}")

    results = compute_similarities(model, queries)

    # Print detailed results
    for i, r in enumerate(results, 1):
        print(f"\nTest {i}:")
        print(f"  Original:  {r['original']}")
        print(f"  Paraphrase: {r['paraphrase']}")
        print(f"  Negative:   {r['negative']}")
        print(f"  ")
        print(f"  Positive similarity: {r['positive_similarity']:.4f}")
        print(f"  Negative similarity: {r['negative_similarity']:.4f}")
        print(f"  Margin (higher = better): {r['margin']:.4f}")

        # Check if model correctly ranks paraphrase > negative
        if r["positive_similarity"] > r["negative_similarity"]:
            print(f"  ✓ CORRECT: Paraphrase ranked higher than negative")
        else:
            print(f"  ✗ WRONG: Negative ranked higher than paraphrase!")

    # Compute aggregate metrics
    positive_sims = [r["positive_similarity"] for r in results]
    negative_sims = [r["negative_similarity"] for r in results]
    margins = [r["margin"] for r in results]

    accuracy = sum(
        1 for r in results if r["positive_similarity"] > r["negative_similarity"]
    ) / len(results)

    print(f"\n{'-'*80}")
    print(f"Summary Statistics:")
    print(f"{'-'*80}")
    print(
        f"Accuracy (positive > negative): {accuracy:.1%} ({int(accuracy * len(results))}/{len(results)})"
    )
    print(
        f"Average positive similarity: {np.mean(positive_sims):.4f} ± {np.std(positive_sims):.4f}"
    )
    print(
        f"Average negative similarity: {np.mean(negative_sims):.4f} ± {np.std(negative_sims):.4f}"
    )
    print(f"Average margin: {np.mean(margins):.4f} ± {np.std(margins):.4f}")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "avg_positive_sim": np.mean(positive_sims),
        "avg_negative_sim": np.mean(negative_sims),
        "avg_margin": np.mean(margins),
        "results": results,
    }


def main():
    print("=" * 80)
    print("Testing LoRA Adapter vs Baseline")
    print("=" * 80)
    print()
    print("This test compares:")
    print("  1. Baseline: Untrained sentence-transformers/all-MiniLM-L12-v2")
    print("  2. LoRA: Adapter trained on domain-specific triplets")
    print()
    print("Expected: The LoRA model should:")
    print("  - Assign higher similarity to paraphrases (positive pairs)")
    print("  - Assign lower similarity to related but different queries (negatives)")
    print("  - Show larger margins between positive and negative similarities")
    print()

    # Create test queries
    queries = create_test_queries()
    print(f"Testing on {len(queries)} medical query triplets...\n")

    # Evaluate baseline
    baseline_model = load_baseline_model()
    baseline_results = evaluate_model(baseline_model, "Baseline (No LoRA)", queries)

    # Evaluate LoRA model
    lora_path = Path("/Users/yovadia/code/semantic-router/models/medical-cache-lora")
    if not lora_path.exists():
        print(f"\n❌ ERROR: LoRA adapter not found at {lora_path}")
        print("Please ensure the model was trained or downloaded.")
        return

    lora_model = load_lora_model(str(lora_path))
    lora_results = evaluate_model(lora_model, "LoRA Adapter", queries)

    # Compare models
    print(f"\n{'='*80}")
    print("Model Comparison")
    print(f"{'='*80}")
    print(f"\nMetric                          Baseline      Corrected     Improvement")
    print(f"{'-'*80}")
    print(
        f"Accuracy (positive > negative)  {baseline_results['accuracy']:.1%}         "
        f"{lora_results['accuracy']:.1%}         "
        f"{(lora_results['accuracy'] - baseline_results['accuracy']) * 100:+.1f}%"
    )
    print(
        f"Average positive similarity     {baseline_results['avg_positive_sim']:.4f}      "
        f"{lora_results['avg_positive_sim']:.4f}      "
        f"{lora_results['avg_positive_sim'] - baseline_results['avg_positive_sim']:+.4f}"
    )
    print(
        f"Average negative similarity     {baseline_results['avg_negative_sim']:.4f}      "
        f"{lora_results['avg_negative_sim']:.4f}      "
        f"{lora_results['avg_negative_sim'] - baseline_results['avg_negative_sim']:+.4f}"
    )
    print(
        f"Average margin                  {baseline_results['avg_margin']:.4f}      "
        f"{lora_results['avg_margin']:.4f}      "
        f"{lora_results['avg_margin'] - baseline_results['avg_margin']:+.4f}"
    )

    # Conclusion
    print(f"\n{'='*80}")
    print("Conclusion")
    print(f"{'='*80}")

    if lora_results["avg_margin"] > baseline_results["avg_margin"]:
        improvement = (
            (lora_results["avg_margin"] - baseline_results["avg_margin"])
            / baseline_results["avg_margin"]
            * 100
        )
        print(f"✓ The LoRA adapter shows {improvement:.1f}% improvement in margin!")
        print(
            f"  This indicates better semantic understanding of domain-specific queries."
        )
    else:
        degradation = (
            (baseline_results["avg_margin"] - lora_results["avg_margin"])
            / baseline_results["avg_margin"]
            * 100
        )
        print(f"✗ The LoRA adapter shows {degradation:.1f}% degradation in margin.")
        print(
            f"  This suggests the training data or approach needs further investigation."
        )

    print()


if __name__ == "__main__":
    main()
