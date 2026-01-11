#!/usr/bin/env python3
"""
Test the medical domain cache embedding LoRA model.

Compares baseline vs LoRA-trained model for medical queries.
"""

import torch
from sentence_transformers import SentenceTransformer
from peft import PeftModel
import numpy as np
from typing import List, Tuple


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


def test_cache_similarity(
    model,
    cache_query: str,
    similar_queries: List[str],
    dissimilar_queries: List[str],
    model_name: str,
) -> Tuple[float, float]:
    """
    Test cache embedding performance.

    Returns:
        (avg_similar_score, avg_dissimilar_score)
    """
    cache_emb = model.encode(cache_query, convert_to_numpy=True)

    # Compute similarities for similar queries
    similar_scores = []
    for query in similar_queries:
        emb = model.encode(query, convert_to_numpy=True)
        score = cosine_similarity(cache_emb, emb)
        similar_scores.append(score)

    # Compute similarities for dissimilar queries
    dissimilar_scores = []
    for query in dissimilar_queries:
        emb = model.encode(query, convert_to_numpy=True)
        score = cosine_similarity(cache_emb, emb)
        dissimilar_scores.append(score)

    avg_similar = np.mean(similar_scores)
    avg_dissimilar = np.mean(dissimilar_scores)
    margin = avg_similar - avg_dissimilar

    print(f"\n{model_name}:")
    print(f"  Similar queries avg:     {avg_similar:.4f}")
    print(f"  Dissimilar queries avg:  {avg_dissimilar:.4f}")
    print(f"  Margin:                  {margin:.4f}")

    return avg_similar, avg_dissimilar


def main():
    print("=" * 80)
    print("Medical Domain Cache Embedding Test")
    print("=" * 80)

    # Test cases: cache query, similar queries, dissimilar queries
    test_cases = [
        {
            "cache": "What are the symptoms of diabetes?",
            "similar": [
                "What are the signs and symptoms of diabetes mellitus?",
                "How does diabetes present clinically?",
                "What symptoms indicate diabetes?",
            ],
            "dissimilar": [
                "How is diabetes treated?",
                "What causes diabetes?",
                "How to prevent diabetes?",
            ],
        },
        {
            "cache": "How to diagnose hypertension?",
            "similar": [
                "What are the diagnostic methods for high blood pressure?",
                "How do doctors diagnose hypertension?",
                "What tests confirm hypertension?",
            ],
            "dissimilar": [
                "What are the risk factors for hypertension?",
                "How to treat high blood pressure?",
                "What medications lower blood pressure?",
            ],
        },
        {
            "cache": "What causes heart disease?",
            "similar": [
                "What are the underlying causes of cardiovascular disease?",
                "What leads to heart disease development?",
                "What are the etiological factors of cardiac disease?",
            ],
            "dissimilar": [
                "How can heart disease be prevented?",
                "What are the symptoms of heart disease?",
                "How is heart disease treated?",
            ],
        },
        {
            "cache": "What are the side effects of chemotherapy?",
            "similar": [
                "What adverse effects can occur from cancer chemotherapy?",
                "What are the toxic effects of chemotherapy?",
                "What complications arise from chemotherapy treatment?",
            ],
            "dissimilar": [
                "What types of chemotherapy drugs are available?",
                "How effective is chemotherapy?",
                "When is chemotherapy indicated?",
            ],
        },
        {
            "cache": "How is COVID-19 transmitted?",
            "similar": [
                "What are the transmission routes of the coronavirus?",
                "How does COVID-19 spread between people?",
                "What are the modes of SARS-CoV-2 transmission?",
            ],
            "dissimilar": [
                "What are the symptoms of COVID-19?",
                "How is COVID-19 treated?",
                "How effective are COVID-19 vaccines?",
            ],
        },
        {
            "cache": "What is the treatment for pneumonia?",
            "similar": [
                "How do you treat pneumonia infections?",
                "What medications are used for pneumonia?",
                "What is the therapeutic approach for pneumonia?",
            ],
            "dissimilar": [
                "What causes pneumonia?",
                "How is pneumonia diagnosed?",
                "What are the symptoms of pneumonia?",
            ],
        },
        {
            "cache": "What are the risk factors for stroke?",
            "similar": [
                "What increases the risk of cerebrovascular accident?",
                "What predisposes someone to stroke?",
                "What are the risk factors for cerebrovascular events?",
            ],
            "dissimilar": [
                "What are the symptoms of stroke?",
                "How is stroke treated?",
                "How to prevent stroke?",
            ],
        },
        {
            "cache": "How to manage chronic pain?",
            "similar": [
                "What are the approaches to managing persistent pain?",
                "How do you treat long-term pain conditions?",
                "What are the strategies for chronic pain management?",
            ],
            "dissimilar": [
                "What causes chronic pain?",
                "How is chronic pain diagnosed?",
                "What medications treat acute pain?",
            ],
        },
    ]

    # Load models
    print("\nLoading models...")
    baseline_model = load_baseline_model()
    print("✓ Baseline model loaded")

    lora_model = load_lora_model("models/medical-cache-lora")
    print("✓ LoRA model loaded")

    # Run tests
    baseline_margins = []
    lora_margins = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Test Case {i}: {test_case['cache']}")
        print(f"{'=' * 80}")

        # Baseline
        base_similar, base_dissimilar = test_cache_similarity(
            baseline_model,
            test_case["cache"],
            test_case["similar"],
            test_case["dissimilar"],
            "Baseline Model",
        )
        baseline_margins.append(base_similar - base_dissimilar)

        # LoRA
        lora_similar, lora_dissimilar = test_cache_similarity(
            lora_model,
            test_case["cache"],
            test_case["similar"],
            test_case["dissimilar"],
            "LoRA-Trained Model",
        )
        lora_margins.append(lora_similar - lora_dissimilar)

        # Improvement
        improvement = (lora_similar - lora_dissimilar) - (
            base_similar - base_dissimilar
        )
        improvement_pct = (improvement / (base_similar - base_dissimilar)) * 100
        print(f"\nImprovement:")
        print(f"  Absolute margin gain:  {improvement:+.4f}")
        print(f"  Relative improvement:  {improvement_pct:+.1f}%")

    # Overall summary
    print(f"\n{'=' * 80}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 80}")

    avg_baseline_margin = np.mean(baseline_margins)
    avg_lora_margin = np.mean(lora_margins)
    overall_improvement = avg_lora_margin - avg_baseline_margin
    overall_improvement_pct = (overall_improvement / avg_baseline_margin) * 100

    print(f"\nAverage Margins:")
    print(f"  Baseline:     {avg_baseline_margin:.4f}")
    print(f"  LoRA-trained: {avg_lora_margin:.4f}")
    print(f"\nOverall Improvement:")
    print(f"  Absolute:     {overall_improvement:+.4f}")
    print(f"  Relative:     {overall_improvement_pct:+.1f}%")

    print(f"\nTraining Details:")
    print(f"  Dataset: MedQuAD (NIH, CDC, FDA)")
    print(f"  Queries: 47,457")
    print(f"  Training samples: ~230,000")
    print(f"  Base model: sentence-transformers/all-MiniLM-L12-v2")
    print(f"  Method: LoRA fine-tuning with MNR loss")
    print("=" * 80)


if __name__ == "__main__":
    main()
