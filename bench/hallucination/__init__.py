"""Hallucination Detection Benchmark for Semantic Router.

This package provides end-to-end evaluation of the hallucination detection pipeline
through the router + Envoy stack.
"""

from .evaluate import HallucinationBenchmark
from .datasets import HaluEvalDataset, CustomDataset, get_dataset

__all__ = [
    "HallucinationBenchmark",
    "HaluEvalDataset",
    "FinancialFactEvalDataset",
    "CustomDataset",
    "get_dataset",
]
