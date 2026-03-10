"""
Cache-Specific Embedding Training Pipeline
==========================================

This package implements domain-specific embedding model training for semantic caching,
following the methodology from "Enhancing Semantic Caching with Domain-Specific Embeddings"
(arXiv:2504.02268v1).

Key Features:
- Contrastive learning for cache-optimized embeddings
- Synthetic data generation pipeline
- Domain-specific fine-tuning (coding, medical, math, etc.)
- LoRA-based parameter-efficient training
- Cache-specific evaluation metrics (Precision@K, Recall@K, MRR)

Modules:
- dataset_builder: Real and synthetic data collection
- synthetic_data_generator: Paraphrasing and hard negative mining
- train_cache_embedding_lora: Main training script with contrastive losses
- evaluate_cache_model: Evaluation framework
- losses: Contrastive loss functions (Triplet, InfoNCE, MNR)
- common_utils: Shared utilities
"""

__version__ = "0.1.0"
__author__ = "Semantic Router Team"

from .common_utils import setup_logging

__all__ = ["setup_logging"]
