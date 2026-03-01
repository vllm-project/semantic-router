#!/usr/bin/env python3
"""
Embedding generation for ML model selection training.

Uses the SAME Qwen3-Embedding model that the router uses via Candle.
This ensures training embeddings match inference embeddings exactly.

The router uses: Qwen/Qwen3-Embedding-0.6B (1024-dim)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Default embedding model - MUST match router's Candle/Qwen3 model
# Router uses: Qwen/Qwen3-Embedding-0.6B via Candle (1024-dim)
# We use the same model via sentence-transformers for identical embeddings
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Model aliases - qwen3 is the default and matches the router
EMBEDDING_MODELS = {
    # Qwen3 - SAME model as router uses (768-dim) - RECOMMENDED
    "qwen3": "Qwen/Qwen3-Embedding-0.6B",  # 1024-dim, matches router exactly
    # Alternative 768-dim models (for comparison/fallback)
    "gte": "thenlper/gte-base",  # 768-dim, original GTE
    "mpnet": "sentence-transformers/all-mpnet-base-v2",  # 768-dim
    "e5": "intfloat/e5-base-v2",  # 768-dim
    "bge": "BAAI/bge-base-en-v1.5",  # 768-dim
    # Larger Qwen models (different dimensions)
    "gte-qwen2": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",  # 1536-dim
    # Smaller/faster models
    "minilm": "sentence-transformers/all-MiniLM-L12-v2",  # 384-dim
}


class EmbeddingGenerator:
    """Generate embeddings for text queries using the same model as the router."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize embedding generator.

        Args:
            model_name: HuggingFace model name or alias (default: qwen3)
            cache_dir: Directory to cache model files
            device: Device to run on ("cpu", "cuda", "mps")
        """
        # Resolve model alias
        if model_name in EMBEDDING_MODELS:
            model_name = EMBEDDING_MODELS[model_name]

        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        print(f"  (This is the SAME model the router uses via Candle)")

        # Qwen models require trust_remote_code=True
        is_qwen = "qwen" in model_name.lower()

        self.model = SentenceTransformer(
            model_name,
            cache_folder=cache_dir,
            device=device,
            trust_remote_code=is_qwen,  # Required for Qwen models
        )
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"✓ Embedding model loaded (dim={self.dim})")

        if self.dim != 768:
            print(f"  ⚠ Warning: Router expects 768-dim embeddings, got {self.dim}-dim")

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings (N x dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text."""
        return self.encode([text], show_progress=False)[0]


def generate_embeddings_for_queries(
    queries: List[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    cache_dir: Optional[str] = None,
    cache_file: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate embeddings for a list of queries.

    Args:
        queries: List of query strings
        model_name: Embedding model name
        batch_size: Batch size for encoding
        cache_dir: Model cache directory
        cache_file: Optional file to cache embeddings

    Returns:
        Dict mapping query -> embedding
    """
    # Check cache first
    if cache_file and os.path.exists(cache_file):
        print(f"✓ Loading cached embeddings from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return dict(data["embeddings"].item())

    # Generate embeddings
    generator = EmbeddingGenerator(model_name, cache_dir)
    embeddings = generator.encode(queries, batch_size=batch_size)

    # Create mapping
    result = {q: emb for q, emb in zip(queries, embeddings)}

    # Save cache
    if cache_file:
        cache_path = Path(cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, embeddings=result)
        print(f"✓ Saved embeddings cache to {cache_file}")

    return result
