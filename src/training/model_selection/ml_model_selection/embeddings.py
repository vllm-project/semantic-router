#!/usr/bin/env python3
"""
Embedding generation for ML model selection training.

Uses the SAME Qwen3-Embedding model that the router serves through its model runtime.
This ensures training embeddings match inference embeddings exactly.

The router uses: Qwen/Qwen3-Embedding-0.6B (1024-dim)
"""

from pathlib import Path

import numpy as np
from feature_snapshot import build_snapshot, cache_key

# Optional sentence-transformers import; the service only needs it when it
# actually generates embeddings, not to start and serve health checks.
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# Default embedding model - MUST match the router's Qwen3 embedding model
# Router uses: Qwen/Qwen3-Embedding-0.6B through the model runtime (1024-dim)
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
        cache_dir: str | None = None,
        device: str = "cpu",
        revision: str | None = None,
    ):
        """
        Initialize embedding generator.

        Args:
            model_name: HuggingFace model name or alias (default: qwen3)
            cache_dir: Directory to cache model files
            device: Device to run on ("cpu", "cuda", "mps")
            revision: Model revision to load; None loads the default branch
        """
        # Resolve model alias
        if model_name in EMBEDDING_MODELS:
            model_name = EMBEDDING_MODELS[model_name]

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers is required for embedding generation. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        print("  (This is the SAME model the router serves through its model runtime)")

        # Qwen models require trust_remote_code=True
        is_qwen = "qwen" in model_name.lower()

        self.model = SentenceTransformer(
            model_name,
            cache_folder=cache_dir,
            device=device,
            revision=revision,
            trust_remote_code=is_qwen,  # Required for Qwen models
        )
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding model loaded (dim={self.dim})")

    def encode(
        self,
        texts: list[str],
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


def _load_cached_embeddings(
    path: Path, key: str, queries: list[str]
) -> dict[str, np.ndarray] | None:
    """Return the cached vectors, or None when the file is absent or was made for other inputs."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if str(data["inputs_id"]) != key:
            return None
        cached = dict(zip(data["queries"].tolist(), data["vectors"], strict=True))
    if any(query not in cached for query in queries):
        return None
    return {query: cached[query] for query in queries}


def generate_embeddings_for_queries(
    queries: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    cache_dir: str | None = None,
    embeddings_dir: str | None = None,
    model_revision: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Generate embeddings for a list of queries.

    Args:
        queries: List of query strings
        model_name: Embedding model name or alias
        batch_size: Batch size for encoding
        cache_dir: Model cache directory
        embeddings_dir: Directory for the embeddings cache; None disables it. A cache
            file is named after the model, revision, and query set, so changing any of
            them never reuses another run's vectors.
        model_revision: Model revision to pin; None is recorded as unpinned, which
            cannot detect an upstream change to the model.

    Returns:
        Dict mapping query -> embedding
    """
    model_id = EMBEDDING_MODELS.get(model_name, model_name)
    key = cache_key(queries, model_id, model_revision)
    cache_path = (
        Path(embeddings_dir) / f"embeddings_{key}.npz" if embeddings_dir else None
    )

    if cache_path is not None:
        cached = _load_cached_embeddings(cache_path, key, queries)
        if cached is not None:
            print(f"Loading cached embeddings from {cache_path}")
            return cached

    generator = EmbeddingGenerator(model_name, cache_dir, revision=model_revision)
    vectors = np.asarray(generator.encode(queries, batch_size=batch_size))
    build_snapshot(
        queries, model_id, generator.dim, model_revision=model_revision
    ).check_vectors(vectors, len(queries))

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            inputs_id=np.array(key),
            queries=np.array(queries, dtype=str),
            vectors=vectors.astype(np.float32),
        )
        print(f"Saved embeddings cache to {cache_path}")

    return dict(zip(queries, vectors, strict=True))
