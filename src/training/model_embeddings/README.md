# Embedding model training

The multilingual MoM embedding collection has five artifacts and three narrow
in-tree owners:

| Artifact | Local owner |
| --- | --- |
| `llm-semantic-router/mmbert-32k-yarn` | `mmbert_32k/` foundation continuation |
| `llm-semantic-router/mmbert-embed-32k-2d-matryoshka` | `mmbert_32k/` bi-encoder |
| `llm-semantic-router/mmbert-rerank-32k-2d-matryoshka` | `mmbert_32k/` cross-encoder |
| `llm-semantic-router/multi-modal-embed-small` | `multimodal/small/` |
| `llm-semantic-router/multi-modal-embed-large` | `multimodal/large/` |

Each owner contains its public artifact manifest, production configuration,
training/evaluation entrypoints, dependency contract, and lightweight tests.
The older `cache_embeddings/` and `domain_adapted_embeddings/` directories are
research utilities rather than owners of these five artifacts.

Use [`../model_artifacts.json`](../model_artifacts.json) for the complete
classifier and embedding mapping.
