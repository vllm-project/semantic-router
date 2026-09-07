# Model training

This tree is the canonical source for producing, evaluating, and packaging the
models maintained with Semantic Router. Public artifacts map to their local
owners in [`model_artifacts.json`](model_artifacts.json).

## Top-level ownership

| Directory | Responsibility |
| --- | --- |
| `model_classifier/` | intent, policy, feedback, modality, PII, and safety classifiers |
| `model_embeddings/` | text, reranking, and multimodal embedding families |
| `model_eval/` | cross-family evaluation utilities |
| `model_experiment/` | experiments that are not release owners |
| `model_selection/` | learned model-selection research |

Each release-owning family has a focused directory with a README,
machine-readable configuration, explicit data/output paths, train and export
entrypoints, and lightweight contract tests. Model weights, datasets, caches,
checkpoints, and run logs stay outside Git.

Start with the public [model training guide](../../website/docs/training/training-overview.md)
to choose a model family. Use the README beside a trainer for its environment,
data preparation, commands, evaluation, and artifact format.
