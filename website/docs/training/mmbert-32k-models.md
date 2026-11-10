---
title: Train Vela Embedding
sidebar_label: Embedding
---

# Train Vela Embedding

Vela Embedding finds candidates for the Router's recall features: the
response cache, memory and RAG. It adapts the shared
[Vela Encoder](https://huggingface.co/vllm-sr/Vela-1.0-Encoder-307M) and
supports a choice of encoder depth and output dimension. Embeddings serve
recall only; the decision model judges what recall finds.

To use the published model without training, follow
[Embeddings](../model-runtime/guides/embeddings.md).

## Embedding model: bi-encoder

The embedding model encodes queries and documents independently into normalized
vectors. Precompute document vectors, then compare a query vector with your
index using cosine similarity or a dot product.

Train with query-positive pairs and useful negative documents. Add semantic
similarity or paraphrase examples when your application also compares requests,
groups related text, or detects repeated questions. Include the languages and
domains your index will serve.

## Choose depth, dimension, and context

| Setting | Available choices | Tradeoff |
| --- | --- | --- |
| Encoder depth | 3, 6, 11, 22 layers | Fewer layers reduce encoder computation |
| Dimension | 64, 128, 256, 512, 768 | Smaller embedding vectors reduce index storage and comparison cost |
| Input limit | Up to 32,768 tokens | Longer inputs require more memory and time |

Training supervises all 20 depth/dimension combinations. Evaluate the
combination you intend to deploy. Use the same model revision, depth, and
dimension for queries and indexed documents, and rebuild the index when these
settings change.

## Prepare a training run

From a repository checkout, create the locked uv environment for this trainer:

```bash
uv sync --locked --project src/training/model_embeddings/mmbert_32k
```

The default PyPI torch wheel is a CUDA build on Linux and the platform default
on macOS. A release run still uses the digest-pinned image recorded beside the
trainer. ROCm uses PyTorch's `cuda` device name.
[requirements.txt](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_embeddings/mmbert_32k/requirements.txt)
stays for anyone who installs with pip.

Choose one starting point:

- **New task:** download Vela Encoder and initialize the task from that complete
  checkpoint.
- **Continue a task:** download Vela Embedding and select
  `initialization: "continued_task"`. This keeps the trained encoder while
  starting a new optimizer.
- **Resume an interrupted run:** use `--resume` to restore its optimizer and
  progress.

Prepare separate training and development corpora, a sampling plan, and a
training configuration. The
[training workflow reference](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_embeddings/mmbert_32k#train-a-new-task-from-a-standard-base)
provides the JSON formats, supported losses, optional teacher supervision, and
configuration fields.

The commands below assume those files are ready under `/data/retrieval`.
Set `VELA_TRAIN_CONFIG_SHA256` to the SHA-256 of your configuration file:

```bash
export PYTHONPATH="$PWD"

uv run --locked --project src/training/model_embeddings/mmbert_32k \
  python -m src.training.model_embeddings.mmbert_32k.newbase_training \
  --config /data/retrieval/task.json \
  --config-sha256 "${VELA_TRAIN_CONFIG_SHA256:?Set the configuration SHA-256}" \
  --output /data/retrieval/run --device cuda
```

The configuration selects `task: "embedding"`, datasets, input budgets,
training steps, and evaluation intervals. Start with a small budget and
inspect the first development results before extending the run.

## Evaluate the trained checkpoint

Evaluate a saved checkpoint against the same development corpus used for your
baseline:

```bash
export PYTHONPATH="$PWD"

uv run --locked --project src/training/model_embeddings/mmbert_32k \
  python -m src.training.model_embeddings.mmbert_32k.newbase_scoring \
  --model /data/retrieval/run/step-100 --task embedding \
  --known-dev /data/retrieval/validation --split validation \
  --output /data/retrieval/evaluation --device cpu --token-budget 32768
```

Replace the checkpoint path with a step your run saved. Use development
results to choose a checkpoint, then run a separate final comparison on
held-out data. Compare retrieval, similarity, and multilingual transfer, and
measure short and long inputs separately, along with latency and memory at
each deployed depth/dimension.

A full MMTEB result requires its complete selected benchmark and matching
evaluation protocol. A task subset can diagnose gaps, but cannot establish an
overall benchmark score or leaderboard rank.

## Deploy the result

Declare the checkpoint as an embedding deployment and name it where recall
uses it ([Embeddings](../model-runtime/guides/embeddings.md)); a store indexed
with the earlier model must be re-indexed. Test recall through a real request
before routing traffic to it.

## Earlier mmBERT workflows

The original foundation and embedding recipes remain in the
[workflow README](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_embeddings/mmbert_32k).
Their checked-in historical configurations reproduce the earlier mmBERT
workflow. Use the Vela starting points above for new training.
