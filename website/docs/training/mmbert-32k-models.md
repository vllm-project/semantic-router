---
title: Train Vela Embedding and Reranker
sidebar_label: Embedding and Reranking
---

# Train Vela Embedding and Reranker

Use Vela Embedding to find relevant documents efficiently, then Vela Reranker
to improve the order of a smaller candidate set. Both adapt the shared
[Vela Encoder](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M)
and support a choice of encoder depth and output dimension.

To use the published models without training, follow
[Embeddings and reranking](../installation/runtime/embeddings.md).

## Embedding model: bi-encoder

The embedding model encodes queries and documents independently into normalized
vectors. Precompute document vectors, then compare a query vector with your
index using cosine similarity or a dot product.

Train with query-positive pairs and useful negative documents. Add semantic
similarity or paraphrase examples when your application also compares requests,
groups related text, or detects repeated questions. Include the languages and
domains your index will serve.

## Reranking model: cross-encoder

The reranker reads a query and a candidate document together, then returns a
relevance score. Run it on the candidates returned by retrieval.

Train with complete candidate lists and relevance judgments. Hard negatives
from your retrieval system help the model distinguish plausible but incorrect
results. Keep documents with unknown relevance separate from judged negatives.
The score orders candidates; it is not a probability that a document is correct.

## Choose depth, dimension, and context

| Setting | Available choices | Tradeoff |
| --- | --- | --- |
| Encoder depth | 3, 6, 11, 22 layers | Fewer layers reduce encoder computation |
| Dimension | 64, 128, 256, 512, 768 | Smaller embedding vectors reduce index storage and comparison cost |
| Input limit | Up to 32,768 tokens | Longer inputs require more memory and time |

Training supervises all 20 depth/dimension combinations. Reranker has a trained
scoring head for each combination; reducing its dimension does not skip encoder
layers. Evaluate the combination you intend to deploy.

For embedding, use the same model revision, depth, and dimension for queries
and indexed documents. Rebuild the index when these settings change. For
reranking, the input budget covers the query, document, and special tokens
together.

## Prepare a training run

From a repository checkout, create an isolated environment with the appropriate
PyTorch build for your accelerator and install the
[training dependencies](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_embeddings/mmbert_32k/requirements.txt).
ROCm uses PyTorch's `cuda` device name.

Choose one starting point:

- **New task:** download Vela Encoder and initialize the task from that complete
  checkpoint.
- **Continue a task:** download Vela Embedding or Reranker and select
  `initialization: "continued_task"`. This keeps the trained encoder and heads
  while starting a new optimizer.
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

python -m src.training.model_embeddings.mmbert_32k.newbase_training \
  --config /data/retrieval/task.json \
  --config-sha256 "${VELA_TRAIN_CONFIG_SHA256:?Set the configuration SHA-256}" \
  --output /data/retrieval/run --device cuda
```

The configuration selects `task: "embedding"` or `task: "reranker"`, datasets,
input budgets, training steps, and evaluation intervals. Start with a small
budget and inspect the first development results before extending the run.

## Evaluate the trained checkpoint

Evaluate a saved checkpoint against the same development corpus used for your
baseline:

```bash
python -m src.training.model_embeddings.mmbert_32k.newbase_scoring \
  --model /data/retrieval/run/step-100 --task embedding \
  --known-dev /data/retrieval/validation --split validation \
  --output /data/retrieval/evaluation --device cpu --token-budget 32768
```

Replace the checkpoint path with a step your run saved. For a reranker, set
`--task reranker`. Use development results to choose a checkpoint, then run a
separate final comparison on held-out data.

For embedding, compare retrieval, similarity, and multilingual transfer.
For reranking, keep candidate lists fixed and compare ranking metrics such as
nDCG. Measure short and long inputs separately, along with latency and memory
at each deployed depth/dimension.

A full MMTEB result requires its complete selected benchmark and matching
evaluation protocol. A task subset can diagnose gaps, but cannot establish an
overall benchmark score or leaderboard rank.

## Deploy the result

Use [local model bindings](../installation/runtime/in-process.md) to select the
checkpoint and serving engine. Test the selected depth, dimension, and input
limit through [route preview](../installation/runtime/lifecycle-diagnostics.md).
An ONNX deployment needs graphs exported from the same trained weights.

## Earlier mmBERT workflows

The original foundation, embedding, and reranker recipes remain in the
[workflow README](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_embeddings/mmbert_32k).
Their checked-in historical configurations reproduce the earlier mmBERT
workflow. Use the Vela starting points above for new Vela task training.
