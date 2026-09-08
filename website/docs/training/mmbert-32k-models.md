---
title: mmBERT-32K Foundation, Embedding, and Reranking
sidebar_label: mmBERT-32K Models
---

# mmBERT-32K foundation, embedding, and reranking

Three models form one progressive text-retrieval family:

```text
mmBERT base encoder
  -> 32K YaRN masked-language continuation
     -> bi-encoder embedding training
     -> cross-encoder reranking training
```

The foundation supplies long multilingual representations. The embedder makes
retrieval efficient by encoding inputs independently. The reranker spends more
compute on a small candidate set by reading each query-document pair jointly.

## Shared foundation

[`mmbert-32k-yarn`](https://huggingface.co/llm-semantic-router/mmbert-32k-yarn)
is a ModernBERT-family encoder with 22 Transformer layers, hidden size 768, and
about 307 million parameters. It uses a 256K vocabulary and extends the base
8,192-token position range to 32,768 tokens with YaRN rotary-position scaling.

The checked workflow continues masked-language-model training on nine CC-100
language streams. It packs full 32K sequences with explicit document
boundaries and applies 30% token masking. The production configuration uses
30,774 sequences, one epoch, learning rate `1e-5`, BF16, per-device batch 1,
and gradient accumulation 16.

This stage changes the encoder's language representation and long-context
behavior; it does not add a routing head.

## Embedding model: bi-encoder

[`mmbert-embed-32k-2d-matryoshka`](https://huggingface.co/llm-semantic-router/mmbert-embed-32k-2d-matryoshka)
uses the 32K foundation as a bi-encoder:

```text
query -----------------> shared encoder -> normalized query vector
document --------------> shared encoder -> normalized document vector
                                            cosine/dot-product similarity
```

Because the two inputs are encoded independently, document vectors can be
precomputed and searched at scale.

“2D Matryoshka” means the loss supervises two axes:

- **Embedding dimension:** one trained vector can be truncated to 768, 512,
  256, 128, or 64 dimensions.
- **Encoder depth:** intermediate layers receive useful retrieval supervision,
  enabling layer-selectable representations.

The checked training path combines BGE-M3 retrieval examples with optional
AllNLI examples and uses a multiple-negatives ranking objective wrapped by
Sentence Transformers `Matryoshka2dLoss`. The production configuration uses a
32K maximum length, one epoch, batch 16 with accumulation 2, learning rate
`2e-5`, BF16, and STS-B for semantic-similarity evaluation.

## Reranking model: cross-encoder

[`mmbert-rerank-32k-2d-matryoshka`](https://huggingface.co/llm-semantic-router/mmbert-rerank-32k-2d-matryoshka)
reads a query and candidate together:

```text
[query, candidate] -> shared 32K encoder -> CLS representation -> relevance score
```

Joint attention is more expensive than bi-encoder search, but it can model
fine-grained interactions between the query and candidate. Use it after initial
retrieval, not over an entire corpus.

The model attaches scoring heads at layers 3, 6, 11, and 22 and at dimensions
768, 512, 256, 128, and 64. That creates 20 layer/dimension heads. Training
averages binary relevance loss across all heads, using BGE-M3 query-positive-
negative records. The production configuration uses three negatives per query,
one epoch, batch 16 with accumulation 2, learning rate `2e-5`, BF16, and
gradient checkpointing.

The heads make early-layer scores trainable; the trainer itself still computes
all hidden states. A runtime must implement early termination separately if it
wants a latency saving.

## Run the checked configurations

Install the family dependencies and set local data and output paths:

```bash
python -m pip install --requirement \
  src/training/model_embeddings/mmbert_32k/requirements.txt

export PYTHONPATH="$PWD/src"
export MMBERT32K_FOUNDATION_DATA=/path/to/tokenized-cc100-32k
export MMBERT32K_FOUNDATION_OUTPUT=/path/to/mmbert-32k-yarn
export MMBERT32K_BGE_DATA=/path/to/bge-m3-data
export MMBERT32K_EMBEDDER_OUTPUT=/path/to/mmbert-embed-32k-2d
export MMBERT32K_RERANKER_OUTPUT=/path/to/mmbert-rerank-32k-2d
```

Inspect each resolved command before starting a large run:

```bash
python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/foundation.json \
  --stage train --print-command

python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/embedder.json \
  --print-command

python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/reranker.json \
  --print-command
```

Remove `--print-command` to run. Prepare the foundation dataset first with
`--stage prepare`; use the exact preparation and data-integrity procedure in
the [workflow README](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_embeddings/mmbert_32k).

## Evaluate before release

For the foundation, verify long-context loading and masked-language loss on a
held-out corpus. For the embedder, report retrieval Recall@k and semantic
similarity at every supported dimension and selected layer. For the reranker,
report ranking metrics and classification quality for all 20 heads, not only
the largest final-layer head.

Also measure latency and memory at the exact layer, dimension, sequence length,
and batch size you intend to deploy. “Matryoshka” provides choices; it does not
make every choice equally accurate.
