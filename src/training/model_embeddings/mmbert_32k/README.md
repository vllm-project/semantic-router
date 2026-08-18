# mmBERT-32K training lineage

This directory is the canonical in-repository source for the related mmBERT
foundation, embedding, and reranking artifacts:

| Family | Trainer | Published artifact |
| --- | --- | --- |
| 8K → 32K YaRN masked-LM continuation | `foundation.py` | `llm-semantic-router/mmbert-32k-yarn` |
| 2D Matryoshka bi-encoder | `embedder.py` | `llm-semantic-router/mmbert-embed-32k-2d-matryoshka` |
| 2D Matryoshka cross-encoder | `reranker.py` | `llm-semantic-router/mmbert-rerank-32k-2d-matryoshka` |

The implementations were canonicalized from
[`semantic-router/Model-training`](https://github.com/semantic-router/Model-training)
`main` at commit `3bc41e1322ee5a53e08d18eb940855dec53c1539`.
`provenance.json` records the source blob for every imported implementation.
The source repository is Apache-2.0 licensed, as is this repository.

## What was normalized

- There is one trainer per artifact family. `foundation_data.py` is the data
  preparation helper for the foundation trainer, not a competing trainer.
- Hugging Face model and dataset revisions are explicit in the checked-in JSON
  configs and are forwarded by the trainers.
- Data, output, cache, and credential locations are not tied to the original
  machines. Required local paths come from environment variables.
- Python packages and the public ROCm validation image are pinned in
  `requirements.txt` and `runtime.json`.
- JSONL discovery and random seeds are deterministic.
- Foundation continuation now sets the Transformers 4.57.6 official YaRN
  `rope_scaling` configuration before model construction and validates every
  instantiated rotary layer. The recipe explicitly uses SDPA because
  ModernBERT's Flash Attention 2 branch does not consume this config-driven
  rotary implementation in the pinned Transformers release.
- The foundation corpus is generated directly from pinned CC-100 iterable
  streams. It preserves row order within each language, inserts an explicit
  tokenizer SEP/EOS document boundary, emits only full unpadded 32K examples,
  and writes a canonical `packing_manifest.json` plus detached digest. The
  manifest binds quotas, source-prefix counters/hashes, packed-token digest,
  fixed-size Arrow schema, and every Arrow shard's size and SHA-256.
- Foundation training validates that complete handoff before loading model
  weights. Missing/non-canonical manifests, changed revisions, wrong row or
  sequence lengths, padded masks, and changed Arrow shards fail closed. The
  output `training_receipt.json` records the canonical manifest digest, Arrow
  hashes, per-language strong-ETag/consumed-prefix content receipt, and the
  explicit data-governance acknowledgement.
- Reranker exports retain both `classification_heads.pt` and
  `matryoshka_config.json`; release validation must upload both files with the
  encoder and tokenizer.

The objectives and architecture remain those in the upstream code: MLM with
YaRN position scaling, Sentence Transformers `Matryoshka2dLoss` over MNRL, and
BCE across the reranker's layer/dimension heads. The corrected foundation
recipe changes how YaRN is instantiated, not the MLM objective: it uses the
official config-driven implementation instead of a post-load buffer scan that
could silently patch zero ModernBERT layers.

## Reproducible configuration

The three configs record artifact-compatible parameters recovered from the
published model metadata. Set paths explicitly before resolving a command:

```bash
export PYTHONPATH="$PWD/src"
export MMBERT32K_FOUNDATION_DATA=/path/to/tokenized-cc100-32k
export MMBERT32K_FOUNDATION_OUTPUT=/path/to/mmbert-32k-yarn
export MMBERT32K_BGE_DATA=/path/to/bge-m3-data
export MMBERT32K_EMBEDDER_OUTPUT=/path/to/mmbert-embed-32k-2d
export MMBERT32K_RERANKER_OUTPUT=/path/to/mmbert-rerank-32k-2d
```

Materialize the shared BGE-M3 corpus at the revision recorded in both configs:

```bash
hf download Shitao/bge-m3-data \
  --repo-type dataset \
  --revision a69db8b86e9c1767d193ee0de95e5c4001a71eae \
  --local-dir "$MMBERT32K_BGE_DATA"
```

The foundation preparation command below streams CC-100 itself, using the
revision, ordered nine-language list, and exact 30,774-sequence target in
`configs/foundation.json`. The stable allocation is 3,420 examples each for
`en`, `zh-Hans`, and `de`, then 3,419 each for `fr`, `es`, `ru`, `ar`, `ja`,
and `ko`. (`zh-Hans` is the actual config name in the pinned CC-100 loader;
the old `zh` argument was not executable.) The pinned loader revision contains
only a dataset script, so the exact path streams and incrementally decompresses
the StatMT URLs declared by that script. It rejects weak ETags and validates a
checked-in strong ETag and Content-Length for every language. Document size is
fail-closed before tokenization at 8 MiB and tokenized size at 1,048,576 tokens.
Arrow uses fixed 32K list features and a writer batch of 16, keeping each writer
window bounded instead of buffering a large default batch.

The loader Git revision hashes loader code only; it does **not** hash the remote
StatMT objects. Likewise, because this recipe intentionally stops each huge XZ
stream once its language quota is filled, `compressed_prefix_sha256` covers
exactly the compressed bytes consumed, not the complete XZ object. ETag and
Content-Length are version checks, not substitutes for a full-object
cryptographic hash. The manifest states these scopes explicitly and never
claims full-file immutability.

Preparation therefore uses a two-pass audit/replay contract. First create an
audit dataset to discover the exact consumed prefixes, then replay into the
training dataset while requiring those byte counts and hashes for every
language. Both commands receive the checked-in
`acknowledge_cc100_license_unknown=true` flag; it records informed use of an
unlicensed dataset card and does not clear the separate release gate:

```bash
export MMBERT32K_FOUNDATION_AUDIT=/path/to/mmbert32k-cc100-audit
python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/foundation.json \
  --stage prepare --output_dir "$MMBERT32K_FOUNDATION_AUDIT"
python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/foundation.json \
  --stage prepare \
  --source_prefix_contract_dir "$MMBERT32K_FOUNDATION_AUDIT"
```

An audit-only dataset is intentionally rejected by foundation training; only a
successful replay marks every prefix contract as verified. Packing remains
disk-backed and bounded by the capped current source document, tokenizer
result, one 32K payload, and the small Arrow writer window.

Inspect the fully resolved commands without importing PyTorch:

```bash
python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/foundation.json \
  --stage prepare --print-command
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

Remove `--print-command` to execute. Additional arguments after the runner
options are delegated last and therefore provide explicit, reviewable
overrides for smoke runs.

The original producer-facing launcher names are retained as portable wrappers
around the same config runner:

```bash
bash src/training/model_embeddings/mmbert_32k/run_rope_training.sh --print-command
bash src/training/model_embeddings/mmbert_32k/run_bge_style_training.sh --print-command
bash src/training/model_embeddings/mmbert_32k/run_rerank_2d_matryoshka_training.sh --print-command
```

`run_rope_training.sh` trains by default. Set `MMBERT32K_STAGE=prepare` to
materialize its pinned CC-100 input first. `MMBERT32K_PYTHON_BIN` selects the
Python executable; all data and output locations remain the explicit
environment variables above.

Install the Python layer inside the image recorded in `runtime.json`:

```bash
python -m pip install --requirement \
  src/training/model_embeddings/mmbert_32k/requirements.txt
```

PyTorch is intentionally supplied by the ROCm image rather than replaced by a
PyPI wheel. Record the image digest, resolved config, source commit, and output
checksums with every run.

The 30,774-example run is not divisible by accumulation 16. The trainer uses
ceiling optimizer-step counts, flushes the final six-example window, and scales
that window by six rather than sixteen. No examples or tail gradients are
silently dropped.

## Source-to-artifact facts and gaps

### Foundation

The published `training_config.json` records batch size 1, gradient
accumulation 16, one epoch, LR `1e-5`, MLM probability `0.3`, and both
retrieval masking and EWC disabled. The card reports 30,774 CC-100 sequences.
The original sample IDs and prepared dataset fingerprint were not published,
so the checked-in reconstruction is deterministic and auditable but cannot be
bit-identical to the historical corpus.

There is a second, important historical gap: at revision
`72a23a6640489471eb4ff7ad3ec5bc80af8a27de`, the released Hugging Face
`config.json` contains `max_position_embeddings=32768` but no `rope_scaling`.
That is evidence for the released artifact's context declaration, not evidence
that an official YaRN configuration was persisted. The imported upstream
post-load patch searched rotary modules for `dim`/`head_dim`; Transformers
4.57.6 ModernBERT rotary modules expose neither in that form, so it could warn
after patching zero layers and continue. This corrected recipe therefore must
not be described as a bit-exact reproduction of the historical weights. It is
a fail-closed, reproducible YaRN continuation recipe for a new run: it persists
both `max_position_embeddings=32768` and the full `rope_scaling` object
(`rope_type`, factor, original context, and beta values), forces `sdpa` or
`eager`, and refuses to train if any actual rotary layer is not YaRN-backed.

The upstream code contains both `warmup_steps=100` and `warmup_ratio=0.1`; its
implementation uses the ratio whenever it is non-zero, and the config preserves
that behavior.

### Foundation data-governance gate

The pinned CC-100 dataset card does not declare a dataset license, and the
corpus contains Common Crawl-derived web content. Repository ownership does
not resolve rights in that underlying content. The generated manifest therefore
sets publication to `blocked-pending-data-governance-review`. Do not redistribute
the packed corpus or publish weights trained by this reconstruction until the
project has reviewed the [CC-100 dataset card](https://huggingface.co/datasets/statmt/cc100),
the [Common Crawl terms of use](https://commoncrawl.org/terms-of-use), privacy,
provenance, and the intended release jurisdiction. A full-object archival/hash
decision is part of that gate; consumed-prefix hashes alone are only run audit
evidence. Preparation and training both fail closed unless the operator passes
the explicit unknown-license acknowledgement. The manifest and training
receipt retain that acknowledgement together with the Common Crawl terms URL;
it is an audit record, not a representation that release rights were granted.

### Embedder

The published artifact reports 32,768 tokens, batch size 16 with accumulation
2, one epoch, LR `2e-5`, and dimensions `768,512,256,128,64`. Historical
training mounted an uncommitted `sentence-transformers-fix` directory. The
artifact records `sentence-transformers 5.3.0.dev0`; the pinned reproducible
replacement is the public `5.3.0` release with Transformers `4.57.6`. This
replacement requires a short compatibility smoke before a full run.

The upstream launcher enabled AllNLI even though the model card only names the
BGE-M3 corpus. The config preserves launcher behavior and pins both AllNLI and
STS-B revisions. Change that choice only as an explicit experiment.

### Reranker

The published `training_args.json` is represented directly: 32,768 tokens,
three hard negatives, batch size 16 with accumulation 2, one epoch, LR `2e-5`,
and 20 heads across layers `3,6,11,22` and dimensions
`768,512,256,128,64`. The card names a `cfli/bge-m3-data` mirror that is no
longer anonymously resolvable; upstream documentation names the public
`Shitao/bge-m3-data`, so its immutable revision is the reproducible source in
both embedding configs.

The source computes every intermediate hidden state before scoring a selected
head. It trains early-exit-compatible heads but does not itself implement
compute-saving early termination.

## Tests

The contract suite is standard-library-only and does not download models. It
also exercises exact quotas, source-order packing with a fake tokenizer,
padding-aware real-length handling, config persistence, and fail-closed rotary
validation. It additionally covers manifest/Arrow tampering, audit/replay
prefix contracts, weak/same-ETag source changes, document caps, and full/tail
gradient accumulation. A real Datasets/PyArrow integration test runs whenever
those pinned dependencies are installed:

```bash
python -m unittest discover \
  -s src/training/model_embeddings/mmbert_32k/tests \
  -p 'test_*.py'
```

The canonical imported trainers are single-process, matching upstream. A
future distributed launcher should wrap these trainers without duplicating
their model or loss implementations.
