# Cross-model KV mapper artifacts

Ridge-mapper files for cross-model KV transfer
([#2976](https://github.com/vllm-project/semantic-router/issues/2976)).
This family owns the on-disk contract the router names with
`x-vsr-kv-mapper-id`. Apply is a vLLM KVConnector.

Weights, activations, and evaluation dumps stay outside Git.

## Layout

A published mapper is a directory:

```
qwen3-14b-32b-full_head-fp16-tp1-h8-s<src>-t<tgt>-b1/
  manifest.json
  weights.safetensors
  SHA256SUMS
```

Tensor keys: `target.{layer}.k.W`, `target.{layer}.k.b`, and the same for `v`.

`mapper_id` includes the full immutable Hugging Face weight commits, dtype, and
KV head count. Branch names and tags must be resolved before fitting and cannot
be written into an artifact. `bundle_version` (`b1`, `b2`, …) is a re-fit of the
same commits.

## Install

```bash
cd src/training/kv_mapper
pip install -r requirements.txt
```

## Contract tests

From the repository root:

```bash
python3 -m unittest discover -s src/training/kv_mapper/tests -p 'test_*.py'
```

## Fit

## Collect

`collect.py` is the run metadata and window/layer-subset contract (numpy, CI).
`hooks.py` attaches k_norm/k_proj and v_proj hooks and reads K after RMSNorm
and before RoPE. `collect_run.py` writes `run.json`, `tokens.npy`, and atomic
per-sequence chunks under `source/` and `target/`. It streams and shuffles the
pinned corpus with `--seed`, then captures `--num-sequences` token windows of
`--seq-len` with `--window-stride`. `--fitting-token-step` selects every fourth
token from each distinct sequence by default. Each chunk stores K and V with
shape `(layers, sampled_tokens, num_kv_heads, head_dim)` in float32. Chunks have
SHA-256 sidecars and are checked before a resumed run skips them. `run.json`
records the exact token fingerprint, dataset revision, and fitting recipe.
Model revision names are resolved to full weight commits before loading, and
those commits are stored in `run.json`.
Windows stay within each corpus document; short documents are skipped, and no
synthetic transition is created between adjacent documents.
The models must use the same tokenizer vocabulary so positions stay paired.
That script needs torch, transformers, and datasets; it is not part of
`make test-training-contracts`.

```bash
PYTHONPATH=. python3 src/training/kv_mapper/collect_run.py \
  --source-model Qwen/Qwen3-14B --source-revision <sha> \
  --target-model Qwen/Qwen3-32B --target-revision <sha> \
  --dataset-revision <sha> --dtype bf16 \
  --num-kv-heads 8 --head-dim 128 --output-dir /tmp/kv-collect
```

## Fit

`fit.py` ranks source layers by single-source per-head affine OLS R² averaged
over K and V, then fits a centered ridge with bias (keys and values separately),
and writes the A1 directory via
`write_fitted_artifact`.
`fit_run.py` validates every captured chunk, loads the sampled rows into host
RAM, selects one shared source-layer list per target, fits K and V, writes the
artifact, and reads it back to verify checksums. It stores completed target
fits under `.fit-work/` for restart after interruption. A full 500-sequence
Qwen3 run needs a high-memory machine; this script has not yet been validated
on the model pair.
Publishing refuses to overwrite an existing artifact. Use `--bundle-version 2`
for a new fit of the same pinned model pair and precision.

```bash
PYTHONPATH=. python3 src/training/kv_mapper/fit_run.py \
  --run-dir /tmp/kv-collect --output-dir /tmp/kv-artifacts \
  --pair-slug qwen3-14b-32b --topk 8 --ridge-alpha 0.01
```

```bash
PYTHONPATH=. python3 -m unittest src.training.kv_mapper.tests.test_fit
```

## Eval

`eval.py` computes KV-space rel_err / cosine / R² and paired bootstrap CIs on
per-item scores (same examples, every arm). `eval_report.py` reads a JSON dump
from a GPU run and writes the report. Inject / HellaSwag / CoQA collection
stays on the GPU box; this PR is the CI math.

```bash
PYTHONPATH=. python3 src/training/kv_mapper/eval_report.py \
  --items /tmp/inject_items.json --output /tmp/inject_report.json
```

`items.json` shape: `metric`, optional `reference` (default `cold`), and
`arms` mapping each arm to records of `{ "id": "example-id", "score": 0.0 }`.
Every arm must have the same unique example IDs; record order may differ.

`model_eval_run.py` runs held-out HellaSwag validation with the pinned source
and target revisions from an artifact. It uses a seeded random sample with
stable example IDs. Each item scores the same endings against four target-model
cache arms: cold target, mapped source, raw source, and zero KV. The report
contains paired bootstrap intervals for total-log-probability HellaSwag
accuracy, length-normalized accuracy, and mean gold-ending log probability,
plus per-layer-averaged pre-RoPE K/V relative errors. Per-item files include
ending token counts so both accuracy rankings can be reproduced. The scorer
checks its cached continuation against a full forward pass in a small Qwen3
model test. The GPU runner needs torch, transformers, and datasets in addition
to the artifact dependencies.

```bash
PYTHONPATH=. python3 src/training/kv_mapper/model_eval_run.py \
  --artifact /tmp/kv-artifacts/<mapper-id> \
  --dataset-revision <hellaswag-sha> --count 100 --seed 42 \
  --output-dir /tmp/kv-eval
```

This run measures teacher-forced target cache injection. Connector reuse and
fallback require their own integration evaluation.

## Distill

`distill.py` adds a second stage on top of the ridge fit, following KV-Lingo
([arXiv 2609.32610](https://arxiv.org/abs/2609.32610)). The maps keep their
shape and the same `features @ W + b` application, so the artifact, the
connector and the headers do not change. Ridge makes the mapped cache close to
the target's own cache. The second stage trains the maps so the target predicts
the same next tokens from the mapped cache as from its own: the source prefills
the prefix, the target reads the continuation on the mapped cache, and the loss
is the KL from the target's own distribution. Both models stay frozen and only
the maps receive gradients.

`distill_run.py` reads a stage-1 artifact, streams chat conversations from a
pinned dataset revision, cuts each one before an assistant reply, holds out
`--val-count` conversations, and trains with AdamW (no weight decay), a cosine
schedule with 5% warmup and clipping at 1.0. It writes the same layout under
the next `bundle_version` (or `--bundle-version`) and records the recipe and
the validation KL before and after under `calibration.stage2`. It refuses to
overwrite an existing artifact.

The defaults are the recipe measured on Qwen3-14B to Qwen3-32B (#2976): each W
stays frozen and a rank-16 correction is trained in the NoRA form
([arXiv 2608.31036](https://arxiv.org/abs/2608.31036)), and each tensor's rate
is `--lr` times the RMS of its map, because value maps there are about fifty
times smaller than key maps. Training every entry of W (`--rank 0`) needs a
rate near `1e-5` to come close; at `1e-4` it lowers the chat KL while plain
text gets worse than the ridge fit. Check the result on held-out plain text as
well as on chat, and compare it with `model_eval_run.py` on the same items as
its stage-1 artifact. Needs torch, transformers and datasets.

```bash
PYTHONPATH=. python3 src/training/kv_mapper/distill_run.py \
  --artifact /tmp/kv-artifacts/<stage1-mapper-id> --pair-slug qwen3-14b-32b \
  --dataset HuggingFaceH4/ultrachat_200k --dataset-revision <sha> \
  --split train_sft --output-dir /tmp/kv-artifacts
```

```bash
PYTHONPATH=. python3 -m unittest src.training.kv_mapper.tests.test_distill
```

## KL evaluation

`kl_eval_run.py` measures how close each artifact's mapped cache keeps the
target to its own cache, on held-out chat (cut before an assistant reply, like
the stage-2 data) and on plain text (the first tokens of documents from a
second corpus). For every sample the target's own distribution and the source
prefill are computed once and each artifact's maps run on them, so artifacts
are paired by sample. The report gives the mean KL from the target's own
distribution and the gold-token NLL per artifact, with paired bootstrap
intervals against the first `--artifact`. On Qwen3-14B to 32B the ridge mapper
already matches the cold target on HellaSwag, so this is the measure that
separates mappers there. Needs torch, transformers and datasets.

```bash
PYTHONPATH=. python3 src/training/kv_mapper/kl_eval_run.py \
  --artifact v1=/tmp/kv-artifacts/<stage1-mapper-id> \
  --artifact stage2=/tmp/kv-artifacts/<stage2-mapper-id> \
  --chat-revision <ultrachat-sha> --text-revision <wikipedia-sha> \
  --output /tmp/kv-eval/kl.json
```
