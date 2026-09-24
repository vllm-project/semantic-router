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

`mapper_id` pins Hugging Face weight revisions, dtype, and KV head count, not a
routing alias. `bundle_version` (`b1`, `b2`, …) is a re-fit of the same
revisions.

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
