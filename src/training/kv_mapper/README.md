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

Collect, fit, and evaluation entrypoints land in later PRs.
