# vllm-sr-plugins

vLLM plugins that serve vLLM Semantic Router models on vLLM **without modifying
vLLM**. They reuse vLLM's scheduler, continuous batching, kernels and hardware
platforms, and register through vLLM's documented plugin entry points.

Status: prototype (v0). It serves Decision 2.0 full Qwen3.5 packages. Vela 1.0
classifiers need no plugin class: vLLM's native ModernBERT sequence
classification serves them. Results, limitations and next steps are in
[`src/training/decision2/v2/serving/records/`](../training/decision2/v2/serving/records/).

## Contents

| Entry point | Name | What it adds |
| --- | --- | --- |
| `vllm.general_plugins` | `vllm_sr_decision2` | Registers `Decision2Qwen3_5ForScoring`, a pooling model: vLLM's text-only Qwen3.5 backbone plus the Decision 2.0 candidate head in FP32, with no LM head. |
| `vllm.endpoint_plugins` | `vllm_sr_decisions` | `POST /v1/decisions` (alias `/v1/system_one`): the Decision 2.0 System One contract of the package runtime. |

- **`Decision2Qwen3_5ForScoring`** (`decision2/model.py`) loads a package directory directly:
  `backbone/*.safetensors` into vLLM's `Qwen3_5Model` and `decision_head.safetensors` into
  `CandidateHead` (`decision2/head.py`, the same parameters and arithmetic as the training head).
  It applies vLLM's own Qwen3.5 text-model config hook, so the gated-delta state is FP32 as the
  checkpoint's `mamba_ssm_dtype` says, and positions are one-dimensional.
- **`CandidatePooler`** (`decision2/pooler.py`) serves the `token_classify` pooling task (the built-in task with
  per-request output lengths, accepted by both vLLM model runners). Each request
  carries its option-endpoint and query positions in `PoolingParams.extra_kwargs["decision2"]`.
  The pooler copies those hidden rows out of every prefill step, so chunked prefill works, and
  returns one FP32 logit per candidate when the prompt is complete.
- **`/v1/decisions`** (`decision2/endpoint.py`, `decision2/service.py`; `/v1/system_one` is an
  alias) opens the served package
  like `Decision2.from_pretrained` does: every file is checked against `MODEL_MANIFEST.json` and the
  model identity against the scored identity. Prompt rendering, tokenization, calibration, Score
  offsets and answer formatting then use the **package's own vendored runtime**
  (`decision2/_vendor/dev2model`), imported under a private module name. Each valid question is one
  engine request; a request's questions run concurrently.

## Tested build

One vLLM build is tested: **vLLM `0.29.1rc1.dev187+gaf1c01499.rocm723`** in the ROCm image
`decision20-train-fast:host2` (PyTorch `2.12.0+git6bbd260`, Transformers 5.17.0, Triton 3.7.1,
ROCm 7.2, MI325X / gfx942). Pooler, weight-loader and endpoint-plugin interfaces change between
vLLM releases, so the plugin logs a warning on any other build. Moving to a new vLLM needs the unit
tests and the parity panels to pass on it first.

## Serve a Decision 2.0 package

```bash
pip install --no-deps src/vllm-sr-plugins          # into the environment that has vLLM
export VLLM_PLUGINS=vllm_sr_decision2,vllm_sr_decisions
PKG=/path/to/DEV2.0-0.8B                          # a downloaded package directory
vllm serve "$PKG" --runner pooling \
  --hf-config-path "$PKG/backbone" --tokenizer "$PKG" \
  --hf-overrides '{"architectures": ["Decision2Qwen3_5ForScoring"]}' \
  --dtype bfloat16 --max-model-len 16384 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.2 --served-model-name DEV2.0-0.8B
```

`VLLM_PLUGINS` must name both plugins: vLLM loads endpoint plugins only when they are listed,
and the list also filters general plugins. `--max-model-len` is the package's
`max_input_tokens`; longer questions are answered with `max_length_exceeded`, never truncated.
Keeping `--max-num-batched-tokens` at least that large lets most prompts prefill in one step.

```bash
curl -s localhost:8000/v1/decisions -H 'Content-Type: application/json' -d '{
  "state": {"ticket": "The invoice total is wrong and I was charged twice."},
  "questions": {
    "route": {"type": "choice", "instructions": "Which team should handle this?",
              "criteria": {"billing": "Payments and invoices", "tech": "Product defects"}},
    "urgent": {"type": "noul", "instructions": "Does the customer report a double charge?"}
  }
}'
```

The response is `{"model", "answers", "usage"}` exactly as `Decision2.system_one` returns it.

## Serve a Vela 1.0 classifier

No plugin is involved; this uses vLLM's `ModernBertForSequenceClassification`:

```bash
vllm serve /path/to/Vela-1.0-Encoder-307M-Domain --runner pooling --dtype float32 \
  --max-model-len 32768 --served-model-name vela-domain
curl -s localhost:8000/classify -H 'Content-Type: application/json' \
  -d '{"model": "vela-domain", "input": "Why do bond prices fall when interest rates rise?"}'
```

## Limits of v0

- One model per engine. Separate engines share a GPU through `--gpu-memory-utilization`.
- Served package profile: `qwen-full` with the shared candidate head. Base-bound adapter packages
  (27B), bf16z-compressed packages, Qwen3 backbones and Kai-native encoders are not served yet.
- Prefix caching is off for hybrid pooling models in vLLM, so a shared state is re-read by every
  question.
- A request without positions (for example a raw `/pooling` call) gets a NaN logit, which System
  One reports as `invalid_model_output`.

## Tests

The tests run where PyTorch (and, for the pooler and registration tests, vLLM) is installed, for
example inside the image:

```bash
cd src/vllm-sr-plugins && python3 -m unittest discover -s tests -v
```
