# Decision 1.0 in 🤗 Transformers (`trust_remote_code`)

The Decision 1.0 section of the shared remote-code API (`API.md`, written with the Decision 2.0 auto_map work):
the same calls, loading options, refusals and pipeline, with the differences below. Code:
[`decision1/`](decision1/); staging, parity, Hub and smoke tools: `stage1.py`, `parity1.py`, `hub1.py`,
`smoke1.py`; record: [`../records/dev1-automap-2026-10-01.md`](../records/dev1-automap-2026-10-01.md).

```python
from transformers import AutoConfig, AutoModel, pipeline

repo = "llm-semantic-router/Decision-1.0-Sol-2B"
config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
model = AutoModel.from_pretrained(repo, trust_remote_code=True)  # cuda:0 if a GPU is visible, else CPU
response = model.system_one(state=..., questions={...})
decide = pipeline("decision", model=repo, trust_remote_code=True)
```

## Repository files

| File | Content |
| --- | --- |
| `config.json` | Unchanged Decision file map (`decision_format: vllm-sr-decision`, format 1) plus `model_type: decision1`, `architectures: [Decision1Model]`, `auto_map` (`AutoConfig`, `AutoModel`) and `custom_pipelines.decision`. Key order and formatting of the existing keys are kept. |
| `configuration_decision1.py` | `Decision1Config`: exposes the file map. |
| `modeling_decision1.py` | `Decision1Model`: loads the files named by `config.json` (and nothing else) from one commit. |
| `pipeline_decision1.py` | `Decision1Pipeline` (task `decision`). |
| `decision1_system_one.py` | Request checks, candidate order, answers. |
| `decision1_vela.py` | Kai / Lex / Route: ModernBERT encoder vendored from Transformers 4.57.6 (SDPA path, YaRN RoPE), three typed paths, heads. |
| `decision1_qwen.py` | Eos / Sol / Nox / Lux: Transformers' `Qwen3_5TextModel` plus the FP32 candidate-endpoint head. |

Weights, tokenizers, calibration and every other file are unchanged. The repositories have no
`MODEL_MANIFEST.json`; Route's root `MANIFEST.json` gains the hashes of the changed and added files.
Imports: the standard library, `torch`, `transformers`, `safetensors`, `huggingface_hub`. No network
calls besides Hugging Face downloads, no telemetry.

## Numerics: the native Decision 1.0 runtime

| Family | Precision | Batching | Limit |
| --- | --- | --- | --- |
| Encoders (Kai, Lex, Route) | FP32 everywhere; during a call the published runtime's settings apply (no MHA fast path, no TF32) and are restored afterwards; ROCm GPUs get contiguous post-RoPE Q/K/V as in the native runtime | questions stably sorted by type, physical batches of 8, padded to the batch maximum; each path runs over the whole batch | 1,024 tokens |
| Decoders (Eos, Sol, Nox, Lux) | GPU: BF16 backbone (its stored precision) under BF16 autocast, FP32 head; CPU: FP32, gated-delta layers bound to the PyTorch references | questions in order, physical batches of 8, padded to a multiple of 32 | 16,384 tokens |

Prompts are rendered and tokenized segment by segment exactly as the native runtime does: Decoders use the
pointer-v2 prompt (candidate endpoints, global query; Nox renders a null Choice description as its key,
the others keep `null`); encoders use the marker layout (`[CLS] <type> question: … [SEP] [MASK] candidate
[SEP] … state [SEP]`). Calibration: the decoders' temperature (`config.json` for Eos, `temperature.json`
for Sol, Nox and Lux); the encoders use raw probabilities. Noul defaults are the runtime's per family.

## `system_one` (differences from Decision 2.0)

- Answers have the System One schema the vLLM-SR Decision runtime serves for these models; Choice and
  Score `confidence` is its `decision_type_aware_v1` statistic (Choice: top-two margin; Score: one minus
  the variance relative to a uniform distribution). The Decision 2.0 runtime reports its own statistic.
- Errors use the 2.0 forms: a malformed question is answered `{"type", "error": "invalid_question"}`
  and the others are answered. **Admission is per request, as in the native 1.0 runtime:** if any question
  of a request exceeds the limit, every question is answered `max_length_exceeded` (2.0 Qwen packages
  answer the fitting questions). Malformed `state` or `questions` raise `ValueError`.
- No `choice` / `noul` / `score` helpers (neither runtime has them).

## Tokenizer

Decoders: `AutoTokenizer.from_pretrained(repo)` works (tokenizer files at the root). Encoders keep their
tokenizer under `native/tokenizer`: `AutoTokenizer.from_pretrained(repo, subfolder="native/tokenizer")`.

## Requirements

Encoders: Transformers 4.57 or later (checked on 4.57.6, 5.17.0 and 5.18.0). Decoders: Transformers
5.17 or later (Qwen3.5; checked on 5.17.0 and 5.18.0); flash-linear-attention is optional on GPUs.

## Serving compatibility

The vLLM-SR Decision runtime (`xunzhuo/decision-runtime`, `58cd660b5`) accepts only the Decision keys
in a root `config.json`. Catalogs pin earlier revisions, so serving is unaffected; before a catalog moves
to a revision with these keys, `parse_decision_config` must ignore `model_type`, `architectures`,
`auto_map` and `custom_pipelines`. Repository Python is never selected by that runtime.
