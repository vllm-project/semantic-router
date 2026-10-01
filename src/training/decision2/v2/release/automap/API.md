# Decision models in 🤗 Transformers (`trust_remote_code`)

One API for every Decision model repository, Decision 2.0 (`Decision-2.0-*`) and Decision 1.0
(`Decision-1.0-*`): stock `transformers` downloads the repository and runs it through the repository's
own code, which wraps the family's native runtime. There is no text generation and no chat API. The
common surface comes first; [Divergences](#divergences-between-decision-10-and-20) lists every place
where the two families differ.

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer, pipeline

repo = "llm-semantic-router/Decision-2.0-Eos-0.8B"  # or e.g. "llm-semantic-router/Decision-1.0-Sol-2B"
config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)  # 1.0 encoders: see Tokenizer
model = AutoModel.from_pretrained(repo, trust_remote_code=True)  # cuda:0 if a GPU is visible, else CPU
response = model.system_one(state=..., questions={...})
decide = pipeline("decision", model=repo, trust_remote_code=True)
response = decide({"state": ..., "questions": {...}})
```

Code: Decision 2.0 in this directory (`configuration_decision2.py`, `modeling_decision2.py`,
`pipeline_decision2.py`, carried into each package by the release builder; record
[`../records/dev2-automap-2026-10-01.md`](../records/dev2-automap-2026-10-01.md)); Decision 1.0 in
[`decision1/`](decision1/) with `stage1.py`, `parity1.py`, `hub1.py`, `smoke1.py` (record
[`../records/dev1-automap-2026-10-01.md`](../records/dev1-automap-2026-10-01.md)).

## Repository files

| File | Content |
| --- | --- |
| `config.json` | The repository's existing root file plus `model_type`, `architectures`, `auto_map` (`AutoConfig`, `AutoModel`) and `custom_pipelines` (`decision`). Existing keys keep their order and formatting. |
| `configuration_<family>.py` | The config class (`model_type` `decision2` or `decision1`); it only exposes `config.json`. |
| `modeling_<family>.py` | The model class: a `PreTrainedModel` whose `from_pretrained` loads the repository through its native runtime. |
| `pipeline_<family>.py` | The `decision` pipeline. |
| runtime | 2.0: the package's `decision2/` directory (unchanged). 1.0: flat `decision1_system_one.py`, `decision1_vela.py` (Kai / Lex / Route encoders), `decision1_qwen.py` (Eos / Sol / Nox / Lux decoders). |

Weights, tokenizers, calibration and every other file are byte-identical. 2.0 repositories record the
added files in `MODEL_MANIFEST.json`; 1.0 repositories have no `MODEL_MANIFEST.json` (Route's root
`MANIFEST.json` gains the hashes of the changed and added files).

The remote code is self-contained: it imports only the Python standard library, `torch`, `transformers`,
`safetensors`, `huggingface_hub` and the runtime files shipped in the same repository. It makes no network
calls other than Hugging Face downloads of the repository (and, for the 27B, its pinned base) and sends no
telemetry. Transformers' dynamic-module loader copies only flat `*.py` files next to `config.json`, so the
2.0 runtime package directory `decision2/` is copied from the downloaded repository into the same
dynamic-module directory, checked against the repository's file hashes, and imported relative to the
remote code.

## `system_one`

`model.system_one(*, state, questions)` is the family's native System One call (2.0:
`decision2.Decision2.system_one`, which vLLM-SR serves at `POST /v1/decisions`; 1.0: the System One
schema the vLLM-SR Decision runtime serves for those models):

- `state`: text, a JSON object or a JSON array.
- `questions`: a non-empty mapping from question ID to a question:
  - `{"type": "choice", "instructions": ..., "criteria": {<key>: <description or null>, ...}}` (2–255 keys);
  - `{"type": "noul", "instructions": ..., "criteria": {"false": ..., "true": ...}}` (`criteria` optional);
  - `{"type": "score", "instructions": ..., "criteria": [<level 0>, <level 1>, ...]}` (2–10 levels).
- Returns `{"model": <model name>, "answers": {<question ID>: <answer>}, "usage": {"input_tokens": n,
  "output_tokens": 0}}`, answers in question order:
  - Choice: `{"type", "choice", "probabilities", "confidence"}`;
  - Noul: `{"type", "noul"}` (P(true));
  - Score: `{"type", "score", "probabilities", "confidence", "legend"}` (`score` is the expected level);
  - a question that cannot be answered: `{"type", "error"}` with `max_length_exceeded` (never truncated),
    `invalid_question` or `invalid_model_output`. A malformed question is answered `invalid_question` and
    the others are still answered; malformed `state` or `questions` raise `ValueError`.

`model(state=..., questions=...)` (`forward`) is the same call. Neither family has `choice` / `noul` /
`score` helpers.

## Loading

`AutoModel.from_pretrained(repo_or_dir, trust_remote_code=True, **kwargs)`:

- `repo_or_dir`: a Hub ID (downloaded with `huggingface_hub.snapshot_download` into the standard cache) or
  a local download (`hf download <repo> --local-dir <dir>`).
- Hub options: `revision`, `cache_dir`, `token`, `local_files_only`, `force_download`. Config, code and
  weights come from one commit.
- Device: `device` or `device_map` names one device (`"cpu"`, `"cuda"`, `"cuda:N"`, `N`, a `torch.device`,
  `{"": device}` or `"auto"` for the runtime default). The runtime runs on one device.
- Numerics are the native runtime's, so `dtype` / `torch_dtype` accept only `None` or `"auto"` and
  `attn_implementation` only `None` or `"sdpa"`.
- 2.0 runtime options pass through: `threads`, `bf16_resident`, and for the 27B `base_path` (a local copy
  of the pinned base revision).
- Everything else is refused with a `TypeError`.

Before any model code or weight is used, the native checks run. 2.0 checks every repository file against
`MODEL_MANIFEST.json`, the loaded parameter count and the scored model identity. 1.0 loads only the files
named by `config.json`, all from one commit. The Hugging Face cache stores files as links; the 2.0 runtime
refuses links, so its check runs on a temporary hard-link view of the downloaded revision (copies only
where hard links are impossible), which is removed after loading.

`model.to(device)`, `model.cuda()` and `model.cpu()` reload the repository on the new device through
the same native path, so a moved model answers exactly as one loaded there. Dtype casts (`half()`,
`float()`, `bfloat16()`, `to(dtype)`), `train()`, `save_pretrained()` and `push_to_hub()` are refused:
the repository itself is the distributable package. `model.runtime` is the native runtime object and
`model.config` the repository's `config.json`.

## Pipeline

`pipeline("decision", model=repo, trust_remote_code=True)` (task `decision`, model class `AutoModel`)
accepts `{"state": ..., "questions": {...}}`, `state=..., questions=...` keywords or a list of requests,
and returns the `system_one` response (or a list). `device` / `device_map` work as above; `batch_size`
must be 1 (the runtime already batches the questions of one request).

## Divergences between Decision 1.0 and 2.0

| Topic | Decision 2.0 (`Decision-2.0-*`) | Decision 1.0 (`Decision-1.0-*`) |
| --- | --- | --- |
| Over-length admission | Per question: a question over the package's limit is answered `max_length_exceeded`; the fitting questions of the same request are answered. | Per request, as the native 1.0 runtime: if any question exceeds the limit, every question of the request is answered `max_length_exceeded`. Limits: encoders 1,024 tokens, decoders 16,384. |
| `confidence` (Choice, Score) | One minus the normalized entropy of `probabilities` (`1 − H(p) / ln K`, clipped to [0, 1]). | The vLLM-SR Decision runtime's `decision_type_aware_v1`: Choice the top-two margin; Score one minus the variance relative to a uniform distribution. |
| Tokenizer | At the repository root for every tier: `AutoTokenizer.from_pretrained(repo)`. | Decoders (Eos, Sol, Nox, Lux) at the root; encoders (Kai, Lex, Route) under `native/tokenizer`: `AutoTokenizer.from_pretrained(repo, subfolder="native/tokenizer")`. |
| CPU vs GPU numerics | GPU: BF16 backbone under BF16 autocast with an FP32 head; CPU: FP32 throughout. CPU answers equal the native CPU runtime's, but not always the GPU-scored ones (spot checks: 3–4 of 200 decisions differ, drift ≤ 0.005). | Decoders as 2.0 (GPU BF16 / FP32 head, CPU FP32), with the same caveat. Encoders run FP32 on both devices under the published runtime's settings (no MHA fast path, no TF32; restored after each call), so CPU and GPU agree to rounding. |
| Base download | The 27B is an unmerged LoRA adapter (under `adapter/`, so Transformers' PEFT auto-detection does not redirect `AutoModel` to the base). Loading fetches exactly the pinned `Qwen/Qwen3.8-27B` files at the pinned revision into the same cache (honouring the Hub options) and checks each file's SHA-256, or takes `base_path`. All other tiers are self-contained. | None: every repository is self-contained. |
| Requirements | Transformers 5.17 or later (checked on 5.17.0 and 5.18.0). | Encoders: Transformers 4.57 or later (checked on 4.57.6, 5.17.0, 5.18.0); decoders: 5.17 or later (checked on 5.17.0, 5.18.0). |
| Integrity | `MODEL_MANIFEST.json` and `verify_bundle` over every file. | Files named by `config.json`; Route's `MANIFEST.json` hashes. |

Decision 1.0 prompt rendering is the native runtime's, segment by segment: decoders use the pointer-v2
prompt (candidate endpoints, global query; Nox renders a null Choice description as its key, the others
keep `null`); encoders use the marker layout (`[CLS] <type> question: … [SEP] [MASK] candidate [SEP] …
state [SEP]`), sorted stably by type in physical batches of 8. Calibration: the decoders' temperature
(`config.json` for Eos, `temperature.json` for Sol, Nox and Lux); the encoders use raw probabilities.

## Kernels and devices

On a GPU, Transformers uses the flash-linear-attention and causal-conv1d kernels of the Qwen3.5
gated-delta layers when they are installed, else its PyTorch reference implementation (slower, same
operations, last-digit differences). Those kernels are GPU-only but Transformers binds them at import,
so on CPU the model runs the reference implementations even when the kernel packages are installed.
This applies to the Qwen3.5 tiers of both families (2.0 0.8B–27B; 1.0 decoders).

## Serving compatibility

The vLLM-SR Decision runtime for Decision 1.0 (`xunzhuo/decision-runtime`, `58cd660b5`) accepts only the
Decision keys in a root `config.json`. Catalogs pin earlier revisions, so serving is unaffected; before a
catalog moves to a 1.0 revision with these keys, `parse_decision_config` must ignore `model_type`,
`architectures`, `auto_map` and `custom_pipelines`. Repository Python is never selected by that runtime.

## Parity

A repository revision with this API is published only if, on every scored prompt the release checks
(and mlx-diag for 2.0), the `AutoModel` path gives 0 answer changes against the native runtime on the same
device and kernels, with the maximum probability drift reported.
