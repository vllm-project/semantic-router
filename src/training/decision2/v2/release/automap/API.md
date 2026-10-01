# Decision models in 🤗 Transformers (`trust_remote_code`)

One API for every Decision model repository (Decision 2.0, and Decision 1.0 for uniformity): stock
`transformers` downloads the repository and runs it through the repository's own code. There is no
text generation and no chat API.

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer, pipeline

repo = "llm-semantic-router/DEV2.0-0.8B"
config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModel.from_pretrained(repo, trust_remote_code=True)  # cuda:0 if a GPU is visible, else CPU
response = model.system_one(state=..., questions={...})
decide = pipeline("decision", model=repo, trust_remote_code=True)
response = decide({"state": ..., "questions": {...}})
```

## Repository files

| File | Content |
| --- | --- |
| `config.json` | The repository's existing root file plus `model_type`, `architectures`, `auto_map` (`AutoConfig`, `AutoModel`) and `custom_pipelines` (`decision`). |
| `configuration_<family>.py` | The config class (`model_type` of the family, e.g. `decision2`); it only exposes `config.json`. |
| `modeling_<family>.py` | The model class: a `PreTrainedModel` whose `from_pretrained` loads the repository through its native runtime. |
| `pipeline_<family>.py` | Optional: the `decision` pipeline. |

The remote code is self-contained: it imports only the Python standard library, `torch`, `transformers`,
`huggingface_hub` and the runtime files shipped in the same repository. It makes no network calls other
than Hugging Face downloads of the repository (and, for a base-bound adapter, its pinned base) and sends no
telemetry. Transformers' dynamic-module loader copies only flat `*.py` files next to `config.json`, so a
runtime that lives in a package directory (Decision 2.0: `decision2/`) is copied from the downloaded
repository into the same dynamic-module directory, checked against the repository's file hashes, and
imported relative to the remote code.

## `system_one`

`model.system_one(*, state, questions)` is the package runtime's `system_one` (Decision 2.0:
`decision2.Decision2.system_one`), which is also what vLLM-SR serves at `POST /v1/decisions`:

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
    `invalid_question` or `invalid_model_output`.

`model(state=..., questions=...)` (`forward`) is the same call. Helpers named `choice` / `noul` / `score`
exist only where the native runtime has them; the Decision 2.0 runtime has none.

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
- Runtime options pass through: Decision 2.0 `threads`, `bf16_resident`, and for base-bound adapters
  `base_path` (a local copy of the pinned base revision).
- Everything else is refused with a `TypeError`.

Before any model code or weight is used, the native checks run: Decision 2.0 checks every repository file
against `MODEL_MANIFEST.json`, the loaded parameter count and the scored model identity, and a base-bound
adapter fetches exactly its pinned base files at the pinned revision (into the same cache, honouring the
Hub options) and checks each file's SHA-256. The Hugging Face cache stores files as links; the runtime
refuses links, so the check runs on a temporary hard-link view of the downloaded revision (copies only
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

## Kernels and devices

On a GPU, Transformers uses the flash-linear-attention and causal-conv1d kernels of the Qwen3.5
gated-delta layers when they are installed, else its PyTorch reference implementation (slower, same
operations). Those kernels are GPU-only but Transformers binds them at import, so on CPU the model runs
the reference implementations even when the kernel packages are installed.

## Parity

A repository revision with this API is published only if, on every scored prompt the release checks
(and mlx-diag), the `AutoModel` path gives 0 answer changes against the native runtime on the same
device and kernels, with the maximum probability drift reported.
