# 🤗 Transformers auto_map / trust_remote_code for the six DEV2.0 repositories (2026-10-01)

User request 2026-10-01 10:53 UTC+8 (coordinator note 11:20, worker 4c0a68cd): every DEV2.0 model supports standard
Hugging Face usage through `auto_map` / `trust_remote_code`, so stock `transformers` downloads and runs it. Release
worker, worktree `vllm-sr-dev2-automap` (branch `xunzhuo/decision-2-automap`); state
[`dev2-automap-state.md`](dev2-automap-state.md); API spec [`../automap/API.md`](../automap/API.md) (shared with the
Decision 1.0 auto_map worker). Node receipts are copied under [`dev2-automap-2026-10-01/`](dev2-automap-2026-10-01/);
per-prompt answers stay on the node. Times are UTC. Everything stays private.

## Result

**Every DEV2.0 package now loads with stock 🤗 Transformers** —
`AutoModel.from_pretrained(repo, trust_remote_code=True)` (and `AutoConfig`, `AutoTokenizer`,
`pipeline("decision", ...)`) — **through its own native runtime, with 0 answer changes against the native runtime
on every scored prompt and on mlx-diag for all six models (max drift 0.0)**, under Transformers 5.17.0 and 5.18.0.
Published as runtime-only revisions on top of the BF16-resident rollout (weights byte-identical; private; the
collection unchanged). Status at the worker's handoff (07:40Z): 0.6B, 0.8B, 2B, 4B published; 9B and 27B releases
running on node E (27B uploaded `2b7508e9`, its post-download checks running; 9B in its 5.18 parity).

| Model | AutoModel vs native, Transformers 5.17: answers changed (max drift) | 5.18 | New `main` | Final decision (gate items) | Hub smoke 5.17 / 5.18 |
| --- | --- | --- | --- | --- | --- |
| DEV2.0-0.6B | 0 of 11,053 (0) | 0 of 11,053 (0) | `08b00e07cb90472e4184e3b91f3188c0ead8935b` | `469b4d7d…` (13 / 13) | pass / pass |
| DEV2.0-0.8B | 0 of 11,053 (0) | 0 of 11,053 (0) | `1188dd33cc2d69381d3c37c9259b4f4163a666a6` | `591a9ae2…` (7 / 7) | pass / pass |
| DEV2.0-2B | 0 of 11,053 (0) | 0 of 11,053 (0) | `ffe291b4401930040914339592cd9fc59b9a3f87` | `6e15df74…` (7 / 7) | pass / pass |
| DEV2.0-4B | 0 of 11,053 (0) | 0 of 8,778 (0) | `3785b7b963d2f56de0e44f9ec638c814c5ee6499` | `799be1b4…` (7 / 7) | pass / pass |
| DEV2.0-9B | 0 of 11,053 (0) | — | — | — | — / — |
| DEV2.0-27B | 0 of 11,053 (0) | 0 of 11,053 (0) | `2b7508e95a243d71e854a0743e14c1514f4c29dc` | — | — / — |

The 5.18 column for 4B covers typed-final, css15 and public231 (its mlx-diag parity uses that run's own autotune
cache and ran under 5.17 only). Gate items: the successor profile (0.6B, 27B) adds its R1–R7 items.

## 1. Design

```python
from transformers import AutoModel, pipeline

model = AutoModel.from_pretrained("llm-semantic-router/DEV2.0-0.8B", trust_remote_code=True)
model.system_one(state=..., questions={...})        # {"model", "answers", "usage"}, the native response
pipeline("decision", model="llm-semantic-router/DEV2.0-0.8B", trust_remote_code=True)({"state": ..., "questions": {...}})
```

- **Files (repository root):** `configuration_decision2.py` (`Decision2Config`, `model_type` `decision2`; it only
  exposes `config.json`), `modeling_decision2.py` (`Decision2Model`, a `PreTrainedModel`), `pipeline_decision2.py`
  (`Decision2Pipeline`, task `decision`). `config.json` keeps every pointer field and gains `model_type`,
  `architectures`, `auto_map` (`AutoConfig`, `AutoModel`) and `custom_pipelines` (`decision` → `AutoModel`).
  `MODEL_MANIFEST.json` lists their SHA-256 (`files_sha256`, `remote_code`); `verify_bundle` covers them.
- **The model is the package's own runtime.** `Decision2Model.from_pretrained` resolves the repository (the config's
  commit on the Hub, or a local download), copies the package's `decision2/` runtime next to the remote code in
  Transformers' dynamic-module cache — each file checked against the manifest; Transformers itself copies only flat
  modules — imports it relative to `modeling_decision2`, and calls `Decision2.from_pretrained`. Everything the
  native path checks is checked: every file against `MODEL_MANIFEST.json`, the loaded parameter count, the scored
  model identity, and for the 27B adapter the 28 pinned base files at the pinned revision (fetched with the same
  Hub options into the same cache). `system_one` / `forward` / the pipeline call the native `system_one`.
- **Hugging Face cache links.** The runtime refuses links and the cache stores files as links, so a cached
  revision is loaded from a temporary hard-link view next to its blobs (copies only where hard links fail),
  removed after loading.
- **Device and numerics.** `device` or `device_map` names one device (`"auto"` = the runtime default, cuda:0 if
  visible); `dtype` only `None` / `"auto"` (FP32 on CPU; BF16 autocast with BF16-resident Linear weights and an
  FP32 head on a GPU, as natively); `to(device)` / `cuda()` / `cpu()` reload the package on that device through the
  native path; casts, `train()`, `save_pretrained()` and `push_to_hub()` are refused.
- **Kernels.** On a GPU, Transformers uses flash-linear-attention / causal-conv1d when installed, otherwise its
  PyTorch reference (as natively). Those kernels are GPU-only but bound at import, so on CPU the model gives each
  Qwen3.5 gated-delta layer a forward that names the reference functions (per layer; nothing global changes).
- **PEFT auto-detection.** An `adapter_config.json` at the repository root makes `AutoModel` (and `pipeline`) load
  the base named in it and apply the adapter with Transformers' PEFT integration, never calling the remote code.
  The 27B package keeps its adapter under `adapter/`, so `find_adapter_config_file(repo)` is `None` and `AutoModel`
  reaches `Decision2Model`; the integration test shows the redirect with a root copy (section 4).

## 2. Runtime change (`0cbf1033e`, shared module, separate commit)

With `auto_map` in the root `config.json`, the vendored loader's `AutoTokenizer.from_pretrained(package)` (which in
Transformers 5 always loads `AutoConfig` first, without `trust_remote_code`) asked on the terminal whether to run
the remote code — up to 15 s, and the prompt text landed on stdout ahead of the native card example's JSON.
`Decision2.from_pretrained` now runs with `TIME_OUT_REMOTE_CODE = 0` (restored afterwards), so the question is
refused at once and the tokenizer comes from `tokenizer_config.json`, the same fallback as before. Weights,
numerics and answers are unchanged (parity below); the packages' runtime is the BF16-resident runtime plus this.

## 3. Pipeline (shared modules, called out)

- `build.py` / `layout.py` / `card.py` (`83e04b0dc`): the remote code is copied into every build (`automap_source`
  pins it like `runtime_source`); the card gains "Use with 🤗 Transformers" (`check_rendered` requires it when the
  package ships the remote code).
- `examples.py` (`08f183855`): `automap` (AutoConfig / AutoTokenizer / AutoModel / forward / pipeline vs a native
  run, bit-identical), `automap-card` (the card's Transformers block, local or `--hub`), `automap-parity`,
  `compare-answers`; `parity --answers` writes per-prompt answers.
- `release.sh` / `gate.py` (`8e8082440`): the automap steps before and after upload, the Hub smoke (`automap-hub`,
  fresh `HF_HOME`, host network, token file mounted read-only; `--hub-site` adds Transformers 5.18); gate item
  `7_transformers_remote_code`. A leased GPU is now passed as its own render node plus `/dev/kfd` with
  `ROCR_VISIBLE_DEVICES=0` (never all of `/dev/dri`), as shared nodes require.

## 4. Tests

- **stdlib** (`v2.release.tests.test_automap`, 13, in the image with torch; the release suite passes): config fields only with remote code;
  copies byte-identical; `auto_map` names classes that exist; the remote code imports only flat relative modules
  plus stdlib / torch / transformers / huggingface_hub; the card section (example compiles, names the repository,
  the same request as the native example, base note, tested versions; `check_rendered` requires it); card block
  selection; answer-file comparison; gate item 7; the runtime's prompt refusal is scoped to the load.
- **Image integration** (`v2.release.tests.automap_integration`; tiny Qwen3 / Qwen3.5 hybrid `qwen-full` and Qwen3
  LoRA `qwen-adapter` packages built by the release builder):
  - GPU (node E GPU6, FLA kernels; mirror `07e2c3ffe`): native vs AutoModel / pipeline **bit-identical** for all
    three; the card's Transformers block passes; and the offline-cache checks below, plus `model.to("cpu")`
    (reloads on CPU through the native path; same decisions).
  - CPU: Qwen3 full and adapter bit-identical; Qwen3.5 recorded as skipped (the image's PyTorch has no CPU LAPACK).
    From an offline Hugging Face cache layout: AutoModel by repository ID through the hard-link view (view removed
    afterwards), casts / dtype / `save_pretrained` refused, the `adapter/` layout loads `Decision2Model`, and **with
    an `adapter_config.json` (and weights) at the root AutoModel loads `Qwen3Model`** — Transformers' PEFT
    detection redirects to the named base and applies the adapter itself, bypassing the remote code.
- **CPU spot checks with a standard PyTorch** (2.12.0+cpu overlay; Transformers 5.17): 0.8B (Qwen3.5; FLA on the
  AutoModel path only, so all 18 gated-delta layers get reference forwards) and 0.6B (Qwen3): the examples
  bit-identical to the native CPU run and **200 / 200 typed-final prompts identical (drift 0)**. Against the
  GPU-scored predictions the CPU (FP32) answers differ: 0.8B 4 / 200 and 0.6B 3 / 200 decisions, drift ≤ 0.005.

## 5. Parity: AutoModel vs the native runtime

Node E (GPU6 / GPU7, one render node per container), the scored images (`host2` `f83b1d10`; 27B `latest`
`dbe5f32b`), FLA 0.5.2 + causal-conv1d 1.7.0 required, a fresh copy of each scored run's frozen Triton autotune
cache, the rollout's prompts and sealed predictions relayed from node A (SHA-256 lists compared). `release.sh`
without upload on the draft specs (`automap.sh --verify`; 4B mlx-diag in its own run with its own cache, `--mlx`):
native `parity-pre` writes every prompt's answers, `automap-parity-pre` the same prompts through
`AutoModel.from_pretrained(package, trust_remote_code=True)` in a fresh process, and `compare-answers` compares
them answer by answer. Every answer of the 10,653 prompts (11,053 answers: typed-final 1,600, css15 6,547,
public231 231, mlx-diag 2,275) is **identical — 0 changes, max drift 0.0, for all six models**, and the native
answers still reproduce the scored predictions (0 changes; drift ≤ 2e-14; 0.6B and 27B 0.0).

Transformers 5.18.0 (`automap.sh --tf518`: the wheel installed `--no-deps` as an overlay on the image, AutoModel
parity on the verified package) against the same native 5.17 answers: **0.6B, 0.8B, 2B and 27B 10,653 / 10,653 prompts
identical, 4B 8,378 / 8,378 (three panels), drift 0.0**; 9B runs in its release chain. The text paths of `modeling_qwen3` / `modeling_qwen3_5` are unchanged
between 5.17.0 and 5.18.0 (only multimodal code differs).

The release builds re-run all of this before the upload (section 6). The verify and release packages differ only
in `MODEL_MANIFEST.json` `builder.source_commit`.

## 6. Publication

After the BF16-resident rollout's record reached integration (`587c0e490`, all six released): integration merged,
`make_automap.py final` (each final decision carries the rollout's final decision forward; the spec equals the
verified draft except `gate_receipt`), then per tier `automap.sh --tf518` and `automap.sh --release` on node E:
`main` must equal the superseded revision, `hf_headroom.sh` (52.37 / 100 GB), `release.sh --upload --collect
--already-collected` with full native parity before and after the real download, the AutoModel steps before and
after, the card's Transformers block from the Hub in a fresh `HF_HOME` under Transformers 5.17.0 and 5.18.0
(`automap-hub`, `automap-hub-tf518`: the only downloaded commit must be the uploaded one), readback, gate seal
(item 7 included) and the collection readback; then card HTTP, links and `gate evaluate`. Nothing is deleted
(no `permanently_delete_lfs_files`, so no history rewrite).

- **DEV2.0-0.6B:** the first run uploaded `25669e2d` and stopped at the Hub smoke (the image sets
  `HF_HUB_OFFLINE=1`, so the "fresh" container was offline; fixed in `539d2769f`); the completed run
  (`--resume 25669e2d`) published **`08b00e07cb90472e4184e3b91f3188c0ead8935b`** (the packages differ only in
  `MODEL_MANIFEST.json` `builder.source_commit`), gate 13 / 13, Hub smoke 5.17 and 5.18 bit-identical, post-checks
  ok. Receipts under `0p6b/release/` (and `0p6b/release-interrupted/`).
- **DEV2.0-0.8B** `1188dd33cc2d69381d3c37c9259b4f4163a666a6` and **DEV2.0-2B**
  `ffe291b4401930040914339592cd9fc59b9a3f87`: gate 7 / 7, Hub smoke 5.17 and 5.18 bit-identical, post-checks ok.
- **DEV2.0-4B** `3785b7b963d2f56de0e44f9ec638c814c5ee6499`: gate 7 / 7, Hub smoke 5.17 and 5.18 bit-identical,
  post-checks ok (mlx-diag parity from its own `--mlx` run, as in the rollout).
- **DEV2.0-9B, 27B:** running at the handoff (27B uploaded `2b7508e9…`; its parity-post, AutoModel-post and
  the two fresh-cache Hub smokes, each downloading the 52 GB base, follow). See the state file for the steps
  that complete this record.

## 7. Limits

- **CPU.** Works with a standard PyTorch (0.8B / 0.6B spot checks above), slowly; CPU (FP32) answers differ from the
  GPU-scored predictions (0.8B 4 / 200, 0.6B 3 / 200 typed-final decisions; drift ≤ 0.005). The images' ROCm PyTorch
  has no CPU LAPACK, so Qwen3.5 models (0.8B–27B) cannot run on CPU inside them, natively or through AutoModel.
- **Kernels.** GPU answers equal the scored ones with FLA 0.5.2, causal-conv1d 1.7.0 and the persisted autotune
  cache; without the kernels Transformers' reference path is slower and differs in the last digits (as natively).
- **27B.** The first load downloads the 28 pinned Qwen3.8-27B files (about 52 GB) into the Hugging Face cache;
  peak GPU memory is the native one (about 52 GB BF16-resident at the release examples' sizes, about 110 GB
  allocated on the longest evaluated input).
- **One device.** No `device_map="auto"` sharding (it means the runtime default); no dtype casts; `to()` reloads.
- **AutoTokenizer** without `trust_remote_code` prompts (standard for repositories with remote code);
  `pipeline("decision")` takes one request per call (`batch_size` 1).
- **Versions.** Tested with Transformers 5.17.0 and 5.18.0 (huggingface_hub 1.31, tokenizers 0.23, PyTorch 2.12,
  Python 3.12); 5.18 needs huggingface_hub ≥ 1.31.
