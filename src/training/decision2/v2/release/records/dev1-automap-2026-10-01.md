# Decision 1.0 in 🤗 Transformers (`auto_map` / `trust_remote_code`) — 2026-10-01

User request (2026-10-01 11:10 UTC+8): give every Decision 1.0 model the same Hugging Face `auto_map` /
`trust_remote_code` support as Decision 2.0. Release engineering, worker `1.0-automap`, branch
`xunzhuo/decision-1-automap`. API: [`../automap/API-decision1.md`](../automap/API-decision1.md) (the
Decision 1.0 section of the shared `automap/API.md`). State log: [`dev1-automap-state.md`](dev1-automap-state.md).

## Outcome

| Repository | Before | PR | After (`main`) | Parity (8,378 scored prompts) | Readback | Fresh-cache smoke |
| --- | --- | --- | --- | --- | --- | --- |
| Decision-1.0-Kai-0.6B | `9d6872cd` | [#1](https://huggingface.co/llm-semantic-router/Decision-1.0-Kai-0.6B/discussions/1) merged | `69aef406` | 0 changes, max drift 2.4e-7 | pass | pass |
| Decision-1.0-Lex-0.6B | `6c5e3d48` | [#1](https://huggingface.co/llm-semantic-router/Decision-1.0-Lex-0.6B/discussions/1) merged | `1f9750a7` | 0 changes, 2.6e-7 | pass | pass |
| Decision-1.0-Route-0.6B | `c7a31eb0` | [#3](https://huggingface.co/llm-semantic-router/Decision-1.0-Route-0.6B/discussions/3) merged | `a5b21dff` | 0 changes, 2.4e-7 (vs native Kai runtime, same GPU) | pass | pass |
| Decision-1.0-Sol-2B | `a1c9f252` | [#2](https://huggingface.co/llm-semantic-router/Decision-1.0-Sol-2B/discussions/2) merged | `fc210c8f` | 0 changes, 0.0 (bit-identical) | pass | pass |
| Decision-1.0-Nox-4B | `eab48e99` | [#2](https://huggingface.co/llm-semantic-router/Decision-1.0-Nox-4B/discussions/2) merged | `f098bdec` | 0 changes, 0.0105 (16 long CSS15 prompts) | pass | pass |
| Decision-1.0-Lux-9B | `8db79130` | [#2](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B/discussions/2) merged | `a31b9e2c` | 0 changes, 0.0103; 0.0 with the published FLA profile | pass | pass |
| Decision-1.0-Eos-0.8B | `a66df1b5` | [#3](https://huggingface.co/llm-semantic-router/Decision-1.0-Eos-0.8B/discussions/3) merged | `bbdc2221` | 0 changes, 0.0 vs the native Eos runtime on the same GPU and kernels (the stored set is not reproducible, see below) | pass | pass |
| DEV2.0-Route-0.6B (private) | `72a2d317` | [#1](https://huggingface.co/llm-semantic-router/DEV2.0-Route-0.6B/discussions/1) open (PR only) | unchanged (never merged by us) | AutoModel = native and native before = after the `api.py` fix: identical on 68 prompts (clean venv, Transformers 5.18, CPU); PR smoke pass | — | — |

Every PR was opened against the head current at staging (`parent_commit`), after checking that no open PR
touched the same files; the heads included subin's merges of the same morning (neutral hardware wording on
Eos / Sol / Nox / Lux; new Route weights). Merges re-checked both conditions. No discussion or PR of
anyone else was touched.

## What changed in each repository

- Added (flat, next to `config.json`): `configuration_decision1.py`, `modeling_decision1.py`,
  `pipeline_decision1.py`, `decision1_system_one.py`, `decision1_vela.py`, `decision1_qwen.py`,
  `decision1_rocm_conv.py` — the same seven files in every repository; Apache-2.0 headers; imports only the
  standard library, `torch`, `transformers`, `safetensors`, `huggingface_hub` (Triton optional, guarded).
- `config.json`: every Decision key kept in place (order and formatting), plus `model_type: decision1`,
  `architectures: [Decision1Model]`, `auto_map` (`AutoConfig`, `AutoModel`), `custom_pipelines.decision`.
- `README.md`: a "Use with 🤗 Transformers" section (requirements, a runnable snippet with the card's own
  example request, pipeline, device, precision and limit notes), and the sentences that said Transformers
  cannot load the model corrected in place (each had to match the live card exactly once, else staging stops).
- Route's root `MANIFEST.json`: hashes of the changed and added files (it was consistent before the change).
- Weights, tokenizers, calibration and all other files: byte-identical (readback by LFS SHA-256 / blob ID).

Commits: remote code as uploaded at `f1fddc759` (`v2/release/automap/decision1/`, SHA-256 in the stage receipts
under the node work root); tools `stage1.py`, `parity1.py`, `reference_kai_native1.py`,
`reference_eos_native1.sh`, `equivalence1.sh`, `hub1.py`, `smoke1.py`, `publish1.sh`; tests
`v2/release/tests/test_automap1.py` (stdlib; includes Transformers' remote-code import scan).

## Reference runtimes

- Stored predictions of the scored 1.0 comparator runs (JevArena v3 typed-final 1,600 and CSS15 6,547, plus
  public 231): Kai `r2-kai1-repeat`, Lex `r3-lex`, Eos `r4-eos1`, Sol / Nox `m1-adopt`, Lux `r1-lux1-repeat`
  (node A frozen runtime). They were produced by the native 1.0 runtimes of the runtime-bearing revisions
  (Kai `7185f514`, Lex `ee8e74d9`, Eos `3c2d6326`, Sol `0665a411`, Nox `0bb83350`, Lux `bd45a30a`), whose
  weights are byte-identical to the current mains.
- Route has no scored run. Its reference is the published Kai runtime (`7185f514` code, exactly as Kai's
  `systemone.py` runs it) loaded with Route's native weights, on the same GPU (`reference_kai_native1.py`).
- The vLLM-SR Decision runtime (`xunzhuo/decision-runtime`, `58cd660b5`) was read for the request / response
  contract, prompt policies (Nox renders a null Choice description as its key), limits and calibration.

Native details reproduced that the first draft missed and that mattered: Kai / Lex / Route run with the MHA
fast path and TF32 off (as `systemone.py` sets them; now scoped to the call and restored), type-sorted physical
batches of eight computed over the whole batch per path, and ROCm contiguous post-RoPE Q/K/V. With the fast
path on, Kai and Lex still had 0 changes (drift 4.5e-7 / 6.6e-7).

## Parity

Same image as the scored runs (ROCm PyTorch `2.12.0+git6bbd260`, Transformers 5.17.0, FLA 0.5.2), one
MI325X (node E GPU7 / GPU6), loading through `AutoModel.from_pretrained(<staged repo>, trust_remote_code=True)`.
An answer change is a different Choice, Noul side (p > 0.5) or Score argmax level (release `compare_answers`);
drift is the largest absolute difference of any probability, `noul` or `score`.

| Model | Answer changes | Missing | Over-length rows agree | Token counts differ | Max drift (typed / CSS15 / public) | Prompts with any drift > 1e-6 |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| Kai | 0 | 0 | 448 / 448 | 0 | 2.4e-7 / 0.0 / 1.3e-7 | 0 |
| Lex | 0 | 0 | 448 / 448 | 0 | 2.6e-7 / 0.0 / 1.8e-7 | — |
| Route (vs native reference) | 0 | 0 | 448 / 448 | 0 | 2.4e-7 / 0.0 / 1.3e-7 | 0 |
| Sol | 0 | 0 | 4 / 4 | 0 | 0.0 / 0.0 / 0.0 | — |
| Nox | 0 | 0 | 4 / 4 | 0 | 0.0 / 0.0105 / 0.0 | 16 |
| Lux | 0 | 0 | 4 / 4 | 0 | 0.0 / 0.0103 / 0.0 | — |
| Lux, published FLA l2norm profile | 0 | 0 | 4 / 4 | 0 | 0.0 / 0.0 / 0.0 | 0 |
| Eos | **47** (14 / 32 / 1) | 0 | 4 / 4 | 0 | 0.031 / 0.023 / 0.021 | 8,366 |
| Eos with its native ROCm convolution | **71** | 0 | 4 / 4 | 0 | 0.039 / 0.047 / 0.018 | 8,295 |
| **Eos vs the native Eos runtime, same GPU, same pinned FLA l2norm configs** | **0** | 0 | 4 / 4 | 0 | 0.0 / 0.0 / 0.0 | 0 |
| (native Eos unpinned vs the stored set) | 56 (24 / 32 / 0) | — | — | — | 0.039 / 0.023 / 0.021 | 8,294 |
| (native Eos pinned vs the stored set) | 46 (14 / 32 / 0) | — | — | — | 0.031 / 0.023 / 0.021 | 8,294 |
| (native Eos unpinned vs pinned) | 30 (30 / 0 / 0) | — | — | — | 0.051 / 0.0 / 0.0 | 1,600 |

- The encoders' remaining drift is only the Score expected value (summed in float64 here, in FP32 on the device
  natively). Route's numbers equal Kai's because Route keeps Kai's Score path.
- Nox's drift is confined to 16 CSS15 prompts of 6.1k–8.1k tokens; the other 8,362 prompts are bit-identical.
  With the FLA l2norm launch configurations the native Sol / Nox / Lux runtimes pinned (`FLA_CACHE_MODE=strict`),
  Lux is bit-identical on every prompt, so this drift is FLA autotuning, an environment property.
- **Eos** cannot be held to its stored set: the native Eos runtime itself (revision `3c2d6326` code, run by the
  scored-run collector `inference.run`, same image, byte-identical files) reproduces it with 46–56 answer changes
  and drift on nearly every prompt, and two native runs that differ only in FLA's l2norm launch configurations
  differ by 30 answers on typed-final. Eos's native runtime is the only 1.0 runtime that never pinned those
  configurations, so its stored predictions carry that run's autotune picks. Against the native Eos runtime on the
  same GPU with the same pinned configurations (node D GPU1, `eos-d/`), the uploaded remote code is bit-identical
  on all 8,378 prompts (0 changes, drift 0.0, 0 prompts with any drift), including Eos's native ROCm convolution
  (vendored as `decision1_rocm_conv.py`, gfx942, batch × length ≥ 2048, as the native runtime installs it).

Equivalence of the parity-tested and the uploaded code (`equivalence1.sh`, CPU, clean venv): identical
responses for Kai, Lex and Route (231 prompts each), Sol (40), Nox (30) and Lux (25); the uploaded files differ
only in Eos-gated code and the Hub download helper.

## Verification on the Hub

- PR revisions (`refs/pr/N`), clean venv (Python 3.12, torch 2.14.1 CPU, Transformers 5.18.0, huggingface_hub
  1.33.0): `AutoConfig` → `Decision1Config`, tokenizer, `AutoModel` → `Decision1Model` (loaded parameters
  571,909,635 for each encoder; 753,446,208 / 1,883,930,944 / 4,208,383,488 / 7,940,895,744 for Eos / Sol / Nox / Lux), `pipeline("decision")` equal to `system_one`,
  over-length → `max_length_exceeded` on every question, malformed question → `invalid_question`: all pass.
- After merge: readback of every file at the new `main` (above); fresh-cache smoke (empty Hugging Face cache)
  that also runs each card's Python block verbatim: pass for all seven (Transformers 5.18.0, CPU).
- Also checked locally: Transformers 4.57.6 (encoders) and 5.17.0.

## DEV2.0-Route-0.6B (a collaborator's private 2.0 package; PR only)

Prepared after the 2.0 auto_map release published its remote code on DEV2.0-0.6B (`25669e2d`), mirroring
that revision (`stage_dev2route1.py`): the three files `configuration_decision2.py`, `modeling_decision2.py`,
`pipeline_decision2.py` byte for byte (checked against DEV2.0-0.6B's `MODEL_MANIFEST.json` and the 2.0
branch source), the 2.0 `config.json` fields (sorted rendering as in 2.0), `MODEL_MANIFEST.json` with the new
hashes, a `remote_code` section naming its source package and an amended runtime equivalence note, a 2.0-style
card section (example built from the repository's `QUESTIONS.json`), and one sentence corrected. The package's
own runtime predates the 2.0 BF16-resident rollout and the 2.0 loading fix, so `decision2/api.py` gets only that
fix (commit `0cbf1033e`: Transformers' remote-code prompt answered "no" while the vendored loader reads the
tokenizer), applied as one change. Native answers through the package runtime before and after the change are
identical on 68 prompts (public 231 and typed-final, CPU), and the `AutoModel` path answers identically to the
native runtime on the same prompts (clean venv, Transformers 5.18.0); the PR-revision smoke passes. PR #1 is open
for the owner's review and is not merged by us.

## Divergences and follow-ups

- **vLLM-SR Decision runtime:** `parse_decision_config` (`58cd660b5`) accepts only the Decision keys in a root
  `config.json`. Deployed catalogs pin earlier revisions, so serving is unaffected, but a catalog move to the new
  revisions needs that parser to ignore `model_type`, `architectures`, `auto_map` and `custom_pipelines`
  (Route's card points at that runtime too).
- **Admission:** 1.0 keeps its native request-level admission (any over-length question → every question
  `max_length_exceeded`); 2.0 Qwen packages answer the fitting questions.
- **Confidence:** 1.0 reports the Decision runtime's `decision_type_aware_v1`; 2.0 reports its runtime's own.
- **Encoder tokenizer:** under `native/tokenizer` (`AutoTokenizer.from_pretrained(repo, subfolder=...)`).
- **Spec location:** the 1.0 section lives in `automap/API-decision1.md` until the 2.0 worker's `API.md` is
  integrated; then it folds in.
- Transformers prints two cosmetic warnings when the decoders' tokenizer is read from the repository root
  (a `decision1` config read as a plain config; the Mistral-regex notice). Tokenization is unaffected (token
  counts equal on every prompt; Sol bit-identical).

## GPU-hours

Parity and references on one MI325X at a time: ≈ 0.95 GPU-hour (runs of 75–340 s each plus loads): node E
GPU6 / GPU7 and, for the Eos references, node D GPU1, each under this worker's own lease, released after each
job. CPU work (smokes, equivalence, DEV2.0-Route checks) on node E.
