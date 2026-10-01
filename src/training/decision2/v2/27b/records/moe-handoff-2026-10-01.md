# 27B MoE milestone (MoE-1): frozen package and release hand-off notes (2026-10-01)

**Status: NOT ISSUED (2026-10-01 11:45Z).** Neither hand-off condition holds: the package did **not** pass successor
items 1–7 against DEV2.0-27B (A20r) — items 1 and 4 fail (`moe-results-2026-10-01.md`) — and its private Decision
Index run (scored 11:38Z on node A; values in the private report only) does **not** clear the 27B-class frontier
bar. These notes stay as the record of the frozen package and of what a Gemma MoE release would need, should a
later Gemma-based candidate qualify. Nothing was built or published.

## Frozen package (`decision2-27b-moe-package/1`, frozen 2026-10-01T08:57:35Z, before any formal collection)

| Field | Value |
| --- | --- |
| Location | node B `/data/dev2/runs/27b-moe/MOE-Git-soup/package/PACKAGE.json` + `../checkpoint/`; verified copy on node A `/data/dev2/runs/27b-moe/index-stage/MOE-Git-soup/` (`SHA256SUMS`, `STAGED.json`) |
| `PACKAGE.json` SHA-256 | `ffb11e1c5b525234f806853f6b1c12095602c664fc84663fea454f7011721ed6` |
| Model identity (checkpoint + base files) | `model_sha256` `9165bed76da491d12b33df365e226c3d3a1a941d7caed060cf40d280d9abc115` |
| Checkpoint | `peft-lora/1`: PEFT LoRA on the attention + dense-MLP projections (205 targets; experts, routers and the shared-expert gate frozen) plus the FP32 candidate head; soup of MOE-Git-s1 / s2 `checkpoint-0002676` (exact rank concatenation, head averaged, max relative error 4.5e-7), rank 64 / α 128, dropout 0.05 |
| Base | `google/gemma-4-26B-A4B-it@4d7ae4984b7db7de8f8457170b3f1a419ee76d52` (tree `b05e076d09618da6071bcda007e8702e9b4ab4badd999189a64a8bb589753a31`; every file SHA-256-bound through the LoRA source fingerprint) |
| Experts kernel | `grouped_mm` (amendment 1; pinned in the checkpoint metadata, used for training and inference) |
| Prompt | `decision2-bos-segmented-options-global-query-v1` (the shared segmented prompt behind Gemma's BOS token) |
| Limit | 32,768 tokens per question; nothing truncated |
| Calibration | T = 1 (`package/calibration.json`, binding the rejected CAL698 fit .465 / .381 / .481; CSS-pilot ECE worsened) |
| Runtime used for every score | image `sha256:dbe5f32b…` (torch 2.12.0+git6bbd260, HIP 7.2, Transformers 5.17.0, PEFT 0.21.0), SDPA attention, `training.model.infer` (FP32-resident, BF16 autocast, FP32 head) |
| Formal run | node B `/data/dev2/runs/27b-moe/formal/MOE-Git-soup` (sealed `674fee4e…`; stored predictions under `output/`), mlx-diag `formal/MOE-Git-soup-mlx` (scored on node A `runs/27b-moe/mlx-diag/MOE-Git-soup`) |
| Gates and verdicts | node B `/data/dev2/runs/27b-moe/gates/` (`VERDICTS-20261001T094035Z.json` `9474a46e…`) |

## Parameters (safetensors headers; `package/params.json`)

- **Loaded 25,310,379,550** = base text decoder 25,233,141,790 (the vision tower is not loaded) +
  rank-64 adapter 74,342,400 + head 2,895,360.
- **Active per token 3,899,768,350** = loaded − routed experts 22,837,985,280 + top-8 of 128 of them
  (1,427,374,080); adapter and head count as active.
- For comparison, DEV2.0-27B (A20r) loads 26,096,775,168, all active.

## Latency and memory (node B GPU7, MI325X)

- **Decision forward, BF16-resident, batch 1** (M1's 64-prompt SELECT roster, ≈ 145 tokens): p50 / p95
  **114.6 / 120.5 ms**, 50.7 GB resident (A20r 83.6 / 88.8 ms, 52.2 GB, FLA kernels).
- **Formal path per prompt** (FP32-resident + BF16 autocast): typed FINAL 128.4 / 133.7 ms, CSS15 132.8 / 185.4,
  public 231 133.3 / 281.7 (A20r 118.5 / 145.1, 103.8 / 232.6, 106.2 / 414.8).
- **Memory ceiling of the FP32-resident path** (one batch per request, path check on synthetic rows): ≈ 101 GB of
  weights; 16 questions × 16K tokens fit (≈ 257K padded tokens), 20 × 16K and 32 × 16K exhaust 256 GB. A serving
  runtime needs memory-bounded question chunks (with a parity check) or a parity-checked BF16-resident path for
  very large multi-question requests.

## Licence (from the gate record)

- Gemma 4 is plain **Apache-2.0** (the card's licence link resolves to the verbatim Apache License 2.0); it is not
  under the Gemma Terms of Use or the Gemma Prohibited Use Policy of Gemma 1–3. The repository is ungated and ships
  no `NOTICE` file.
- A derived release (adapter + head served on the unmodified base, or merged weights) may be Apache-2.0 and must
  include the licence text (§4(a)), carry a prominent modification notice — an adapter and decision head trained
  on top of `google/gemma-4-26B-A4B-it@4d7ae498` (§4(b)) — keep upstream attribution notices if any (§4(c); none
  ship), and use "Gemma" / "Google" only to describe the origin (§6). The card names the exact base repository and
  revision as the direct weight origin.
- Data terms are A20r's (`a20` mixture, SELECT700, CAL698); Rune-26B-A4B and Decider 35B-A3B were never teachers or
  weight sources.

## What the release builder needs (it has no Gemma MoE path today)

`v2/release/build.py` and `v2/release/runtime/qwen.py` support only Qwen checkpoints (`qwen-full`, `qwen-adapter`).
A Gemma MoE release needs, as separate reviewed commits with tests:

1. **A profile** (e.g. `gemma-moe-adapter`, runtime family `decision2-gemma-moe-adapter`) in `v2/release/layout.py`:
   base-bound PEFT adapter + head, the base pinned by repository revision and per-file SHA-256.
2. **Runtime loading**: the vendored `training.model` modules from a commit with the MoE support
   (`decision_model.from_base(..., experts_implementation="grouped_mm")`, `encoder_for` so the BOS prompt is used —
   `qwen.py` currently encodes with the plain prompt), SDPA attention, Transformers ≥ 5.17.
3. **Residency**: the BF16-resident runtime (`5dc962b00`) handles `nn.Linear`; the fused expert tensors and
   `grouped_mm` under autocast need their own parity check (0 answer changes vs this package's formal predictions)
   before BF16 residency is used; otherwise ship FP32-resident (≈ 101 GB weights).
4. **Builder checks**: the parameter identity check is Qwen-specific (`text_parameter_count`); add the loaded /
   active counts above (`v2/27b/moe/moe_params.py` computes them from safetensors headers).
5. **Long requests**: memory-bounded chunking of very large multi-question requests (see the memory ceiling).
6. **Licence / card**: an Apache-2.0 Gemma 4 lineage entry in `v2/release/licence.py`, the modification notice,
   the card lineage `google/gemma-4-26B-A4B-it` → LoRA rank 64 + head.
7. **Item 8 (C1 post-key)**: like A20r's, it waits for a release pre-build; a C1 spec draft follows the template
   `v2/eval/sealed/c1-postkey/dev2-27b-a20r.json` with the adapter spec `v2/27b/moe/adapters/moe-lora-cal.json`
   (calibration = the package's T = 1 binding). This track never runs C1.

## Release-panel standing (for the coordinator)

Post-key v3 68.75 (T .812, H .582): vs A20r −3.61 [−6.54, −1.18], vs AutoJev-27B −3.39 [−5.60, +0.79], vs Eikos-27B
−0.54 [−3.17, +3.90], vs Jebadiah-27B +3.27 [+0.61, +6.10]; mlx-diag Choice + Noul −.036 [−.052, −.020] vs A20r;
tier gates (item 5) pass. A release of this package would carry the item 1 / item 4 regressions on its card.
