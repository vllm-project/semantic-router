# `m3/teachers/lux1/xl/` — own-Lux targets on the XL rows that lack them (M3b, private)

**Teacher:** own Decision 1.0 Lux, `llm-semantic-router/Decision-1.0-Lux-9B@bd45a30aee8c84032791c245c70f86dee5389cc8`
(revision attested per row). These are our own model's targets, the clean default. They carry no third-party caveat,
unlike `m3/teachers/autojev27/`.

**Rows:** every row of `m3/mixtures/xl/mx-xl-full` ∪ `mx-xl-short` (r1) that had no own-Lux target (the
`*.lux1.missing.jsonl` lists there). XL-full ids come first, then XL-short-only ids, each part sorted by id, cut into
five waves:

| Wave | Prompts |
| --- | ---: |
| `w1`–`w4` | 60,000 each |
| `w5` | 12,215 |

That is 252,215 prompts in total. `w4` holds the last 32,554 XL-full rows and the first 27,446 XL-short-only rows.

## Files per wave

- **`w<k>.targets.jsonl`:** `{id, input_sha256, teacher_probs}`, sorted by id. Join by `id` to the XL recipe rows and
  check `input_sha256`. The other own-Lux targets of these recipes are `m2/teachers/lux1/rp-v2/wave1`–`wave4` and
  `m3/pk1/lux1/A0-train.canonical.jsonl` (A0s-strict rows).
- **`w<k>.attestation.jsonl`:** one line per prompt, sorted by id, with these fields:
  - `input_sha256` of the training row;
  - `source_input_sha256`, the digest of the native prompt the teacher answered, recomputed from the training row;
  - `model_id`, `model_revision`, `revision_attested`, `adapter_version`, `backend`, `model_config_sha256`,
    `runtime_matches_validated`;
  - `target`, and for a prompt without a target `no_target_reason` / `native_error`;
  - `node`, `gpu`, `image_id`.
- **`w<k>.report.json`:**
  - per type (Choice / Noul / Score): rows, targets, prompts without a native answer and why, argmax accuracy vs gold,
    mean gold probability, normalized entropy and Brier;
  - SHA-256 of the targets, attestation, prompts, rows and teacher output;
  - `provenance`: guard receipt, launcher, image, collector summary, wall time and GPU-hours.
- **`coverage.json`** (uploaded with `w1`): for each XL recipe and control, the rows with own-Lux targets before and after
  each wave, and the rows no wave covers.

## How the targets were made

- **Teacher run:** node B GPU7, one wave per process.
  - Launcher, image, source mirror and node-B Triton autotune cache: those of own-Lux RP-v2 waves 1–4 (Milestone 2).
  - Collector: `inference.run --backend lux --over-budget-invalid` (native, no chat API).
- **Conversion:** on node A with `v2.data.m2.targets` from an exact mirror of a pushed commit. Every receipt is checked for:
  - model identity and an attested revision;
  - the validated runtime;
  - the prompt digest, bound to the training row's native prompt;
  - the teacher output answering exactly the wave's prompt file.

  Before conversion the output was copied node B → node A with its SHA-256 checked at both ends. After the upload, each
  revision was downloaded again and every file's SHA-256 compared.
- **Image disclosure:**
  - The node-B Lux image (`sha256:ce895822…`) has no causal-conv1d kernel; Transformers computes the same convolution
    with its reference PyTorch fallback.
  - The eval track found that this image changes 43 of 8,778 Lux1 argmax answers against the kernel-equipped image.
  - Own-Lux RP-v2 waves 1–4 used the same image, so these targets are consistent with every earlier own-Lux target.

## Coverage (own-Lux targets, rows of the recipe)

| Recipe | Rows | Before | After all five waves | Left without a target |
| --- | ---: | ---: | ---: | ---: |
| `mx-xl-full` | 340,698 | 128,144 | 340,698 | 0 |
| `mx-xl-short` | 355,855 | 123,876 | 355,855 | 0 |
| `cx-xl-a7v1-full` | 191,605 | 12,413 | 170,089 | 21,516 |
| `cx-xl-a7v1-short` | 167,071 | 11,172 | 157,460 | 9,611 |
| `cx-xl-v2v1-full` | 296,514 | 127,186 | 225,542 | 70,972 |
| `cx-xl-v2v1-short` | 282,695 | 123,974 | 227,074 | 55,621 |

- **Control-only rows:** 100,957 rows of the four controls are in neither XL recipe, and no wave covers them. They
  train on gold labels unless a later wave covers them.
- **r2 rows:** H7 / H8 rows of the r2 recipes have no teacher targets and train on gold labels (M3b amendment 2).
