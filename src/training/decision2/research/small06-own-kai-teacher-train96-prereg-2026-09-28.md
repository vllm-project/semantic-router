# Own Kai 0.6B teacher: bounded TRAIN-only native probability screen

**Prospective technical diagnostic; no student update or protected evaluation.**
This screen asks whether our pinned Decision 1.0 Kai model returns usable
Choice, Noul and Score probability vectors on short, source-stratified rows of
the exact rights-clean v2 TRAIN used by the official-Qwen 0.6B control. Its
result cannot select a release checkpoint or count as JevArena/JevBench gain.

## Frozen sources and prior evidence

- Teacher: our `llm-semantic-router/Decision-1.0-Kai-0.6B` revision
  `7185f514f54b8f93c55998b1e8f9c5cc67f0d029`, native file manifest
  SHA-256 `c1bf07ab1c4c3fa1f819256d3de858d1ed87869bdfa663553280d7e78b88bee4`.
  Its verified direct Decision architecture is derived from our own Vela
  encoder, with native System One Choice/Noul/Score outputs. It is a teacher,
  never a Qwen student's weight initializer in this experiment.
- Student comparison, if separately proposed later: official
  `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`
  shared-head control, fixed rights-clean v2 TRAIN, 466 updates and
  `562/700` SELECT. This screen performs **zero** student updates.
- Exact TRAIN SHA-256
  `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`;
  pinned prior Kai-native conversion manifest SHA-256
  `a2591978481e019f482686df69d55f213a970f51ce465c6a45e6910bc9d3dca5`,
  accepted native TRAIN SHA-256
  `fc8496c76c84b37f0a9308b13a270b69aab608defb8a9e4a3f62011352b6976c`,
  whole-group quarantine SHA-256
  `1bd6a566eaac72f374a5e391ee6613fed4d667870d9c79e2fac8bc152789379d`.
  The published Kai cap is 1,024 native tokens without silent truncation.
- The [completed AutoJev external-teacher KL arm](autojev-kl06-student-result-2026-09-28.md)
  already used this same official-Qwen student start/data/schedule, attaching
  3,690 Choice/Noul soft distributions at weight `.05`. It **failed** its frozen
  SELECT gate: `510/700` versus hard-label control `562/700`, with family-macro
  Brier `.240292` versus `.142634`. A different teacher alone does not reverse
  that negative result; no whole-TRAIN KL run is authorized by this screen.

## Completed CPU admission inventory, before teacher inference

The exact official Qwen tokenizer (`tokenizer.json` SHA-256
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`)
and unchanged native renderer reproduce **4,094,489** TRAIN tokens. The
read-only inventory verifies source and native conversion hashes and emits
only aggregates. No SELECT, CAL, DEV, FINAL or public benchmark label was read.

| Slice | Original TRAIN | Kai-admissible | Excluded for native length |
| --- | ---: | ---: | ---: |
| Rows | 7,455 | 6,262 | 1,193 |
| Choice rows | 3,908 | 3,127 | 781 |
| Noul rows | 3,031 | 2,672 | 359 |
| Score rows | 516 | 463 | 53 |
| Qwen-rendered tokens | 4,094,489 | **1,302,130** | **2,792,359** |
| Choice tokens | 2,482,423 | 702,265 | 1,780,158 |
| Noul tokens | 1,322,873 | 472,266 | 850,607 |
| Score tokens | 289,193 | 127,599 | 161,594 |

The excluded records are 1,139 stage-4 composition and 54 stage-3 replay
rows; their Kai packed lengths are minimum 1,026, median 1,809, p95 4,789 and
maximum 6,950. Of 1,987 original stage-4 rows, only 848 remain Kai-admissible;
their Qwen token exposure is 441,983 of 3,124,797. Eligible Score contains
only **86** three-level rows. This teacher has no native prediction for 68.2%
of the Qwen student's TRAIN token exposure; any future comparison must keep
all excluded rows as hard-label training, never shorten or discard them.

## One fixed screen

The local implementation is
[`small06_own_kai_teacher_probe.py`](small06_own_kai_teacher_probe.py),
SHA-256 `445a7261c36914df83bbbff5c080146f5ce0db47ab3a4dae576c7492a8bb3f29`,
on signed code commit `a9f0f2adbe82de602349fc6e598dd66be7128428`.
Its CPU tests use synthetic rows and cover deterministic group selection,
native request semantics and probability-key rejection. The mirrored source
hash was checked before the CPU inventory. The fixed one-per-group roster has
SHA-256 `4f56daee2b5f9f58451228114b6360ef62a8e503b3a61d361a9538d82dd4cebf`:

| Type | Source/level strata | Count |
| --- | --- | ---: |
| Choice | 16 stage-4 + 16 other | 32 |
| Noul | 16 stage-4 + 16 other | 32 |
| Score | 16 stage-4 + 8 other three-level + 8 other non-three-level | 32 |

Rows are chosen by a fixed SHA-256 ordering of source group and record ID,
without consulting gold. Each of the 96 TRAIN labels is used **only after**
native inference for aggregate correct count, gold probability and half-Brier.
The model receives state, instruction and candidate descriptions, not gold.
The original option keys and order are preserved; Noul uses its native yes
probability and Score uses all ordered-level probabilities. Private prompts,
IDs, labels and vectors are not written by this script.

Use the existing pinned ROCm runtime image digest
`sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`
and its existing Kai/Lex Python environment: Python `3.12.13`, PyTorch
`2.12.0+git6bbd260`, HIP `7.2.53211`, Transformers `4.57.6`, tokenizers
`0.22.2`, safetensors `0.8.0`, NumPy `2.5.3`. Verify the exact Kai HF
revision metadata, 30 native files, model manifest and runtime before loading.
Reserve one currently free GPU for at most **0.15 GPU-hours** including
loading and 96 requests. Output must be a new mode-0600 aggregate receipt in
the private experiment area. Stop on a missing source, changed hash, runtime
mismatch, unexpected exception, nonfinite/vector-key mismatch or budget cap;
do not retry with a different checkpoint, renderer or truncated input.

**Technical gate:** all 96 answers must be valid finite, normalized vectors
with exact source option keys and no context overflow. **Evidence screen:**
report every type/stratum, top-label ties, hard-label agreement, mean gold and
top probability, and half-Brier. To consider *only* a separately preregistered
source-specific retention treatment, require Choice at least `20/32`, Noul at
least `20/32`, Score at least `12/32`, each type half-Brier at most `.25`, and
three-level Score at least `3/8`. These small TRAIN quotas are screening gates,
not proof of transfer. A failed technical or evidence screen means **HOLD**
for this own-Kai soft teacher on the current data. A pass still requires an
independent source-overlap audit and matched-token loss/control preregistration;
it does not authorize student training or claim any formal improvement.

The Kai1 inherited corpus has an earlier 42-near-CSS warning and lacks exact
published record IDs. This screen cannot establish teacher source-disjointness
from protected transfer tasks. Such uncertainty must remain visible in any
later student analysis; these 96 TRAIN answers do not settle it.
