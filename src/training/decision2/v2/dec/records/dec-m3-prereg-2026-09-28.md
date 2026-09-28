# Decoder track Milestone 3 preregistration (2B / 4B on data v2 full-M), 2026-09-28

Status: **frozen before any Milestone 3 training arm is launched** (≈21:50 UTC+8).
Only data preparation has run: the mixture build and own-1.0 teacher labels (below).
Code: `v2/dec` at `5e2d6f094` (recipe-id mixtures, sharded labels, label merge,
spec) plus the commit that adds this record (merge subset mode, decoder-checkpoint
teachers). Development readouts (SELECT, CAL, typed DEV1600, CSS pilot1430) are
never release scores; JevArena v3 and public JevBench 231 are **post-key
same-panel** comparisons.

## Coordinator instructions implemented (Milestone 3)

- 4B: Nox full fine-tuning on data v2 full-M (pk1 A0s, the two A7h shortcut
  families and their 752 A0s rows excluded) with own-Nox soft targets as a trust
  region and A7 as retention (Nox's own data); three seeds + uniform soup; vs the
  adopted Nox1 run 56.470 (or a stricter same-limit control).
- 2B: Sol with the same recipe, three seeds + soup; vs the adopted Sol1 run 45.580.
- AutoJev-27B soft-target variant as soon as its targets are published (they were,
  21:27 UTC+8), with a matched own-teacher control; provenance caveat disclosed.
- Promotion: calibrated development proxy with the tie rule; formal post-key runs
  only for finalists, packaged at the largest supported limit against the stricter
  same-limit comparator; `mlx-diag` reported.

## Data (hash-verified; built on node B from exact mirrors)

Mixture `m3-v2m-ret` (spec `specs/m3-v2m-ret.json`, `074993af…`; builder
`v2.dec.build_mixture`, Eos/Sol/Nox tokenizer `06b95093…`):
**TRAIN `13804ac6d6c4b3f55f4db35d3bebad8575a9604eca84cfb9fad6a1afd5cff79b`,
56,198 rows / 29,249,047 native tokens, Choice / Noul / Score 16,645 / 26,782 / 12,771.**

| Component | Source (private HF dataset revision) | Rows | Tokens | Notes |
| --- | --- | ---: | ---: | --- |
| A0s | pk1 `m3/pk1/A0s` @ `d8eae3e4…` (registry `m3/registry.json`) | 6,547 | 3,977,352 | recipe pool A0s (7,299 ids) minus the 752 rows of `natural_cosmos_qa` / `natural_squad2_answerability` |
| H5, H1, H6, H3, E11, G2, G6, G4h | `m2/arms/<ARM>/train.jsonl` @ `ed87a03a…` | 12,328 / 7,879 / 4,179 / 1,534 / 2,882 / 4,325 / 3,465 / 1,119 | 14,570,946 | rows listed by recipe **mx-v2-full-M** (`m2/mixtures/mx-v2-full-M.ids.jsonl` @ `5c602c5a…`, pinned `5b8d5d29…` = the published recipe hash); H1: 1 duplicate input dropped |
| V1S | v1 `v2/arms/A6g`, `A6h` @ `5c0255ed…` | 3,664 | 945,968 | recipe pool V1S; 12 duplicate inputs dropped |
| A7 retention | A7 v3 `v2/a7/arms/*` @ `3a4efc77…` (the A7 release-candidate revision), view `dec10-stage4v2` (Sol/Nox fourth round) | 8,276 (C/N/S 5,246 / 2,083 / 947) | 9,754,781 | whole groups, stratified, budget = 0.5 × the recipe components' tokens (9,747,133); 2,545 duplicates of A0s rows dropped (current or pre-renumbering input hash) |

All inputs are the data track's and A7 track's published, gated files (the C1
v1.1 dataset-level independence check covered them; no `mlx-diag` source, no A7x).
SELECT700 `32a4352d…` (checkpoint selection) and **CAL698 `19cc1a8c…`** (every
calibration in this milestone) form the data directory
`/data/dev2/runs/dec/m3/data-sel700-cal698` on node B.

### Teacher files (all 56,198 rows; package temperatures; `v2.dec.teacher_label` + `v2.dec.merge_labels`)

| Teacher | File SHA-256 | Temperature | Argmax = gold (Choice / Noul / Score) |
| --- | --- | ---: | --- |
| own Nox 1.0 `llm-semantic-router/Decision-1.0-Nox-4B@cde2a68d` (3 shards, node B GPU0–2) | `2afe3048cc089243a4ad58c26c17151fdaa339bff136f22642aeafaaf4ad8575` | 1.32314 | .792 / .777 / .541 |
| own Sol 1.0 `llm-semantic-router/Decision-1.0-Sol-2B@ce0c018a` (2 shards, GPU3–4) | `947bc65b15f881aafaae4e3b0fda0885a78f752451dc1187663854d03d0ae4d4` | 1.30036 | .721 / .725 / .490 |
| AutoJev-27B `denis-pplx/autojev-27b@6f5b557e` (research & data; node A) | AJ-M `m3/teachers/autojev27/rp-v2/aj-m.targets.jsonl` @ `3a99bf1c…` `ba52dd86…`; AJ-A0s `m3/teachers/autojev27/pk1/A0s-train.targets.jsonl` @ `9bb9790b…` `97a071af…` | as published | covers all 41,375 v2-pool rows and all 6,547 A0s rows of this mixture (id and input hash); none of the 8,276 A7 retention rows |

The AutoJev teacher files for the J arms are composed with `merge_labels --part
<own-1.0 file> --override <AJ-A0s> --override <AJ-M> --override-subset`: AutoJev
on the 47,922 recipe rows, the arm's own-1.0 teacher on the 8,276 retention rows.
Their hashes are recorded at composition, before the J arms launch. **Provenance
caveat (disclosed on every record/card of a J candidate):** AutoJev's public
training pipeline reportedly used SFT data generated with a closed OpenAI model;
AutoJev is a third-party decision model and never a weight origin.

## Arms (one epoch each; seeds s1 = 20260926, s2 = 20260927, s3 = 20260928)

Common recipe **"M3F"**: full fine-tuning of the own 1.0 start (backbone + head,
FP32 weights and AdamW, BF16 autocast, FP32 head/loss), **backbone lr 5e-6, head
lr 5e-5**, weight decay 0.01, 5% warmup then cosine to 10%, objective CE + 0.5
Brier + **0.5 KL(teacher ‖ student) on every row**, token-budget micro-batches
(≤ 32,768 tokens, ≤ 64 rows; ≥ 64 rows per update), max length 8,192 (no
truncation), gradient checkpointing, SELECT700 matrix-v1 over 8 evenly spaced
checkpoints (earliest tie-break). The lower learning rate than E8F (1e-5 / 1e-4)
is the preregistered response to M2's half-data full fine-tuning result, which
eroded Nox 1.0 at 1e-5 without a teacher.

| Arm | Start | Teacher (KL 0.5) | Cap (GPU-h, incl. preflight + postrun) |
| --- | --- | --- | ---: |
| N4T-s1/s2/s3 | Nox 1.0 `@cde2a68d` | own Nox on all rows | 1.6 each |
| N4J-s1/s2/s3 | Nox 1.0 | AutoJev on recipe rows, own Nox on retention rows | 1.6 each |
| S2T-s1/s2/s3 | Sol 1.0 `@ce0c018a` | own Sol on all rows | 0.9 each |
| S2J-s1/s2/s3 | Sol 1.0 | AutoJev on recipe rows, own Sol on retention rows | 0.9 each |

J vs T at the same tier and seed is the single-factor teacher contrast (same
data, order, recipe, seed). Every arm: zero-step + one-step preflight and gate
(`drive_arm.sh`); a failed preflight, nonfinite loss, OOM or cap breach stops the
arm (recorded, not rerun). Postrun: CAL698 temperatures of the BEST checkpoint, one
typed DEV + CSS pilot readout (node B, image `dbe5f32b…`).

**Order (node B GPU0–4):** GPU0–2 N4T-s1/s2/s3 then N4J-s1/s2/s3; GPU3 S2T-s1,
S2T-s3, S2J-s1; GPU4 S2T-s2, S2J-s2, S2J-s3. A 0.8B follow-up (lowest priority)
is registered separately by amendment before it launches.

## Artifact, finalist and formal rules (fixed now)

Proxy P = 100·√(T·H) (typed-DEV family macro × CSS-pilot task-median macro-F1),
Choice / Noul / Score counts read separately; same-image 1.0 controls from M2
(node B, `dbe5f32b…`): **Nox 1.0 P 52.46 (C/N/S 460 / 228 / 378); Sol 1.0 P 43.27
(410 / 218 / 311).**

1. **Artifact per (tier, teacher):** the uniform FP32 soup of the three seeds'
   BEST checkpoints (`v2.dec.soup`) if its development P ≥ the seeds' mean P;
   otherwise the median-P seed. The soup's development readout uses CAL698
   temperatures fitted with `calibrate_ckpt`.
2. **Finalist:** P(artifact) > P(1.0) − 4 (not clearly below the 1.0 under the
   eval track's tie rule), every typed-DEV type count ≥ 75% of the 1.0's (Nox
   345 / 171 / 284, Sol 308 / 164 / 233), all answers valid. If both the T and
   the J artifact of a tier are finalists, both take formal runs (tie rule).
3. **Formal (finalists only):** the artifact with CAL698 temperatures fitted at
   **16,384 tokens** goes to the private staging repo
   `llm-semantic-router/dev2-dec-staging` (folder `m3/`), is downloaded on node A
   and hash-checked, then collected once with the eval track's frozen runner
   (image `f83b1d10…`, persisted per-run autotune cache, typed-final + css15 +
   public231, 20-item smoke first, 16,384 tokens) on the node A GPU the
   coordinator returns to the decoder track. Same-limit controls for each tier with a finalist:
   Nox 1.0 / Sol 1.0 through the shared renderer at 16,384 tokens with package
   temperatures (`adapter-spec-infer-1p0.json`). **Comparator = the stricter of the
   adopted 1.0 run (Nox1 56.470, Sol1 45.580) and the 16K same-limit control.**
   `mlx-diag` is collected for every formal finalist.
4. **Release reading:** a candidate qualifies if its v3 paired 95% lower bound
   vs the comparator is > 0 on node A and no decision type collapsed (typed-FINAL
   type count ≥ 75% of the comparator's and not a single constant answer).
   Per-type, per-task, public-tier and `mlx-diag` regressions are disclosed; the
   paired human-transfer difference against the tier's best measured peer
   (Decider 2B / Decider 4B) is reported for the coordinator's first-tier
   judgment. A qualifying candidate is reported immediately with its staging
   checkpoint, profile `qwen-full` and scored run directory; anything else is
   HOLD with the readouts recorded.
