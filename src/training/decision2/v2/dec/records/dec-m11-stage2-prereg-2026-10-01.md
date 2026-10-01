# Decoder M11 stage 2 — preregistration (4B LH + IB1 + IB2 breadth; 2026-10-01)

Written 2026-10-01 ≈15:45 UTC+8 (07:45Z), before any stage-2 row was sampled or any stage-2 GPU job. It replaces the
"Stage 2" section of [`dec-m11-prereg-2026-10-01.md`](dec-m11-prereg-2026-10-01.md) (`b8c3caf22`), whose IB1-only
arms and three seeds the coordinator's instruction of 2026-10-01 15:28 UTC+8 superseded: "4B stage 2 (start now on
free M11 GPUs) … (a) + IB1 + IB2 at a preregistered ratio; (b) the same without the in-distribution families (`w2c`,
`isarc`, `hover`, `gsm2`) … matched tokens vs LH where feasible; 2 seeds plus a soup each". Everything else in the M11
prereg (confidentiality, isolation, stop rules, the 110 GPU-h cap) is unchanged. Index numbers stay private; no Index
row is read for selection or tuning; IB DEV is the breadth signal.

## Inputs (release-safe, private dataset `llm-semantic-router/decision-2.0-training-data`)

| Block | Revision / path | TRAIN rows / SHA-256 | DEV rows / SHA-256 | In-distribution families |
| --- | --- | --- | --- | --- |
| IB1-r3 | `31b200a3` `m6/ib1/` (record merged `6391e2843`) | 24,325 / `1e1b08f3…` | 2,067 / `3f56aa41…` | `w2c`, `isarc` |
| IB2 | `c5dbdd0a` `m6/ib2/` (record merged `7d9841c36`) | 24,518 / `ee137efa…` | 1,272 / `ab009fb1…` | `hover`, `gsm2` |

Copies: node A's HF read-back files (SHA-256-equal to the published revisions), relayed to nodes E / F over the
temporary transfer key and hash-checked. The 4B base mixture is M10's TRAIN `m10-4b-base` (`c385406e…`, 58,739 rows)
with N4XF's own-Lux teacher (`e2ff27ce…`), as LH.

## Arms (the LH recipe exactly; only TRAIN differs)

| Arm | TRAIN |
| --- | --- |
| **`4b-LH`** (reference) | M10's LH soup (`m10/soup/LH/build/LH-soup`, the release candidate's weights) — not retrained |
| **`4b-LHB`** | base subsample ∪ an IB sample from all 15 IB1 / IB2 families |
| **`4b-LHBx`** (transfer-only ablation) | the **same** base subsample ∪ an IB sample from the 11 families other than `w2c`, `isarc`, `hover`, `gsm2` |

Recipe: `train_dec --init base` from `Qwen/Qwen3.5-4B-Base@1001bb4d…`, LoRA r 128 / α 256 / dropout .05, LoRA LR 1e-4,
fresh head (`--head-init-seed 20261001`, head LR 1e-4), own-Lux KL 1.0 with `--teacher-partial` (base rows keep their
teacher; IB rows are gold-only), token batching ≤ 32,768 tokens / ≤ 64 rows, 64 rows per update, one epoch, max length
8,192, `even8` checkpoints, SELECT700 `matrix-v1` selection. **Seeds 20260926 / 20260927** (two per arm); the arm
artifact is the uniform FP32 soup of the two seeds' BEST checkpoints, each merged into FP32 first (`m10_merge.py`).
Both seeds must finish for a soup (otherwise the arm has no artifact; disclosed).

## Ratio and matched tokens (`ops/m11/m11_s2data.py`, CPU, in the decoder image)

- Token unit: the trainer's own head-readout encoding (`training.model.decision_model.encode`, Qwen3.5-4B-Base
  tokenizer, max length 8,192), so **T** = LH's TRAIN tokens exactly as the trainer counts them.
- **IB share 25% of T.** The IB token target is **B = min(round(0.25·T), tokens of the transfer-only pool)**, the same
  B for both arms, so both arms keep the **identical** base subsample of T − B tokens and differ only in which IB
  families fill B. Each arm's total is T within sampling granularity (matched to LH).
- **IB sample:** whole groups (`group_id`), stratified by family in proportion to each family's tokens in the arm's
  pool (the blocks' natural mix); within a family, groups in a seeded shuffle (`random.Random(20261001)` keyed by
  family) are taken while the family stays within its quota. A pool no larger than B is taken whole.
- **Base subsample:** the same procedure on M10's TRAIN by family, quota (T − B)·(family tokens / T), seed 20261001.
  LH saw all of it; the stage-2 arms trade ≈ B tokens of the released mixture for breadth (disclosed; this is the
  matched-token design the coordinator asked for).
- Lines are copied byte for byte (base, then IB1, then IB2, each in file order; the trainer shuffles by seed). The
  builder reports rows / tokens per family and block, the longest row (must be ≤ 8,192 tokens, otherwise the build
  stops), group counts and every SHA-256, and runs on both nodes: the two builds must be byte-identical.
- The stage-2 data lock (a separate record) pins the outputs and writes `m11/data/READY-4b-s2.json` on both nodes after
  it is pushed; every seed re-hashes TRAIN and the teacher against it.

## Retention probes and IB DEV

- **Probe TRAIN-overlap.** M10's 13-gram check (`m10_probes.py overlap --kind train`) against **both** stage-2 TRAIN
  files (IB2's `gsm2` is built from GSM8K train, the probes' GSM8K source); every probe item with a hit in either file
  is dropped from the stage-2 probe gold (`m10-probes.4b.gold.jsonl`, node A). LH and both arms are scored on that
  same reduced gold.
- **IB DEV panel `ib-dev`.** IB1 DEV then IB2 DEV (3,339 rows, byte for byte) rendered gold-free exactly as M9's HR2 DEV
  slice (`ops/m8/m8_data.to_prompt`; one question per prompt), built on node A (`ops/m11/m11_ibdev.py prompts`), gold
  kept on node A (`/data/dev2/private/dec/m11/ib-dev.gold.jsonl`), prompts relayed to E / F. Scored per family
  (invalid answers wrong); **IB DEV macro** = the family macro over every IB family with DEV rows; **transfer macro** =
  the same without `isarc`, `hover`, `gsm2` (`w2c` has no DEV rows); paired group bootstrap vs LH (2,000 draws, seed
  20261001).

## GPUs and order (amendment to the M11 layout: co-tenancy)

Stage 1 holds node E GPU0–2 and node F GPU2 / 3 / 6 for training; node E GPU3 and node F GPU7 only read. Node F GPU4–5
(amendment 1), node E GPU4–7 and node F GPU0–1 are never used. Stage 2 therefore trains **co-tenant** on the readout
GPUs:

| Node / GPU | Stage-2 chain | Pre-warm |
| --- | --- | --- |
| F GPU7 | `4b-LHB` s1, then s2 | s1's preflight fills node F's `4b-train` cache alone (s2 runs after s1 on the same GPU) |
| E GPU3 | `4b-LHBx` s1, then s2 | the same on node E |

- Per node, two new caches `4b-train` / `4b-read`, each a `cp -a` copy of the node's M10 decoder cache.
- A stage-2 chain holds its own lock (`chains/gpuN-s2.flock`) and writes its lease entry to `gpuN.lock/owner.dec-m11-s2`
  (track `dec-m11`); the GPU's `owner` stays M11's (stage-1 readouts keep writing it). Each seed starts only with
  ≥ 60 GB free VRAM on its GPU (the co-tenant rule). Readouts and training then share the GPU; numerics do not depend
  on co-tenancy (deterministic kernels from a persisted cache), so the stage-1 sentence "reads never share a GPU with
  training" no longer holds for E GPU3 / F GPU7 (disclosed).
- Post (same GPU, after both seeds of the arm are terminal): read the LH reference on the node as `4b-LH-<node>` if not
  read (LH's soup is copied F → E through node A, hash-checked); merge, soup, read the arm. Panels: dev, css-pilot,
  ht-dev2, score5t-dev, hs1-dev, pn1-dev, m10-probes, ib-dev (16,384 tokens, T = 1, `4b-read`). Parity reports:
  `4b-LH-f` vs M10's stored `4b-LH` readouts, and `4b-LH-e` vs `4b-LH-f`.

## Development gates (`m11_rules.py --tier 4b`; each arm vs `4b-LH` read on the arm's node)

A point passes when:

1. **M10's gate** with LH as the reference: M8 eligibility (type floor c_t ≥ c_t,LH − 0.03·n_t; no typed-DEV family below
   LH − 0.10; Noul `rule_precedence` ≥ LH − 0.01·n; HT-DEV v2 not FLAG), Score5-typed-DEV check half without COLLAPSE
   and without WARN unless LH has WARN, and retention-probe macro Δ vs LH with paired 95% CI upper ≥ 0;
2. **yes-bias guard:** `hs1-dev` false-yes on `hs1_unmet_condition` ≤ LH + 0.10;
3. **breadth:** IB DEV macro Δ vs LH ≥ 0 (point estimate).

**Finalists (≤ 2):** passing points with an HT-DEV v2 GAIN first, then the larger IB DEV macro Δ. Report only:
transfer macro, per-family IB DEV, `4b-LHB − 4b-LHBx` (HT-DEV v2, probes, IB DEV, transfer macro), typed T / H3,
`pn1-dev`.

## Formal, successor, C1 and Index

- Formal for ≤ 2 passers on node F (M10's path: `M6_4B_NODE=F`, image `dbe5f32b`, `run_same_panel --isolate`, copies of
  node B's frozen masters `formal/m5/cache-frozen{,-mlx}`), smoke first, then collection, mlx-diag; scored on node A.
- **Bar for items 1–7:** if the LH release is published (`DEV2.0-4B` main carries LH's weights) when the first
  stage-2 formal collection starts, the paired bar is LH's sealed formal run (`formal/m10/m10-4b-LH`, the same weights
  and path); otherwise DEV2.0-4B's bar as in M10. The record states which applied; the other comparison is reported.
- **The custodian's C1 content recheck (IB1 and IB2 are in the rescan roots) is required before any C1 scoring of a
  stage-2 model**; item 8 is a hand-off only. A frozen finalist that passes items 1–7 gets a release hand-off (package
  path, a C1 item-8 spec, the recheck requirement) and a private Index request to the eval track.

## Budget and stop rules

Stage 2 keeps the M11 prereg's 40 GPU-h line: arm caps **6 GPU-h** each (two seeds, preflights, co-tenancy slow-down),
readouts / merges / soups / LH reference reads 6, formal 8. A failed preflight stops the arm (no rerun, no replacement
seed); a seed stops at its arm cap; no stage-2 seed starts above 95 M11 GPU-h cumulative.

## Code (committed before use)

`ops/m11/`: `m11_s2data.py`, `m11-s2prep.sh`, `m11-s2chains.sh`, `m11-s2post.sh`, `m11_ibdev.py`; `m11-soup.sh`,
`m11-lines.sh`, `m11-score.sh` and `m11_rules.py` gain the 4B tier (the running stage-1 chains use the earlier mirror and
are unaffected); tests in `tests/test_m11.py`.
