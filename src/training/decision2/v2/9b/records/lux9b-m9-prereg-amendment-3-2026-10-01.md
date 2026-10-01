# 9B Milestone 9, amendment 3: stage 3 = the released K-a13 recipe plus IB1-r3 + IB2 at matched tokens (K-a13IB) and its transfer-only ablation (K-a13IBX)

Written 2026-10-01 ≈19:20 UTC+8 (11:20Z), before any stage-3 job (CPU build or GPU) and before any stage-2 readout
(the four stage-2 seeds are training or just finished training; no stage-2 arm has been read). Assigned by the
coordinator after the stage-1 verdict (COORDINATION 2026-10-01 18:25: "Stage 3, added by the coordinator
(amendment 3)"). It adds stage 3; nothing in the [preregistration](lux9b-m9-prereg-2026-10-01.md) (`8570a5896`),
[amendment 1](lux9b-m9-prereg-amendment-1-2026-10-01.md) (`0b84e0db4`) or
[amendment 2](lux9b-m9-prereg-amendment-2-2026-10-01.md) (`51c80ddc9`) changes.

## Why

- **Stage 1** ([result](lux9b-m9-stage1-result-2026-10-01.md), `8ab82f069`): changing the training method (a rank-128
  LoRA, from the base or from Lux) does not carry the 4B result to 9B. L9 loses held-out transfer (HT-DEV v2 FLAG),
  and both adapters carry more near-miss yes-bias than K-a13. DEV2.0-9B already keeps the base's knowledge on the
  retention probes (.792 vs Qwen3.5-9B-Base .781), so there is nothing to recover.
- The private Index analysis (IX1; numbers stay private) places 9B's deficits in **coverage families absent from
  training** (tool-call decisions, spam / phishing, stance / sarcasm, a grounding threshold effect), not in lost
  knowledge. IB1-r3 and IB2 cover several of them (tool-call relevance and readiness, SMS / YouTube spam, stance,
  sarcasm, claim verification); phishing and grounding stay uncovered (COORDINATION 15:55).
- So stage 3 adds breadth on the proven recipe: the released K-a13 recipe, unchanged except for the data.

## Arms (three seeds each; K-a13's construction)

| Point | Seed TRAIN | Native tokens per seed |
| --- | --- | ---: |
| **K-a13IB** | x60 cut to 60,183,732 − 10,173,355 = 50,010,377 tokens + every IB1-r3 and IB2 TRAIN row (24,325 + 24,518 rows; 10,173,355 tokens) | ≈ 60.18M |
| **K-a13IBX** (transfer-only ablation) | x60 cut to 60,183,732 − 7,504,930 = 52,678,802 tokens + IB without the in-distribution families `isarc`, `w2c` (IB1), `hover`, `gsm2` (IB2) (7,504,930 tokens) | ≈ 60.18M |

- **Matched tokens and the ratio.** Every seed sees K-a13's 60,183,732 native tokens (plus the cut's overshoot, at
  most +1% of the x60 part). Every kept IB row appears exactly once (the stage-2 IB dose) and x60 fills the rest, so
  the IB token share is 16.9% (K-a13IB) and 12.5% (K-a13IBX).
- **The x60 cut** is M6's KH cut: whole groups, stratified by pool × source × task type × language as the x60 recipe
  (`lux9b.m3_data.recipe_budget`), seed `20261001:m9-s3:keep`, overshoot tolerance 1%. Per-row native tokens and
  pools come from the x60 ids file (`mx-xl-full-r2.ids.jsonl`, `7843afb7…`, copied from node A). One seed serves both
  builds, so K-a13IB's x60 rows are a subset of K-a13IBX's (nested by construction; checked).
- **Recipe = K-a13's** (the M4 K-s1..s3 trainer contract):
  - full fine-tuning of Lux 1.0 `bd45a30a…` (`--init decision1`), backbone LR 1e-5, head LR 1e-4;
  - AdamW wd .01, warm-up .05 then cosine, one epoch, max length 8,192, gradient checkpointing;
  - CE + 0.5·Brier on gold + 1.0·KL(own Lux ‖ student) on every x60 row (the kept rows' own-Lux target lines of
    `cdcd99c1…`, byte for byte);
  - token batching ≤ 32,768 tokens / ≤ 64 rows, 64 rows per update; `even8` checkpoints with SELECT700 `matrix-v1`
    selection per seed;
  - **seeds 20260926, 1, 2 (K-a13's own).**
  - The only change besides the data: **IB rows train on gold only** (`--teacher-partial`), as in stage 2. Lux 1.0's
    targets on families it does not cover would teach its deficits; the ⅔-Lux interpolation stays the trust region.
- **Artifact = K-a13's construction.**
  - The uniform FP32 soup of the three seeds' BEST checkpoints (node C, CPU).
  - Then the uniform FP32 soup [arm soup, Lux 1.0, Lux 1.0], i.e. α = ⅓ toward the arm soup, with **K-a13's own
    checkpoint-form Lux 1.0** (`m3/pf-D-s1-zero/run/checkpoint-0000000`; node A, CPU).
  - **No α line.** The 9B yes-bias grows with distance from Lux, K5-a12 (α ½) failed item 4, and the released
    recipe's α is ⅓.
  - If the soup module refuses K-a13's Lux checkpoint as a member (an initialisation-source mismatch with the M9
    trainer), the zero-step checkpoint of K-a13IB-s1 replaces it (the same Lux 1.0 weights written by the M9 trainer,
    checked by its preflight), disclosed.
  - A failed seed is not rerun: two finished seeds make a two-seed soup; one makes the point from that seed alone.
    Both cases are disclosed.
- **Build:** `lux9b/m9_data.py --match-tokens` through `m9/stage3.sh build` on node C (host python3). The script
  hash-checks x60, the ids file and the IB TRAIN files first. `data/READY3.json` pins both builds; the chains re-hash
  against it before each seed.
- **Item 6:** each stage-3 TRAIN is a subset of x60 ∪ IB1-r3 TRAIN ∪ IB2 TRAIN. Both receipts list 0 groups (x60's;
  `exposure/ib1-ib2-train.json` `9fd9d3db…`).

## Placement and stop rules (node C, `m9/chains3.sh`)

- **Wave 1:** GPU3 K-a13IB-s1 first, alone through its one-step run (pre-warm; `status/prewarm-s3.DONE`); then GPU6
  K-a13IB-s2 and GPU7 K-a13IB-s3. Node C GPU6–7 are lease-checked: IX1 released them at 09:53Z, and their previous
  owner files are archived.
- **Wave 2:** K-a13IBX-s1 / s2 / s3 on GPU2 / GPU1 / GPU4 as their stage-2 chains end (GPU4 after node C's L9IBX
  merges).
- Node C GPU0 is never used. Node B GPU6 is not used.
- Seed cap 4.5 GPU-h; a failed preflight stops the arm; no seed starts once node C's M9 GPU-hours reach 100.
- Soups `m9/post-c.sh` KIB / KIBX (CPU). Node A: `m9/post-a3.sh` (pull, the ⅓ point, readouts on GPU6 / GPU7,
  scoring, rules).

## Readouts, gates and finalists

- **Gates:** node A, the stage-1 path, the same eight panels and the same seven development gates vs C0
  (`m9_rules.py` unchanged):
  - typed / family floors;
  - the Noul `rule_precedence` floor (≥ C0 − 4 of 400);
  - the Score5-typed-DEV floor;
  - HT-DEV v2 not FLAG;
  - **the yes-bias guard**: Y1 PN1-dev clean gold-no Δ ≤ +.02 with CI lower ≤ 0; Y2 hop ≥ −.03; Y3 `hs1-dev`
    false-yes ≤ +.05;
  - MLX-DEV-9B upper bounds ≥ 0;
  - retention upper bound ≥ 0.

  Outputs: typed readout `readout/m9-s3.json`, rules `select/9b-finalists-s3.json`.
- **Report only:** IB1 / IB2 DEV per family (C0, both points, the α 1 arm soups, and L9IB / L9IBX when read);
  contrasts K-a13IB − K-a13IBX, K-a13IB − L9IB, K-a13IBX − L9IBX (HT-DEV v2, probes, PN1, typed).
- **Readings (preregistered):**
  - **H4** (breadth on the proven recipe): K-a13IB passes the seven gates.
  - **H5:** the IB DEV gain over C0 survives the ⅓ interpolation (the point vs C0 and vs its α 1 soup).
  - **H6:** the in-distribution families' contribution (K-a13IB − K-a13IBX).
- **Finalists:** at most two stage-3 passers (an HT-DEV v2 GAIN first, then the larger typed T); at most four formal
  candidates in M9 overall (amendment 2).
- **Formal:** the stage-1 path, after the newest COORDINATION notes are re-read, a lock record is pushed and
  `status/formal-s3.GO` is written (`formal-chain.sh` with `M9_STAGE=3`). The successor choice among every passer of
  items 1–7, across stages, keeps the preregistered order.
- **C1:** the custodian's C1 content recheck precedes item 8 (IB-trained).
- **Release:** any successor package is built on top of the 9B forward-budget fix revision that release publishes
  from `main` `41cb6a08` (COORDINATION 18:10).

## Budget

Stage 3 ≈ 6 seeds × ≈ 2.8 GPU-h (preflights included) + readouts ≈ 1.5 ≈ 18 GPU-h. Cap 30 GPU-h, inside the 70 GPU-h
the preregistration set for IB stages (stage 2 ≈ 16). The M9 cap stays 120 (16.3 used at 10:19Z).
