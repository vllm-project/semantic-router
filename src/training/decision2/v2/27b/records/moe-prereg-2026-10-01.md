# 27B MoE milestone (MoE-1): preregistration (2026-10-01)

Signed before any GPU job of this milestone. Assignment: COORDINATION 2026-09-30 23:50 (user-approved MoE base
exploration, in parallel, on idle GPUs). Gate record: `moe-gate-2026-10-01.md`. Development readouts are never release
scores. JevArena v3 is post-key ("post-key same-panel"); public 231 is a public-subset rerun; C1 is never opened by
this track. Nothing goes to Hugging Face during the milestone.

## Question and target

Can an official MoE base (≈ 3.5–3.8B active of 25–35B), trained with DEV2.0-27B's A20r recipe, **beat AutoJev-27B
significantly** (post-key v3 > 72.133 with paired 95% lower bound > 0, human transfer not significantly below), and
pass successor items 1–7 against DEV2.0-27B = A20r (scored run node B `/data/dev2/runs/27b/M4-A20r-soup/formal`,
72.360)?

## Fixed inputs

- **Bases** (licence-clean, pinned, hash-verified; gate record): G-it `google/gemma-4-26B-A4B-it@4d7ae498`, G-pt
  `google/gemma-4-26B-A4B@24548b62`, Q-it `Qwen/Qwen3.5-35B-A3B@59d61f3c`, Q-pt `Qwen/Qwen3.5-35B-A3B-Base@0f081307`,
  at `/data/dev2/models/moe/<name>` on both nodes.
- **Runtime:** image `sha256:dbe5f32b…` (torch 2.12.0+git6bbd260 / HIP 7.2, Transformers 5.17.0, PEFT 0.21.0,
  FLA 0.5.2, causal-conv1d 1.7.0), one GPU per job, launcher `v2.27b.moe.launch` (lease track `27b-moe`).
- **Trainer:** the MoE pipeline `v2/27b/moe/pinned/moe-pipeline-2026-10-01.tar` (`69dc8622…`) = the BEST368 pipeline
  that trained A20r plus the MoE patch; the Qwen dense path is unchanged (test).
- **Data (A20r's, frozen):** `a20` TRAIN `4aa0dc964505682b2840fa5167a7ec14983e5a0cea427480bed1996f1befc0d4`
  (A0s-strict pk1 ∪ A6g/A6h ∪ 20M A7 tokens; 56,969 rows), SELECT700 `32a4352d…`, CAL698 for readouts only. Its C1
  source check (M4 `c1-sources-a20.json`, 0 hits) and overlap exposure (0 groups) carry over unchanged; the exposure
  check is rerun for finalists.
- **Contract = M4-A20r's:** LoRA rank 32, α 64, dropout 0.05; LoRA lr 2e-5, head lr 1e-4, backbone lr 1e-6 (unused),
  AdamW wd 0.01; 5% warm-up then cosine over one pass (3,561 updates); `ce_brier` 0.5; head dim 256; micro-batch 1 ×
  16; gradient checkpointing; FP32 parameters, BF16 autocast, FP32 head; save every 446 updates plus the last;
  BEST = SELECT700 family-macro accuracy, then Brier, then earliest; seeds s1 20260926, s2 20260928; a fresh training
  autotune cache per arm-seed seeded from T0 (`1933eb36…`).
- **Base-specific, not a recipe change:** LoRA targets are each base's non-expert projections (gate record): Gemma
  attention + dense MLP, Qwen-MoE attention / gated-delta + shared expert; experts, routers and the shared-expert gate
  frozen. Gemma runs behind BOS with a 4,736-token admission limit (all 56,969 rows; Qwen-MoE at 4,096 as A20r).
  Trainable parameter counts are recorded by the probe.

## Stage P0: experts-kernel probe and preflights (per base, before any training)

- **Probe** (`experts_probe.py`, cap 0.5 GPU-h each; G-it and Q-it; the -pt variants share the architecture and kernel
  choice): 64 SELECT rows under `eager` vs `grouped_mm`, and rank-32 LoRA fwd + bwd speed on 12 `a20` rows.
  **Pin rule (per family):** `grouped_mm` if it runs, changes 0 of 64 argmaxes vs eager, keeps max |Δp| ≤ 0.01, and is
  faster per training row; otherwise `eager`. The pin is recorded in amendment 1 before any onestep; the checkpoint
  metadata carries it, so training and inference use the same kernel.
- **Preflights per arm-seed:** admission (every row admitted at the base's limit), one-step, reload (32 SELECT rows,
  0 argmax changes, max |Δp| ≤ 1e-4). A failure is recorded and stops that arm-seed; nothing is rerun.

## Stage A: base screen (matched budget)

- **Cells** = seed 1 of the full contract on each base, run concurrently: MOE-Git-s1 node A GPU3, MOE-Gpt-s1 node A
  GPU4, MOE-Qit-s1 node A GPU5, MOE-Qpt-s1 node B GPU6. Node B GPU7 serves probes, readouts and formal runs.
- **Screen point:** `checkpoint-0000892` of each cell (the second save, a quarter of the pass; rows, order and the
  warm-up + cosine schedule over 3,561 updates are identical across cells). Matched dense reference: **M4-A20r-s1
  `checkpoint-0000892`** on node B (same data order, schedule, rank and trainer), read the same way.
- **Screen readout** (node B GPU7; formal limit 32,768; T = 1; typed-dev, css-pilot, ht-dev2; Qwen-MoE on the FLA
  kernel path with a persisted autotune cache, Gemma SDPA; `v2.eval.dev_readout` with
  `--htdev2-reference` = the dense matched reference's ht-dev2 predictions): P_dev = 100·√(T_dev·H_pilot), H_dev2.
- **Screen rules** (applied in order; a dropped cell is stopped at once):
  1. collapse: a typed-DEV type with one answer category ≥ 95%, Choice or Score at or below chance, or > 1% invalid;
  2. HT-DEV v2 FLAG (ΔH_dev2 ≤ −0.02) vs the dense matched reference;
  3. proxy: P_dev ≥ 8 below the best of {all cells, dense matched reference};
  4. at most one variant per family continues (higher P_dev; within |ΔP| < 2 the higher H_dev2), and at most two cells
     in total.
  If no cell survives, all cells stop, there is no Stage B, and the milestone records the attribution.
- **G1 (projection)** after ≈ 300 updates of every cell: if a cell's projected full attempt exceeds its 13 GPU-h cap,
  or the plan below exceeds 60 GPU-h, the slowest cell stops first (Q-pt, then G-pt), recorded as a budget stop.

## Stage B: main run on the winner(s)

- The surviving cell(s) train to completion (they are seed 1). **Seed 2 of the best surviving cell** starts at the
  screen decision on a freed GPU of the node that holds its base; a second surviving cell also gets a seed 2 only if
  the running total plus projections stays ≤ 60 GPU-h (otherwise it is attribution only, not a finalist).
- **Soup:** exact uniform rank concatenation of the two seeds' BEST checkpoints (rank 64 / α 128, head averaged;
  `v2/27b/lora_soup.py` exactness check), as A20r.
- **Control:** A20r itself (dense Qwen3.8-27B, same data, contract, seeds and soup) is the matched control; the formal
  paired difference soup − A20r is the base effect (dense 27B vs MoE base) at the A20r recipe.
- **Development gates on each soup** (node B, 32K, typed-dev + css-pilot + ht-dev2 + CAL698 fit under the 23:15 rule):
  collapse (rule 1 above); HT-DEV v2 not FLAG vs A20r (`m5` reference `.5655`); P_dev not ≥ 8 below A20r's 78.99.
  Passing soups go to formal, **at most three formal finalists** (the soups; nothing else).

## Formal runs and verdicts (node B, post-key same-panel, 32,768 tokens)

- Typed FINAL, CSS15, public 231; seal, report (`--tier 27B`, loaded and active parameters), then mlx-diag (node A
  scores). Per type and family, Score levels, human transfer per task, latency (BF16 decision p50 / p95 on 64 fixed
  SELECT prompts, as M1), loaded total and active parameters.
- Paired (5,000 draws) vs A20r (DEV2.0-27B), AutoJev-27B (72.133), Eikos-27B, Jebadiah-27B.
- **Successor items 1–7 vs DEV2.0-27B (A20r):** 1 v3 lower bound > 0; 2 human transfer not significantly below; 3 no
  type collapsed (`gates types`); 4 mlx-diag card-eligible (Choice + Noul) not significantly below; 5 tier gates
  (v3 ≥ 64.92; H not below AutoJev / Eikos / Jebadiah); 6 no overlap exposure; 7 `gates public231` not REGRESSION.
- **"Beats AutoJev":** v3 > 72.133, paired lower bound vs AutoJev-27B > 0, H not significantly below AutoJev.
- **Tie-break** among passing finalists: highest paired v3 lower bound vs AutoJev, then vs A20r, then smaller active
  parameters.
- **Item 8:** a successor's frozen package (adapter + head + pinned base identity + experts kernel) and a C1 post-key
  spec go to the eval custodian; this track never runs C1.
- **Naming** follows the base: DEV2.0-26B-A4B (Gemma) or DEV2.0-35B-A3B (Qwen). Replacing the 27B tier vs adding a new
  family member is the coordinator's call.

## Early stops and budget

- Nonfinite loss, OOM, a cap hit or a failed preflight stops that arm-seed (recorded, never rerun). Exit 139 keeps the
  M3 rule (≤ 2 exact recoveries within the arm-seed cap).
- **Caps:** 13.0 GPU-h per full attempt, 14.0 per arm-seed including preflights; **milestone 60 GPU-h** (wall-clock ×
  GPUs, probes, preflights and evaluation included). Before every post-training GPU stage the running total must cover
  it; priority when tight: seed 2 of the best cell > its soup readout > formal > mlx-diag > a second cell's seed 2 >
  optional peer references.
- **Plan** (s = seconds per update from the probe / G1): probes 0.6; preflights ≈ 1.2; screen 4 × 892·s;
  continuation ≤ 2 × 2,669·s; seed 2 1–2 × 3,561·s; readouts 5 × 0.35 + soups ≤ 2 × 0.4; formal ≤ 2 × 1.0; mlx-diag
  ≤ 2 × 0.2. At s = 10: ≈ 45 GPU-h with one winner.
- **Optional peer references** (post-key v3 of Rune-26B-A4B and Decider 35B-A3B BF16, opponents only): only if
  ≤ 2 GPU-h in total, after the finalists' formal runs, with a native adapter that passes a parity smoke.

## Disclosures if a candidate is ever released

Base repo and revision (direct weight origin), Apache-2.0 with the modification notice, total loaded and active
parameters, LoRA rank 64 after the soup with experts frozen, the experts kernel, Gemma's BOS prompt and 4,736-token
training limit (if Gemma), the A7 dose (≈ 20M tokens, Choice-heavy), T = 1 or CAL698, 32K limit, C1 status.

## Amendment 1 (2026-10-01 00:55 UTC+8; after the P0 probes, before any training job)

Receipts: node A `/data/dev2/runs/27b-moe/MOE-{Git,Qit}-probe/{probe/experts-probe.json,check/experts-check.json}`
(0.052 + 0.017 GPU-h Gemma, 0.098 + 0.032 GPU-h Qwen-MoE).

- **Probe results** (rank-32 LoRA fwd + bwd per training row, first 12 `a20` rows):

  | | Gemma-4-26B-A4B-it | Qwen3.5-35B-A3B |
  | --- | --- | --- |
  | Text / routed / active-per-token parameters (text decoder) | 25,233,141,760 / 22,837,985,280 / 3,822,530,560 | 34,152,051,328 / 32,212,254,720 / 2,946,429,568 (+ 508.6M LM head, unused) |
  | LoRA targets / trainable (adapter + head) | 205 / 40,066,560 | 310 / 44,438,016 |
  | eager s per row | 3.05 | 6.91 |
  | `grouped_mm` s per row | 0.66 | 1.07 |
  | Peak allocated | ≈ 103 GB | ≈ 139 GB |
  | Random-head parity `grouped_mm` vs eager (64 rows) | 13 argmax changes, max Δp .048 | 22 changes, max Δp .073 |

- **The preregistered pin rule was mis-specified and is replaced (disclosed deviation).** An untrained random head
  gives near-tied logits, so any BF16 rounding difference flips argmaxes; the rule could not separate a wrong kernel
  from rounding, and eager is infeasible (≈ 49 / 110 s per update, 48+ GPU-h per arm). Replacement, decided before any
  training: `experts_check.py` compares last hidden states at the readout positions of 32 SELECT rows against an FP32,
  autocast-off, eager reference. **Pin `grouped_mm` if its BF16 relative L2 error is ≤ 1.25 × eager's at the median
  and at the max.**
  - Gemma: eager .185 / .338, `grouped_mm` .193 / .359 → ratios 1.05 / 1.06 → **`grouped_mm`**.
  - Qwen3.5-MoE: eager .234 / .349, `grouped_mm` .198 / .419 → ratios 0.85 / 1.20 → **`grouped_mm`**.
  - Both kernels sit at the same distance from FP32, so the random-head flips were rounding, not a kernel fault. The
    checkpoint metadata pins `grouped_mm` for training and inference.
- **Caps (from the measured speed; the rows average ≈ 440–460 tokens vs the probe's ≈ 330):** Gemma ≈ 13–14 s per
  update → 15.0 GPU-h per full attempt / 16.0 per arm-seed; Qwen3.5-MoE ≈ 17–21 s → 20.0 / 21.0. The launcher's
  maximum becomes 21 h. G1 (≈ 300 updates) re-projects from logged speeds.
- **Screen cells, for cost: three, not four.** G-it on node A GPU3, Q-it on node A GPU5, Q-pt on node B GPU6. **G-pt
  is not trained:** Gemma's pretrained start read far worse as frozen features (M1: SELECT .515 vs .676 for -it), while
  Qwen-MoE has no stage evidence, so both of its variants are screened. Node A GPU4 is held for the winner's seed 2
  (from the screen decision); node B GPU7 serves readouts and formal runs. Screen rule 4 (one variant per family)
  applies to Qwen.
- **Budget projection (60 GPU-h):** P0 0.2; preflights ≈ 1.2; screen to update 892 ≈ 3.4 + 2 × 4.7 = 12.8; winner
  continuation ≤ 14.1 (Qwen) / 10.1 (Gemma); seed 2 ≤ 18.8 / 13.5; readouts, soup readout, formal and mlx-diag ≈ 7.
  Worst case ≈ 54, Gemma winner ≈ 45. **A second surviving cell gets no seed 2** (attribution only); if G1 projects
  above 60, Q-pt stops first.

## Amendment 2 (2026-10-01 01:25 UTC+8; after a setup failure of the first one-step, before any training update)

- **Finding (affects how the 27B records describe their trainer; no result changes):** the 27B launcher runs every
  container with working directory `/code` and `python3 -m`, and `-m` puts the working directory ahead of
  `PYTHONPATH`. So the vendored BEST368 tar mounted at `/pipeline` was never imported. A20r's own provenance
  (`M4-A20r-s1/full/run/provenance.json`, `code_sha256`: `train.py` `0111bbfd…`, `data.py` `632bd605…`, `loss.py`
  `c6fc39ed…`, `plan.py` `c39d706e…`, `source.py` `ef7b3017…`, `decision_model.py` `1ab1e49b…`, `lora.py` `7049e35e…`)
  equals the repository files at M4's mirror `1b56e482b`, not the BEST368 files (`train.py` `f5a1a1a0…`). **The A20r
  trainer is the repository trainer.** Today's integration head has the same `train/data/loss/plan/source` bytes, and
  `decision_model.py` / `lora.py` differ only by this milestone's MoE commit (dense paths unchanged).
- **Change:** the cells train with the mirror's `training.model.train`, byte-frozen at `0111bbfd…` (a repository test
  pins it), through `v2.27b.moe.train_moe`, which passes the experts pin to `from_base`, sets Gemma's BOS encoder and
  prompt version, and writes a receipt beside the run. The MoE pipeline tar of amendment 1 is withdrawn (never used).
- **Setup failure, recorded:** the first one-step of MOE-Git-s1, MOE-Qit-s1 and MOE-Qpt-s1 exited 2 after ≈ 1.3 s
  (0.00035–0.00037 GPU-h each): argparse rejected `--experts-implementation` because the frozen trainer has no such
  option. No model code ran. As with Milestone 1's zero-step launch incident, this is an infrastructure failure, not a
  preflight outcome: the one-step and reload are repeated **once** under `-r2` receipts with the fixed entry point; a
  second failure stops the cell. Admission receipts stand.
