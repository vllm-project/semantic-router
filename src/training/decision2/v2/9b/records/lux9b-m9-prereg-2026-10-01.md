# 9B Milestone 9: a rank-128 LoRA from Qwen3.5-9B-Base with the candidate head on the released 9B mixture (preregistration)

Status: frozen 2026-10-01 ≈15:10 UTC+8, before any Milestone 9 GPU job. Assignment: COORDINATION 2026-10-01 14:45
("9B M9: L9, a rank-128 LoRA from Qwen3.5-9B-Base, plus an optional LoRA-on-Lux control; retention probes and a
yes-bias guard; stage 2 + IB1 / IB2; node A GPU6–7 + node C GPU1–7 (lease-checked); 120 GPU-h"), with the 14:30
(pre-warm), 14:25, 13:45, 13:40 and 10:55 (Index policy) notes and the chain rule. Goal: a **successor** to the
released `llm-semantic-router/DEV2.0-9B` (K-a13 at T = 1; `main` `b4f65fa8`, weights of package `53bac735`), tested
under successor items 1–8 against the released T = 1 run `/data/dev2/runs/release/dev2-8b-t1-derived` (post-key
same-panel v3 67.737, T .8106, H .5660). Development readouts are never release scores; v3 / public 231 are post-key
same-panel comparisons. Nothing goes to Hugging Face unless an M9 model passes items 1–8 and the coordinator approves.
No Index row is read for selection or tuning; the private Index is run only on frozen finalists, with IX1's harness.

## Why this lever

- **4B M10** (`dec-m10-results-2026-10-01.md`): a rank-128 LoRA on Qwen3.5-4B-Base with the candidate head, trained
  on the released 4B mixture at matched tokens (LH), beat DEV2.0-4B by +4.19 [+0.10, +9.88] post-key and passed items
  1–7. The released 4B lineage (from Nox 1.0) had lost knowledge / maths against its base on Index-row-free retention
  probes; every base-start arm recovered it.
- **27B M5:** on the same rows the L128 adapter beat full fine-tuning on HT-DEV v2 by +.039 [+.021, +.057]
  (adapters keep human transfer that full FT loses).
- **9B's binding problem** (M5–M8): the near-miss / paraphrase yes-bias that grows with distance from Lux 1.0. It
  sank K5-a12's multilingual item (mlx-diag PAWS-X yes-rate .676 vs K-a13 .624) and blocked every M7 / M8 line point
  (PN1-dev clean gold-no yes-rate above K-a13's). A from-base adapter is far from Lux in weights but is trained to
  Lux's distributions on every row (the own-Lux KL term of the K recipe); whether its yes-bias follows Lux or the base
  is unknown, so it is guarded explicitly below.
- Lux 1.0 itself descends from the post-trained Qwen3.5-9B (`c2022362…`), not from the base checkpoint; the probes
  read the base, Lux 1.0 and DEV2.0-9B so the retention question (H1) is answered at 9B rather than assumed.

## Hypotheses

- **H1 (retention):** L9 keeps more of Qwen3.5-9B-Base's knowledge / maths than DEV2.0-9B (retention-probe macro Δ vs
  C0 with a 95% CI above 0).
- **H2 (transfer):** L9 is at least level with DEV2.0-9B on HT-DEV v2 (GAIN or TIE) without the yes-bias that ended
  M7 / M8.
- **H3 (lineage, if L9-Lux runs):** the same adapter from Lux 1.0 isolates the start point: L9 − L9-Lux on the
  probes, HT-DEV v2 and the yes-bias screens (report only).

## Stage 1 arms (matched tokens: one pass over the released 9B TRAIN)

| Arm | Start | Trained | Readout | Seeds | Artifact |
| --- | --- | --- | --- | --- | --- |
| **C0** | DEV2.0-9B = K-a13 (`m4/K-a13-build/soup`, FP32, the released weights) | — (reference) | head | — | — |
| **L9** | Qwen3.5-9B-Base `68c46c4b3498877f3ef123c856ecfde50c39f404` | LoRA r 128, α 256, dropout .05 on every text projection the trainer targets (as at 4B), LR 1e-4; **fresh candidate head** (shared init `--head-init-seed 20261001`), LR 1e-4 | head | 20260926, 20260927 | merged FP32 soup of the two seeds' BEST |
| **L9-Lux** (optional lineage control) | Lux 1.0 `bd45a30aee8c84032791c245c70f86dee5389cc8` (the K recipe's `/model`) | the same LoRA (`--init decision1`); Lux's head continued at LR 1e-4 | head | 20260926, 20260927 | merged FP32 soup |
| **B0** (retention ceiling only) | Qwen3.5-9B-Base | zero-step only, label-token readout (`--readout label_token`) | label-token | 20260926 | the zero-step checkpoint (untrained base) |

- **Data, identical for L9 and L9-Lux** = K-a13's training data byte for byte: x60 `train.jsonl`
  `a66131b1165513128e5a51af78587c4ddefcc836773e724c4514a592e33fc0e0` (122,651 rows, 60,183,732 native tokens) and
  own-Lux targets `teacher.jsonl` `cdcd99c1550d35531df0982ae85116fe3ea5035bc9a09c1ef7b5b3063a32f2c9` (every row; strict
  coverage as in K, no `--teacher-partial`); SELECT700 `32a4352d…`, CAL698 `19cc1a8c…` (isolation only). The M7
  quarantine group is not in x60 (checked). No new data source, so the exposure receipt of x60 (`groups: []`) carries
  over (item 6).
- **Objective and schedule = the K recipe's** except the adapter / start: CE + 0.5·Brier on gold + 1.0·KL(own Lux ‖
  student) on every row; AdamW wd .01; warm-up .05 then cosine; one epoch; token micro-batches ≤ 32,768 tokens / ≤ 64
  rows, 64 rows per update; max length 8,192 (longest TRAIN row 7,754 native tokens); gradient checkpointing; `even8`
  checkpoints with SELECT700 `matrix-v1` selection per seed (earliest tie). `v2.dec.train_dec` from the M9 mirror.
- The trainer contract per seed (start revision, data hashes, flags) is written by the trainer and must equal this
  record; L9 − L9-Lux differs only in `--init` / start and the head's initialisation.
- **Merge and soup:** each LoRA seed's BEST is merged into its pinned start in FP32 (`v2/dec/ops/m10/m10_merge.py`,
  agreement with the adapter form on 128 SELECT rows reported), then the uniform FP32 soup (`v2.dec.soup`). A
  one-seed arm (the other seed failed) is read as that seed alone and disclosed. There are no α lines toward Lux or
  DEV2.0-9B: averages across different starts are not defined, and the L9-Lux α line is not preregistered.
- Third-party Decision models are never starting points or teachers; the own-Lux 1.0 targets are the released
  recipe's.

## Development readouts (never v3, C1, mlx-diag, public 231 or Index rows)

- **Path:** node A GPU6–7, image `f83b1d10…` (the 9B lineage image, FLA overlay), `v2.dec.infer_dec` from the M9
  mirror at 16,384 tokens and **T = 1** (every screen reads argmax answers or Noul yes / no, which do not depend on
  temperatures); MLX-DEV-9B through `v2.dec.eval_rows` (raw probabilities, as M7). M9's node-A readout cache is one
  copy of `m4/triton-cache` (the cache M7 / M8 readouts used). **C0 is read fresh on this path**, and its answers are
  compared with M7's stored `ref-ka13` readouts (parity reported). Every development contrast is paired within the M9
  path.
- **Panels:** typed DEV (1,600), CSS pilot (1,430), HT-DEV v2 (1,944), Score5-typed-DEV (800), `hs1-dev`, PN1 dev
  (1,974), MLX-DEV-9B (6,147 rows, M7's panel `d07fe654…`) and the **retention probes**.
- **Retention probes (M9 panel):** the 4B M10 probe panel (MMLU `validation` + `dev`, ARC `validation` Challenge +
  Easy, a ≤ 1,000-item GSM8K `train` hold-out; native Choice questions; already free of every Decision Index suite
  row and of the 4B TRAIN by word 13-grams) **minus every item with a word 13-gram (or whole short stem) in the 9B
  TRAIN file x60** (`m10_probes.py overlap --kind train`, CPU on node A). Read for B0 (ceiling), Lux 1.0, C0 and the
  arm soups. Gold stays on node A.
- Scoring on node A (CPU): typed DEV / CSS pilot `v2.dec.dev_readout`; HT-DEV v2 paired vs C0
  (`v2/dec/ops/m7/m7_htdev2.py`); Score5-typed-DEV (`m8_rules.py score5t`); probes (`m10_probes.py score`, paired
  bootstrap 2,000 draws); `hs1-dev` (`v2.data.hs1.validity`); PN1 dev (`lux9b.m7_rules pn1` vs C0); MLX-DEV-9B
  (`v2.dec.mlx_dev compare` C0 → X).

## Development gates (per arm soup X, against C0 on the M9 path)

1. **Typed floors:** every type c_t ≥ c_t,C0 − 0.03·n_t; every typed-DEV family F_f ≥ F_f,C0 − 0.10.
2. **Noul floor:** `rule_precedence` c ≥ c_C0 − 0.01·400.
3. **Score floor:** the Score5-typed-DEV check half without COLLAPSE, and without WARN unless C0's check half WARNs.
4. **HT-DEV v2:** GAIN or TIE vs C0 (ΔH_dev2 > −0.02).
5. **Yes-bias guard (preregistered here):**
   - Y1 PN1 dev near-miss: the clean gold-no yes-rate Δ vs C0 ≤ +0.02 **and** its paired group-bootstrap 95% CI
     (2,000 draws, `m7_rules`) has a lower bound ≤ 0 (not significantly above C0). The margin blocks every recorded
     failure (K5-a12 +.056, KD1 ½ +.059) while not failing a model that is level with C0 by sampling noise.
   - Y2 PN1 dev true paraphrase: hop yes-rate Δ ≥ −0.03 (no general "no" bias; M7's hop guard).
   - Y3 `hs1-dev` unmet-condition false-yes rate (`f3_false_yes_rate`) Δ vs C0 ≤ +0.05.
6. **Multilingual (as M7 / M8):** MLX-DEV-9B Noul-ML and Choice-ML paired 95% upper bounds vs C0 ≥ 0.
7. **Retention:** the probe macro (mean of MMLU, ARC, GSM8K accuracy) Δ vs C0 with a paired 95% CI upper bound ≥ 0
   (not significantly below C0).

Reported, never gated: PN1 PAWS-X-6 and all-8 yes-rates, `hs1-dev` family accuracies, CSS-pilot H3, SELECT700, the
retention ceiling (B0) and Lux 1.0, MLX-DEV-9B per cell, and L9 − L9-Lux on every readout.

**Finalists (≤ 2):** every passing arm soup, at most two; if more pass than slots (only possible after stage 2), an
HT-DEV v2 GAIN first, then the larger typed-DEV T. The newest COORDINATION notes are re-read before the pick is
fixed. No passer → no formal run; stage 1 reports development only.

## Formal and successor (finalists only)

- **Formal-path parity first:** DEV2.0-9B's weights are collected once on the M9 formal path (below) and compared
  with the stored bar run `release/dev2-8b-t1-derived`; 0 answer differences → the stored run is the bar, otherwise
  the M9-path run is the paired bar (both reported).
- **Formal:** the 9B procedure (M7 / M8) on node A GPU6 / GPU7, image `f83b1d10…`, one copy of the frozen
  `formal-m3/triton-cache` (tree `af623300…`; refused otherwise), 16,384 tokens, over-length invalid; smoke, typed
  FINAL + CSS15 + public 231, mlx-diag; gold-free seal, report, compares (5,000 draws), gates. The runtime is the M9
  mirror's `v2.dec.infer_dec` (it reads merged LoRA checkpoints); the parity run above makes the bar path-matched.
- **Successor items vs I** (the released T = 1 run): 1. v3 `ci95.low` > 0; 2. `axis_ci95.H.delta.high` ≥ 0; 3. every
  `gates types` verdict `OK`; 4. card-eligible mlx-diag Choice + Noul `lux9b.mlx_paired` `ci95.high` ≥ 0 vs
  `formal-m4/K-a13-16k-mlx`; 5. tier gates: vs adopted Lux1 16K (65.808) `ci95.low` > 0, `H.delta.high` ≥ 0 vs Lux1
  and Nimble v2, types `OK`; 6. no overlap exposure (x60's receipt; no new rows); 7. `gates public231` vs I not
  `REGRESSION`; 8. C1 post-key (the eval custodian, from a frozen package and a C1 successor spec).
- **Choice among passers of 1–7:** the highest v3 `ci95.low` vs I; within 0.25 of it, an HT-DEV v2 GAIN, then the
  higher formal ΔH.
- **Calibration:** T = 1 ships (13:40 policy). An optional **CAL-only Noul temperature-plus-bias fit** for the chosen
  finalist: (T_noul, b) fitted by NLL on the CAL698 Noul rows only (`v2/eval/ix1/calib.py`'s fit), applied post hoc
  to the finalist's stored formal and development predictions (Noul answers can change; Choice / Score cannot). It is
  a package candidate only if (a) typed-DEV Noul Brier and ECE-10 do not worsen and typed-DEV Noul accuracy does not
  fall, (b) items 1–7 recomputed on the T+b predictions still pass with a v3 point estimate ≥ the T = 1 run's, and
  (c) the release runtime serves a Noul bias with exact parity (today it does not: a release hand-off and a
  coordinator decision). Otherwise T = 1; the study is reported either way. No Index row is used.
- **Hand-off:** a frozen package dir on node A (checkpoint, `SHA256SUMS`, calibration, scored run, gate files), a C1
  item-8 spec draft for the custodian, a release-package request (card facts: weights from Qwen3.5-9B-Base through a
  merged LoRA; the own-Lux teacher and the x60 mixture unchanged) and a private Index request for IX1. Any model
  trained on IB data needs the custodian's C1 content recheck first.

## Stage 2 (amendment-gated; no stage-2 GPU job before its amendment is pushed)

- **IB1:** when `xunzhuo/decision-2-training` carries a release-safe IB1 record (IB1-r3), an amendment fixes, before
  any IB1 GPU job: **W+IB1** = the stage-1 winner's recipe + the IB1 TRAIN block (gold labels; no teacher term on IB1
  rows, `--teacher-partial`) and **W+IB1−ID** = the same without the IB1 in-distribution families, two seeds each,
  merged soups, the same gates plus the IB1 DEV slices as diagnostics. "Winner" = the stage-1 finalist with the best
  formal v3 that passes items 1–7; if none, the arm soup with the best development standing (gates passed, then
  HT-DEV v2, then typed T).
- **IB2** (when it lands release-safe): a further amendment on the same pattern.
- Stage-2 finalists go through the same formal / successor path; the custodian's C1 content recheck precedes C1.

## Budget, GPUs, chains and stop rules

- **Cap 120 GPU-h** (wall clock × GPUs, including preflights, merges, readouts, parity and formal).

  | Item | Cap (GPU-h) |
  | --- | ---: |
  | L9 (2 seeds incl. preflights; a 9B 60M-token seed took ≈ 2.35 h full-FT) | 9 |
  | L9-Lux (optional) | 9 |
  | B0 zero-step, merges, soups | 2 |
  | Development readouts (≈ 6 points) | 5 |
  | Formal (parity run, ≤ 2 finalists with mlx-diag) | 4 |
  | Stage 2 IB1 (amendment) | 40 |
  | Stage 2 IB2 (amendment) | 30 |
  | Contingency (never for reruns) | 21 |

  A seed stops at 4.5 GPU-h. No new seed starts once the milestone has used 100 GPU-h.
- **Preflights per seed:** zero-step, one update, the `preflight_dec` gate (parity against the start + bitwise
  reload). A failed preflight stops that arm (no rerun, no replacement seed). A reproduced ROCm fault, OOM or
  nonfinite loss stops the arm. Failed steps are recorded and never rerun to fill GPUs.
- **GPUs:** node C GPU1–5 for training (lease owner files `track=9b-m9`, written only when the GPU's lease is free or
  released; **node C GPU0 is foreign and never used**), node A GPU6–7 for readouts and formal. Node C GPU6–7 stay free
  for IX1 / release private-Index work unless a stage-2 amendment leases them. Each node-C container gets only its
  GPU's render node plus `/dev/kfd` with `ROCR_VISIBLE_DEVICES=0`, `--network none`.
- **Autotune cache:** node C's M9 training cache is one copy of node A's 9B training cache `m4/triton-cache`;
  **L9-s1's preflights run alone first (pre-warm, 14:30 convention)**; the other seeds start after its one-update run.
- **Chain rule:** scripts run only from an exact mirror (`mirror_to_node.sh`, content manifest verified); chains are
  launched in a separate step after the mirror is verified; liveness by PID and container name, never `pgrep -f`;
  the first log line is confirmed. Poll every ≤ 30 min with a state commit (`records/m9-state.md`); hand off near 5 h.
- **Storage:** node C `/data/dev2/runs/9b/m9/` (training), node A `/data/dev2/runs/9b/m9/` (readouts, soups, formal).

## Code (committed before use)

`v2/9b/lux9b/m9/` (node-C launcher with GPU isolation, seed runner, chains, data staging), `v2/9b/lux9b/m9_probes.py`
(the x60 probe subset), `v2/9b/lux9b/m9_rules.py` (gates 1–7 and finalists); readout, scoring and formal wrappers
follow before their first use. Shared modules are reused unchanged (`v2.dec.train_dec`, `preflight_dec`,
`infer_dec`, `eval_rows`, `soup`, `m10_merge.py`, `m10_probes.py`, `m7_htdev2.py`, `m8_rules.py`, `lux9b.m7_rules`,
`v2.dec.mlx_dev`).
