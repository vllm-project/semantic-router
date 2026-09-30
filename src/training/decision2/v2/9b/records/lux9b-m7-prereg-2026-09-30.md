# 9B Milestone 7: a PN1-r2 continuation of the five K seeds against a matched-token K-mix continuation, with multilingual and human-transfer development screens (preregistration)

Status: frozen 2026-09-30 before any Milestone 7 GPU job. The CPU builds (both continuation files, their exposure
receipts and MLX-DEV-9B) ran first. Goal: a **successor** to the released `llm-semantic-router/DEV2.0-9B` (K-a13 at
T = 1, weights of package `53bac735`), tested under successor items 1–8 against the released T = 1 run
`/data/dev2/runs/release/dev2-8b-t1-derived` (post-key same-panel v3 67.737, T .8106, H .5660). Development readouts
are never release scores; v3 is post-key. Nothing goes to Hugging Face.

## What M6 left, and the lever

- **M6's near miss.** K5-a12 = ½·(five-seed K soup) + ½·Lux 1.0 scored 69.362, **+1.62 [−0.19, +2.41]**. Typed rose
  (+.030) and human transfer was level (+.006). It failed item 1 (lower bound) and item 4: card-eligible mlx-diag
  −.010 [−.020, −.001]; Noul −.017 [−.034, .000], Japanese −.033.
- **The mlx loss is a PAWS-X yes-bias that grows with distance from Lux.** The predicted-yes rate on mlx-diag's
  PAWS-X Noul part (gold .50), from the stored formal predictions (a diagnosis of M6's failure; M7 never reads
  mlx-diag for selection):

  | Model | yes (all 7) | non-English |
  | --- | ---: | ---: |
  | Lux 1.0 (same-renderer 16K) | .550 | .555 |
  | K-a13 (released) | .624 | .625 |
  | U-a13 | .629 | .630 |
  | KN-a12 | .663 | .667 |
  | K5-a12 | .676 | .680 |

  K5-a12's PAWS-X losses against K-a13 sit in ja (−5 of 100), de (−3) and zh (−3); Choice lost 3 of 882.
- **PN1-r2 targets exactly that.** It is paraphrase-style multilingual Noul built to cut near-miss / swap yes-rates
  (4,364 rows, 491,023 tokens; ja 2,236, zh 948, de 822, ru 148, ar 108, ko 102), cleared for released models at
  dataset revision `27b1d2f1` (02:25 note). It is a strong lever: at 2B, decoder M7's S7P (PN1-r2 ×3 in a full run)
  moved the PN1-dev yes-rate from .925 to .527 (near-miss .653 → .156) but also cut the true-paraphrase (hop) yes-rate
  .996 → .938. Hence a hop guard.
- **Design choice: a short continuation of each of the five K seeds, then the soup.** Evidence and cost:
  - The K5 soup is the part that works: P 69.84 against its seeds' mean 65.51. A continuation of the *soup* with
    x60 replay would pull it back toward single-seed optima. Continuing each *seed* keeps every member near its own
    optimum, and the five-member soup keeps the averaging.
  - A continuation costs about 0.35 GPU-h per member (6.2M tokens) against about 3 GPU-h for a new 60M-token seed.
    Retraining seeds with PN1-r2 would give at most two new seeds in this budget, and a 7-seed soup would dilute
    PN1 to 2/7 of its members.
  - The matched-token control (C) is the same continuation on K-mix replay only, so P − C isolates PN1-r2 from
    "more steps on x60".
- **Protection.** Noul keeps M6's `rule_precedence` floor and typed keeps M6's type and family floors plus the typed
  gain requirement. Human transfer uses HT-DEV v2 (04:10 note), which replaces the CSS-pilot H3 condition.

## Fixed configuration

| Field | Value |
| --- | --- |
| Members (starts) | the SELECT-chosen BEST of M4 K-s1 / K-s2 / K-s3 (checkpoints 1,624 / 1,420 / 1,424) and M6 K-s4 / K-s5 (1,629 / 1,621): the five members of M6's K5 soup |
| Trainer | `v2/dec/train_dec.py` with the new `--init decision2` (`f24a8f46c`, a shared-module change with test): full continuation of a full Decision 2.0 checkpoint that keeps the root `full_training_source` (Lux 1.0), so soups with Lux stay defined; the direct start goes to `continued_from`. The decision1 / base paths are unchanged |
| Hyper-parameters | the K recipe's (full fine-tuning, CE + 0.5·Brier + 1.0·KL(own Lux) where a teacher row exists, wd .01, warmup .05 then cosine to 10%, one epoch, token micro-batches ≤ 32,768 tokens / ≤ 64 rows, ≥ 64 rows per update, max length 8,192), except the **peak learning rates, halved to backbone 5e-6 / head 5e-5** (a continuation restarts from the K seeds' final 10% LR; 5e-6 / 5e-5 is the decoder full-FT recipe under which PN1-r2 had its measured effect), fresh AdamW state, `--teacher-partial` in both arms, and **one checkpoint at the final update** (SELECT700 at the start and the end, recorded; SELECT is blind to the lever, so it picks nothing) |
| Seeds | member k continues with seed 20260930 + k in both arms (P and C paired by member and seed) |
| Runtime | image `sha256:f83b1d10…` + FLA 0.5.2; node A GPU6 (P) and GPU7 (C); M7 training autotune cache = one copy of `m4/triton-cache` (as M5 / M6) |
| Readouts | `v2.dec.calibrate_ckpt` then `v2.dec.infer_dec` from the incumbent's runner mirror `3277dec9d` (CAL698 per-type temperatures), 16,384 tokens: typed DEV 1,600, CSS pilot 1,430, HT-DEV v2 1,944, PN1 dev 1,974; `v2.dec.eval_rows` on MLX-DEV-9B (raw probabilities, as SELECT700). Noul yes / no and every argmax answer, hence every screen, are temperature-invariant |
| Code | mirror `7168f08643eb9297c7f457799dc06223056be656` on node A: `lux9b/m7_data.py`, `lux9b/m7_rules.py`, wrappers and chains `lux9b/m7/`, spec `lux9b/specs/m7-topup.json`; 9B tests 89 pass (13 skipped); `v2.dec.tests.test_dec` 28 pass in the image; `lux9b/m7/dryrun_test.sh` 30 checks pass |

## Data (node A `/data/dev2/runs/9b/m7/data/`; every builder guard passed)

Build `m7-topup` (spec `m7-topup.json`; manifest `b2390d288dd6beb5…`; `v2.common.eval_only` guard and row check;
isolation from SELECT700, CAL698, the PN1 dev slice and `hs1-dev`; C1 registry + MASSIVE / PAWS-X / XNLI source guard:
0 denied):

| TRAIN | Rows | Native tokens | C / N / S token share | train.jsonl | teacher.jsonl |
| --- | ---: | ---: | --- | --- | --- |
| **P** | 19,201 (10,473 replay + 8,728 PN1-r2) | 6,198,838 | .260 / .542 / .198 | `e6a74ea99132d63a84d57dd7b10e701dffde9417ee3544cb0e0ad7268b8d89a0` | `ede96e74a1eedf9537e48efb7171b7409e202d60a6a41f4f2e70b339a9ab7794` (10,473 replay rows; PN1 gold only) |
| **C** | 12,443 (replay) | 6,198,202 | .311 / .458 / .231 | `eb55dbb22e87353b7bf33a92186c4c7167c90a02d8c3a1519faff31e4de8ec5e` | `babd72c570cd850dc7a338783ab20c7d01f2a1f3e91c32888323e43982bb88a7` (every row) |

- **Replay:** whole x60 groups (x60 `a66131b1…`, own-Lux targets `cdcd99c1…`) stratified by pool × source × type ×
  language (178 strata), in `sha256("20260930:m7:replay\0" + group)` order per stratum. P's replay budget 5.0M
  (realized 5,216,792); C's budget is bisected so that C's tokens equal P's (gap 0.01%), and P's replay groups are
  contained in C's (checked).
- **PN1-r2:** `m4/pn1/arms/pn1.train.jsonl@27b1d2f1` `c1cec06b…`, all 4,364 rows twice (copy 2 ids suffixed `~r2`);
  PN1 is 45% of P's rows and 16% of its tokens. No PN1 row shares an id, input or group with x60.
- **Exposure** (`v2.eval.overlap_effects exposure`, K-a13 groups `2194716a…`): P `47153775…`, C `ae8fd45d…`; both
  `groups: []`, `methods_agree: true`. The members' own TRAIN is x60 (`14e7c0ca…`, `groups: []`).
- **MLX-DEV-9B** (the multilingual development check): the decoder's MLX-DEV panel (relayed from node B over the
  direct node link, SHA-256 checked: panel `100ae4e7…`, index `6ffa4b84…`) minus every group sharing a group id, row
  id or input hash with x60 or PN1-r2 (1,285 groups). **6,147 rows / 2,517 groups**, every cell kept: Noul H5
  answerability 3,968 and H8 MIRACL 563; Choice MTOP 272 and JCommonsenseQA 270; Score A7q 368, A7k 255, A7s 256,
  Spanglish 195 (panel `d07fe654…`, index `31dfbe58…`, manifest `d16406db…`). Disclosed: 133 kept A7q groups share
  an OASST prompt line with another rated reply in x60, as do 30 H5 and 1 MIRACL group. MLX-DEV Noul is
  answerability / relevance, not paraphrase; it guards multilingual behavior in general, while PN1 dev reads the
  paraphrase construction.

## Arms

| Arm | Continuation of each K seed on | Artifact |
| --- | --- | --- |
| **P** (treatment) | P (x60 replay + PN1-r2 ×2) | P5 = uniform FP32 soup of P-m1..P-m5 |
| **C** (matched-token control) | C (x60 replay at P's tokens) | C5 = uniform FP32 soup of C-m1..C-m5 |

The soup is the artifact by construction (M6's seed rule already chose the soup of these five members: 69.84 ≥
65.51); members are not read individually.

## Early stop after member 1

- P-m1 and C-m1 (member 1 = K-s1, both with preflights) are read at α 1 on PN1 dev (T = 1).
- **P continues only if** (`lux9b.m7_rules early`):
  1. its clean gold-no yes-rate is at least 0.02 below C-m1's (the lever works in family);
  2. its hop yes-rate is not more than 0.03 below C-m1's (no general "no" bias);
  3. its final SELECT700 family-macro accuracy is not more than 0.02 below C-m1's (typed protection).
- Clean gold-no = gold-no rows of `pn-near` / `pn-name` / `pn-twin` outside the constructions PN1-r2 dropped for
  label noise (`pn-near` in es / fr / ar / ru / ko; Russian `pn-name`), so the noisy dev labels (02:25 note) do not
  drive it.
- A stopped P runs no further member and no line; C always runs (it is the control and a candidate line).

## Lines and the development rule (never v3, never mlx-diag)

- θ(α) = α·S + (1 − α)·Lux 1.0 for S ∈ {P5, C5}, α ∈ {⅓, ½, ⅔} (FP32, repeated soup members; the Lux end point
  `m3/pf-D-s1-zero/run/checkpoint-0000000`). The soups themselves (α 1) are not read (M6's α-1 soup failed the
  floors).
- **Anchor R** = a fresh M7 re-read of K-a13 (`m4/K-a13-build/soup` with M6's `ref-ka13` CAL698 file), expected to
  reproduce M6's `ref-ka13` (T .9250, C / N / S 799 / 338 / 343, RP 338).
- **α is eligible** (`lux9b.m7_rules alpha`, exact rationals where counts exist) if:
  1. every type keeps c_t ≥ c_t,R − 0.03·n_t;
  2. every typed-DEV family keeps F_f ≥ F_f,R − 0.10;
  3. Noul `rule_precedence` keeps c_RP ≥ c_RP,R − 0.01·n_RP (≥ 334 for R's 338);
  4. **HT-DEV v2 is not FLAG**: ΔH_dev2 > −0.02 against the eval track's 9B reference `9b-m4-K-a13`
     (`v2.eval.dev_readout --htdev2-reference`);
  5. **PN1 dev guard** vs R: hop yes-rate ≥ R's − 0.03 and clean gold-no yes-rate ≤ R's;
  6. **MLX-DEV-9B guard** vs R (`v2.dec.mlx_dev compare`, group-clustered paired bootstrap, 2,000 draws): Noul-ML
     and Choice-ML 95% upper bounds ≥ 0.

  The CSS-pilot H3 is reported only. G(α) = T(α) − T(R); **no pick** if nothing is eligible or G\* < 0.01; else
  **α\* = the smallest eligible α with G ≥ 0.75·G\***; the pick is dropped if P < P(R) − 8.
- Report only: P − C at equal α on every readout (the PN1 effect at matched tokens), and both lines against
  K5-a12 (M6's finalist, re-read on the screens) and Lux 1.0 (PN1 dev and MLX-DEV-9B).

## Finalists, formal runs and the successor test

- **At most two finalists**, the non-dropped picks of P5 and C5 in that priority (hypothesis order); a line with no
  pick, or a stopped arm, yields none, and slots are not refilled. The newest COORDINATION notes are re-read first.
- Each finalist gets a lock record (checkpoint and calibration SHA-256, rule-output hashes, cache tree), committed
  and pushed before its formal run.
- **Formal** (`lux9b/m7/formal.sh`, M6's with M7 paths): runner mirror `3277dec9d`, node A GPU6 / GPU7, image
  `f83b1d10…`, `formal-m7/triton-cache` = one copy of the frozen `formal-m3/triton-cache` (tree `af623300…`,
  refused otherwise), 16,384 tokens, over-length invalid; smoke, typed FINAL + CSS15 + public 231, mlx-diag; seal,
  report, compares (5,000 draws); gates. Then `hs1-dev` for the finalist and the incumbent (diagnostic), the 23:15
  shipped-calibration rule (`ship_cal.sh`) and the T = 1 derivation if it ships T = 1.
- **Successor items** against I = the released T = 1 run:
  1. v3 `ci95.low` > 0;
  2. `axis_ci95.H.delta.high` ≥ 0;
  3. every `gates types` verdict `OK`;
  4. card-eligible mlx-diag Choice + Noul `lux9b.mlx_paired` `ci95.high` ≥ 0 vs `formal-m4/K-a13-16k-mlx`;
  5. tier gates: vs adopted Lux1 16K (65.808) `ci95.low` > 0, `H.delta.high` ≥ 0 vs Lux1 and Nimble v2, types `OK`;
  6. no overlap exposure (receipts above; a line point also carries x60's);
  7. `gates public231` vs I not `REGRESSION`;
  8. C1 post-key guard: a frozen package on node A plus a C1 successor spec for the eval custodian, who collects.
- **Choice among passers of 1–7:** the highest v3 `ci95.low` vs I; within 0.25 of it, the coordinator's 09:50 hint
  decides (an HT-DEV v2 GAIN, then the higher formal ΔH). That one goes to the custodian at once with checkpoint +
  `SHA256SUMS`, calibration, scored run, T = 1 derivation, gate files, the C1 spec and the disclosures.
- **Disclosures for any PN1-trained successor:** Tatoeba (CC BY 2.0 FR / CC0) with per-sentence attribution from
  the PN1 release; the PN1-r2 blind-review error rate (1.96% [0.54, 4.94]); the C1 custodian content recheck the PN1
  record asks for (Tatoeba text matched C1 items in the v1.2 rescan) goes with the item-8 hand-off.

## Budget, chains and stop rules

- **Chains** (each uploaded, size + SHA-256 verified, then launched in a separate step with `m7/launch.sh`; liveness
  by PID and container name, never `pgrep -f`):
  - `m7-gpu6`: P-m1 (preflights) → P-m1-e1 → early rule → (if P continues) P-m2..P-m5 → P5 line.
  - `m7-gpu7`: R's PN1 dev → C-m1 (preflights) → C-m1-e1 → R's typed DEV, CSS pilot, HT-DEV v2, MLX-DEV-9B and
    screens → C-m2..C-m5 → C5 line → report-only K5-a12 and Lux 1.0 screens.
  - Then the rules (CPU, `m7/rules.sh`), the locks and `m7-post.sh` per finalist.
- **Projection ≈ 8 GPU-h:** two preflight trios ≈ 0.8, ten continuations ≈ 3.7, member-1 and reference readouts
  ≈ 0.6, six line points ≈ 1.8, report-only references ≈ 0.25, two formal runs ≈ 0.8. **Hard cap 24 GPU-h**; a
  continuation starts only if used + 0.8 ≤ 24 (member 1: + 1.3). The budget decides, never results.
- **Per-attempt caps** (wall-clock × GPUs): a continuation 0.8; a preflight trio 0.6; a line point's readouts 0.5;
  a formal run with mlx-diag 0.6. An attempt past its cap is stopped at the next check and recorded.
- Preflights (zero-step, one update, parity against the start checkpoint + bitwise reload) must PASS or the arm
  stops, without retry. A reproduced ROCm fault, OOM or nonfinite loss stops that arm. No failed arm is rerun to fill
  GPUs. If C fails, P cannot be tested and the milestone stops and reports.
- GPU-hours = wall-clock × GPUs, including preflights, readouts and formal runs; data builds and soups are CPU.
- Storage: every artifact stays on node A (`m7/NAME-build/soup`, `m7/NAME-cal/`, `SHA256SUMS` re-hashed once).
  Nothing is uploaded to Hugging Face.
