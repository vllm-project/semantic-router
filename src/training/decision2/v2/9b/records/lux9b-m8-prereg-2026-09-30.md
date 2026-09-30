# 9B Milestone 8: cross-size distillation from DEV2.0-27B (A20r) as top-ups of the five K seeds, against M7's matched K-mix control line (preregistration)

Status: frozen 2026-09-30 before any Milestone 8 GPU job. Only the CPU prompt build ran first. Goal: a
**successor** to the released `llm-semantic-router/DEV2.0-9B` (K-a13 at T = 1, weights of package `53bac735`),
tested under successor items 1–8 against the released T = 1 run `/data/dev2/runs/release/dev2-8b-t1-derived`
(post-key same-panel v3 67.737, T .8106, H .5660). Development readouts are never release scores; v3 is post-key.
Nothing goes to Hugging Face. M8 runs in parallel with the end of M7 (another worker); M7's worktree, runs and
GPU6–7 are not touched, and M7's K5 members and C line are read-only inputs.

## Why this lever

- **The teacher is our strongest model.** DEV2.0-27B = A20r (`llm-semantic-router/DEV2.0-27B@5323310327e5…`):
  post-key v3 72.36 (T .896, H .584) and C1 v1.2 57.56, against DEV2.0-9B's 67.74 (T .811, H .566) and 53.77.
  Its lead is on both typed decisions and human transfer, and it is a same-family model with the same prompt
  rendering, so its distributions are a denser target than gold on the same rows.
- **What 9B has tried:** own-Lux KL is the K recipe's own teacher; AutoJev-27B soft targets on the human-rated rows
  (M6 KA) were stopped because Noul `rule_precedence` fell 74 items; the A7 dose broke Noul (M5); the five-seed soup
  is the part that works (M6). M7's top-up design (continue each K5 member ~6.2M tokens, re-soup, α line toward Lux
  1.0) costs ~0.3 GPU-h per member and already has a matched control: arm C (own-Lux targets on the same rows).
- **The multilingual failure that sank K5-a12 (item 4)** is a PAWS-X yes-bias. Among M7's development screens,
  the PN1-dev clean gold-no and PAWS-X-6 yes-rates track it (K-a13 .262 / .626; K5-a12 .318 / .652), while
  MLX-DEV-9B does not (K5-a12 Noul-ML +.012 [+.007, +.018] vs K-a13, against formal mlx-diag −.010). M8 therefore
  keeps M7's whole PN1-dev guard (hop **and** clean gold-no) next to MLX-DEV-9B. On this rule M7's C line has no
  eligible point (clean gold-no .280 / .336 / .378 at ⅓ / ½ / ⅔ > .262), which is the bar the D lines must clear.

## Teacher targets (DEV2.0-27B scored runtime, T = 1)

- **Runtime = the scored run `27b/M4-A20r-soup/formal` of A20r**, on node A: `v2.27b.typed_collect_kernel` from
  that run's mirror `ff660322a76b…` in the 27B kernel image `dbe5f32b…`; checkpoint `M4-A20r-soup/soup/checkpoint`
  (identity `2e07451107a2…`, the released package's model files); base Qwen3.8-27B; `--calibration` = the package's
  T = 1 file (`518e19cd…`, every temperature 1.0); 32,768 tokens; `HIP_FORCE_DEV_KERNARG=1`; each job on a private
  copy of the run's frozen autotune cache (`M4-A20r-soup/formal/triton-cache`, tree hashes before / after recorded).
- **Parity first:** the first 80 typed FINAL gold-free prompts through the same job must reproduce the scored run's
  stored predictions (same answers, probabilities within 1e-4; `teacher-parity/parity.json`). A FAIL stops M8.
- **Prompts** (`lux9b.m8_data prompts`, spec `lux9b/specs/m8-kd.json` `2fdc646d…`, code `60cecd999`; node A
  `m8/data/m8-prompts/build`, manifest `4777b12b…`): one gold-free prompt per row of M7's C TRAIN
  (`eb55dbb2…`: 12,443 rows, 6,198,202 native tokens), criteria in option order; the runtime's `question_to_row`
  rebuilds each row's prompt text exactly (checked for every row with the shared `segments`). Shards (by position
  mod 3): `c2c94848…` (4,148), `adc60a1f…` (4,148), `60375c72…` (4,147).
- **Human-rated rows S** = M6's frozen soft-target rule (`lux9b.m6_data.human_rated`, the KA spec's pools and
  sources) on each row's XL r2 pool: 3,153 rows / 898,622 tokens (Choice 504, Noul 540, Score 2,109; A7q 807,
  H1 622, A7s 430, H6 413, H8 200, V1:A6h 184, V1:A1 146, A0s-strict 116, H5 113, A7k 75, V1:A5 47);
  `human.jsonl` `7efa4984…`.
- **Conversion** (`lux9b.m8_data teacher`): every prediction must be valid, bind the teacher identity, the T = 1
  calibration and 32K, and hash to its prompt; the distribution is taken over the row's option keys (Noul from
  P(true)). D1's teacher file covers every row; D2's covers S only. Each arm's train.jsonl is C's TRAIN byte for
  byte. Diagnostics against gold and the own-Lux targets are recorded, never used for selection.

## Arms (each a top-up of the five K5 members at M7's token count, then a soup and the α line)

| Arm | Teacher term on | Loss per row | Artifact |
| --- | --- | --- | --- |
| **D1** | every row: A20r (T = 1) | CE(gold) + 0.5·Brier(gold) + **λ·KL(A20r ‖ student)**, λ = 1.0 | KD1 = uniform FP32 soup of D1-m1..m5 |
| **D2** | human-rated rows S only: A20r (T = 1) | S: as D1; every other row (typed and other non-human rows): CE + 0.5·Brier on gold | KD2 = soup of D2-m1..m5 |
| **C** (control, M7, reused) | every row: own Lux 1.0 | CE + 0.5·Brier + 1.0·KL(own Lux ‖ student) | C5 (M7's line) |
| KDX (only if D1 and D2 both continue) | — | — | uniform soup of all ten D members (= ½ KD1 + ½ KD2) |

- **λ = 1.0**: the teacher term keeps the K recipe's weight, so D1 − C isolates the teacher (A20r vs own Lux) at an
  identical objective and D2 − D1 isolates where the teacher is applied (human rows only vs every row).
- **Everything else is M7's C exactly** (`m7/cont.sh`): members = the SELECT-chosen BEST of M4 K-s1..s3 and M6
  K-s4 / K-s5; member k continues with seed 20260930 + k; `v2.dec.train_dec --init decision2`, full fine-tuning,
  peak LR backbone 5e-6 / head 5e-5, wd .01, warmup .05 then cosine, one epoch, token micro-batches ≤ 32,768 tokens
  / ≤ 64 rows, ≥ 64 rows per update, max length 8,192, `--teacher-partial`, one checkpoint at the final update;
  **trainer code = M7's training mirror `7168f08643eb…`** (the one C used; `train_dec.py` is also unchanged at
  `60cecd999`); image `f83b1d10…` + FLA 0.5.2; M8's training autotune cache = one copy of `m4/triton-cache` (M7's
  source).
- **Reusing M7's C line as the control.** It is exactly token-matched (the same TRAIN file) and the recipe is
  identical apart from the teacher targets, which is verified per member: `lux9b.m8_rules recipe` compares each D
  continuation's trainer provenance contract with C's same member (and D-m1's zero-step preflight with M7's
  `pf-C-m1-zero`) — every field must be equal except the arm name, the teacher file hash and the teacher row count,
  and the training code, image, model source and TRAIN statistics must match; the teacher row count must equal the
  build's. A FAIL stops that arm. So no new control is trained.
- Exposure (item 6): the D arms train on C's rows only; M7's receipt for C (`exposure/C.json` `ae8fd45d…`) has
  `groups: []`, and the members' own TRAIN is x60 (`groups: []`). The teacher adds no rows.

## Early stop after member 1

- D1-m1 and D2-m1 (member 1 = K-s1, seed 20260931, both with preflights) and C-m1 (M7's, re-read in M8 on typed
  DEV, CSS pilot and HT-DEV v2; its PN1 dev screen is M7's) are read at α 1 and T = 1 (every screen below reads
  argmax answers or Noul yes / no, which do not depend on the temperatures).
- **A D arm continues only if, against C-m1** (`lux9b.m8_rules early`):
  1. its final SELECT700 family-macro accuracy is not more than 0.02 below C-m1's (typed protection);
  2. its typed-DEV Noul `rule_precedence` count is not more than 0.01·n (4 of 400) below C-m1's (M6's floor; the
     failure mode of M6's AutoJev teacher);
  3. its PN1-dev true-paraphrase (hop) yes-rate is not more than 0.03 below C-m1's (the hop guard);
  4. its HT-DEV v2 H is not 0.02 or more below C-m1's (no FLAG against the control).
- A stopped arm runs no further member and no line; KDX needs both arms. Nothing is refilled.

## Lines and the development rule (never v3, never mlx-diag, never C1 or public 231)

- θ(α) = α·S + (1 − α)·Lux 1.0 for S ∈ {KD1, KD2, KDX}, α ∈ {⅓, ½, ⅔} (FP32; the Lux end point
  `m3/pf-D-s1-zero/run/checkpoint-0000000`); the soups (α 1) are not read.
- Readouts per point (M7's, runtime mirror `3277dec9d`, CAL698 per-type temperatures, 16,384 tokens): typed DEV
  1,600, CSS pilot 1,430, HT-DEV v2 1,944, PN1 dev 1,974, MLX-DEV-9B (M7's panel `d07fe654…`, 6,147 rows).
- **Anchor R** = M7's re-read of K-a13 (`m7/ref-ka13`, which reproduces M6 exactly; read-only).
- **The rule is M7's, unchanged** (`lux9b.m7_rules alpha`, exact rationals): a point is eligible if
  1. every type keeps c_t ≥ c_t,R − 0.03·n_t;
  2. every typed-DEV family keeps F_f ≥ F_f,R − 0.10;
  3. Noul `rule_precedence` keeps c_RP ≥ c_RP,R − 0.01·n_RP (≥ 334 for R's 338);
  4. HT-DEV v2 is not FLAG against the eval track's 9B reference `9b-m4-K-a13` (ΔH_dev2 > −0.02);
  5. PN1 dev vs R: hop yes-rate ≥ R's − 0.03 and clean gold-no yes-rate ≤ R's;
  6. MLX-DEV-9B vs R: Noul-ML and Choice-ML paired 95% upper bounds ≥ 0.

  G(α) = T(α) − T(R); **no pick** if nothing is eligible or G\* < 0.01; else **α\* = the smallest eligible α with
  G ≥ 0.75·G\***; the pick is dropped if P < P(R) − 8. The CSS-pilot H3 is reported only.
- Report only: D − C at equal α on every readout (the distillation effect at matched tokens and objective), the C5
  line under the same rule, K5-a12 (M6's finalist), and the teacher-target diagnostics.

## Finalists, formal runs and the successor test

- **At most three finalists**: the non-dropped picks of KD1, KD2, KDX, in that priority. A line with no pick, or a
  stopped arm, yields none; slots are not refilled. The newest COORDINATION notes are re-read first.
- Each finalist gets a lock record (checkpoint and calibration SHA-256, rule-output hashes, cache tree), committed
  and pushed before its formal run; then `SHA256SUMS` of `m8/NAME-build/soup` + `m8/NAME-cal`.
- **Formal** (`lux9b/m8/formal.sh`, M7's with M8 paths): runner mirror `3277dec9d`, image `f83b1d10…`,
  `formal-m8/triton-cache` = one copy of the frozen `formal-m3/triton-cache` (tree `af623300…`, refused
  otherwise), 16,384 tokens, over-length invalid; smoke, typed FINAL + CSS15 + public 231, mlx-diag; seal, report,
  compares (5,000 draws), gates. Then `hs1-dev` (diagnostic), the 23:15 shipped-calibration rule and the T = 1
  derivation when T = 1 ships.
- **Successor items** against I = the released T = 1 run: 1. v3 `ci95.low` > 0; 2. `axis_ci95.H.delta.high` ≥ 0;
  3. every `gates types` verdict `OK`; 4. card-eligible mlx-diag Choice + Noul `lux9b.mlx_paired` `ci95.high` ≥ 0
  vs `formal-m4/K-a13-16k-mlx`; 5. tier gates: vs adopted Lux1 16K (65.808) `ci95.low` > 0, `H.delta.high` ≥ 0 vs
  Lux1 and Nimble v2, types `OK`; 6. no overlap exposure (receipts above); 7. `gates public231` vs I not
  `REGRESSION`; 8. C1 post-key guard: a frozen package on node A plus a C1 successor spec for the eval custodian,
  who collects (one successor per tier baseline; if M7 also hands one over, the coordinator decides).
- **Choice among passers of 1–7:** the highest v3 `ci95.low` vs I; within 0.25 of it, an HT-DEV v2 GAIN, then the
  higher formal ΔH (the 09:50 hint). That one goes to the custodian with checkpoint + `SHA256SUMS`, calibration,
  scored run, T = 1 derivation, gate files, the C1 spec and the disclosures.
- **Card of any successor:** names **DEV2.0-27B (A20r, `5323310327e5`) as the distillation teacher** (soft targets
  at T = 1 from its scored runtime on the K-mix top-up rows; for D2 on the human-rated rows only), states that the
  weights remain Lux 1.0-derived full fine-tunes of Qwen3.5-9B (the teacher's weights are not in the package), and
  carries the per-type / per-task and language regressions against the released model.

## Budget, GPUs, chains and stop rules

- **GPUs:** node A GPU2–4, lent by the ~27B track after M5's lane A finished (no M5 job running; checked at
  launch). Every job writes only the lease entry `gpuN.lock/owner.9b-m8` and refuses a GPU whose ~27B owner entry
  is not idle or that shows allocated VRAM; the owner entry is never rewritten. GPU6–7 may be added after M7 ends,
  only by an amendment.
- **Chains** (each uploaded, size + SHA-256 verified, then launched separately with `m8/launch.sh`; liveness by PID
  and container name, never `pgrep -f`):
  - `m8-gpu2`: teacher parity → shard 0 → (shards 1–2 done) teacher build (CPU) → D1-m1 (preflights) → D1-m1 α-1
    readouts → early D1 → (if D1 continues) D1-m2..m4 → (D1-m5 done) KD1 line.
  - `m8-gpu3`: (parity PASS) shard 1 → (build) D2-m1 (preflights) → readouts → early D2 → D2-m2..m4 → KD2 line.
  - `m8-gpu4`: (parity PASS) shard 2 → C-m1 α-1 re-read → member 5 of each continuing arm → (both continue) KDX line.
  - Then the rules (CPU, `m8/rules.sh`), the locks and `m8-post.sh` per finalist (GPU2–4).
- **Projection ≈ 7.5 GPU-h:** teacher parity + three shards ≈ 0.8, two preflight trios ≈ 0.4, ten continuations ≈
  2.7, member-1 and control re-reads ≈ 0.3, nine line points ≈ 2.0, three formal runs ≈ 1.2. **Hard cap 24 GPU-h**;
  a continuation starts only if used + 0.8 ≤ 24 (member 1: + 1.3). The budget decides, never results.
- **Per-attempt caps** (wall-clock × GPUs): teacher parity 0.2; a teacher shard 0.6; a continuation 0.8; a
  preflight trio 0.6; a line point's readouts 0.5; a formal run with mlx-diag 0.6. An attempt past its cap is
  stopped at the next check and recorded.
- Preflights (zero-step, one update, parity against the start checkpoint + bitwise reload) and the recipe checks
  must PASS or the arm stops, without retry. A reproduced ROCm fault, OOM or nonfinite loss stops that arm. No failed
  arm is rerun to fill GPUs. A teacher parity FAIL or a failed teacher build stops the milestone.
- GPU-hours = wall-clock × GPUs, including preflights, teacher jobs, readouts and formal runs; data builds, soups
  and rules are CPU.
- Storage: every artifact stays on node A (`m8/…`, `formal-m8/…`); nothing is uploaded to Hugging Face.
