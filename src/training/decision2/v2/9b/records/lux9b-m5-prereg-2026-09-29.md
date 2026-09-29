# 9B Milestone 5: a family-equal A7 typed-curriculum dose on the K recipe, with and without the own-Lux trust region on the dose (preregistration)

Status: frozen 2026-09-29 before any Milestone 5 GPU step (the two TRAIN builds and the overlap screen below ran on CPU;
the power check below re-read existing formal runs on CPU). Goal: a **successor** to the released 9B package
(`llm-semantic-router/DEV2.0-8B@53bac735…`, being renamed DEV2.0-9B; K-a13 = ⅓·K soup + ⅔·Lux 1.0, BF16, T = 1, 16K)
under the coordinator's successor rule (COORDINATION 2026-09-29 16:05, item 3), tested against the released revision's
T = 1 scored run `/data/dev2/runs/release/dev2-8b-t1-derived` (post-key same-panel v3 67.737, T .8106, H .5660). Development
readouts are never release scores; v3 is post-key. Nothing goes to Hugging Face during the milestone.

## Evidence and design

- **Required effect.** Pairing the existing post-key runs against the T = 1 run (CPU, `m5/power/`, `v2.eval.gates paired`):
  K-a13 (CAL698 run) 0.000 [0, 0] (temperatures change no answers); U-a13 +0.129 [−2.199, +1.014]; KN-a12 +0.244 [−1.514,
  +2.282] (H −.002 [−.029, +.031]); DW +0.834 [−4.483, +3.374] (T +.041 [+.022, +.059], H −.014 [−.097, +.026]). A lower bound
  above 0 therefore needs about +2 v3 over K-a13, i.e. typed T ≈ .86 (+.05) with human transfer held at .566.
- **Lever chosen: the A7 typed-curriculum dose** (the only lever with a large measured typed effect). ~27B: a family-equal
  ~5M-token A7 slice raised typed accuracy by +.099 (+.081, +.118) over its matched control (constraint competition .38 → .57,
  exception stack .59 → .79). 9B M3: DW (≈ 73M gold-only A7 Stage tokens, α ½) had the best 9B typed FINAL (.8519; exception
  .882, ledger .662). The K recipe's x60 holds only a uniform third of every XL r2 pool, so its ≈ 17M A7 Stage tokens are
  dominated by long families (stage4_relations 18.7%, stage4_dense_table 13.1%). A7 is licence-clean (project-generated
  curricula with program-oracle labels; registries a7-v1 `29bc7ea8…`, a7-v2 `f32bb58b…`) and C1-independent (registry,
  27B recheck, the 9B builder's name-key guard, embedding scan).
- **Second factor: the trust region on the dose rows.** Own-Lux targets agree with gold on 87.8% of the dose rows (Choice
  .890, Noul .823, Score .890; stage4_automaton .449), so KL(own Lux) on those rows pulls against gold on about one row in
  eight. K put KL on every row; D's A7 rows were gold-only.
- **Levers not chosen (reasons).** AutoJev-27B soft targets: its human-transfer lead over K-a13 is FLUTE (H without FLUTE
  ≈ .562 vs .565), every AutoJev distillation arm so far (0.6B, 4B, 9B B, 27B M2-K) failed to raise human transfer, and no
  XL r2 human row has AutoJev targets. A fresh XL r2 subsample or more K seeds: expected gain well below +2 v3. α
  re-selection on the M4 lines: the M4 readouts were already at the formal 16,384-token limit on the FLA kernel path; the
  rule below applied to the M4 K line picks K ½ (report-only), whose sibling KN-a12 is +0.24 v3 and significantly below
  K-a13 on card-eligible mlx-diag (−.020 [−.029, −.010]). The research & data paraphrase-style Noul arm (PN1) has no
  dataset revision yet; it is left for Milestone 6 unless it lands with own-Lux targets before the M5 lines are read (it
  would then need an amendment before use).
- **Known blind spot.** No development panel sees mlx-diag; KN-a12 (α ½) lost .020 on its card-eligible parts. The α rule
  keeps M4's "smallest α with three quarters of the best gain" to limit drift the development panels cannot see.

## Fixed configuration (every training arm; as Milestone 4's K)

| Field | Value |
| --- | --- |
| Start | own `llm-semantic-router/Decision-1.0-Lux-9B@bd45a30a…` (node A `/data/decision20-20260926/models/Decision-1.0-Lux-9B`), 7,940,895,744 parameters |
| Trainer | `v2/dec/train_dec.py` (byte-identical to the M4 training mirror `9d90212dd`), full fine-tuning, every hyper-parameter as M4 (backbone lr 1e-5, head lr 1e-4, wd .01, warmup .05, one epoch, token micro-batches ≤ 32,768 tokens / ≤ 64 rows, ≥ 64 rows per update, max length 8,192, even8 checkpoints, SELECT700 `matrix-v1`) |
| Objective | CE + 0.5·Brier + 1.0·KL(own Lux ‖ student) on every row that has a teacher record |
| Runtime | image `sha256:f83b1d10…` + FLA 0.5.2; node A **GPU6–7** only; M5 training autotune cache = one copy of `m4/triton-cache` |
| Readouts | `v2.dec.calibrate_dec` (arms) or `calibrate_ckpt` (soups, references), then `infer_dec`, from the incumbent's runner mirror `3277dec9d` (the M4 soups / formal mirror; `training/model/infer.py` changed since, so newer mirrors would change the adapter identity), CAL698 per-type temperatures, typed DEV 1,600 + CSS pilot 1,430 at 16,384 tokens |
| Code | data `4c375723f` (`lux9b.m3_data` dose section; `m5/data.sh`), rules `4e8d963e3` (`lux9b/m5_rules.py`, `mlx_paired.py`, `score_levels.py`), wrappers `facd4f846` + `a7b326315` (`lux9b/m5/`) |

## Data (frozen; node A `/data/dev2/runs/9b/m5/data/`; all builder guards passed)

- **Recipe = M4's x60, byte-identical:** extracting the x60 ids from the new files reproduces `train.jsonl` `a66131b1…` and
  `teacher.jsonl` `cdcd99c1…` (122,651 rows, 60,183,732 native tokens).
- **Dose `a7fe24`:** 48,700 rows / 23,997,646 native tokens of A7 Stage curricula (A7g 9.21M, A7o 7.54M, A7i 4.95M, A7p 2.29M;
  English 14.52M, Chinese 9.48M; Choice / Noul / Score 17.75 / 4.33 / 1.92M), selected **family-equal** over the 39 families
  (water-filled share 1,151,372 tokens; 13 families at the full share, 26 exhausted), whole groups, order
  sha256(`20260929:m5-dose:` + id). Eligible = XL r2 or A7-only-control (`cx-xl-r2-a7v1-full`) members not in x60, with own-Lux
  targets (M4 waves + c-w1 `1a6b5a80…`), ≤ 4,096 tokens, no denied source (C1 name-key registry `db00d4fe…`, MASSIVE, PAWS-X,
  XNLI), isolated from SELECT700 / CAL698, no duplicate input.

| TRAIN | Spec (sha256) | Rows / native tokens | train.jsonl | teacher.jsonl | manifest |
| --- | --- | --- | --- | --- | --- |
| **KD** `m5-kd-x60-a7fe24` | `156efd16804bda5ab7cb1af0c8d4778fc8586d745b8c49a97ea7547ff6acf110` | 171,351 / 84,181,378 | `86a41afb7f2a82159e8a5edfc29d342f81f75b5411cb45e863ada5c24fbbcbb8` | `0f9e6ed25ac35d0fd49a50292fbf847a1ffbe89cf7c5ec2e5a073593fb0e787a` (every row) | `417667b9d470348a364b2e0b49a0c01399eb210084ff7c6831d2e8416871014d` |
| **KG** `m5-kg-x60-a7fe24g` | `42fa21b19b85e10a494a79393124382e2d25cf82c73fbb421b77c2018097803f` | the same file | `86a41afb…` (byte-identical) | `cdcd99c1550d35531df0982ae85116fe3ea5035bc9a09c1ef7b5b3063a32f2c9` (= M4 x60; no dose ids) | `91a5c5ff4b985ea9155a44a289c58dc22125ba8a0a8c1571ee979964bc365efd` |

- **Overlap exposure** (`v2.eval.overlap_effects exposure`, the K-a13 groups file `2194716a…`, 305 excluded groups):
  `exposure.json` `5639cfa3d7e5522c7108e917aa7fc0eed1b676974646fc3e0138c5729b5b3d92`, 171,351 rows, `groups: []`,
  `methods_agree: true`. KG's train is the same file.

## Arms, seeds and contrasts

| Arm | TRAIN | Teacher term | Seeds (run names) |
| --- | --- | --- | --- |
| **KD** — K + dose, trust region everywhere | KD | 1.0·KL on all 171,351 rows | 20260926 (`KD-s1`, with preflights), 1 (`KD-s2`) |
| **KG** — K + dose, dose rows gold-only | KG | 1.0·KL on the 122,651 x60 rows; `--teacher-partial` (dose rows: CE + Brier only; the loss is normalised by all rows, so every x60 row's KL gradient equals KD's) | 20260926 (`KG-s1`, with preflights), 1 (`KG-s2`) |

- Contrasts (development, reported only, never selection): **KD − KG** = KL on the dose rows (seed-paired s1/s1, s2/s2 and
  artifact-level); **KD − K** (M4 artifact) = the dose on the K recipe, confounded with 84M vs 60M tokens (disclosed).
- **Seed rule (M4, two-seed form):** an arm's artifact is the uniform FP32 soup of its two SELECT-chosen seeds if the soup's
  proxy P ≥ the seed mean, otherwise its primary seed `-s1`. Lines are built from the soup as soon as it exists; if the rule
  picks the primary seed, those line readouts are discarded and the line is rebuilt from `-s1`.

## Lines and the development-only α rule (never v3)

θ(α) = α·S + (1 − α)·Lux 1.0 (FP32, repeated soup members; Lux end point `m3/pf-D-s1-zero/run/checkpoint-0000000`), α ∈ {⅓, ½,
⅔, 1}; α = 1 is S itself.

| Line | S | α grid |
| --- | --- | --- |
| **KD** | KD artifact | ⅓, ½, ⅔, 1 |
| **KG** | KG artifact | ⅓, ½, ⅔, 1 |
| **UM5** | ½·KD artifact + ½·KG artifact (members KD, KG, Lux × 4 / 2 / 1 / 0) | ⅓, ½, ⅔, 1 |

**Anchor R** = a fresh M5 readout of the incumbent's scored checkpoint `m4/K-a13-build/soup` (`ref-ka13`; expected to
reproduce M4's K ⅓ readout: T .925, C/N/S 799 / 338 / 343, H3 .562, P 73.22; if it does not, R is still the fresh readout and
the difference is recorded). A fresh Lux readout `ref-lux` is report-only. `lux9b.m5_rules alpha`, exact rationals:

- α is **eligible** if (i) c_t(α) ≥ c_t(R) − 0.03·n_t for Choice, Noul and Score; (ii) H3(α) ≥ H3(R) (CSS-pilot three-task
  mean, the eval track's recommended human-transfer screen); (iii) no typed-DEV family below F_f(R) − 0.10. Equality is
  eligible.
- G(α) = T(α) − T(R); G* = the largest G over eligible α. **No pick** if nothing is eligible or G* < 0.01. Otherwise
  **α\* = the smallest eligible α with G(α) ≥ 0.75·G\***.
- Proxy drop (line-local): the pick is dropped if its P < P(R) − 8.
- The M4 Lux-anchored rule is also computed, report-only.

## Finalists, formal runs, successor test

- **At most three finalists:** the non-dropped picks of KD, KG, UM5, in that fixed priority (hypothesis order). A line
  without a pick (or whose arm stopped; UM5 needs both artifacts) yields no finalist; slots are not refilled.
- Each finalist gets a short **lock record** (checkpoint and calibration sha256, rule output hashes, cache tree hash)
  committed and pushed **before its formal run**; lines may lock and run as they finish (progressive updates). The rules
  read development readouts only, so no formal result can change a later pick.
- **Formal:** `lux9b/m5/formal.sh` with runner mirror `3277dec9d`, node A GPU6 or GPU7, image `f83b1d10…`, `formal-m5/triton-cache`
  = one copy of the frozen `formal-m3/triton-cache` (source tree `af623300…`, as M4), 16,384 tokens, over-length invalid:
  smoke, typed FINAL + CSS15 + public 231, mlx-diag; seal, report, paired compares (5,000 draws, seed 20260927).
- **Successor rule** (16:05 note), each item vs the **current incumbent run** I = the latest M5 successor already handed off,
  else the released T = 1 run:
  1. v3 `ci95.low` > 0 (`gates paired` vs I);
  2. human transfer not significantly below: `axis_ci95.H.delta.high` ≥ 0 (same file);
  3. no type collapsed: every `v2.eval.gates types` verdict `OK`;
  4. mlx-diag card-eligible overall (Choice + Noul, all languages; the XNLI-based Score part excluded) not significantly
     below: `lux9b.mlx_paired` `ci95.high` ≥ 0 vs I's mlx answers (for the released run: `formal-m4/K-a13-16k-mlx`, identical
     answers);
  5. the 9B tier gates (eval record `m4-dev2-9b-gates-2026-09-29.md`): vs adopted Lux1 16K (`eval/m1/d1-lux1-autotune-cache`,
     65.808) `ci95.low` > 0; `H.delta.high` ≥ 0 vs adopted Lux1 and vs Nimble v2 (`eval/m2/q6-nimble2`); types `OK`;
  6. no overlap exposure (above; UM5 inherits it);
  7. JevBench: public 231 (easy / standard / hard) reported, not selected on — unless the eval track's JevBench decision is in
     COORDINATION when the successor is chosen, in which case it applies as written there (COORDINATION is re-read first).
- **Choice among passers:** the passer with the highest v3 `ci95.low` vs I (ties: priority) is handed to the coordinator at
  once as the successor (release handoff: checkpoint + `SHA256SUMS`, calibration, scored run, T = 1 derivation, gate files).
  Each remaining passer is then re-tested against the new successor's scored run (CPU) and handed off only if it passes
  again.
- **Shipped calibration** per the 23:15 rule on the finalist's development readouts (`m5/ship_cal.sh`): CAL698 only if it
  worsens neither ECE nor Brier on typed DEV and CSS pilot, else T = 1 (`m5/derive_t1.sh`, the release's retemper-undo;
  answers unchanged).
- Also reported: T / H axes, per type, per-task CSS15 vs I and vs Lux1, Score level usage (`lux9b.score_levels`: histogram,
  per-level recall incl. level 0), public 231, typed Brier / ECE, mlx-diag (type macro incl. the NC part, en / non-en, card
  lines, per language).

## Schedule, budget and stop rules

- Chains (each step via `m5/chain-step.sh`, launched with `m5/launch.sh` after `upload_chain.sh` verifies size and SHA-256;
  liveness by PID / container name):
  - GPU6: `ref-ka13`, `KD-s1` (preflights), `KD-s2`, `KD-soup`, `KD-a13`, `KD-a12`, `KD-a23`.
  - GPU7: `ref-lux`, `KG-s1` (preflights), `KG-s2`, `KG-soup`, `KG-a13`, `KG-a12`, `KG-a23`.
  - Then `UM5-a1`, `-a13`, `-a12`, `-a23` on whichever GPU frees, the rules (CPU), locks and formal runs.
  - If an arm stops, its GPU runs nothing of that arm; it may take the other arm's `-s2` once that arm's preflight has passed.
  - GPU7 steps wait at step boundaries while the eval track's C1 event 3 lease entry is active.
- **Budget:** projected ≈ 15.5 GPU-h (≈ 3.4 h per 84M-token seed at the M4 rate, preflights ≈ 0.4, ≈ 14 readouts ≈ 0.9, three
  formal runs ≈ 0.6). **Hard cap 24 GPU-h**; a step whose projected completion would exceed it does not start (budget only,
  never results). **Per-attempt caps** (wall-clock × GPUs): a training seed including its in-arm SELECT / CAL / dev readouts
  4.6; a preflight trio 0.5; a readout (incl. its soup) 0.3; a formal run 0.5 — an attempt past its cap is stopped at the next
  check and recorded.
- Preflights (zero-step, one update, parity + bitwise reload) must PASS or the arm stops, without retry. A reproduced ROCm
  fault, OOM or nonfinite loss stops that arm. No failed arm is rerun to fill GPUs.
- GPU-hours = wall-clock × GPUs including preflights, readouts and formal runs; data builds and soups are CPU.
- Storage: every artifact stays on node A (`m5/NAME-build/soup`, `m5/NAME-cal/`, `SHA256SUMS` re-hashed once); nothing is
  uploaded to Hugging Face during the milestone.
