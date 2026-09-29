# Decoder Milestone 6 — preregistration (successors for DEV2.0-4B / 2B / 0.8B; 2026-09-29)

Written 2026-09-29 ≈16:50 UTC+8, before any M6 GPU job, on integration `28f05ace7`. Assignment: coordinator note
2026-09-29 16:05 (user directives, optimization round 1). Development readouts are never release scores; v3 /
public 231 are post-key same-panel comparisons; `mlx-diag` is a diagnostic panel. A data lock with every SHA-256
follows this record before any training launch. Nothing goes to HF during the milestone.

## Goal and bars

Per tier, find a successor that passes the successor rule (16:05 note) against the current revision's T = 1
scored run, and report its paired CI against the best same-size peer. The milestone goal is a significant lead
over that peer (4B: Decider 4B +1.27 [−5.71, +4.31] today; 2B: Decider 2B +3.94 [−2.48, +5.67]; 0.8B: keep the
lead over Intern-0.8B). Priority: 4B, then 2B, then 0.8B.

| Tier | Current revision (T = 1) | Weights | Scored run (node A) | T = 1 binding | v3 |
| --- | --- | --- | --- | --- | ---: |
| 4B | `DEV2.0-4B@452f1332` | N4XF soup `11b5ca1c…` | `formal/m4/m4-N4XF-soup-nodeA` | `runs/release/dev2-4b-t1-derived` (REPORT `57fa23ef…`); node-B equal-answer reference `formal/m5/m5-ref-N4XF-soup` | 63.151 |
| 2B | `DEV2.0-2B@5ad3e9a3` | S2T soup `073bd1f2…` | `formal/m3/m3-S2T-soup-nodeA` | `runs/release/dev2-2b-t1-derived` (REPORT `08745acf…`) | 53.437 |
| 0.8B | `DEV2.0-0.8B@f458c34c` | E8F soup `60356482…` | `formal/m2/m2-E8F-soup-nodeA` | `runs/release/dev2-0p8b-t1-derived` (REPORT `1f5cf33d…`) | 50.236 |

Peers (post-key v3): Decider 4B 61.882, Jet v6.2 60.375; Decider 2B 49.499; Intern-0.8B 43.535.

## Evidence behind the design

1. **Mixture, 4B.** The only significant within-tier recipe gain so far is the XL r2 full mixture: N4XF vs the
   M3-mixture N4LKr +3.61 [+1.22, +12.13] (M4). **2B has never trained on it**: S2T is the M3 mixture `m3-v2m-ret`
   with own-Sol KL 0.5, which also carries 57 rows of r2-excluded groups (disclosed exposure).
2. **9B K-a13** (13:45 note): own-1.0 KL on all rows, then a point on the line back toward the comparator chosen by
   the development α rule (smallest α keeping ≥ ¾ of the best typed gain). The line kept the typed gain and damped
   the per-task human-transfer swings, which narrowed the paired CI. In M6 the comparator is the incumbent, so the
   lines here are anchored on the incumbent.
3. **Interpolation toward own 1.0** (the listed cheap lever). Each incumbent beats its own 1.0 on both axes except
   0.8B's H (E8F .440 vs Eos 1.0 .461). A point between an incumbent and its own 1.0 can only beat the incumbent
   through non-linearity along the line. It is screened cheaply (two points per tier) under the same rule.
4. **A7 typed curricula** (16:00 note, +.099 typed at 27B). These are the A7 sub-arms A7g / A7o / A7p / A7i (A7 v3,
   dataset revision `0b47239e`). All three decoder incumbents and their 1.0 starts already train on them: 4B about
   9.68M A7 tokens; 2B 9.75M `dec10-stage4v2` retention; 0.8B every A7 v2 sub-arm, about 121M tokens. For the decoders
   this is a dose change, not new data. So there is no separate A7 arm. The 2× XL dose (N6D, S6D) raises A7 to about
   19.4M tokens as part of the recipe. C1 independence: the eval recheck of `0b47239e` found no C1 source. C1 is
   exhausted after event 3, so an M6 successor makes no sealed-set claim. Licence: project-generated; A7i is CC-BY.
5. **Own-1.0 KL on all rows.**
   - 0.8B never used a teacher: E8F is CE + 0.5 Brier.
   - 4B uses own Lux 1.0, the cross-tier own 1.0, on all rows except H7 / H8, which are gold-only (6.4% of rows).
     Own-Nox targets lost typed reasoning (N5N −3.09), so 4B keeps own-Lux and extends it to H7 / H8 with the
     published `h-w1` targets.
   - 2B already has own-Sol KL 0.5 on every row. Own-Lux at 2B lowered typed DEV twice (M3 S2L, M4 S2LR).
6. **Variance.** Soup-to-soup variance at 4B is about 2.8 v3 (N4LX vs N4XF). Paired CIs resample items, not
   training runs. Points anchored on the incumbent have much narrower paired CIs against it than independent soups.
7. **Development proxy.** It does not separate close within-tier pairs: 67 of 70 sibling pairs fall inside the tie
   band. So finalists come from the α rule and a fixed priority, not from ranking P.
8. **Paraphrase-style Noul (PN1, research & data).** Generation started on node B GPU3–4 at 16:22 UTC+8; nothing
   has landed. It enters only by amendment (below).

## Training arms

Common settings:

- full fine-tuning from own 1.0, one epoch, CE + 0.5 Brier (+ KL as listed);
- the trainer defaults as in M4 / M5: max length 8,192, token batching ≤ 32,768 tokens / 64 rows, `even8`
  checkpoints, SELECT700 `matrix-v1` selection per seed;
- `drive_arm.sh` preflights: zero-step, one-step, `preflight_dec` load parity + one-step reload;
- the node's decoder image: node B `dbe5f32b…`, node A `f83b1d10…`, where M1 trained 0.8B arms on node A GPU5.

| Arm | Tier | Start | Mixture | Teacher, KL weight | LR backbone / head | Seeds | Node / GPU | Cap (GPU-h) |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| **N6D** | 4B | Nox 1.0 `cde2a68d` | `m6-xl-full-59m`: the XL r2 full recipe at 2 × 14.42% = 28.84% per pool, whole groups, same sampler as `m4-xl-full-29m` (nested if the sampler is threshold-based), about 58.8M native tokens | own Lux 1.0 on **all** rows (published XL waves + `h-w1` for H7 / H8), 1.0 | 5e-6 / 5e-5 | 20260926 / 27 / 28 | B / GPU3 | 6.5 |
| **N6A** | 4B | Nox 1.0 | `m4-xl-full-29m` (`c7d51219…`, N4XF's) | own Lux 1.0 on **all** rows (`e2ff27ce…` + `h-w1` H7 / H8), 1.0 | 5e-6 / 5e-5 | **20260929 / 30 / 31** (new, so that N4XF ∪ N6A is a 6-seed set) | B / GPU4 | 3.5 |
| **S6X** | 2B | Sol 1.0 `ce0c018a` | `m4-xl-full-29m` | own Sol 1.0 on all rows (S2T's labelling recipe, package temperature 1.30036), 0.5 | 5e-6 / 5e-5 | 20260926 / 27 / 28 | B / GPU4 | 2.0 |
| **S6D** | 2B | Sol 1.0 | `m6-xl-full-59m` | own Sol 1.0 on all rows, 0.5 | 5e-6 / 5e-5 | 20260926 / 27 / 28 | B / GPU4 | 3.5 |
| **E6K** | 0.8B | Eos 1.0 `363c4a5e` | `m6-e8f-r2clean`: E8F's `d1dc33fc…` minus every row of the 305 r2-excluded groups (33 groups today) | own Eos 1.0 on all rows, 0.5 | 1e-5 / 1e-4 (E8F's) | 20260926 / 27 / 28 | A / GPU5 | 4.5 |

- **Teacher labels:** own Sol and own Eos labels are made with the M3 own-Sol procedure (`v2.dec.teacher_label`,
  1.0 package temperature). The label cost counts against the arm's cap.
- **Arm artifact** = the uniform 3-seed FP32 soup, as in every M4 / M5 arm, where the soup was ≥ the seed mean. The
  8K postrun readouts of the seeds are reported.
- **Stop rules:**
  - A failed preflight stops that arm, with no rerun and no replacement seed.
  - An arm stops at its cap.
  - An arm with fewer than two finished seeds is dropped. With two, the soup is two-seed and this is disclosed.
  - Failed arms are never rerun to fill GPUs.

## Candidates (CPU builds; no training)

Lines are anchored on the incumbent soup `I`, the released weights: node B `m4/soup/N4XF/build/N4XF-soup` and
`m3/soup/S2T/build/S2T-soup`, and the node-A E8F soup copy verified against `batch2.sha256`. Each point is built with
`v2.dec.soup` by member repetition, so the recorded weights are rational.

- **Toward an arm artifact `A`:** `(1 − β)·I + β·A` for β ∈ {⅓, ½, ⅔, 1}, built as `[I, I, A]`, `[I, A]`,
  `[I, A, A]` and `A`. For three-seed soups, β = ½ is exactly the six-seed uniform soup.
- **Toward own 1.0 `O`** (the FP32 zero-step conversion of the start, which has the same `full_training_source`):
  `(1 − γ)·I + γ·O` for γ ∈ {⅙, ⅓}, built as `[I × 5, O]` and `[I × 2, O]`.
- **4B only, toward the N5BN soup** (M5's lead, already trained): β ∈ {⅓, ½}.

Lines per tier:

- 4B: `L-N6D`, `L-N6A`, `L-N5BN`, `L-Nox`.
- 2B: `L-S6X`, `L-S6D`, `L-Sol`.
- 0.8B: `L-E6K`, `L-Eos`.

## Development readouts (selection only)

- Typed DEV (1,600) and CSS pilot (1,430 items, 3 tasks) via `v2.dec.infer_dec` at **16,384 tokens** (the formal
  limit), on the kernel path that `require_runtime` enforces, then `v2.dec.dev_readout`:
  - T is the family macro and H3 the CSS-pilot three-task mean; H is the median and P = 100·√(T·H);
  - paired bootstraps are against `I`.
- The references `I` and `O` are re-read at 16K on the same node and image as the line points: node B for 4B / 2B,
  node A for 0.8B.
- 4B also runs MLX-DEV. It is report only, not a drop rule: M5 showed it cannot see the paraphrase failure.

## Selection rule (development only; fixed now)

The code is the 9B rule module `v2/9b/lux9b/m4_rules.py` (`alpha` subcommand, `3277dec9d`), reused unchanged with
the incumbent's readout as the reference (`--lux` = `I`) and `--h-field H_mean`. Per line:

- **Eligible** if:
  - every type keeps c_t ≥ c_t,I − 0.03·n_t;
  - H3 ≥ H3_I;
  - no family falls below F_f,I − 0.10.
- **Gain:** G(point) = T(point) − T_I on typed DEV.
- **No pick** if nothing is eligible or G\* < 0.01.
- **Otherwise** the pick is the smallest eligible step (β, or γ) with G ≥ 0.75·G\*.
- **Proxy drop:** a pick ≥ 8 P below the tier's best pick is removed.

**Finalists (≤ 3 per tier), in priority order.** A line without a pick passes its slot to the next line.

| Tier | Slot 1 | Slot 2 | Slot 3 |
| --- | --- | --- | --- |
| 4B | `L-N6D` pick | `L-N6A` pick | the better of the `L-N5BN` / `L-Nox` picks (higher G, then higher P) |
| 2B | `L-S6X` pick | `L-S6D` pick | `L-Sol` pick |
| 0.8B | `L-E6K` pick | `L-Eos` pick | — |

**PN1 amendment (the only planned amendment).** If research & data lands PN1 before N6A's seeds finish, an
amendment adds the 4B arm N6P and its line `L-N6P`, which takes 4B slot 3. PN1 counts as landed only with all of:

- a SHA-256 data lock in the private dataset;
- the preregistered validity check passed: yes(N4XF) − yes(Nox) ≥ .03 with lower bound > 0;
- a clean C1 registry check.

N6P is N5BN's recipe with PN1 TRAIN rows added to the multilingual block, three seeds, a 4.0 GPU-h cap. The
composition is fixed in the amendment before any N6P job.

## Formal runs (finalists only)

Post-key same-panel JevArena v3 (typed FINAL + CSS15) and public 231, collected at 16,384 tokens with the eval
track's frozen runner (`v2/eval/run_same_panel.sh`), then gold-free seal, report and paired CIs on node A. Every
finalist gets a 16K CAL698 fit and the 23:15 calibration rule (`v2.release.dev_calibration`). It ships T = 1 unless
the rule adopts CAL698; answers are unaffected either way.

- **4B:** the M5 procedure.
  - Node-B collection on `dbe5f32b` with a `cp -a` copy of `cache-frozen` (`f6d0f920…`), relay with manifest checks,
    node-A scoring.
  - `mlx-diag` on a copy of `cache-frozen-mlx` (`65d7d38f…`), paired against the node-B reference's `mlx-diag`.
- **2B:** node-B collection after a node-B S2T reference collection (fresh cache, then frozen). The reference must
  reproduce the node-A S2T answers exactly: 0 differing answers on typed FINAL, CSS15 and public 231. If it does not,
  2B finalists are collected on node A GPU5 with S2T's image `f83b1d10`, `HIP_FORCE_DEV_KERNARG=1` and a copy of
  S2T's persisted cache, after a hash-checked checkpoint relay. `mlx-diag` is paired on the same node.
- **0.8B:** node-A GPU5 collection with E8F's image `f83b1d10`, `HIP_FORCE_DEV_KERNARG=1` and a `cp -a` copy of
  E8F's persisted cache, the M2 procedure. `mlx-diag` is paired against `m2-E8F-soup-nodeA-mlx`.

Reported for every finalist:

- v3, T, H and per-type counts;
- the types gate and Score level usage (predicted vs gold per level);
- per-task CSS15 macro-F1 deltas (human transfer);
- public 231 (easy / standard / hard), reported, never selected on;
- `mlx-diag`: overall, the card-eligible parts, non-English per type and per language;
- paired CIs against the incumbent, the tier's own 1.0 and the peers.

## Successor rule (per finalist, against the current revision's T = 1 scored run)

1. v3 paired 95% CI lower bound > 0 (`same_panel compare`, joint bootstrap, 5,000 replicates, seed 20260927).
2. Human transfer not significantly below: `axis_ci95.H.delta.high` ≥ 0.
3. No type collapsed (`v2.eval.gates types`).
4. **`mlx-diag` overall on the card-eligible parts not significantly below.**
   - The card-eligible parts are Choice and Noul; the XNLI-based Score part is NC.
   - Statistic: the type macro over those two parts.
   - Method: a paired strata bootstrap over items within type × language, 5,000 replicates, seed 20260927.
   - Pass: 95% upper bound ≥ 0.
   - The tool is new: `v2/dec/mlx_paired.py`, with tests. The full overall, including Score, is reported.
5. The tier gates still hold:
   - 4B: lower bound > 0 vs the adopted Nox 1.0 (`m1-adopt/nox1`); v3 ≥ 55.7; H not significantly below Decider 4B
     or Jet v6.2.
   - 2B: lower bound > 0 vs Sol 1.0 16K (`formal/m3/sol1-16k`); v3 ≥ 44.5; H not significantly below Decider 2B.
   - 0.8B: lower bound > 0 vs the adopted Eos 1.0 run.
   - Every tier: no type collapsed.
6. **No overlap exposure.**
   - (a) The candidate's new training files have an empty `v2.eval.overlap_effects exposure` receipt against the r2
     rescreen's 305 groups. `m4-xl-full-29m` already has one (`52c7ccad`).
   - (b) Rules 1 and 5 still hold with the 84 flagged items removed (`overlap_effects run`).
   - A point that contains incumbent weights inherits that incumbent's disclosed exposure: 2B 24 groups / 57 rows,
     0.8B 33 groups, 4B none. That is disclosed and flagged to the coordinator. Such a point cannot move the
     successor-vs-incumbent contrast in its own favour, because the incumbent carries the same exposure at a larger
     weight.
7. JevBench (public 231): per the eval decision if it lands before selection; otherwise reported without gating.
   The newest COORDINATION notes are re-read before selection.

**Choosing among passing finalists in a tier:** the highest paired lower bound against the best same-size peer (the
milestone goal), then the higher lower bound against the incumbent, then priority. One successor per tier is handed
to the coordinator as soon as that tier's finalists are scored, with package paths and hashes. Release follows the
16:05 fast path, done by a release worker.

## Budget and GPUs

- **Cap: 36 GPU-h** (wall-clock × GPUs, including preflights, labels, readouts, co-tenant formal jobs and smokes).
- **Per-attempt caps:**

  | Attempt | Cap (GPU-h) |
  | --- | ---: |
  | N6D | 6.5 |
  | N6A | 3.5 |
  | S6X | 2.0 |
  | S6D | 3.5 |
  | E6K | 4.5 |
  | Development builds and readouts, all tiers | 3.0 |
  | References (node-B S2T, 16K reference readouts) | 1.0 |
  | Formal, ≤ 8 finalists incl. `mlx-diag`, CAL fits and smokes | 3.5 |
  | PN1 reserve | 4.0 |
  | Contingency (never for reruns) | 4.5 |

- **At 33 GPU-h cumulative:** no new training; formal work finishes within the cap.
- **GPUs:**
  - Node B GPU3–4 start after research & data's PN1 lend (≤ 1.5 GPU-h) has released its lease entries.
  - Node A GPU5 is free now.
  - Readouts and formal jobs may share a GPU with a training job as recorded co-tenants (the M5 practice) and are
    counted in full.
- **Plan:**
  - GPU3: N6D s1 → s3.
  - GPU4: Sol labels → S6X → N6A → S6D.
  - Node A GPU5: Eos labels → E6K → 0.8B lines → 0.8B formal.
  - 4B and 2B lines and formal runs go on node B as co-tenants when the arms finish.
- **Chain rule (16:00 note):**
  - Upload each script, verify its size and SHA-256, then launch it in a separate step.
  - Check liveness by container name or PID, never by a `pgrep -f` pattern.
  - Confirm the first log line.
  - Before any turn ends with an automatic chain, verify it is live.

## Code (committed before use; mirrors are exact pushed commits)

- `v2/dec/ops/m6/`: prep (mixture builds, teacher composition and labels, locks), chains, lines, readouts,
  formal and score, parameterized by tier from the M5 scripts.
- `v2/dec/mlx_paired.py`, with tests.

No shared module changes are planned. If one is needed, it lands as a separate small commit with tests and is called
out.
