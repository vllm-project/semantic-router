# Decision 2.0 attributable experiment matrix, v1 (2026-09-28)

Owner: research & data track. Consumers: 0.6B encoder, 0.8B–4B decoder and 9B/CLM
tracks. This is a **plan**, not a result. Every number tagged **[M]** is our own
same-panel or development measurement from the linked record at the pinned handoff
commit `be472b957`; every number tagged **[A]** is an author or leaderboard claim that
we have not reproduced. Development readouts (SELECT, CAL, typed DEV, CSS pilot,
arm held-out slices) are never release scores. JevArena v3 is post-key: report it as
"post-key same-panel".

Data arms referenced below (A0–A6, R0–R2, A0p) are defined, hashed and published in
the data-arm registry `arms-v1-2026-09-28.md` next to this file. Until an arm has a
frozen hash there, a contrast that needs it must not start.

Revision v1.1 (same day): adds D4b (staged hard negatives), the D7 replay-dose and
teacher-quality notes, D11 (evidence-removal abstention), and records that Jev
outputs cannot be training targets (TypeSafe MCA §2.3(b)). v1 rows are unchanged.

## 1. Shared protocol (binds every row of the matrix)

1. **One factor per contrast.** Same start repo@revision, native adapter, runtime,
   seed policy, optimizer, LR schedule, max length, update count, batch token budget,
   checkpoint schedule, selection rule and calibration rule; only the factor changes.
   Commit a preregistration (start revision, arm hashes, token/step budget, control,
   selection and stop rule) before launch, then run the load/parity, zero-step and
   one-step+reload preflights. A failed preflight is recorded and the arm stops.
2. **Budget templates.** Tokens are counted with the testbed's own tokenizer and
   native `encode` (including option segments), raw and padded.
   - **S (default, fixed budget).** Control `C(ρ) = A0 ∪ A0-resample(ρ)`; treatment
     `A0 ∪ X(ρ)`. `ρ = min(0.5 × A0 tokens, tokens of arm X)`. A0-resample is a
     whole-group resample of A0 stratified by source × task type × language with a
     fixed seed. One `C(ρ)` per testbed and ρ is shared by every arm with that ρ.
   - **R (fixed A0 budget, alternative when compute is tight).** Replace a
     token-matched stratified slice of A0's *replaceable programmatic* groups
     (stage4 / targeted / pilot families) by X; all human rows and all Score rows of
     A0 stay. Report which template was used; S and R answer different questions.
   - **Scale-up (addition).** `A0 ∪ X` at full arm size, only after S passes.
3. **Selection and calibration.** Fixed checkpoint schedule (8 evenly spaced points);
   select by SELECT700 family-macro (ties → earlier). Fit per-type temperatures
   (Choice, Noul, Score) on CAL700 only and apply them unchanged everywhere.
   Arm held-out slices are read once, at the selected checkpoint.
4. **Readouts.**
   - Dev proxies (decide factors): SELECT700 family-macro; typed DEV 1,600
     (`T_dev` and Choice/Noul/Score separately); CSS pilot 1,430 mean macro-F1
     (`H_pilot`); `P_dev = 100·sqrt(T_dev·H_pilot)`; each arm's group-disjoint
     held-out slice (`AHO-X`, never trained on, from the arm registry); calibration
     on CAL and typed DEV (Brier, ECE10, Score RPS and expected-level MAE).
   - Primary (finalists only, through the eval track's frozen runners): post-key v3
     composite `100·sqrt(T·H)`, `T` typed family-macro, `H` median CSS15 macro-F1,
     plus `H\flute` (median over the 14 tasks without TRAIN exposure; A0 contains 120
     FLUTE TRAIN rows), per-type Choice/Noul/Score, Score expected-level MAE/RPS,
     per-type Brier/ECE, long-input and multilingual slices, paired 95% CIs, true
     loaded parameters. JevBench public 231 reported separately as easy/standard/hard.
5. **Statistics.** Paired group bootstrap (10,000 resamples over item groups) for
   treatment − control. Each testbed first runs **N0** (below) to measure seed noise
   `σ_seed`; an effect counts only when the 95% lower bound is > 0 and
   `|Δ| > 2σ_seed`.
6. **Retention floors (every factor).** Versus its control: no typed-DEV decision type
   falls by more than 3.0 points; `H_pilot` falls by at most 1.5 points; CAL Brier
   worsens by at most 0.010; invalid/over-budget outputs do not increase.
7. **Stop rule (default).** A factor stops when its targeted dev metric fails the
   effect rule in *both* T1 testbeds, or a retention floor fails in either. A stopped
   factor is not rerun with tweaks under the same ID; a new version needs a new
   hypothesis and preregistration. Record GPU-hours (wall-clock × GPUs, including
   preflights and evaluation) for every run, including failures.
8. **Scale-up criterion (default).** Pass in at least one T1 testbed with retention
   intact in the other → one T2 run (2B and/or 4B) with the same arm hash and the
   same relative ρ. Pass at T2 → eligible for a combined mixture (D10) and one
   post-key v3 + public231 readout. The 9B/CLM study (T3) consumes passing arms as
   fixed inputs; it does not decide data factors.

## 2. Testbed tiers

| Tier | Testbed | Start candidates | Notes |
| --- | --- | --- | --- |
| T0 | CPU | — | Arm audits, shortcut baselines, overlap scans (data track). |
| T1a | 0.6B encoder | own Kai-0.6B / Lex-0.6B continuation; new light encoder | Kai's 1,024-token cap: long-evidence arm A3 is measured on T1b, not T1a. |
| T1b | 0.8B decoder | own Eos-0.8B; official Qwen3.5-0.8B-Base | Cheapest decoder; carries A3. |
| T2 | 2B / 4B decoder | own Sol-2B / Nox-4B; official Qwen3.5 Base / general | Confirmation only. |
| T3 | 9B + CLM | official Qwen3.5-9B (frozen representations), own Lux-9B | Readout/objective study (M1–M3) on fixed passing data. |

Measured cost anchor **[M]**: one 466-update 0.6B run on A0 used about 0.31 GPU-hour.

## 3. Factors

Each block: hypothesis · evidence · minimal controlled contrast · primary metric and
dev proxy · stop rule · testbed · scale-up. Defaults from §1 apply unless overridden.

### N0 — seed-noise floor (run first on every testbed)

- **Hypothesis:** none; measures `σ_seed` so later deltas are interpretable.
- **Contrast:** `C(ρ)` with two data-order seeds, everything else identical.
- **Metric:** SD of every dev proxy across seeds (with the paired-bootstrap CI).
- **Testbed:** T1a, T1b (once each). No stop rule.

### D0 — FLUTE exposure disclosure (no training)

- **Evidence [M]:** A0 contains 120 FLUTE official-TRAIN rows; FLUTE is one of the 15
  CSS transfer tasks, so that task is same-task supervised.
- **Rule:** always report `H` and `H\flute`. New bases may use `A0s` (A0 minus FLUTE,
  7,335 rows, registry) when a strict source-isolated control is needed; A0 stays the
  default so that prior runs remain comparable.

### D1 — cross-domain original human labels (arm A1)

- **Hypothesis:** new human-labelled domains raise human transfer `H` more than the
  same tokens of A0.
- **Evidence:** [M] rights-clean v2 added 2,800 GoEmotions human rows and the 0.6B
  official-base package's `H` was 0.480 vs Kai1 0.357, but init, data and tokens all
  changed together; [A] Decider's stage 1 adds a 26-source expansion (8% of 742M
  tokens); [A] the social-science study reports Jev behind the best LLM on 14/15
  human-labelled tasks.
- **Contrast:** template S, X = A1.
- **Metric:** `H_pilot`; `AHO-A1`; finalists `H`, `H\flute`, per-task F1.
- **Stop:** `ΔH_pilot` fails the effect rule in both T1 testbeds.
- **Testbed:** T1a, T1b → T2.

### D2 — programmatically verifiable rules and counterfactuals (arm A2)

- **Hypothesis:** counterfactual-balanced, verifiable families on documented Jev weak
  skills (dates, counting, multi-hop lookup, injection-robust reading, ordering,
  unit comparison), mechanism-disjoint from the typed DEV families
  (attribute_gate, rule_precedence, set_reconciliation, transition_table) and FINAL
  families (constraint_competition, exception_stack, evidence_join,
  resource_ledger), raise typed transfer `T` to unseen mechanisms.
- **Evidence:** [A] Kev hard-v1 +26.3 pp in-family but +0.3 pp on its locked OOD panel
  (interval includes zero); [A] This-That-Model: extra seen synthetic families did not
  transfer to unseen families; [A] Jev jaggedness guide lists those weak skills;
  [M] policy-conflict v1/v2 failed state-removal gates (Noul 67.19%, Choice 57.81% vs
  50%), so A2 uses byte-identical question/options within each counterfactual group
  with balanced gold.
- **Contrast:** template S, X = A2.
- **Metric:** `T_dev` (mechanism-disjoint transfer) *and* `AHO-A2` (in-family),
  reported separately. In-family gain without `T_dev` gain is a negative result.
- **Stop:** `ΔT_dev` fails the effect rule in both T1 testbeds.
- **Testbed:** T1a, T1b → T2.

### D3 — real long-text evidence (arm A3)

- **Hypothesis:** human-labelled decisions over long, multi-paragraph evidence
  (answerable/unanswerable contrast built in) improve long-input decisions without
  hurting short ones.
- **Evidence:** [A] Kev CFPB long narratives (same-source test, AI-adjudicated
  labels); [A] Decider 11,356 teacher-written document questions; [M] QuALITY 7/24 and
  RACE 6/24 on answer-blind evidence-necessity review, ContractNLI hypothesis-identity
  baseline 69.59% vs 52.32% majority (and it is a Decision Bench source) — so A3 uses
  sources with a built-in contrast; [M] the own-Kai continuation's 1,024-token cap
  left 404/6,547 CSS15 answers invalid.
- **Contrast:** template S, X = A3, **token-matched** (long rows are expensive).
- **Metric:** `AHO-A3`; CSS pilot long subset (state > 800 chars, 169 items); finalists:
  CSS15 long-state tasks and public231 hard tier; latency per request.
- **Stop:** no effect on `AHO-A3` and on the CSS pilot long subset.
- **Testbed:** T1b → T2 (not T1a).

### D4 — hard negatives (arm A4 = A4h vs A4r)

- **Hypothesis:** semantically near distractors (Choice) and minimal-pair negatives
  (Noul) improve discrimination and calibration more than random distractors.
- **Evidence:** [M] 0.6B typed Choice 109/800 vs Kai1 277/800 and only 69/400 jointly
  correct original/reordered Choice pairs; [A] CLM trains with hard negatives
  (see synthesis); [A] Kev policy minimal pairs.
- **Contrast (cleanest in the matrix):** `A0 ∪ A4h` vs `A0 ∪ A4r`. A4h and A4r share
  every state, gold option text, option count and gold position; only the distractor
  selection differs (mined near vs uniform random from the same label space).
- **Metric:** typed DEV Choice; `AHO-A4`; option-order flip rate; Choice Brier.
- **Stop:** Choice effect rule fails in both T1 testbeds.
- **Testbed:** T1a, T1b → T2; T3 reuses A4 pools for M3.
- **D4b (staging, only if D4 passes):** same A4h rows as a short final stage (last
  20% of updates) versus mixed from the start, same total budget. [A] CLM reports
  69.2% hard-negative top-1 for a late stage vs a 62.4% peak when mixed from the start.

### D5 — multilingual native tasks (arm A5)

- **Hypothesis:** natively authored non-English decisions improve non-English
  decisions without hurting English (A0 is English 6,085 / Chinese 1,370 rows).
- **Evidence:** [M] prior MASSIVE locale pilots and multilingual-hard r5–r7 records
  (multilingual development panels; MASSIVE itself is held); [A] Jev documentation
  says English is its strongest language.
- **Contrast:** template S, X = A5.
- **Metric:** multilingual development panels (jev-arena-multilingual dev v1/v2,
  multilingual-hard r6/r7) per language; `AHO-A5`; English retention on typed DEV.
- **Stop:** no per-language effect, or English retention floor fails.
- **Testbed:** T1b (multilingual tokenizer) and T1a if its tokenizer covers the
  languages → T2.

### D6 — Score at all levels (arm A6)

- **Hypothesis:** Score fails because supervision is scarce and skewed, not because
  of the head: A0 has 516 Score rows, levels 3–8 only, 102 three-level rows (1.78% of
  tokens). Balanced level counts 2–10 and balanced grades raise typed Score and
  Score calibration.
- **Evidence:** [M] 0.6B development predicted level 0 for 400/400 DEV Score items;
  v3 Score 80/400 vs Kai1 98/400; the RPS objective alone reached 276/700 SELECT
  (Score slice 32→38/90) — objective without data did not help; [M] v6 English pilot
  +11/192 (gate +12) with the middle level 20→43 but the top level 52→41; [M] 9B
  Score-cardinality residual raised Score to 237/400 but lost human transfer;
  [A] Jev guide lists score interpolation as a weak case.
- **Contrast:** template S, X = A6.
- **Metric:** typed DEV Score accuracy, Score RPS and expected-level MAE;
  `AHO-A6` per level count and per grade (interior grades separately); finalists:
  v3 Score slots and per-level-count breakdown.
- **Stop:** Score effect rule fails in both T1 testbeds, or any endpoint grade falls
  by more than 5 points on `AHO-A6` while interior grades rise (the v6 pattern).
- **Testbed:** T1a, T1b → T2 (priority arm: Score is the recurring failure).

### D7 — replay: R0 none / R1 hard labels / R2 soft distributions

- **Hypothesis:** when continuing own 1.0 weights on new arms, soft-distribution replay
  (KL to own 1.0) keeps 1.0 capability and calibration better than hard replay or none.
- **Evidence:** [A] Decider v2.1: hard replay sharpened probabilities; KL to v1 soft
  probabilities plus per-type temperature partly restored them, yet v2.1 stayed below v1
  on its regression held-out (0.784 vs 0.788); [M] Eos own-source soft replay (KL 0.5,
  512 repeated TRAIN rows) failed its dev gate; Sol soft replay stopped at the BF16
  zero-step parity gate; own-Kai TRAIN-only teacher screen was a negative feasibility
  result.
- **Contrast:** same own-1.0 start and same new data; a fixed 25% of the token budget
  is the replay share: R0 fills it with more new data, R1 with the frozen replay prompt
  set RP-v1 and gold labels, R2 with RP-v1 and own-1.0 teacher distributions (KL,
  temperature 1). Only the replay target changes between R1 and R2. Two variants,
  declared in the preregistration: **D7a** new data = A0 ∪ passing arm (RP-v1 prompts
  also occur in A0); **D7b** (Decider-like, preferred for own-1.0 continuation) new
  data = passing arms only, RP-v1 (an A0 slice) is the only old-distribution data.
- **Metric:** typed DEV per-type retention vs start; CAL/typed DEV Brier and ECE;
  `H_pilot`.
- **Stop:** R2 not better than R1 on calibration with retention intact in both T1
  testbeds → keep R1 or R0 as default.
- **Preflight:** teacher targets must reproduce the teacher's own native output on
  RP-v1 (probability parity) before any student step.
- **Testbed:** T1a (Kai/Lex), T1b (Eos) → T2 (Sol, Nox); T3 Lux only if T1/T2 pass.
- **Teacher quality [M] (RP-v1, in-distribution):** Kai/Lex/Eos Score targets are
  near-uniform (normalized entropy 0.86–1.0), so at T1 R2 can only test Choice/Noul
  retention; exclude or separately weight Score replay rows there and say so. Sol/Nox/Lux
  targets are strong on all three types.
- **Dose (after R1 vs R2 is decided):** replay share 10% / 25% / 40% of tokens with the
  winning target type ([A] CLM uses 40%; the Jev-dataroom Japanese replication lost old
  tasks and calibration with none).

### D8 — source diversity at fixed tokens

- **Hypothesis:** at fixed human-label tokens, more distinct sources transfer better
  than more rows from fewer sources.
- **Contrast:** `A0 ∪ A1-diverse(ρ)` vs `A0 ∪ A1-concentrated(ρ)` (same tokens, same
  type mix; concentrated = the two largest A1 sources). Built from A1 only.
- **Metric:** `H_pilot`, then `H`. **Stop:** no `H_pilot` effect in both T1 testbeds.
- **Testbed:** T1b → T2. Run only after D1 passes.

### D9 — option-order augmentation (arm A0p)

- **Hypothesis:** permuted-option copies of A0 Choice/Score rows reduce order and
  option-name sensitivity and raise Choice accuracy.
- **Evidence:** [M] 69/400 jointly correct original/reordered Choice pairs at 0.6B;
  [A] Lux1 card reports 8.33% option-order semantic flips vs Jev 0% on 36 pairs;
  [A] option/rubric reversal study (yes/no AUC 0.938 → 0.232 for one peer).
- **Contrast:** template S, X = A0p (Choice rows re-rendered with a fixed permutation;
  Score rows keep level order — ordinal order is meaningful).
- **Metric:** typed DEV Choice, flip rate on paired reorderings. **Testbed:** T1a, T1b.

### D10 — combined mixture (interaction check)

- Run once, after at least two data factors pass at T2: `A0 ∪ {passing arms}` at a
  frozen mix versus the best single-arm treatment, same budget. Report per-factor
  deltas only from D1–D9, never from D10.

### D11 — evidence-removal abstention (data in arms v2)

- **Hypothesis:** counterfactual copies whose operative evidence is removed, with an
  explicit "cannot be determined" option (Choice) or a false/"not established" target
  (Noul), teach small models to lower confidence when evidence is missing.
- **Evidence:** [A] Lux1 card: evidence-removal confidence drop Kev-9B 53 points, Jev 30,
  Lux 27, Sol 6, Kai 0.6; [A] third-party CLM probes: abstain options chosen < 1.7% on
  ambiguous cases.
- **Contrast:** template S, X = evidence-removed twins of A2/A3 items vs the same
  items without removal. **Metric:** evidence-removal confidence drop, abstention
  calibration, typed DEV retention. **Testbed:** T1a, T1b.

### Readout and objective factors (owned by the size tracks; data dependencies here)

| ID | Factor (control → treatment) | Evidence | Data | Testbed |
| --- | --- | --- | --- | --- |
| M1 | Readout: shared candidate head → Kai-style typed encoder paths / light candidate-readout encoder (0.6B); → CLM state/action dual projection, raw-embedding similarity (9B) | [M] type-separated head 553/700 vs 562/700 and candidate-interaction 333/700 at 0.6B (HOLD); [A] CLM card claims | A0 first, then A0 ∪ passing arms | T1a, T3 |
| M2 | Objective: candidate CE → CE + bidirectional InfoNCE (state↔candidate) | [A] CLM | same | T1b, T3 |
| M3 | Negatives in the contrastive term: in-batch random → mined hard (A4 pools) | [A] CLM | A4 pools | T3 |
| M4 | Calibration: CE → CE + Brier; per-type temperature always fitted on CAL | [A] This-That-Model, proper-scoring papers; [M] 0.8B temperature-transport diagnostic | A0 | T1a, T1b |
| M5 | Score objective: CE → CE + RPS / cumulative-link head, **only on A0 ∪ A6** | [M] RPS on A0 alone failed (276/700) | A6 | T1a, T1b |
| M6 | Distillation from third-party decision teachers (teacher only, never weights) | [M] AutoJev KL 510/700 vs 562/700; Kev v3 teacher missed gates | A0 | deprioritized; Jev excluded (MCA §2.3(b) forbids training on its outputs); other teachers only if their licences allow |
| M7 | Initialization: own 1.0 → official Base/general | [M] measured at 0.6B–27B (see release status) | — | do not repeat on A0; re-test only with the best mixture |

CLM-specific rule (from the brief): relative candidate similarity is not an absolute
Score probability. M1/M2 at 9B must implement and separately validate native Choice,
Noul and Score readouts (e.g. per-level anchored logits plus CAL temperature).

## 4. Sequencing

- **Wave 0 (A0 only, now):** N0 on T1a/T1b; M4; M1 at 0.6B; M1–M2 at 9B on A0. D7 R1
  can start once RP-v1 is frozen; R2 needs teacher targets (registry records which
  tiers are produced by the data track and which are handed to the size tracks).
- **Wave 1 (first arms):** D6 (A6), D2 (A2), D4 (A4h/A4r), D9 (A0p).
- **Wave 2:** D1 (A1), D3 (A3), D5 (A5), D8; then D10.

## 5. Do not repeat (closed negatives at the pinned commit)

0.6B fixed gradient projection (547/700); 0.6B type-separated head (553/700) and
candidate-interaction head (333/700) at the frozen budget; Score RPS on A0 alone;
AutoJev KL 0.6B; own-Kai TRAIN-only teacher on A0; Eos own-source soft replay (KL 0.5,
512 repeated rows); policy-conflict v1/v2 generators; Score curriculum v1–v7 and the
v8 pilots; QuALITY, RACE, LogiQA 2.0, SNLI/OCNLI, VitaminC, TREC DL 2023, HelpSteer2,
FEVEROUS, ShARC, Evidence Inference, ANLI, PeerRead, F1000RD, ConTRoL source arms as
screened. A new version of any of these needs a distinct, preregistered hypothesis.
