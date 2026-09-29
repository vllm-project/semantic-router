# 9B Milestone 6: AutoJev-27B soft targets on the human-rated rows of the K recipe, an HS1 substitution arm and a five-seed K soup, with a Noul floor and an early stop (preregistration)

Status: frozen 2026-09-30 before any Milestone 6 GPU job. The CPU builds, the overlap screen and the AutoJev target guard
below ran first. Goal: a **successor** to the released `llm-semantic-router/DEV2.0-9B@ae6831960dd1…` (K-a13 = ⅓·K soup +
⅔·Lux 1.0, T = 1, 16K), tested under the successor rule (COORDINATION 16:05 item 3; item 7 as of 17:15; item 8 as of
23:40 / 00:30) against the released T = 1 run `/data/dev2/runs/release/dev2-8b-t1-derived` (post-key same-panel v3 67.737,
T .8106, H .5660). Development readouts are never release scores; v3 is post-key. Nothing goes to Hugging Face.

## Evidence and levers

- **Required effect** (M5 power check): a paired lower bound above 0 needs about +2 v3 over K-a13, for example typed T +.05
  at equal H, or H +.034 at equal T. M5's A7 dose raised typed-DEV Score and collapsed Noul `rule_precedence`; own-Lux KL on
  the dose rows kept H3 higher.
- **Main lever: teacher soft targets on the human-rated rows** (coordinator 01:20). Human transfer is where the +2 has to
  come from, and AutoJev-27B (Apache-2.0) is ahead of K-a13 on it (node-B CSS15 H .5867 vs .5660). No 9B arm has put a
  stronger teacher on the human-judgment rows alone: M3 arm B used AutoJev on every recipe row, including generated typed
  rows, and constraint competition collapsed (−.133). Here the typed rows keep their own-Lux targets and every row keeps
  its gold CE + Brier terms. The matched control is the K recipe itself: the same file, seeds and tokens, with own-Lux
  targets on the same rows.
- **Optional levers chosen, each with a matched control:**
  - **HS1** (round-2 data, 00:25 note): all of F3 (`hs1_unmet_condition`, "condition not met" negatives with matched yes
    twins) plus half of F1 (`hs1_quote_check`), by whole groups. It is substituted for an equal-token, stratified slice of
    x60, so K is its matched-token control. Noul is protected by the floor below and by KH's early-stop screen.
  - **A larger K soup:** M4's three K seeds plus the control's two new seeds give a five-seed soup, to tighten variance at
    α ½. It needs no extra training. Its control is M4's three-seed K line.
- **PN1 is not used.** A PN1-trained successor could not be released or C1-guarded inside this milestone:
  - PN1's audits (embedding scan, H7 / H8 isolation, blind label review) are still open.
  - PN1 is built from Tatoeba sentences, and the widened C1 v1.2 rescan matched C1 items to raw Tatoeba text (494
    hits). A PN1-trained model would be class-(a) trained data under the custodian's rule and needs the content
    recheck the PN1 record asks for (22:50 note).
  - Its target, non-English Noul, is guarded by successor item 4 (mlx-diag). It is not a v3 lever.
- **Noul protection:** the development rule gains a `rule_precedence` floor (below), stricter than M5's Noul type floor.

## Fixed configuration (every training arm; as M4's K and M5)

| Field | Value |
| --- | --- |
| Start | own `llm-semantic-router/Decision-1.0-Lux-9B@bd45a30a…` (node A `/data/decision20-20260926/models/Decision-1.0-Lux-9B`), 7,940,895,744 parameters |
| Trainer | `v2/dec/train_dec.py`, byte-identical to the M4 training mirror `9d90212dd` (the only trainer-side difference is the default-off `--score-bias` path in `training/model/infer.py`). Full fine-tuning with every M4 hyper-parameter: backbone lr 1e-5, head lr 1e-4, wd .01, warmup .05, one epoch, token micro-batches ≤ 32,768 tokens / ≤ 64 rows, ≥ 64 rows per update, max length 8,192, even8 checkpoints, SELECT700 `matrix-v1` |
| Objective | CE + 0.5·Brier on gold for every row, + 1.0·KL(teacher ‖ student) on every row with a teacher record |
| Runtime | image `sha256:f83b1d10…` + FLA 0.5.2; node A **GPU6–7** only; M6 training autotune cache = one copy of `m4/triton-cache` (as M5) |
| Readouts | `v2.dec.calibrate_dec` (arms) or `calibrate_ckpt` (soups, interpolations, references), then `infer_dec`, from the incumbent's runner mirror `3277dec9d`, CAL698 per-type temperatures, typed DEV 1,600 + CSS pilot 1,430 at 16,384 tokens |
| Code | `3c1e1a48f` (`lux9b/m6_data.py`, `lux9b/m6_rules.py`, wrappers and chains `lux9b/m6/`, specs `lux9b/specs/m6-*.json`; the 9B test suite passes, 80 tests with 13 skipped) |

## Data (node A `/data/dev2/runs/9b/m6/data/`; all builder guards passed)

| TRAIN | Build (spec sha256) | Rows / native tokens | train.jsonl | teacher.jsonl |
| --- | --- | --- | --- | --- |
| **K** (control) | M4 `m4-k-xl-r2-60m`, used in place | 122,651 / 60,183,732 | `a66131b1…` | `cdcd99c1…` (own-Lux, every row) |
| **KA** | `m6-ka` (`m6-ka-x60-ajS.json` `44f825c2…`) | the same file | `a66131b1…` (copied, verified) | AutoJev-27B on S, own-Lux elsewhere; hash fixed by the procedure below, recorded before training |
| **KH** | `m6-kh` (`m6-kh-x60-hs1.json` `fa1e7891…`; manifest `c8cabcc0…`) | 121,866 / 60,359,187 | `cb15dca2a6e6992c34c6b71bd56dce94b44a063b711f5c7fc31d1830226d769c` | `1f6d0c7d2a253decf38381a8726a0a54b80fe62cce8368e161705f759963fff9` (own-Lux on the 107,894 kept x60 rows; the block has none, `--teacher-partial`) |

- **S, the human-rated rows** (`lux9b.m6_data split`; manifest `a538817d…`, `S.ids.txt` `4fa9063b…`). S holds rows whose gold is
  a human rating or judgment of the text, by a frozen pool / source rule:
  - A7q (OASST1 ratings), A7k (JSTS / KLUE-STS), A7s (AfriSenti / SentiMix), V1:A6h (ArgQ, STS, SAF), V1:A1 (ABCD, SGD);
  - V1:A5 JNLI and KLUE-YNAT; H6 ArgQ, JSTS, KLUE-STS, MLQE-PE and OneStopEnglish; H8 SentiMix-Spanglish;
  - H1 DBpedia-14, MultiWOZ, Taskmaster-2 and SciTail; H5 MTOP;
  - A0s-strict GoEmotions, SNLI and the stage-3 replays that carry their upstream human label.

  Every other row keeps its own-Lux target: the A7 Stage curricula, the project-generated verifiable / programmatic rows,
  and the human-annotated QA / evidence rows (answer facts, not ratings).
  - Size: 30,792 rows / 8,768,575 native tokens (14.6% of x60); Choice / Noul / Score 4,944 / 5,256 / 20,592.
  - Production AutoJev targets cover 9,239 S rows: `aj-a0s-strict` `b88b52e9…` 1,107, AJ-M `ba52dd86…` 6,004 and AJ-SL
    `1254cb47…` 2,128, with input hashes and option keys checked.
  - The other **21,553** rows form the M6 wave: `wave.prompts.jsonl` `728d3c70bb123eb0e583042e5f3a3b4005c6fab1d236cd628c7de4abe70545a4`,
    rows `ff975f38…`. No S row is longer than the longest production prompt (19,244 characters), so none was excluded.
- **The AutoJev wave** (`lux9b/m6/aj_wave.sh`) is the production procedure of research & data's M3a with the identical,
  qualified collector:
  - model `denis-pplx/autojev-27b@6f5b557e…`, the same image, source and invocation;
  - one copy of the frozen 11-entry autotune cache `cff772eb…`;
  - the target guard (pre-run on CPU: PASS, 0 shared ids, inputs or prompt digests with SELECT / CAL / CAL698, 26 AHO /
    SHO files and the 12 gold-free panel prompt files);
  - two shards on GPU6 / GPU7, a 128-prompt production repeat check, then `v2.data.m3.teacher_targets` (qualification
    `07da814c…`; the autotune digest must be unchanged; per-row attestation).

  Provenance: `PROVENANCE.md` `e27f8e1d…`. AutoJev's public training reportedly used closed-model-generated SFT data; this
  is disclosed on any card, and own-Lux stays the clean control. The targets add ids, input hashes and probabilities only,
  no text. **KA's teacher file** = AutoJev on S, own-Lux (`cdcd99c1…`) elsewhere (`lux9b.m6_data ka`). If the wave or its
  conversion fails, KA stops and is not retried.
- **HS1 block:**
  - `m4/hs1/train.jsonl@171e6f0c` `c90ef316…`: every F3 row (10,392 / 4,495,374 tokens), plus F1 groups in
    `sha256("20260930:m6-kh:f1:" + group)` order up to half of F1's tokens (1,790 of 3,589 groups, 3,580 rows,
    2,798,169 tokens).
  - Total 13,972 rows / 7,293,543 tokens (Choice / Noul / Score 3,570 / 8,924 / 1,478; longest 2,558 tokens).
  - It replaces 14,757 x60 rows cut in whole groups, stratified by pool × source × type × language (seed
    `20260930:m6-kh:keep`), leaving 107,894 rows / 53,065,644 tokens.
  - KH's type shares are .316 / .469 / .215, against x60's .318 / .455 / .227.
- **Guards:**
  - Isolation from SELECT700, CAL698 and the HS1 dev split: 0 shared ids, groups or inputs.
  - C1 name-key registry `db00d4fe…` (38 keys) plus MASSIVE / PAWS-X / XNLI: 0 denied. HS1 has no external source.
  - Overlap exposure (`v2.eval.overlap_effects exposure`, K-a13 groups `2194716a…`): KH `656f5456…` 121,866 rows and x60
    `14e7c0ca…` 122,651 rows, `groups: []` and `methods_agree: true` for both. HS1's own A4 overlap audits passed.
  - No typed-FINAL-family generator draws and no panel items (16:40 rule).

## Arms, seeds and contrasts

Runs are named after M4's K-s1..s3: `-s4` = seed 3 and `-s5` = seed 4.

| Arm | TRAIN / teacher | Seeds |
| --- | --- | --- |
| **KA**: AutoJev on S | x60, KA teacher, KL 1.0 on every row | KA-s4 (with preflights), KA-s5 |
| **K**: matched own-Lux control | x60, own-Lux teacher, KL 1.0 on every row (= M4 K byte for byte) | K-s4 (with preflights), K-s5 |
| **KH**: HS1 substitution | KH, KL 1.0 on kept rows, `--teacher-partial` | KH-s4 (with preflights), KH-s5 |

- Development contrasts, report only: KA − K (the teacher on S; seed-paired and at the two-seed soups), KH − K (the block
  against matched tokens), and K5 − K3 (five-seed against three-seed K soup at equal α; K3 = M4's K line readouts).

## Early stop at the first full checkpoint

- An arm's first full checkpoint is the SELECT-chosen BEST of its first seed (`-s4`). Its point θ₁ = ⅓·BEST + ⅔·Lux 1.0 is
  read and compared with the same point of the control's first seed, K-s4 (`lux9b.m6_rules early`).
- **The arm continues only if** P(θ₁) − P(control θ₁) ≥ **+0.5** (the preregistered minimum development gain) **and** its
  protected screen is not below the control's: H3 for KA (the lever's human-transfer screen), typed-DEV
  `rule_precedence` for KH (Noul protection).
- A stopped arm runs no second seed and no line, and is reported. If KA and KH both stop, only the K lines remain.

## Lines and the development rule (never v3)

- θ(α) = α·S + (1 − α)·Lux 1.0 (FP32, repeated soup members; Lux end point `m3/pf-D-s1-zero/run/checkpoint-0000000`).
- Lines at α ∈ {⅓, ½, ⅔, 1} (1 is S itself):
  - **KA** and **KH**: each arm's two-seed artifact.
  - **K5**: soup of M4 K-s1 / K-s2 / K-s3 (checkpoints 1,624 / 1,420 / 1,424) + K-s4 + K-s5.
  - **K2**: soup of K-s4 + K-s5, α ⅓ and ½ only; report only.
- **Seed rule** (M4 / M5): an arm's artifact is the uniform soup if P(soup) ≥ its seeds' mean P, else the primary seed
  `-s4` (two seeds) or the median seed (K5; M4's seed readouts `k1` / `k2` / `k3b` plus the new ones). If the rule picks a
  seed, that line is rebuilt from it and the soup readouts are discarded.
- **Anchor R** = a fresh M6 readout of the incumbent's scored checkpoint `m4/K-a13-build/soup` (`ref-ka13`; expected to
  reproduce T .925, C/N/S 799 / 338 / 343, H3 .5622, P 73.22).
- **The rule** (`lux9b.m6_rules alpha`, exact rationals). α is **eligible** if:
  1. c_t ≥ c_t,R − 0.03·n_t for Choice, Noul and Score;
  2. H3 ≥ H3_R;
  3. every typed-DEV family ≥ F_f,R − 0.10;
  4. **new:** Noul `rule_precedence` c_RP ≥ c_RP,R − 0.01·n_RP (for R at 338 / 400, that is ≥ 334; M5's Noul type floor
     would allow 326).

  Equality is eligible. G(α) = T(α) − T(R), and G* is the largest G over eligible α. There is **no pick** if nothing is
  eligible or G* < 0.01; otherwise **α\* = the smallest eligible α with G ≥ 0.75·G\***. The pick is dropped if its
  P < P(R) − 8.

## Finalists, formal runs and the successor test

- **At most three finalists:** the non-dropped picks of KA, KH and K5, in that fixed priority (hypothesis order). A line
  with no pick, or whose arm stopped, yields none; slots are not refilled. Before any selection, the newest COORDINATION
  notes are re-read.
- Each finalist gets a short **lock record** (checkpoint and calibration sha256, rule output hashes, cache tree hash),
  committed and pushed before its formal run. Lines may lock and run as they finish (progressive updates).
- **Formal** (`lux9b/m6/formal.sh`):
  - runner mirror `3277dec9d`, node A GPU6 or GPU7, image `f83b1d10…`;
  - `formal-m6/triton-cache` = one copy of the frozen `formal-m3/triton-cache` (tree `af623300…`, refused otherwise);
  - 16,384 tokens, over-length invalid;
  - smoke, typed FINAL + CSS15 + public 231, mlx-diag;
  - seal, report and paired compares (5,000 draws);
  - gates from the runner mirror, plus item 7 from the M6 mirror.
- **Diagnostics:** `hs1-dev` for each finalist and for the incumbent (`readout.sh --goldfree-panel hs1-dev`, `hs1.sh`:
  per-family accuracy, F1 quote adoption, F3 false yes). They are never a selection criterion.
- **Successor rule**, each item against the current incumbent run I: the latest M6 successor already handed off, else the
  released T = 1 run.
  1. v3 `ci95.low` > 0 (`gates paired` vs I).
  2. `axis_ci95.H.delta.high` ≥ 0.
  3. Every `gates types` verdict `OK`.
  4. mlx-diag card-eligible Choice + Noul not significantly below: `lux9b.mlx_paired` `ci95.high` ≥ 0 vs I's mlx answers
     (`formal-m4/K-a13-16k-mlx`, identical answers).
  5. The 9B tier gates: vs adopted Lux1 16K (65.808) `ci95.low` > 0; `H.delta.high` ≥ 0 vs Lux1 and vs Nimble v2; types
     `OK`.
  6. No overlap exposure (above).
  7. `gates public231` vs I does not return REGRESSION.
  8. C1 post-key guard: I hand the eval custodian a frozen package on node A plus a C1 successor spec ("Eval runners").
     The custodian collects; I do not.
- **Choice among passers of items 1–7:** the highest v3 `ci95.low` vs I (ties by priority) is handed to the coordinator
  at once, with checkpoint + `SHA256SUMS`, calibration, scored run, T = 1 derivation, gate files, the C1 spec and the
  disclosures. Any other passer is re-tested against the new successor's scored run (CPU).
- **Shipped calibration:** the 23:15 rule (`m6/ship_cal.sh`): CAL698 only if it worsens neither ECE nor Brier on typed DEV
  and the CSS pilot, else T = 1 (`m6/derive_t1.sh`; answers unchanged).
- **Also reported:** T / H axes, per type, per-task CSS15, Score level usage (`lux9b.score_levels`), public 231 by tier,
  typed Brier / ECE, mlx-diag (type macro including the NC part, en / non-en, card lines, per language) and the `hs1-dev`
  readouts.

## Schedule, budget and stop rules

- **Chains** (each step via `m6/chain-step.sh`; each chain file uploaded and size + SHA-256 verified, then launched with
  `m6/launch.sh` in a separate step; liveness by PID and container name):
  - `m6-aj` (GPU6 + GPU7): the AutoJev wave, then the KA teacher build (CPU). The KA teacher hash goes into the state file
    and amendment 1 before training starts.
  - `m6-gpu6`: KA-s4 → its θ₁ → early rule for KA → KH-s4 → its θ₁ → early rule for KH → (if KA continues) KA-s5, KA soup,
    KA line.
  - `m6-gpu7`: `ref-ka13` → K-s4 → its θ₁ → K-s5 → K5 soup and line → K2 soup and line → (if KH continues) KH-s5, KH
    soup, KH line.
  - Then the rules (CPU), the locks and the formal runs.
- **Budget:** projected ≈ 20 GPU-h. That is AutoJev wave ≈ 0.5, six 60M-token seeds with their in-arm SELECT / CAL / dev
  readouts ≈ 16, three preflight trios ≈ 0.75, ≈ 20 development readouts ≈ 1.2, up to three formal runs with mlx-diag
  and `hs1-dev` ≈ 1.
  - **Hard cap 24 GPU-h.** A training step starts only if the GPU-hours used plus 7.1 (a first seed with preflights, plus
    one concurrent seed) or 6.6 (a later seed, plus one concurrent seed) stay within the cap. The budget decides, never
    results.
  - **Per-attempt caps** (wall-clock × GPUs): a training seed including its in-arm readouts 3.3; a preflight trio 0.5; a
    readout including its soup 0.3; the AutoJev wave 1.5; a formal run with mlx-diag 0.5. An attempt past its cap is
    stopped at the next check and recorded.
- Preflights (zero-step, one update, parity + bitwise reload) must PASS or the arm stops, without retry. A reproduced ROCm
  fault, OOM or nonfinite loss stops that arm. No failed arm is rerun to fill GPUs. If the control K fails, the
  treatment arms cannot be tested and the milestone stops and reports.
- GPU-hours = wall-clock × GPUs, including preflights, readouts and formal runs; data builds and soups are CPU.
- Storage: every artifact stays on node A (`m6/NAME-build/soup`, `m6/NAME-cal/`, `SHA256SUMS` re-hashed once). Nothing is
  uploaded to Hugging Face during the milestone.
