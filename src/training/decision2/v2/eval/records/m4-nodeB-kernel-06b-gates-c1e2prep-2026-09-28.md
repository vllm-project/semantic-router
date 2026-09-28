# Node-B re-validation on the kernel image, DEV2.0-0.6B gate checks, C1 event-2 preparation (2026-09-28)

Trigger: coordinator 21:55 (DEV2.0-0.8B approved for private release; node-B re-validation approved on node B GPU7,
shared with the research & data track; then prepare C1 event 2 and run the 0.6B gates). Plan and pre-registration:
`m4-nodeB-revalidation-plan-2026-09-28.md`.

## 1. Node-B comparator re-validation (kernel image `dbe5f32b`)

**How it ran.** Node B GPU7, lease entry `gpu7.lock/owner.eval` (the data track's `owner` entry is theirs). Image
`decision20-train-fast:latest` `dbe5f32b…` (`causal_conv1d` 1.7.0 and FLA 0.5.2 recorded in every `COLLECT.json`).
Code: exact mirrors `33d3aed76` (AutoJev) and `e9cd637b1` (the rest). Adapter modules are the same as the original
runs (hash-checked). Each model passed a 20-item gold-free smoke on typed FINAL, CSS15 and public 231 first, then ran
once. 27B models used a fresh per-run Triton cache. Lux1 used node A's frozen autotune cache: 0 autotuning events.
Predictions were copied to node A and sealed, reported and compared there. Total 1.33 GPU-h.

- **Runner fixes found here** (`e9cd637b1`):
  - The runner appended its end-of-job lines to `owner` instead of the named entry. That touched the data track's
    file twice (14:02 and 14:20). They rewrote the file at 14:27, so nothing of ours remains. Fixed.
  - The data track started its M3b Lux job on GPU7 at 14:27 (through 18:30). The idle-VRAM check then refused the
    next three smokes (first chain). The new `--shared` flag is allowed only with a named lease entry. It skips the
    idle check for an approved co-tenancy and records `shared` and VRAM% at start in `GPU-TIME.json`. Eikos, Jebadiah
    and Lux1 ran shared (26–27% VRAM held by the data track). Both jobs are inference only; outputs are deterministic
    given image and cache (see the Lux1 result), so sharing affects wall time only.

### Results (post-key same-panel v3; paired bootstrap, 5,000 draws)

| Model | New v3 (T / H) | Public 231 | Old node-B v3 (`ce895822`) | Δ new − old [95% CI] | Answers changed vs old (typed / CSS / public) |
| --- | --- | ---: | --- | --- | --- |
| **AutoJev-27B** | **72.133** (.8869 / .5867) | 201 | 72.310 / 200 | −0.18 [−0.69, +0.48] | 11 / 37 / 3 of 2,000 / 6,547 / 231 |
| Eikos-27B (BF16) | 69.290 (.8175 / .5873) | 212 | 69.201 / 212 | +0.09 [−0.50, +0.84] | 15 / 33 / 2 |
| Jebadiah-27B | 65.472 (.7419 / .5778) | 176 | 65.472 / 177 | 0.00 [−0.46, +0.94] | 23 / 28 / 1 |
| Lux1 (9B, frozen node-A cache) | 65.808 (.7762 / .5579) | 183 | 66.268 (node-B `n1`) | — | 7 / 36 / 0 |

Seals (sha256 of `SEAL.json`): AutoJev `5f403059…`, Eikos `77be665d…`, Jebadiah `3a1cf5d0…`, Lux1 `f7fd346d…`.

**Against node A:**

- **Lux1 is bit-identical to node A** (D1 and D2): 0 of 8,778 answers differ, max probability drift 0.0 on every
  panel, v3 65.808 = 65.808 (the old image differed in 43 answers, 66.268).
- **AutoJev:** typed FINAL is **identical** to the node-A run (0/2,000; the old image differed in 11). CSS differs in
  27/6,547 (old 37) and public in 2/231 (old 3). Paired new vs node A −0.18 [−0.49, +0.40].
- Eikos and Jebadiah have no node-A run.

### What the kernel explains

- **All of the Lux1 node-A vs node-B difference.** With the same image (with `causal_conv1d`) and the same frozen
  autotune cache, node B reproduces node A bit for bit. The M1/M2 attribution to hardware ("below the software
  stack") was wrong; the cause was the missing `causal_conv1d` kernel on `ce895822` (the FLA fallback path).
- **AutoJev's typed FINAL difference**, entirely. Its remaining CSS/public differences (27 + 2 slots, ≤ 0.085 drift)
  are consistent with fresh per-run Triton autotuning; the node-A run had its own cache. They do not change v3
  beyond noise.
- **Pre-registered consequence:** the same-node rule becomes a **same-image, same-autotune-cache rule**. A run on
  node B with the node-A image contents and a frozen cache is interchangeable with node A. Without a frozen cache,
  compare on one node as before.

### Decisions (pre-registered)

- **The new runs are the node-B comparators** for every later 27B comparison (disclosed replacement; the old runs
  stay as records). AutoJev-27B `m4/nodeB-kernel/autojev27` 72.133 / public 201. Eikos-27B 69.290 / 212 (still the
  best public). Jebadiah-27B 65.472 / 176.
- **27B threshold recomputed:** 90% × 72.133 = **≥ 64.9** (was 65.1 from 72.31). Disclose that the node-A AutoJev run
  scores 72.31 and that the two are a statistical tie; 64.9 vs 65.1 is inside noise.
- Artifacts: node A `/data/dev2/runs/eval/m4/nodeB-kernel/`; private eval-artifacts `m4/nodeB-kernel/` (commits
  `742bdf50` AutoJev and `3fbeaeb6` the rest), `OPERATIONS.log` included.

## 2. DEV2.0-0.6B gate checks (stored post-key v3 predictions; CPU; node A)

**Candidate.** `m4-t-a7-soup`: private staging `llm-semantic-router/dev2-release-staging-06bm4@62c61c10…`, scored
run `/data/dev2/runs/06b/m4/formal/m4-t-a7-soup` (export manifest `0f96aa39…`, `dev2-06b-causal-8k` adapter, node A
`f83b1d10`). **v3 43.541** (T .3953, H .4796).

Code: `v2/eval/gates.py` (`860b16bd0`, the same code as the 0.8B gates). Output:
`/data/dev2/runs/eval/m4/gates-06b/` and private eval-artifacts `m4/gates-06b/` (commit `8b9c4ac8`).

### (a) Human transfer vs GLiNER2.5-Decide and Bosun

Joint paired bootstrap with per-axis intervals: 5,000 draws, same node, same panels.

| Candidate vs | Δ v3 [95% CI] | Δ H (human transfer) [95% CI] | Δ T (typed) [95% CI] |
| --- | --- | --- | --- |
| GLiNER2.5-Decide (tier leader, 42.524) | +1.02 [−1.77, +7.80] | +0.039 [−0.013, +0.167] | −0.015 [−0.046, +0.016] |
| Bosun v3.1 0.6B (38.524) | +5.02 [−1.35, +7.80] | **+0.137 [+0.010, +0.192]** | **−0.038 [−0.068, −0.010]** |
| Decision 1.0 Kai (own 1.0, 35.938) | **+7.60 [+4.70, +10.76]** | **+0.123 [+0.068, +0.182]** | **+0.033 [+0.008, +0.059]** |
| Decision 1.0 Lex (31.022) | **+12.52 [+3.47, +18.12]** | **+0.214 [+0.033, +0.305]** | **+0.033 [+0.011, +0.056]** |

- **Human transfer is not significantly below either leader.** It is level with GLiNER2.5 (interval includes 0,
  point above) and significantly above Bosun.
- **Disclose:** typed accuracy is significantly below Bosun (−0.038) and level with GLiNER2.5. The v3 gain over both
  comes from human transfer.
- The candidate clearly beats its own 1.0 (Kai1): the paired lower bound is > 0 on the joint score and on both axes.
- Threshold check: 43.54 ≥ 38.3 (90% of GLiNER2.5-Decide 42.52).

### (b) No decision type collapsed (typed FINAL)

A type counts as collapsed when one answer takes ≥ 90% of it, or when the lower bound of its Wilson 95% interval is at
or below chance.

| Type | Candidate acc [95% CI] | Chance | Top share | Verdict | Kai1 | GLiNER2.5 | Bosun |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| Choice (800) | .319 [.287, .352] | .267 | 34% | OK | .346 OK | .391 OK | .456 OK |
| Noul (800) | .552 [.518, .587] | .500 | 78% | OK | .505 **COLLAPSED** (99% true) | .627 OK | .676 OK |
| Score (400) | .333 [.288, .380] | .200 | 64% | OK | .245 OK (two levels only) | .220 **COLLAPSED** (CI ≤ chance) | .207 **COLLAPSED** (CI ≤ chance) |

- **Candidate Score** uses all five levels but leans on the extremes: predictions 0/1/2/3/4 = 60/6/57/20/257; recall by
  level .54/.03/.13/.03/.77. Middle levels are weak; disclose.
- **Candidate Noul** clears chance narrowly (lower bound .518) with 78% of answers on one side; disclose as the
  weakest type. Choice is the weakest relative to peers (.319 vs GLiNER2.5 .391, Bosun .456).
- Lex (for reference): Choice .294 **COLLAPSED** (CI ≤ chance), Noul .502 **COLLAPSED** (91% false), Score .268 OK.

## 3. JevArena-C1 event 2 — prepared, NOT run (waits for the coordinator's batch confirmation)

- **Independence recheck (done, node A, CPU).** Everything landed after the event-1 check (`3a99bf1c`):
  - training-data commits `9bb9790b` (AutoJev `aj-a0s`), `780d2743` (AutoJev **`aj-sl`**) and `12912429`
    (A0s-strict with own-Lux and AutoJev targets): 12 changed files;
  - local pools: `m3a2` (`aj-m`, `aj-a0s`, `aj-sl` prompts, rows and uploads) and the new `m3b` (Lux-XL waves):
    258 files.

  Result: **no C1 source name or provenance in 9.92M rows / 141 files** (names `3be721f3…`). Content scan of all
  67,617 C1 source rows (8-gram shingles, exact ≥ 8 tokens): **0 OVERLAP, 9 REVIEW** (receipt `a1149284…`,
  hits `665bdc60…`). Six are the event-1 weak hits; the three new ones are in the `m3b` Lux-XL prompt pools with
  containment ≤ 0.21 and no exact 8-token span (same kind). **No change: no v1.2.**
- **Script.** `v2/eval/sealed/event2.sh <gpu> <mirror> [lease-name]` (committed; node A). The key is read once from
  stdin. Each model first passes a 20-item typed-FINAL smoke **before** the prompts are decrypted (the event-1 Kev
  launch failure cost a relaunch; this makes such failures free). Then: decrypt prompts (hash-checked) → collect →
  seal every model → delete prompts → decrypt gold (hash-checked) → score → paired compare. A trap removes all
  plaintext and restores the lease on any exit. Every step is logged in `ACCESS.log`.
- **Batch.** DEV2.0-0.6B `m4-t-a7-soup` (its 8K adapter spec and export `0f96aa39…`); Kai1 (`7185f514`), Lex
  (`ee8e74d9`), Bosun v3.1 0.6B (`1d8b6f96` + its base), GLiNER2.5-Decide (`7ee5da4c`, English variant, offline Hub
  cache). All on node A `f83b1d10` or the GLiNER image, as in their stored runs. Any other candidate the coordinator
  adds is one line in `args()`.
- **Needs from the coordinator:** batch confirmation and a node-A GPU (the 0.6B batch needs ~0.1 GPU-h).
- **Update 2026-09-29:** the event ran on the confirmed frozen package `DEV2.0-0.6B@e61b2b44` after a second
  independence recheck and script fixes (as prepared, every preflight would have aborted). Results:
  `m4-dev2-06b-c1-event2-2026-09-29.md`.
