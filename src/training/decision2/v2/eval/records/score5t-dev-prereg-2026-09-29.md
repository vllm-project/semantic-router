# Score5-typed-DEV v1 (`score5t-dev`) preregistration: an out-of-family 5-level typed Score development panel (2026-09-29)

Eval track, development readout only (never a release score, never in v3, charts or cards). Written and pushed before any
`score5t-dev` seed is drawn, any item is generated and any model is run on the panel.

## 1. Why

- 0.6B Milestone 7 (`v2/06b/records/m7-results-2026-09-29.md` §3–4): no development panel reproduces the typed FINAL Score
  collapse. Typed-DEV Score is one 3-level family; Score5-DEV v1 (in-family A7q rows, `score5-dev-2026-09-29.md`) left every
  m6 soup unflagged. Known typed FINAL Score behaviour (400 slots, recorded `gates/types.json`):

  | Checkpoint | Predicted L0/L1/L2/L3/L4 | Top share | Accuracy | Gate verdict |
  | --- | --- | ---: | ---: | --- |
  | `m7-mx-soup` | 1/0/5/0/394 | .9850 | .3225 | COLLAPSED |
  | `m6-mxcx-soup` | 10/0/11/0/379 | .9475 | .3275 | COLLAPSED |
  | `m7-mxcx-soup` | 0/12/43/14/331 | .8275 | .2975 | OK (fails M7 S3: accuracy below always-majority .32) |
  | `m6-mxcxa-soup` | 21/15/44/5/315 | .7875 | .3300 | OK |
  | `m4-t-a7-soup` (released) | 60/6/57/20/257 | .6425 | .3325 | OK |
  | Kai1 (8K) | 223/0/0/0/177 | .5575 (L0) | .2450 | OK |

  Gold is 35/79/83/75/128 (always-4 = .32).
- Coordinator (2026-09-29 14:05): build an out-of-family 5-level typed Score development panel from fresh seeds of our own
  typed-panel generator, disjoint from FINAL, the development panels, C1 and training rows, validated against these
  collapses; node A GPU0/1 shared lease, ≤ 0.3 GPU-h.

## 2. Facts that change the request (verified before this prereg; read-only)

- **Generator.** The typed panels come from `benchmark/generate.py` (`generate(split, seed, groups_per_family)`, L697–804).
  `publication/generate_arena_v3.py` and `scripts/generate_postkey_artifacts_v3.py` render model-card artifacts; they
  generate no items. Typed DEV = `--split dev --seed decision2-public-dev-v1 --groups-per-family 100`; typed FINAL =
  `--split final --seed-file <private 32 bytes> --groups-per-family 100` (`scripts/plan_first_release_v3.py`), generator
  file sha256 `c6569d4c…` (= the current file). Item streams are `Random(sha256(seed‖family‖index))`, so output is
  deterministic given seed and index; an in-memory regeneration of typed DEV (public seed) matched its prompts
  byte for byte. That DEV check is the only generator run before this prereg; no `score5t-dev` seed has been drawn.
- **FINAL Score is one family, not four.** Typed FINAL has four families (`constraint_competition` Choice,
  `exception_stack` Noul, `evidence_join` Choice + Noul, `resource_ledger` Score). Only `resource_ledger` asks Score:
  5 levels, 400 items = 100 groups × 4 variants (base, counterfactual, order, label). FINAL's Score family mix is therefore
  100% `resource_ledger`, and this panel draws only that family. (The M7 record's "Typed FINAL Score spans the four FINAL
  families" is inaccurate: typed FINAL spans four families, its Score one.)
- **Family structure** (`resource_ledger()`, L482–536): initial ∈ 0..4, capacity 4; an add at tick 10 (1–3 units, posted),
  a removal at tick 20 (1–3 units, posted), a tick-30 event (add or remove, 1–2 units, unposted); rows shuffled; the
  counterfactual negates the removal's posted flag; base and counterfactual are swapped with probability 1/2. The gold level is
  the final amount. Hence 90 answer-relevant structures (initial, add units, removal units, removal posted), 360 states once
  event ids are ignored and 2,160 once row order counts. Analytic expected gold per item: L0 .111 / L1 .189 / L2 .200 /
  L3 .200 / L4 .300 (FINAL realised .0875 / .1975 / .2075 / .1875 / .32).
- **FINAL already covers most of the space** (counted on node A, aggregates only): 80 of 90 answer-relevant structures,
  152 of 360 states (76 of 180 group states, since every FINAL group holds both removal-posted values), 276 of 2,160
  ordered states. Fresh draws are text-disjoint by construction (random 6-character event ids), but about 42% of fresh
  groups would repeat a FINAL ledger problem up to ids and row order, and about 89% share an answer-relevant structure with
  FINAL. Text n-gram tools see nothing in these items (fixed template + ids + small integers), so near-duplicates are defined
  structurally (§4).
- **Other panels and training.** Typed DEV uses the four DEV families (no ledger). SELECT700 / CAL700 / CAL698 come from
  `training/data/build_rights_clean_v1.py` and `build_rights_clean_v2.py` (their 90 Score rows are
  `targeted_quantized_median`). No training builder
  imports `benchmark.generate`, and `training/data/build_pilot.py` rejects all eight benchmark families; a count-only scan
  of the 214 training files (2.77M rows) under node A `c1-corpora/training/` found no ledger rows. The generated Score arms
  (A6g, G6: `v2.data.verifiable`; A7g: Decision 1.0 Stage 4 v2 data) come from other generators.

## 3. Source, seeds, size, halves

- Code: `benchmark.generate.generate("final", seed, 1000)` at the build commit; the build asserts the generator file sha256
  equals FINAL's (`c6569d4c…`, full value in the MANIFEST). Only `resource_ledger` rows are kept, with `state` and
  `questions` unchanged (FINAL's exact format, instructions and five ascending criteria).
- Seeds (fresh, public, 32 raw bytes each): fit = SHA-256(`b"decision2-score5t-dev-v1/fit"`), check =
  SHA-256(`b"decision2-score5t-dev-v1/check"`). Their commitments (sha256 of the seed bytes) go in the MANIFEST. The build
  asserts that no generated group id or item id equals a typed FINAL or typed DEV id.
- Per seed, instance indices are walked in increasing order and whole groups (all 4 variants) are accepted until 100 groups
  pass §4; the build aborts if 1,000 indices do not yield 100 (expected ≈ 173 needed).
- **Panel = 200 groups = 800 items: fit half = the 100 fit-seed groups (400 items), check half = the 100 check-seed groups
  (400 items).** Each half has FINAL's size and group structure; the halves are split by seed group, so no group straddles
  them. No level rebalancing: gold follows the generator after the §4 filter; per-half gold is reported next to FINAL's and
  the analytic expectation.

## 4. Disjointness rules

Keys per ledger item: **K0** exact = sha256 of the canonical `{state, questions}` payload (the generator's
`payload_sha256`); **K1** near-duplicate = the same ledger problem up to event ids and row order = (initial, capacity, the
tick-sorted tuple of (tick, kind, units, posted)) + instructions + criteria; **K2** answer-relevant structure = (initial,
add units, removal units, removal posted), disclosure only.

1. **Typed FINAL** (all 1,600 items, from the hash-verified gold-free prompts): a fresh group is dropped if any of its items
   matches any FINAL item on K0 or K1.
2. **Typed DEV, SELECT700, CAL700, CAL698:** K0 match, or K1 match for any row that parses as a ledger state, drops the group;
   each file is also searched for the ledger instruction sentence. Expected: zero.
3. **Training rows:** the data-manifest evidence in §2, plus a fixed-string scan of every file in node A's training-corpora
   manifest (`/data/dev2/private/htdev/iso/training-corpora.json`) for the ledger instruction sentence and the criterion
   sentence "Final amount is 0 units."; any hit is parsed and compared on K0/K1, and matching groups are dropped. Expected:
   zero hits. A looser co-occurrence count (`posted` and `capacity` in one row) is reported, not used to drop. Missing
   manifest files are listed.
4. **C1: not opened.** C1 is 14 tasks from 8 post-cutoff human-labelled public sources
   (`sealed-c1-source-registry-2026-09-28.md`); a programmatic synthetic ledger family is not among them. Disjoint by source;
   no decryption and no access to `/data/dev2/private/sealed/`.
5. **Within the panel:** K0 and ids unique across all 800 items. K1 repeats within and across halves are allowed (FINAL's
   100 groups use 76 group states) and counted.
6. **Disclosed, not dropped:** K2 overlap with FINAL (unavoidable: FINAL covers 80 of 90) and fit/check K1 and K2 overlap.

The FINAL-derived exclusion keys and per-index accept/drop lists stay in private storage (node A build dir); shareable
records carry counts only.

## 5. Rendering and files

- Prompt line `{id, state, questions}` exactly as generated, with id `score5t-` + the first 16 hex of
  sha256(`"score5t-dev-v1:id:" + generator item id`). The build asserts that the native 0.6B adapter's model input for an
  item is byte-identical under its generator id and its panel id. Order: fit groups then check groups, in index order,
  variants in generator order (as FINAL).
- Gold line in HT-DEV format (`task` `score5t/resource_ledger`, `source` `benchmark-generate-resource-ledger`, `half`
  `fit`/`check`, `group_id`, `cluster_id`, `language` `en`, `input_chars`, `long`, `provenance` {seed name, instance index,
  variant, generator item id}, `questions`, `gold`). `gold.decision` is the generator's gold record, asserted equal to
  `gold_record(...)`, so the file also works with the typed scorer's `evaluate_answer`.
- Node A: `/data/dev2/private/panels/goldfree/score5t-dev.prompts.jsonl` (0644), `/data/dev2/private/panels/gold/score5t-dev.gold.jsonl`
  (0600), per-half split files `score5t-dev.{fit,check}.{prompts,gold}.jsonl` next to them (same modes), and
  `/data/dev2/runs/eval/score5t-dev/build-v1/` (MANIFEST, exclusion report, accept/drop lists, build log). The option-key /
  position leak audit runs before the panel is registered by hash in `v2/eval/panels.py` DEVELOPMENT.
- Backup: the private eval-artifacts dataset `score5t-dev/v1/` (panel files, MANIFEST, validation aggregates; not the
  FINAL-derived accept/drop lists) if `hf_headroom.sh` shows room.
- Rules: `score5t-dev` items are never training data, and fit-half use is limited to fitting post-hoc output corrections
  (e.g. per-level offsets); check-half use is limited to checking and selecting them.

## 6. Metrics (per model; full panel, fit half, check half)

Predicted level = the adapter's Score point via `benchmark.score.evaluate_answer` (unique argmax; ties and invalid answers are
wrong), as on typed FINAL. Reported: level histogram over all slots with invalid as its own category; top (modal) share = top
category / all slots (as `gates.py types`); modal level; rare levels (< 2% of slots); invalid/missing count; accuracy with
the Wilson 95% interval (as the gate) and a 95% item-bootstrap interval; always-majority accuracy (panel modal gold level,
smallest level on ties, as M7) and accuracy − always-majority with a paired item-bootstrap 95% CI (5,000 draws, seed
20260929, as M7 S3); macro-F1 over the five gold levels (missing wrong); QWK over answered items; L4 share and gold L4 share.

## 7. Flags (changed from Score5-DEV, with reason)

Score5-DEV's thresholds (COLLAPSE: modal ≥ .60 or ≥ 2 rare levels; WARN: ≥ .40 or 1 rare level) were set for a balanced
in-family panel (always-modal .20). This panel keeps typed FINAL's skewed gold (≈ 30% level 4). Applied to the recorded typed
FINAL histograms above, those thresholds flag every checkpoint, including the released soup (.6425, rare L1) and mxcxa
(.7875, rare L3). They cannot separate the known collapses from the non-collapses, so this panel uses the release gate's own
definition (`v2/eval/gates.py`, `TOP_SHARE = 0.9`, Wilson lower bound vs chance):

- **COLLAPSE:** top share ≥ 0.90, or the accuracy Wilson 95% lower bound ≤ 0.20 (chance = 1/5).
- **WARN** (advisory): not COLLAPSE, and the Wilson 95% upper bound of the top share ≥ 0.90 (the panel cannot exclude
  gate-level collapse).
- **NO-GAIN** (informational; M7 S3's third clause): the paired-bootstrap 95% lower bound of accuracy − always-majority ≤ 0.
- Rare levels are reported, not flagged. "Flagged" in §8 means COLLAPSE.

## 8. Validity criteria (FINAL outcomes validate the tool, as for HT-DEV); primary = the full 800-item panel

- **V1:** `m6-mxcx-soup` and `m7-mx-soup` are COLLAPSE.
- **V2:** `m4-t-a7-soup` (released) and `m6-mxcxa-soup` are not COLLAPSE.
- **V3:** top share orders the five 2.0 checkpoints strictly as on typed FINAL: `m7-mx-soup` > `m6-mxcx-soup` >
  `m7-mxcx-soup` > `m6-mxcxa-soup` > `m4-t-a7-soup`.
- **PASS iff V1, V2 and V3 all hold**; otherwise FAIL, reported plainly with the reason and the next best source.
- Secondary (reported, not pass/fail): V1–V3 on each half separately; Kai1's position (prediction: lowest top share);
  six-model Kendall τ and Spearman ρ against FINAL; per-model |panel − FINAL| top share; WARN and NO-GAIN outcomes (on FINAL
  all five soups are NO-GAIN); the released soup with the frozen autotune cache (sensitivity).
- Predictions (not criteria): `m7-mxcx-soup` not COLLAPSE; always-majority level 4 on both halves.

## 9. Validation collections

- One collection per model on the full panel with its formal (scored) run's settings: package
  `/data/dev2/runs/06b/m1/arms/<arm>/full/best-export`, revision = `COMPLETE.json` `best_export_manifest_sha256`
  (m6-mxcx `3ae009ad…`, m6-mxcxa `02892a86…`, m7-mx `87dcd8fb…`, m7-mxcx `93aafa04…`, m4-t-a7 `0f96aa39…`), adapter spec
  `v2/06b/records/adapters/dev2-06b-causal-8k.json` (8,192-token cap, over-budget invalid), `--extra model_id=dev2-06b/<arm>`,
  image `decision20-train-fast:host2` (`f83b1d10…`, FLA + causal-conv1d kernels). The four m6/m7 soups use a per-run copy of
  the frozen Triton autotune cache `/data/dev2/runs/06b/m6/triton-cache-frozen` (tree `e215f8bd…`), verified unchanged after
  the run; `m4-t-a7-soup` runs without autotune settings, as formally, plus one sensitivity collection with a cache copy.
- Kai1 if cheap: `Decision-1.0-Kai-0.6B` @ `7185f514…`, spec `v2/06b/records/adapters/dev2-06b-kai-native-8k.json`,
  `--extra backend=kai --extra model_id=dev2-06b/kai1-native-8k`, the kai-lex env mounted, no autotune settings; dropped
  and recorded if its 20-item smoke fails.
- Compute: node A GPU0 (else GPU1), shared lease `owner.eval-score5t`, VRAM checked before each job, each job ≤ 15 min, stop
  at 0.3 GPU-h. No reruns except after an infrastructure failure that wrote no predictions (recorded). Seal with
  `v2.eval.htdev.score seal` (prompts + predictions only) before any gold is read.

## 10. After validation

- PASS: a `score5t` block in `v2.eval.dev_readout` (full / fit / check subsets with these flags); the per-half split files
  published for 0.6B Milestone 8 (fit the per-level offsets of `m6-mxcx-soup` on the fit half, check and select on the
  check half); usage text for "Eval runners" in the record.
- FAIL: the reason and the next best source, in the record.
- Either way: record `score5t-dev-2026-09-29.md`, the eval gist file `01-decision-2-eval-peers.md`, GPU-hours, commits;
  merged into `xunzhuo/decision-2-training`.
