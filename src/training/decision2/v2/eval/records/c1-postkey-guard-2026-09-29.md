# JevArena-C1 v1.2 as a private post-key successor guard (successor-rule item 8) — 2026-09-29

> **Status (2026-09-30 00:30 UTC+8): done.** `python3 -m v2.eval.gates c1` returns PASS or REGRESSION and reproduces all
> 12 event-3 paired comparisons exactly. The custodial runner `v2/eval/sealed/c1-postkey.sh` collects, seals and
> scores one frozen package on node A. Baselines are registered for 0.6B, 2B, 4B, 9B and 27B. **DEV2.0-0.6B
> `b2131337` scores 36.92 on C1 v1.2, against 33.02 for the previous revision (+3.89 [+2.16, +5.60], PASS).**
> GPU: 0.041 GPU-h.

Label for every number here: **JevArena-C1 v1.2, post-key (not an independent validation)**. The
independent-confirmation claim stays with the revisions scored in events 1–3
([event-3 record](m4-c1-event3-prep-2026-09-29.md) §0000).

## 1. The rules (coordinator, 2026-09-29 23:40) and where each is enforced

| Rule | Mechanism |
| --- | --- |
| Only use: successor-rule item 8. A successor must not show a significant C1 regression (p < .05, paired) vs the current revision | `gates c1`: REGRESSION when Δ < 0 and the two-sided paired bootstrap p < .05 |
| Never training data | C1 plaintext exists only transiently on node A: prompts on the panel root during one collection, gold in a private temp dir during scoring; the key only on stdin |
| Never selection, in development or among siblings | The runner takes one frozen package (manifest and identity pinned; exact formal-run parity). The custodial ledger `$C1/postkey/LEDGER.jsonl` refuses a second successor against the same tier baseline and a second scoring of the same weights, unless `--approval` records the coordinator's approval. Development models are never scored |
| Card label "JevArena-C1 v1.2, post-key (not an independent validation)" | `score.py --post-key` writes it into the seal and the report; the gate output and the summary carry it too |
| Custodial collection | Smoke and exact parity before the key; predictions sealed before the gold is decrypted; plaintext removed on any exit; one process at a time (lock); each step logged in C1's `ACCESS.log` (`POSTKEY` lines) |
| No per-item outputs in git | Outputs in git are aggregates. The gold-free predictions go only to the private eval-artifacts dataset (as for event 3) |

## 2. The gate: `python3 -m v2.eval.gates c1`

```bash
python3 -m v2.eval.gates c1 --left <successor run> --right <current-revision run> \
  --left-name A --right-name B --gold <decrypted gold on node A> --output OUT.json
```

- **Inputs.** A run is a C1 run directory (`SEAL-C1.json`, `output/sealed-c1.predictions.jsonl`), as the events
  and the runner write it.
- **Checks.**
  - Both seals: the C1 prompts `0b29686f…`, the predictions unchanged since the seal, no missing prompt.
  - The gold hash `c0277771…`.
  - The pinned v1.2 retired list `bbf095c7…` (default path on node A), so 2,840 items are scored.
  - OUT is written outside both run directories.
- **Statistic.** The same comparison as the scoring events: `score.paired_draws`, a group bootstrap by source group
  within each task, 5,000 draws, seed 20260927. `score.paired` now calls the same function; its output is
  byte-identical to before on a fixed synthetic case.
- **p.** The two-sided percentile-bootstrap p-value of those draws: the smallest level at which the percentile
  interval excludes 0. With 5,000 draws, p < .05 exactly when the reported 95% interval excludes 0.
- **Verdict.** REGRESSION when Δ < 0 and p < .05 (the paired 95% interval lies below 0); otherwise PASS. The exit
  status is 0 for both verdicts.
- **Output.** Aggregates only: both C1 totals and per-type values, Δ, CI and p overall and per type, the item set,
  and the runs' seal hashes and labels.
- **Gold access.** The gold stays on node A. `c1-postkey.sh gate` decrypts it for the call and removes it.

**Reproduction of the event-3 paired numbers** (node A, CPU, `c1-postkey.sh reproduce`, 3 min 13 s). The gate was
run on every pair in the event-3 plan and compared with the event's `PAIRED-C1-*.json` files. **All 12 match
exactly**: Δ, CI, per-type Δ and CI, and the item set. Aggregates are in
[`c1-postkey-guard/REPRODUCE-event3.json`](c1-postkey-guard/REPRODUCE-event3.json).

| Event-3 pair (left − right) | Δ C1 [95% CI] | p | Verdict under the item-8 rule |
| --- | --- | ---: | --- |
| DEV2.0-2B − Sol 1.0 16K | +0.68 [−0.91, +2.22] | .42 | PASS |
| DEV2.0-2B − Decider 2B | +3.26 [+1.32, +5.21] | < .001 | PASS |
| DEV2.0-2B − This-That 1.2 | +2.90 [+0.85, +4.99] | .005 | PASS |
| DEV2.0-4B − Nox 1.0 | −1.32 [−3.00, +0.35] | .12 | PASS |
| DEV2.0-4B − Decider 4B | −1.27 [−3.05, +0.49] | .18 | PASS |
| DEV2.0-4B − Jet v6.2 | −2.06 [−3.86, −0.18] | .033 | **REGRESSION** |
| DEV2.0-0.6B (event 2) − Kai 1.0 8K | +10.89 [+8.88, +12.64] | < .001 | PASS |
| DEV2.0-9B − Lux 1.0 16K | +1.80 [+0.55, +3.07] | .004 | PASS |
| DEV2.0-9B − Nimble v2 | +1.00 [−0.69, +2.75] | .25 | PASS |
| DEV2.0-27B − AutoJev-27B | −0.84 [−2.50, +0.69] | .27 | PASS |
| DEV2.0-27B − Eikos-27B | −2.04 [−3.66, −0.44] | .015 | **REGRESSION** |
| Kai 1.0 8K − Kai 1.0 1,024 (reference) | +4.35 [+3.63, +5.04] | < .001 | PASS |

These are peer pairs, not successor checks. They are shown only because they exercise both verdicts on real data,
matching the bold intervals of the event record.

**Resolution.** Paired 95% half-widths on the 2,840 items are 1.26–2.07 C1 points: 1.26 for a close lineage
(DEV2.0-9B vs Lux 1.0) and 1.6–1.9 for most pairs. The guard therefore flags a successor only when it loses more
than about 1.3–2 C1 points. Like the public-231 guard (item 7), it catches large losses and cannot separate
siblings. Its false-alarm rate on sibling noise was not measured, because scoring sibling checkpoints would be
development use.

## 3. The runner: `v2/eval/sealed/c1-postkey.sh` (node A; helpers in `v2/eval/sealed/postkey.py`)

- **`collect --spec SPEC`.** The spec (`dev2-c1-postkey-spec/1`) is one frozen release package with its formal
  runtime, written like a row of `sealed/event3-models.json`. The runner then works in this order:
  1. It plans the spec against the registry `sealed/c1-postkey-baselines.json`.
     - A `successor` is gated against its tier's registered baseline (item 8). The baseline must hold other
       weights.
     - A `current` revision builds its tier's baseline.
     - A spec's `compare` runs are gated for information only.
  2. It verifies, on CPU: the image, the mirror's adapter module, paths, the package manifest and identity, pinned
     files, the frozen cache, and each comparison run's seal file.
  3. It checks the ledger (§1).
  4. On the GPU, it runs a smoke of typed FINAL and public 231 with **exact** parity against the stored formal run.
  5. Only then does it read the key and collect all 2,874 prompts.
  6. It seals the predictions with the post-key label and removes the prompts.
  7. It decrypts the gold, scores on v1.2, runs `gates c1` against each comparison and removes the gold.
  8. It writes `SUMMARY.json` and appends one ledger line.
- **Dry runs.** `--verify-only` (CPU) and `--preflight-only` (smoke and parity) never read the key.
- **Summary contents.** The C1 score, per-type values, the item-8 verdict, GPU-hours, and `baseline_entry`. When
  the successor is released, `baseline_entry` becomes the tier's registry entry.
- **`gate`.** Runs `gates c1` between two sealed runs. **`reproduce`** repeats the event-3 pairs (§2).
- **Tests.**
  - `v2/eval/tests/test_gates.py`: 5 new tests. They cover verdicts, p against the interval, identity with
    `score.paired`, the CLI, and refusals.
  - `v2/eval/sealed/tests/test_postkey.py`: 13 tests. They cover plan refusals, the committed spec and registry,
    runner arguments, the non-selection ledger, `finish`, `reproduce` against event files, script argument checks
    and `score --post-key`.
  - `python3 -m pytest v2/eval/tests v2/eval/sealed/tests -q`: 351 passed, 1 skipped, 2 xfailed. shellcheck and
    black are clean.

## 4. Baselines: where each tier's current-revision run lives

All runs are on node A. They are valid for every revision with the same weights identity (renames and card-only
revisions included). Registry: `v2/eval/sealed/c1-postkey-baselines.json`.

| Tier | Current weights (scored revision, identity) | C1 v1.2 | Run on node A (`/data/dev2/runs/eval/…`) | Seal | Backup (private eval-artifacts) |
| --- | --- | ---: | --- | --- | --- |
| 0.6B | DEV2.0-0.6B `b2131337`, `bb806f31` | 36.92 | `c1-postkey/0.6B/dev2-0p6b-b2131337-20260929T161232Z/cand` (post-key, this record) | `056026ac…` | `c1-postkey/0.6B/…` (`afcbb407`) |
| 2B | DEV2.0-2B `5ad3e9a3`, `073bd1f2` | 45.70 | `m4/c1-event3/cand2b` (event 3, stored) | `1b008844…` | `m4/c1-event3/` (`1561a3a5`) |
| 4B | DEV2.0-4B `452f1332`, `11b5ca1c` | 48.38 | `m4/c1-event3/cand4b` (event 3, stored) | `b73e25f1…` | `m4/c1-event3/` (`1561a3a5`) |
| 9B | DEV2.0-9B = package `DEV2.0-8B@53bac735`, BF16 `b1ed5a71` | 53.77 | `m4/c1-event3/cand9b` (event 3, stored) | `2457cb74…` | `m4/c1-event3/` (`1561a3a5`) |
| 27B | DEV2.0-27B = `DEV2.0-26B@5683c6f0`, `b7fd44e3` | 57.33 | `m4/c1-event3/cand27` (event 3, stored) | `f41d44c5…` | `m4/c1-event3/` (`1561a3a5`) |

- **0.8B** (added 2026-09-30, §8): DEV2.0-0.8B `7d08d0e1`, identity `60356482`, **C1 v1.2 40.17**. The run is
  `m4/c1-event1/e8f` (event 1, stored predictions; seal `fe510c87…`). Backups: eval-artifacts `m4/c1-event1/` and
  `c1-postkey/0.8B/` (`758acf53`).
- **Successors.** The comparison target for a successor is always the registry entry of its tier. In the release
  commit, the successor's `baseline_entry` replaces that entry.

## 5. DEV2.0-0.6B `b2131337` on C1 v1.2 (post-key baseline collection)

- **Package.** The release run's real `hf download` of `b21313375ad77ddf4a8e420fa5195e6e09582043`: manifest
  `816827dd…`, 32 of 32 files match, identity `bb806f31…`.
- **Runtime.** The formal settings of the scored run `m8-s5-b05`:
  - `training.model.infer` with adapter `dev2-06b-causal-8k-sb`: 8,192 tokens, T = 1, and the package's
    `score_bias.json` (`7d3a060f…`, bound to `bb806f31`);
  - image `f83b1d10` and `HIP_FORCE_DEV_KERNARG=1`;
  - a fresh copy of that run's persisted autotune cache (`e215f8bd`; no entry added);
  - the adapter module is byte-identical to the formal run's.
- **How it ran.** Spec `v2/eval/sealed/c1-postkey/dev2-0p6b-b2131337.json`, mirror `2b5e878db`, node A **GPU0**
  under the shared named entry `owner.eval`. GPU0 was idle (0% use and VRAM), and the 0.6B track's entry was idle.
  Times are UTC:

  | Time | Step |
  | --- | --- |
  | 16:12:33 | Verified; ledger clear |
  | 16:12:34–16:13:44 | Smoke: the whole typed FINAL (2,000 answers) and public 231 (231) panels. **0 answers changed, max drift 0.0**, every row with identity `bb806f31` (`training.model.infer` takes no `--max-items`) |
  | 16:13:44 | Key read, prompts decrypted |
  | 16:13:44–16:15:04 | Collection: 2,874 of 2,874 prompts, 0 invalid, 0 over budget, every row with identity `bb806f31` |
  | 16:15:04 | Sealed (`056026ac…`, post-key label); prompts removed; gold decrypted |
  | 16:15:04–16:18:17 | Scored on v1.2 (2,840 items; 34 retired, as in event 3); gate vs the previous revision; gold removed |

  Cleanup logged "prompts and gold decrypted and removed; GPU0 entry owner.eval restored". Nothing was left on the
  panel root or in a temp dir. `ACCESS.log` has 18 `POSTKEY` lines (reproduction and collection).

**Result** ([summary](c1-postkey-guard/SUMMARY-dev2-0p6b-b2131337.json),
[gate](c1-postkey-guard/GATE-C1-dev2-0p6b-b2131337-vs-e61b2b44.json),
[smoke parity](c1-postkey-guard/PARITY-smoke-dev2-0p6b-b2131337.json)):

| DEV2.0-0.6B | C1 | Choice | Noul | Score | Invalid |
| --- | ---: | ---: | ---: | ---: | ---: |
| `b2131337` (current revision, post-key) | **36.92** | 38.46 | 55.69 | 20.14 | 0 |
| `e61b2b44` (previous revision, event-2 predictions on v1.2) | 33.02 | 33.29 | 47.65 | 21.59 | 0 |
| Paired Δ [95% CI] | **+3.89 [+2.16, +5.60]**, p < .001 | +5.16 [+2.65, +7.75] | +8.05 [+4.11, +11.86] | −1.45 [−4.25, +1.19], p .30 | — |

- **Item 8, applied after the fact to the released successor: PASS.** The current revision is significantly above
  the previous one. The gain is in Choice and Noul. Score is level: the per-level offsets did not move C1's Score
  tasks.
- **Declines to disclose** (task macro-F1; languages with ≥ 20 items):
  - `star_rating` 21.3 vs 30.5;
  - `moment_type` 53.2 vs 60.1;
  - `is_rapport` 43.6 vs 49.0;
  - `likelihood` 10.2 vs 13.1;
  - Russian .263 vs .353 (300 items); Finnish .426 vs .500 (68).
  - The largest gains are `event_causality` +24.4, `span_is_event` +15.7 and `hallucination` +13.8; Arabic
    .447 vs .339.
- **Proposed 0.6B card line** (release engineering, card-only pass; the coordinator decides). Add it under the
  existing sealed lines, which keep measuring the previous revision:
  > **JevArena-C1 v1.2, post-key (not an independent validation):** 36.92 on this revision vs 33.02 for the
  > previous revision (+3.89, 95% CI [+2.16, +5.60]; 2,840 items, paired by source group). The independent sealed
  > confirmation above measured the previous revision.

## 6. Eval runners (paste-ready)

```bash
### JevArena-C1 v1.2 post-key successor guard (successor-rule item 8; eval custodian; code at 2b5e878db or later,
### registry with the 0.6B baseline from the commit that adds v2/eval/records/c1-postkey-guard-2026-09-29.md)
# C1 is post-key: never training data, never a selection criterion (development or siblings); cards say only
# "JevArena-C1 v1.2, post-key (not an independent validation)". Prompts and gold never leave node A.
# local: mirror the commit that holds the successor's spec (and the current registry)
src/training/decision2/v2/common/mirror_to_node.sh --path src/training/decision2 node-a <commit>
SRC=<full-sha>-src_training_decision2; S=/data/dev2/src/$SRC/src/training/decision2
# item 8 for one verified successor package on node A (the key only on stdin; ~5 min for 0.6B, ≈ 0.34 × formal wall for larger tiers)
cat <C1 key file> | ssh -o BatchMode=yes "$NA" "nohup bash $S/v2/eval/sealed/c1-postkey.sh collect --gpu <N> \
  --src $SRC --spec <successor spec.json> [--shared] > /data/dev2/runs/eval/c1-postkey/<name>.stdout 2>&1"
#   key-free first: same command with --verify-only (CPU) or --preflight-only (GPU smoke + exact parity), no key
#   result: <job>/SUMMARY.json → item8.verdict PASS | REGRESSION (the job dir is in the stdout);
#   one successor per tier baseline — another candidate needs --approval "<coordinator decision>"
# the gate alone between two sealed C1 runs
cat <C1 key file> | ssh -o BatchMode=yes "$NA" "bash $S/v2/eval/sealed/c1-postkey.sh gate --src $SRC \
  --left <successor run> --right <current run> --left-name A --right-name B --output <out.json>"
```

- **Writing a successor spec.** Copy `v2/eval/sealed/c1-postkey/dev2-0p6b-b2131337.json` (0.6B), or the tier's row
  in `sealed/event3-models.json` (`cand2b`, `cand4b`, `cand9b`, `candidates_27b.f1`). Then set:
  - `role: successor` and a new `name`;
  - `revision`;
  - `model_path` = `package.dir` (the frozen package on node A), `package.manifest_sha256`, and `identity` =
    `package.identity`;
  - `parity.stored` (the successor's stored formal typed-final predictions, with public 231 beside it), `files`
    and `stored_collect`;
  - `cache` (the formal run's persisted autotune cache and its `triton_cache.py` digest);
  - `image` (`host2` or `kernel`).
- **27B successors.** Their packages and caches are staged on node A first, as in event 3.
- **After the release.** Put `SUMMARY.json` → `baseline_entry` into `c1-postkey-baselines.json` → `tiers.<tier>`.

## 7. GPU-hours, artifacts, commits

- **GPU: 0.041 GPU-h**, all on node A GPU0 under a shared lease: smoke 0.019 and collection 0.022. The
  verification, event-3 reproduction, scoring, gates and upload were CPU only.
- **Artifacts** (gold-free; no prompts, gold, salt or key): private `llm-semantic-router/decision-2.0-eval-artifacts`
  `c1-postkey/`, commit `afcbb407`. It holds 29 files:
  - the 0.6B run's seal, report, predictions with their manifest, COLLECT, GPU time, plan, verify, parity, summary,
    gate and log;
  - the 12 reproduction gates and `REPRODUCE.json`;
  - the ledger and the `POSTKEY` access lines;
  - `SHA256SUMS` `c8b6f23e…`. All 28 hashed files were re-downloaded and matched.
- **Node A.** `/data/dev2/runs/eval/c1-postkey/{0.6B/dev2-0p6b-b2131337-20260929T161232Z, reproduce-event3-20260929T160719Z}`
  and the ledger `/data/dev2/private/sealed/c1/postkey/LEDGER.jsonl`.
- **Commits.** `a9fe47a62` (gate, runner, registry, spec, tests), `2b5e878db` (cleanup log wording), and the
  records commit that adds this file, the 0.6B registry entry and `c1-postkey-guard/`. `check_no_private.sh` is
  clean on each.

## 8. Addendum 2026-09-30: DEV2.0-0.8B baseline (coordinator decision, 00:28 UTC+8)

- **Source.** DEV2.0-0.8B's current weights (identity `60356482`) are the event-1 weights, served at T = 1 by
  revision `7d08d0e1`. Event 1 collected them with the CAL698 temperatures (calibration `9f76867d…`: Choice 1.123,
  Noul 1.071, Score 0.395).
- **Derivation (node A, CPU, prompts not decrypted).** The temperatures were undone on the stored predictions with
  the release's own `v2.release.dev_calibration.rescale` (softmax(log q · T), as `retemper_predictions --undo`),
  taking each question's type from its answer. **0 of 2,874 answers change**, both under the release rule and under
  the C1 scorer's rule (`benchmark.score.evaluate_answer`, including its validity checks: Score answers carry
  probabilities, so the scorer takes their unique argmax).
  - Receipt: [`c1-postkey-guard/T1-DERIVATION-dev2-0p8b-event1.json`](c1-postkey-guard/T1-DERIVATION-dev2-0p8b-event1.json).
  - Rounding the expected score instead would show 372 changes, but that is not the scorer's rule.
  - The stored run therefore serves as the baseline unchanged. The T = 1 file (`d4a104fe…`) is kept as evidence.
- **v1.2 value.** `c1-postkey.sh rescore` (new mode, `1f7ea80f6`) scored the sealed event-1 run on v1.2 with the
  post-key label. The gold was decrypted only for the call. Result: **C1 40.17** (Choice 41.00, Noul 59.40, Score
  24.29; 2,840 items, 0 invalid), report `8b494cd2…`.
- **Registry and backups.** The `tiers.0.8B` entry in `c1-postkey-baselines.json` points at `m4/c1-event1/e8f`.
  Backup: eval-artifacts `c1-postkey/0.8B/` (`758acf53`; derivation, T = 1 file, v1.2 report, event seal;
  readback OK).
- **Cost.** No collection and 0 GPU-h, so the approval for up to 0.1 GPU-h was not needed.
