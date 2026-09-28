# M3 item 4 — Decision Index 0.2.1 external stress test: harness readiness

Eval & peers track, 2026-09-28. **Status: the harness is ready, and nothing has been run on
the Index.** No Decision 2.0 candidate has been evaluated, and none may be until the
coordinator freezes a first release. The Index is an external diagnostic. It never merges
into the JevArena main score, and its numbers never enter our rank charts. Every output is
labelled **independent provisional 0.2.1 reproduction**, and the port always emits
`official_equivalent: false`.

Earlier work: [0.2.1 audit](../../../research/decision-index-v021-audit-2026-09-27.md),
[input rebuild audit](../../../research/decision-index-v021-input-rebuild-audit-2026-09-27.md),
[execution freeze](../../../research/decision-index-v021-execution-prereg-2026-09-27.md),
[peer roster](decision-index-peer-roster-2026-09-28.md). Module:
[`external_index021`](../../../external_index021/README.md).

## 1. Upstream revisions checked (row identity)

| Artifact | Revision | What it publishes | Effect on row identity |
| --- | --- | --- | --- |
| Space, audited pin | [`ed7d683e`](https://huggingface.co/spaces/multimodalart/jev-decision-index/tree/ed7d683e6626a4a5ace0086b8bdedd8b1cc58cf4) (2026-09-27 02:15Z) | index `5444deea…`, methodology `23538461…` | Reference for the port's fixture. |
| Space head | [`7cdcea3d`](https://huggingface.co/spaces/multimodalart/jev-decision-index/tree/7cdcea3dd14615192ff2e1f6fd13936a547b55d8) (2026-09-28 00:40Z; still the head when this work ended) | index `a5a4aa0a…`, methodology `2ff75bb8…`; 70 open entrants plus Jev; per-entrant native `results` for each benchmark, including contamination overrides | The protocol is identical to `ed7d683e`: areas, weights, gold set, chance, not-in-index list, added IDs, panel ID, counts and corpus hash. There is no row manifest, no run IDs and no per-row results. Seven commits since the pin added Eikos-27B, Hopper (G) 1.2, JPT-4B and Jet v6.2, removed Hopper 1.1.1, and rescored Jev's MuSR and HLE after exclusions (`34426d6a`). |
| Kit, audited pin | [`19ad28ec`](https://github.com/apolinario/decision-index/tree/19ad28ec9485493cc4f7fc07d91c178f948e6434) | editions 0.1 and 0.2 | The port's scoring base (six files are hash-pinned). |
| Kit head: `main` = branch `edition-0.2.1` | [`87d4650b`](https://github.com/apolinario/decision-index/tree/87d4650b42b377c0291a89c1f1a879f9b31082bf) (2026-09-27 03:40Z) | Native 0.2.1 edition. `data/release-v2.1/{toolret,bright,home-appliances}-subset.json` (SHA-256 `c301d550…`, `ed3a0522…`, `d2d0df92…`, pinned in `editions.py`), `index-0.2.1.json`, `hub/0.2.1/manifest.json`, a board fixture for 13 entrants, and the claim that `score` over the lab's own results reproduces all 67 entrants. | Kept run IDs: ToolRet 1,013 (685 scoreable plus 328 carried-over excluded rows), BRIGHT 307 (220 plus 87), Home 88 kept and 72 dropped. Rules: a query is answerable if any `scoring.scorable_ids` candidate is relevant; Home keeps the lowest row ID of each set of identical states. The row files are the same as 0.2. There is no tag and no release. |
| Author-uploaded runs (HF datasets) | Jebadiah 27B `3e69074b`, Hopper (G) 1.2 `610f97d3`, Jet v6.2 `ef10608b`, Eikos-27B-FP8 `c0ae0856`; JPT 0.8B/4B/9B gated | Complete `results.jsonl.gz` for each run | Used for item-level scorer validation (section 3). The JPT repos answer "requires approval" for our account; we did not request access. |

I found none of the following anywhere: a 0.2.1 git tag or release, the maintainers' per-row
board results, the maintainers' contamination lists or forcing rule, or per-row Index outputs
for Decider 2B/4B, AutoJev, GLiNER2.5, Bosun, Kev, Intern, Nimble, This-That or Decision 1.0.
Those model and code repos hold only aggregates or evaluations on their own benchmarks.

## 2. Gap status against strict 0.2.1

| Gap from the prior audits | Status | Evidence | Remaining |
| --- | --- | --- | --- |
| Home: which copy of each of the 24 duplicate pairs is kept (88 IDs) | RESOLVED | The kit head publishes the list and the rule. On the pinned generator the lowest-row-ID rule reproduces the list. In the rebuilt file, Home rows are sorted by ID and `first` reproduces the list (kept-ID SHA-256 `f76d0ef2…`). This is now enforced. | `last` remains a diagnostic only. |
| ToolRet/BRIGHT answerable queries and whole-group retention | RESOLVED | On the rebuilt suite, the `retrieved_ids` rule and the `scorable_ids` rule both select exactly the upstream 685 and 220 query groups; no group differs. The kept-ID hashes `9c77197f…` and `64523e4a…` are now enforced. The rule now uses `scorable_ids`, as upstream documents. | — |
| ToolRet/BRIGHT chance recomputed per answerable query | RESOLVED | The kit's `static_score` averages per-query random nDCG@10 over the rows it receives, and the port passes only kept rows. On the rebuilt suite the mean is 0.134074 for ToolRet and 0.116040 for BRIGHT, matching the published 0.1341 and 0.116. | — |
| ACOS review-level F1 | RESOLVED | Aligned with the kit: mean F1 over complete reviews, rounded to 4 decimals, multiplied by answered/requests, against chance 0.031. On four public runs the ACOS values are identical. Before this fix, raw differed by up to 1.4e-5 and failures were accounted differently. | The partial-review path is covered by a unit test only; no public run has ACOS failures. |
| RAGTruth always-"hallucinated" chance | RESOLVED | 0.5177 in the port, the Space and the kit's `index-0.2.1.json`. The transform check (section 3) has no failures. | — |
| ForecastBench Brier-to-skill | RESOLVED | `clip((0.25 − Brier)/0.25) × coverage` is identical in the pinned 0.2 spec and in kit 0.2.1. No transform-check failures. | — |
| Area and gold weights | RESOLVED | The port previously used 4-decimal weights and now uses the exact √n rule: 0.258494, 0.258494, 0.200229, 0.182783, 0.1. These equal the Space's `suite.area_weights` and the kit's `area_weights`. The change moves any index by at most 0.003 points. The 13 gold benchmarks weighted 1.2 are identical. | — |
| 442 common exclusions / 717 removed rows | RESOLVED | Exclusions SHA-256 `331df32d…` is the same in both kits, the protocol and the rebuilt suite. 717 = 315 + 330 + 72 by upstream ID, and hash-enforced. | — |
| 150,759 scheduled / 150,317 scoreable | RESOLVED | The rebuilt suite has 151,034 rows after exclusions, of which 150,317 are selected (`keep_ids_sha256` `ca4f8903…`). All four public runs are complete at 150,317 under both scorers. | — |
| Native per-row metrics in the scorer | RESOLVED for the published runs | Section 3. | — |
| **Contamination overrides** (new) | OPEN | The board scores rows found in an entrant's declared training data as wrong. This applies to 7 of 71 entrants on 8 benchmarks; for example, all 979 FinEntity rows for Eikos and 2 BANKING77 plus 3 CLINC150 rows for Jet. Neither the kit nor the port implements it, and the lists and forcing rule are unpublished. | Our own overlap audit of the frozen package against the 150,317 rows (section 5). |
| Native engine parity for our package | OPEN | Needs the frozen package. | The 86-request parity gate (section 5). |
| Board admission | OPEN, out of scope | Maintainer review; the latency rule requires a median of at most 1,000 ms per request on one RTX PRO 6000. | Coordinator. |

## 3. Reproduction checks

**Tests.** Command: `PYTHONPATH=src/training/decision2:/tmp/index021-kit python3 -m unittest discover -s src/training/decision2/external_index021/tests -v`.

- Kit `19ad28ec`: 10/10 passed before the changes and 14/14 after.
- Kit head `87d4650b`: 13/14. The Home-generator test errors by design, because `verify_kit` rejects the kit's changed `scoring/added.py` (`index02.py` and `report.py` also changed).
- ruff, black and codespell pass on the changed files. `make check` stops at a repo-wide codespell baseline on pre-existing typos in unrelated `v2/data` files.

**Published aggregates** (workstation, public JSON only):

| Check | Result |
| --- | --- |
| `published-parity`, 68 rows at `ed7d683e` | Max drift 0.005373 index points (0.005714 with the old rounded weights). |
| `compile_reference --check` at `ed7d683e` | Pass. |
| `compile_reference --check` at `34426d6a` and `7cdcea3d` | Refused by hash, as designed. Protocol content is identical. |
| Replay of all 71 rows at `7cdcea3d` | Max drift 0.005373 with exact weights. |
| Native result → raw/skill for every non-hosted entrant on the 21 non-track headline benchmarks | 2,940 checks at `7cdcea3d` (2,814 at `ed7d683e`). Max \|Δ\| is 0.0001 and none exceed 1.5e-4. A further 1,065 track-skill consistency checks all pass. |

**Item-level check** (CPU only, node A private directory). The suite was rebuilt with kit
`19ad28ec` in about 30 minutes, including one retry: Zenodo truncated one BPoMP file, the
kit's hash check rejected it, and a resumable hash-verified fetch repaired it. The
uncompressed rows hash is `b2b56d6f…`, the added rows `7429f3c9…` and the exclusions
`331df32d…`; the v1 rows and the added rows are byte-identical. Each of the four
author-uploaded runs (file SHA-256 equal to the HF LFS object ID) was scored three ways:
the final port, the port before this work, and kit `87d4650b` `score --edition 0.2.1`.
Only deltas are shown.

| Run | Final port vs kit 0.2.1 (38 benchmarks) | Port vs Space: benchmarks within 1.5e-4 | Headline Δ vs Space, unadjusted | Headline Δ with the Space's overrides substituted | Home `last` − `first` |
| --- | --- | --- | --- | --- | ---: |
| Hopper (G) 1.2 | identical (skill ≤ 5e-5, display rounding) | 38/38 | −0.002 | −0.002 | +0.077 |
| Jebadiah 27B | identical | 38/38 | −0.000 | −0.000 | 0.000 |
| Jet v6.2 | identical | 36/38 (BANKING77 and CLINC150 overridden) | +0.009 | +0.005 | n/a: second copies not run |
| Eikos-27B-FP8 | identical | 37/38 (FinEntity overridden) | +2.33 | −0.004 | +0.038 |

- Kit 0.2.1 on its own reproduces Hopper and Jebadiah at display precision, and Jet's
  author-reported 0.2.1 value equals our unadjusted rescoring.
- The port therefore equals the upstream 0.2.1 scorer on real per-row outputs. The only
  differences from the board are its contamination overrides.
- The Home copy choice moved headlines by at most 0.08 points, but upstream now publishes
  the kept copy.

## 4. Code changes (uncommitted)

| File | Change and reason | Test |
| --- | --- | --- |
| `external_index021/data/protocol-021.json` | Added `area_sizing` (√n rule, Arts fixed at 0.1) and `upstream_kit021`: revision `87d4650b`, the three subset-file SHA-256 values and the three kept-ID SHA-256 values after exclusions. Reason: pin the upstream 0.2.1 row identity. | `compile_reference --check` still passes. |
| `external_index021/protocol.py` | `area_weights()` computes the exact √n weights and checks them against the published 4-decimal weights; `aggregate` uses them. Reason: match the board formula. | `test_area_weights_follow_published_sqrt_rule`; the parity bound tightened to 0.0055. |
| `external_index021/selection.py` | Added `digest` and `check_upstream`: a complete selection must reproduce the upstream ToolRet, BRIGHT and Home kept-ID hashes, and `last` reports a sensitivity status. The answerable rule now uses `scoring.scorable_ids`, matching upstream and our README. | `test_upstream_021_list_check`, `test_relevant_candidate_must_be_scorable`; the generator test asserts that `first` equals the upstream hash and the lowest-ID rule. |
| `external_index021/score.py` | ACOS now takes the mean F1 over complete reviews, rounds to 4 decimals and multiplies by answered/requests, as the kit does. The claim text states when the row IDs match upstream. | `test_acos_partial_review_leaves_f1_mean_but_counts_answered_rows` |
| `external_index021/README.md` | Documents the changes above. | — |

## 5. Run procedure for one frozen release package

**Prerequisites**

1. The coordinator freezes release R: package repo and revision, weights SHA-256,
   calibration SHA-256, inference-source revision and container image digest. Do not run
   the Index on any unfrozen checkpoint, and do not select or tune anything on it.
2. The suite is built; reuse it. It lives in the node A private directory
   `/data/dev2/private/eval/index021/` (mode 700, 16 GB workspace). Re-run `suite verify`
   with kit `19ad28ec`, then the port's `verify-suite`. Expect 150,317 rows, status
   `upstream_021_row_ids_matched` and keep hash `ca4f8903…`.
3. Run a contamination audit (CPU): exact and normalized-text overlap between R's
   training, selection and calibration data (at the private dataset revision) and the
   `state` and question text of the 150,317 selected rows. Record the overlapping run IDs.
   The board's method is unpublished, so we declare our own.

**Engine adapter requirements** (a kit `Engine` subclass, passed as `module:Class`)

- It calls the packaged native Choice/Noul inference path of R, with R's own prompt
  rendering, calibration and limits. It never uses the kit's `transformers` engine or a
  base-model readout. The Index has no Score questions.
- Every question is answered. Every supplied option gets a finite probability, the
  probabilities sum to 1 within 0.01, and the chosen key is one of the options.
- It raises `Unsupported` only for declared capacity limits. It never truncates, drops
  options, adapts the prompt to a benchmark, or reads gold.
- Size of the task: 322,459 questions (307,759 Choice, 14,700 Noul), up to 151 options per
  Choice question (14,500 questions have more than 26), and up to 64 questions per request.
- `provenance` and `runtime()` report R's hashes, the image digest, the torch/ROCm
  versions and the GPU class. The engine file is hash-locked for the run, the seed is
  fixed, and there is one process per GPU; record whether requests are batched.

**Steps**

1. **Compatibility and native parity, 86 requests.** The sample `compat-86.jsonl.gz` is
   already built on node A with kit `87d4650b`
   (`suite sample --edition 0.2.1 --n 86 --single-requests`, seed 20260919). Its SHA-256
   is `1356ceaf…`; it covers 44 benchmarks, and every row is in the selection. Run it
   (a) through the harness (`python -m decision_index run --rows compat-86.jsonl.gz --engine …`)
   and (b) through R's own native entry point outside the harness, on the same node, GPU
   class and image.
   - Gate: 86/86 valid or declared-unsupported responses, identical chosen keys and
     `Unsupported` decisions, and max |Δp| ≤ 1e-4 (report the value).
   - On any mismatch, stop, fix the adapter and repeat once.
2. **Full run, once.** `python -m external_index021 run --suite-dir …/suite-0.2 --engine <module:Class> --out …/runs/<R>`,
   under one node A GPU lease. Resume only after an interruption; errors are retried,
   while `unsupported` and `abstained` results are final. The run covers 150,317 requests.
3. **Home sensitivity (optional diagnostic, not a gate).** Repeat step 2 with
   `--home-policy last` in resume mode (24 more requests), then run `sensitivity`.
4. **Score (CPU, about 1 minute each).** Score the same `results.jsonl` with the port
   (`score`, producing `index-021-provisional.json`) and with kit `87d4650b`
   (`score --edition 0.2.1`).
   - Gate: all 38 benchmarks agree within 1.5e-4 and the headline agrees within 0.01. On
     the four public runs the two scorers were identical.
   - If the audit found overlaps, report an override-adjusted value, with its declared
     forcing rule, next to the unadjusted one.
5. **Receipt (public-safe).** R's hashes; suite, scorer, engine and image hashes; counts of
   ok, unsupported, abstained and error; per-benchmark, area and headline values from both
   scorers; the audit result; median and p95 latency; GPU-hours; and the label. Rows,
   predictions and logs stay in the node A private directory.

## 6. Cost estimate

One node A GPU: AMD gfx942 (SKU M3250101, MI325X class), single process.

| Item | Requests | GPU-hours at 0.15–0.5 s/request |
| --- | ---: | ---: |
| Compatibility sample through the harness | 86 | 0.004–0.012 |
| The same 86 through the native reference path | 86 | 0.004–0.012 |
| Full run | 150,317 | 6.3–20.9 |
| Home second copies (optional) | 24 | ≤ 0.003 |
| **Total** | **150,513** | **about 6.3–20.9**, plus 0.1–0.3 h of load and warm-up per session |

- Sharding over k GPUs divides wall time only; GPU-hours stay the same.
- Requests average 2.15 questions. If R does one forward pass per question, cost scales
  with questions, so re-estimate from step 1's measured time before starting step 2.
- Scoring, the audit and the suite checks are CPU only.

**Optional peer adapter-fidelity check.** This is not needed for scorer validation, which
was done on CPU above. Run Hopper (G) 1.2 at `71d991f4` on node A with the frozen image
`decision20-train-fast:host2` (the runtime of its M2 native run). Wrap its packaged
decider in a kit `Engine`, run the same 86 requests, and compare each request with its
public run: expect at least 99% identical choices, report max |Δp|, and expect some CUDA
versus ROCm drift. This costs under 0.1 GPU-hours including load. A full peer replication
(150,317 requests, 6.3–20.9 GPU-hours) is not recommended.

## 7. Labelling and policy

- Label every result "independent provisional 0.2.1 reproduction". A result is an
  official 0.2.1 rank only if the maintainers accept it.
- Index results never merge into the JevArena main score, and never appear in rank
  charts or cards next to our same-panel numbers.
- Published Decision 1.0 rows from the same edition, pinned by Space revision, may appear
  only inside this external diagnostic.
- No checkpoint selection or tuning on the Index. Rows and predictions stay private on
  node A; publish only public-safe receipts.

## 8. Decisions for the coordinator

1. **Reference revision.** Keep the preregistered `ed7d683e`, or amend it to `7cdcea3d`.
   The two are protocol-identical: the newer one adds four entrants and rescores Jev, and
   the scoring does not change either way.
2. **Scorer of record.** I recommend dual scoring (port plus kit `87d4650b`) with the
   agreement gate in step 4.
3. **Contamination.** Approve the overlap-audit method, and decide whether to report an
   override-adjusted value.
4. **Home sensitivity.** Keep it as an optional diagnostic, or drop it.
5. **JPT runs.** Optionally request access to the gated JPT runs (manual approval,
   CC BY-NC).

**Remaining blockers for an official-equivalent claim**

1. No release is frozen, so the native adapter and the 86-request parity gate are not done.
2. The board's contamination lists and forcing rule are unpublished, so we can only
   self-audit.
3. Maintainer acceptance: the latency test on their hardware and review of a pull request.
4. The maintainers' per-row board results are private, so board parity is shown only for
   four public runs: two exact, and two exact after substituting the published overrides.
