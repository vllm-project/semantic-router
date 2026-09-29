# Own-Lux targets for the H7 / H8 rows of the XL r2 recipes — wave `h-w1` (2026-09-29)

This follows the [preregistration](m3b-lux-h-prereg-2026-09-29.md) (`2737f5d95`). The code is `5946bb753`: exact
mirrors on node A and node B, content manifest `fbcf7a4a…`, 2,484 files. Every check passed and the wave is published.
With it, **every row of all eight XL r2 recipes and controls has an own-Lux target**. That includes mx-xl-full-r2
(365,970 / 365,970) and mx-xl-short-r2 (373,577 / 373,577).

**Private revision `75e557f170979bdbc428b6ea698a2047e2d2a5cd`** of `llm-semantic-router/decision-2.0-training-data`,
tree `m3/teachers/lux1/xl/`. It adds `h-w1.{targets.jsonl,attestation.jsonl,report.json}` and `coverage-r2.json`, and
refreshes `README.md` ([source](hf-lux1-xl-readme.md)). Join targets by `id` to the H7 / H8 rows of the r2 recipes and
check `input_sha256`. The format is that of `w<k>.*` and `c-w<k>.*`.

## Wave

| Wave | Prompts | Answered | Over budget / invalid | Rows C / N / S | Argmax vs gold C / N / S | Wall | GPU-h |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| h-w1 | 25,664 (H7 4,292, H8 21,372) | 25,664 | 0 / 0 | 6,964 / 12,170 / 6,530 | 0.934 / 0.693 / 0.533 | 1,786 s | 0.496 |
| repeat (256 of h-w1) | 256 | 256 | 0 / 0 | — | — | 55 s | 0.015 |

- **Native tokens:** 28,498,792: H7 16,122,520 and H8 12,376,272.
- **Mean gold probability, C / N / S:** 0.885 / 0.666 / 0.512.
- **Brier, C / N / S:** 0.097 / 0.449 / 0.689.

| File | SHA-256 |
| --- | --- |
| `h-w1.targets.jsonl` | `679ef009aba751c23061b406319ba5c6f255089361fee5baefafaabac12dc2c3` |
| `h-w1.attestation.jsonl` | `9fc691d9aff89be5ecb8fd7d8f38c2849b1d4e40b546947730d3dd51c1a13409` |
| `h-w1.report.json` | `0c8d8d78dfedbe43d0a259ef28914df4d21a712ad287da8508797b902ae3f762` |
| `coverage-r2.json` | `ecb6dc36e19b5847d3b11d60d18a7f2d79190f680ec4fb41706e47b6ed629bab` |
| `README.md` | `26b484af6575acb61990f5b6ec2ee3af5445a9438be54a7ca2b0909b56fb9f5c` |
| teacher output (node A / node B copy) | `ff530f2f96a24dbfda4ad002a5c85976f09be41c10cf551630e9e6d40ed86a0a` |
| prompts `lux-xl-h-w1.prompts.jsonl` | `a535cd15c33041f6888b1bcead9714bf4cb09e03eaf023c71872f4ea82367f2d` |
| rows `lux-xl-h-w1.rows.jsonl` | `2c49cb658245a73ba0079d5052883626cbda5048b1b28e3b8874cc2533b80205` |
| repeat prompts `lux-xl-h-w1-r256.prompts.jsonl` | `53520513da9ff91a20d8156e198f683e136c7ba8c6a08fcab59d65ca02e70fa7` |

## Checks (all passed)

1. **Prompt set** (`v2/data/m3/luxxl_gap_nodeA.sh`, node A, CPU):
   - The node-A missing lists of mx-xl-full-r2 and mx-xl-short-r2 are byte-identical to the files at `10053613…`.
   - `h7.train.jsonl` / `h8.train.jsonl` match the r2 manifest (`7c4133b0…` / `1f19e5ab…`).
   - The prompt set is the union of both lists: 25,664 ids. All 17,945 short-recipe ids are in the full list.
   - Every id is an H7 / H8 TRAIN row. The build gave one wave.
2. **Guard** on the wave: PASS.
   - Not-a-row, not-TRAIN, prompt-differs, shared id, shared input hash, shared prompt digest: 0 each.
   - 29 protected basenames: the 27 of w1–w5 / c-w1 / c-w2 plus `h7.aho.jsonl` and `h8.aho.jsonl`.
   - 8 shared normalized states (informational).
   - The repeat file: PASS, 0 shared states.
   - No SHO slice was read.
3. **Transfer:** node A → local → node B, gzip, with SHA-256 checked locally and on node B; mode 0600.
4. **Frozen runtime:** checked by `v2/data/m3/luxxl_gap_queue.sh` (`02c232ee…`) before it started.
   - Image `decision20-lux-runtime:latest` = `sha256:ce895822…`.
   - Launcher `teach.sh` `49e50c6e…`, container source mirror `5f5cd80a…`.
   - **Triton cache** 1,762 files, tree `299151ab…`, identical at every check: before the wave (19:41:36 UTC), after it
     (20:11:22) and after the repeat run (20:12:17). It is recorded in `provenance.teacher_run.triton_cache.checks`.
5. **Run check** (`luxxl provenance`):
   - launcher rc 0;
   - `collected_now = input_items = 25,664`, `previously_completed = 0`, `over_budget_rows_now = 0`;
   - pinned Lux1 id and revision, attested revision, validated runtime, no runtime differences;
   - the collector log shows the causal-conv1d fallback, i.e. the image caveat below.
6. **Repeat check** (`luxxl repeat`): PASS, recorded in `provenance.repeat_check`.
   - 256 prompts, the first 256 in `sha256("m3b-lux-h-repeat:" + id)` order, run in a fresh process 20:11:22–20:12:17 UTC.
   - **256 / 256 answers bitwise-identical**: 0 validity and 0 argmax mismatches, max drift 0.0.
   - The M2 comparator agrees: 256 compared, max abs diff 0.0.
7. **Conversion** (`v2.data.m2.targets`, node A, exact mirror):
   - `no_target` is empty.
   - The attestation has one line per prompt with the fields of the earlier waves.
   - The path-leak check and the private-value guard on the report are clean.
8. **Readback:** the publish script downloaded the revision and every uploaded file matched.
9. **Independent re-download** (fresh directory on node A):
   - The script is `v2/data/m3/luxxl_verify_h.sh` (`a62374af…`). It does not use the `v2` modules. It ran from a local
     copy, byte-identical to the committed file, before that file was committed.
   - the five files equal the upload;
   - the target ids equal the union of the two r2 missing lists downloaded again from `10053613…`: full-r2 25,664 /
     25,664 and short-r2 17,945 / 17,945;
   - `input_sha256` equals the TRAIN row's for 25,664 / 25,664;
   - every `teacher_probs` is normalized;
   - 25,664 / 25,664 attestation lines carry the Lux id, attested revision `bd45a30a`, the validated runtime, node B,
     GPU 7 and image `ce895822`;
   - the tree at the new revision has 27 files, the `w1`–`w5`, `c-w1`, `c-w2` and `h-w1` waves plus README and
     coverage files;
   - the dataset is private.

**Image disclosure (unchanged):** image `ce895822…` has no causal-conv1d kernel, so Transformers uses its reference
PyTorch convolution. The eval track found that this changes 43 of 8,778 Lux1 argmax answers against the
kernel-equipped image. All own-Lux targets (RP-v2 waves 1–4, XL w1–w5, c-w1, c-w2, h-w1) come from this image.

## Coverage of the r2 recipes (`coverage-r2.json`)

The starting point is the r2 manifest's own-Lux counts. `c-w2` was still listed as pending there.

| Recipe | Rows | In the r2 manifest | After `c-w2` | After `h-w1` |
| --- | ---: | ---: | ---: | ---: |
| **`mx-xl-full-r2`** | 365,970 | 340,306 | 340,306 | **365,970 (100%)** |
| **`mx-xl-short-r2`** | 373,577 | 355,632 | 355,632 | **373,577 (100%)** |
| `cx-xl-r2-nogap-full` | 340,306 | 340,306 | 340,306 | 340,306 |
| `cx-xl-r2-nogap-short` | 355,632 | 355,632 | 355,632 | 355,632 |
| `cx-xl-r2-a7v1-full` | 191,536 | 191,536 | 191,536 | 191,536 |
| `cx-xl-r2-a7v1-short` | 167,067 | 167,067 | 167,067 | 167,067 |
| `cx-xl-r2-v2v1-full` | 295,998 | 261,782 | 295,998 | 295,998 |
| `cx-xl-r2-v2v1-short` | 282,419 | 257,563 | 282,419 | 282,419 |

- **Rows left without a target:** 0 in every recipe.
- **Upload gate:** `--require-full` on both release recipes.
- **For size tracks:**
  - H7 / H8 rows no longer have to train on gold labels only.
  - A run that trained r2 with gold-only H7 / H8 differs from one with these targets. State which one was used.
- **AutoJev:** targets for XL rows stay deferred.

## GPU-hours (node B GPU7) and timing

- **Teacher processes: 0.511 GPU-h** (wave 0.496, repeat 0.015).
  - The prereg predicted 0.39 + < 0.1.
  - The wave took 1.28× the time fitted to w1–w5; the long H7 rows are slower per token.
- **Lease:**
  - taken 19:41:35 UTC, queue finished 20:12:17, released 20:14:07;
  - the previous owner entry is saved in `/data/dev2/runs/data/m3b-lux/lease-owner-before-h-w1.json`;
  - no other job used GPU7.
- **Node A (CPU):** prompt build and guards 19:38–19:39 UTC; conversion, upload and readback 20:14–20:16 UTC.
- **Driver:** a committed copy of `luxxl_publish.sh` at `5946bb753` (`h1`), run locally 19:46–20:16 UTC.
  - Events: `/tmp/m3b/luxxl-publish.events.jsonl` (local) and node A's `…/m3b/lux-xl/publish.events.jsonl`.
  - Receipts on node A: `/data/dev2/runs/data/m3b/lux-xl/publish/h-w1/` (`published.json`, `readback.txt`,
    `repeat.json`, `provenance.json`, `triton-cache.jsonl`) and `…/lux-xl/verify-h-w1/`.
- **On node B:** everything was written under `/data`.

## Deviations

None from the preregistration.
