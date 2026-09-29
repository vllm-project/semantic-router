# DEV2.0-0.6B successor (`m8-s5-b05`): training-data credits from the M6 mixtures

The weights of `m8-s5-b05` are byte-identical to `m6-mxcx-soup`, the uniform soup of the six M6 seeds
(`m6-cx` and `m6-mx` × s1–s3). This directory credits every upstream source that has rows in the union of
the six seed train files. Counts, names, licences and hashes only: no row text was copied or printed.

## Result

- **232,754 distinct rows** (by `id`; `id` and `input_sha256` are 1:1 in the union). The per-file sum is
  356,331, because BASE sits in all six files and 97,389 rows are in both families.
- **89,771 project-generated rows** (Decision 1.0 stage 1–4 generators, Decision 2.0 programmatic and verifiable
  generators; no third-party text) and **142,983 rows from 37 public datasets**.
- **No source is non-commercial, research-only, of unknown licence or restricted.** Six excluded source keys have
  0 rows: MASSIVE (an mlx-diag source), FLUTE (a CSS15 task), and Cosmos QA and the SQuAD 2.0 answerability
  projection (both from Decision 1.0 and legacy).
- **Lux:** 330,143 teacher entries (141,261 for m6-cx and 188,882 for m6-mx) cover all 232,754 rows. The
  97,389 ids in both files have identical `input_sha256` and probabilities.
- **Package licence:** `package_licence` over the unchanged weight-lineage components (the DEV2.0-0.6B package
  and Qwen3-0.6B-Base@`da87bfb6`) returns `apache-2.0`. The training data are attributions, not lineage
  components, as on the current card.

`m6-sources.json` → `sources` has one entry per upstream dataset with its arms, distinct and per-file rows,
source keys, licence, flags and registry entries. `card` holds the ready-to-paste strings, and
`vs_current_card` the difference from the current spec.

## Method

1. **Count on node A (read-only, `python3 -c`; nothing written on node A).** `count_m6_sources.py` reads the
   six `m6-{cx,mx}-s{1,2,3}.train.jsonl` files, the two recipe ids files and the two Lux teacher files. It
   then:
   - counts rows once per `id` by the row's `source` field;
   - joins each id to its arm (`pool`) through the recipe ids files and requires the same `source` there;
   - splits `legacy:stage3_replay` by the builder-recorded `original_source` label into CLINC150 128,
     BANKING77 81 and project-generated 435 (the registry notes 435 generated rows);
   - checks teacher coverage, input hashes and agreement between the two teacher files.

   Before this, read-only probes listed only key names, `source` values and provenance labels. Every
   Decision 1.0 generated row carries a generator label, not a dataset.
2. **Build locally.** `build_m6_sources.py` does the following and fails on any mismatch:
   - maps the 55 source keys to 37 upstream datasets plus project-generated;
   - takes licences from the registries (`data/a7/license-registry-a7-v2.json`,
     `data/records/license-registry-{v2,m3b,v1}.json`, `data/a7/license-registry-a7-v1.json`, first match) and
     checks each card licence against the registry text;
   - sets the policy flags (NC, research-only, unknown, restricted) by regex over each registry's licence and
     redistribution text;
   - cross-checks against `m6-verify.json` and the six mixture reports (file hashes, rows, per-family and
     per-arm union rows, task types);
   - calls `v2/release/licence.py` `package_licence`.

Commands (run in this directory):

```bash
B=$(base64 -w0 count_m6_sources.py) && ssh -o ConnectTimeout=20 -o BatchMode=yes root@<node A> "python3 -c \"\$(echo $B | base64 -d)\"" > m6-union-counts.json
python3 build_m6_sources.py
```

## Inputs (SHA-256; full list in `m6-sources.json` → `inputs`)

| Input | SHA-256 |
| --- | --- |
| node A `m6-cx-s1/s2/s3.train.jsonl` | `ad925c37…86fc`, `420fd988…4489`, `9131be73…8aa5` |
| node A `m6-mx-s1/s2/s3.train.jsonl` | `a277bd45…8eae`, `5fafa72d…d8c0`, `e484c99b…f653` |
| node A `m6-cx.lux1.jsonl`, `m6-mx.lux1.jsonl` | `4be17f38…ec74`, `6b5d8ffa…48f2` |
| node A recipe ids `cx-xl-r2-a7v1-full`, `cx-xl-r2-nogap-full` | `dd141181…44b2`, `4a74a06e…3bca` |
| node A `inputs/INPUTS.json`, `m6-verify.json` (= the local record) | `df630280…068f`, `3bb733c5…291c` |
| `count_m6_sources.py` | `4a6fd0b924783371afd0293b987f74681bfcbdb3d761c9d562f5ebca653d80e5` |
| `m6-union-counts.json` (node A output) | `517183443984a45bd6e85c70204e4f697d5101bc345cc9ff44de9eac0eea1e92` |
| `build_m6_sources.py` | `08c2363a57d263e967cca0bd506f52563a4223604ab65f72498238c428aafefd` |

## Caveats

- **Licences are the registries' entries,** recorded by the data track with pinned evidence. They were not
  re-fetched from upstream today (no network use).
- **Credit follows the text.** Twins, evidence-removal, relevance and coverage rows are programmatic edits of
  upstream text, so they are credited to that upstream (for example SQuAD 2.0, HotpotQA, GSM8K and WinoGrande),
  not to project-generated tasks.
- **Informational flags (not policy flags):**
  - MultiNLI uses the non-SPDX OANC terms (7,168 rows).
  - QuAC's dataset card says MIT; the registry uses the official CC BY-SA 4.0 (2,244 rows).
  - SentiMix Hinglish and AfriSenti Swahili are tweets under the publishers' CC BY 4.0, and platform terms may
    also apply (12,860 rows).
  - TyDi QA, MLQE-PE, 2WikiMultihopQA and MuSiQue contain CC BY-SA Wikipedia text (21,289 rows).
  - 14 datasets are CC BY-SA (32,867 rows).

  None of these changes the package licence, and no data is redistributed.
- **The Score offsets are outside the M6 union.** The `m8-s5-b05` per-level Score offsets (five numbers) were
  fitted on the 400-row score5t-dev fit half, drawn fresh from the benchmark's own Score generator (no
  third-party text).
- **The current card's "Not used" line must change.** It names SQuAD 2.0 answerability, but M6 uses SQuAD 2.0
  passages as E11 evidence-removal twins (3,362 rows). Cosmos QA and the SQuAD 2.0 answerability projection
  are still absent.
- **The current spec says "three seeds"** in `origin.summary` and `card.text.details`. This release is six
  seeds: two recipes × three.
