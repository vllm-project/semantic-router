# IB3-r2: maths multiple choice with a per-option check (`m6/ib3/`)

> **Status: RELEASE-SAFE (preregistered blind review passed).**
>
> - 3 gold errors in 216 reviewed TRAIN rows: 1.39% (exact 95% CI 0.29–4.01%); population-weighted 1.39%.
> - Bars: ≤ 5.0% with an upper bound ≤ 8.0%, a weighted error ≤ 5.0%, and no family with an exact 95% lower bound
>   above 5%. All three hold.
> - The 3 rows are removed from TRAIN. Every audit passed. See `status.json`.
>
> This revision **replaces IB3 round 1** in this folder (round 1 had four families and failed its review; it stays at
> revision `c2401ab4`). IB3-r2 keeps only round 1's maths family. The phishing (`wpd`, `phiu`) and product-search
> (`esci`) rows are not here.

One family, `mqa`: is the proposed option the correct answer to a maths word problem?

- Every problem appears twice: once with the key option (yes) and once with another option (no). So yes = no for every
  problem and the proposed answer alone does not predict the label.
- The key is verified by executing the problem's annotated formula. A problem is kept only if exactly one option
  matches the result (within 1%) and that option is the key. The "no" option is more than 5% away from the result.

**No row of the Decision Index suite is included:** every candidate group with an exact, normalized or 13-gram match
to any suite row was dropped. No Jev-derived benchmark repository is a source. The preregistration is
`v2/data/records/ib3-r2-prereg-2026-10-02.md`, which builds on IB3's `ib3-prereg-2026-10-01.md` (branch
`xunzhuo/decision-2-training`). The rows are exactly round 1's `mqa` rows minus the review findings. No family is
in-distribution.

## Files

| File | Rows | Groups (problems) | Native tokens (Qwen3.5-0.8B-Base) |
| --- | ---: | ---: | ---: |
| `ib3.train.jsonl` (Noul) | 8,752 | 4,376 | 1,519,651 |
| `ib3.dev.jsonl` (split `select`; diagnostic only, never selection) | 426 | 213 | 73,517 |

The path also holds:

- per-row tokens (`*.tokens.jsonl`) and the freeze manifests;
- `build.json` (candidate build) and `final.json` (every drop, by reason);
- `stats.json` (sizes, labels and per-problem balance) and `isolation.json`;
- `audits/` (count-only receipts);
- `license-registry-ib3.json` (MathQA only), `status.json` and `registry.json` (the SHA-256 of every file).

## Source and licence

| Family | Target family | TRAIN | DEV | Source | Licence |
| --- | --- | ---: | ---: | --- | --- |
| `mqa` | maths MCQ, per-option check | 8,752 | 426 | MathQA (Amini et al. 2019) `train.json`; problems from AQuA-RAT (Ling et al. 2017) | Apache-2.0 |

No ShareAlike text. English only.

**Considered and not included:**

- Claim-level grounding: WiCE was dropped because its evidence text comes from third-party websites under no stated
  licence. VitaminC and FEVER were dropped because of per-article Wikipedia terms, as in earlier rounds.

## Construction and audits (counts only)

- **Index rows (G0).** The text leaves of every suite row (selected and added rows) are compared with IB3-r2's data
  leaves after NFKC normalisation, casefolding and `\w+` tokenisation (raw equality ≥ 20 characters, whole-leaf
  equality ≥ 6 tokens, sentence or leaf equality ≥ 8 tokens, any shared word 13-gram). 116 candidate groups were
  dropped whole. Positive controls: 2,000 / 2,000 suite rows rendered as candidates and 2,000 / 2,000 perturbed copies
  flagged.
- **Index URLs and hosts (G0u).** No candidate holds a URL. Positive controls: 500 / 500 and 500 / 500 flagged.
- **Held-out panels (G2).** Overlap scans against the protected inventory and a supplement (development panels,
  calibration rows, HR2 DEV, IB1 DEV of rounds 1 and 3, IB2 DEV):
  - 5 groups with a quarantining hit were dropped;
  - 294 DEV groups near TRAIN were dropped from DEV;
  - the re-scan of pass 1 and the final-file re-scan were clean.
- **C1 (G3).** MathQA is not a C1 registry dataset, and the source-term scan found none. The sealed directory was not
  read.
- **Names (G1).** No protected hit beyond three lineage lines that name another IB3 source.
- **Shortcuts (G4).** State removed .500 and option only .500, against a majority of .500. Proposed answer alone .543,
  under the .550 threshold.
- **Balance (G5).** yes = no for every problem. **Isolation (G7).** No shared id, group or input between IB3-r2 TRAIN,
  IB3-r2 DEV, IB1 DEV, IB2 DEV, HS1 dev, PN1 dev and HR2 DEV. **Leak guard:** clean.

## Blind review

- **No screen stage:** `mqa` passed IB3's screen (1 of 10).
- **Decisive review:** 216 rows (one per problem), none from a row or problem round 1 had sampled.
  - Two fresh reviewers answered every row blind, in two packets each, the second in another order. They agreed on
    214 of 216 (κ 0.98). A third reviewer answered the 2 splits.
  - **3 errors:** in each, the stored key was judged not to be the correct option (both reviewers on 1; the third
    reviewer on 2 splits).
  - Round 1's two `mqa` findings (one review error, one screen disagreement) are removed too.
- Limitation: the reviewers are one model family (AI adjudication, not human annotation).

## How to use

Add `ib3.train.jsonl` as one block against a matched-token control, alone or with IB1 / IB2. The DEV slice is a
diagnostic only. **Any C1-scored model trained on IB3-r2 needs the custodian C1 content recheck first.**
