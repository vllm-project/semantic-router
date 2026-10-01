# IB3: third licence-clean breadth block for the Decision Index task families (`m6/ib3/`)

> **Status: NOT RELEASE-SAFE.** The decisive blind label review failed the preregistered bars:
>
> - 29 gold errors in 216 reviewed TRAIN rows: 13.43% (exact 95% CI 9.18–18.71%);
> - population-weighted 10.65% (bootstrap 95% CI 6.39–15.42%);
> - three families above the per-family bar: `phiu` 12 of 54, `wpd` 8 of 54, `esci` 8 of 54 (`mqa` 1 of 54).
>
> The bars are ≤ 5.0% with an upper bound ≤ 8.0%, a weighted error ≤ 5.0%, and no family with an exact 95% lower bound
> above 5%. The two reviewers agreed with each other on 214 of 216 items: the errors are rows whose label a reader
> cannot check from the row (mostly phishing URLs that look benign). The 29 rows are removed from TRAIN. Every audit
> passed. Do not use this block in a release candidate without a new, passing review. See `status.json`.

Training rows for task families where Decision 2.0 is weakest and that IB1 / IB2 could not cover:

- phishing links (`wpd`: a URL alone) and phishing pages (`phiu`: a site's homepage URL and page title);
- product-search relevance (`esci`: does a product exactly match a shopper's query; English, Spanish, Japanese);
- maths multiple choice with a per-option check (`mqa`: is the proposed option the correct one).

Every family is a yes / no question with yes = no inside every declared cell (URL shape, URL length, query, problem),
so no declared surface property predicts the label. The maths keys are verified by executing each problem's annotated
formula; the product items use only the Exact and Irrelevant classes; the phishing labels are the sources' blocklist
labels.

**No row of the Decision Index suite is included:** every candidate group with an exact, normalized or 13-gram match
to any suite row was dropped, and so was every group with a URL or host that occurs in any suite row. No Jev-derived
benchmark repository is a source. The preregistration is `v2/data/records/ib3-prereg-2026-10-01.md` with amendments 1
and 2 (branch `xunzhuo/decision-2-training`).

**In-distribution disclosure.** `esci` is the train split of the Shopping Queries Dataset, whose test rows the Index
uses. Only `split = train` rows were read, and every row matching an Index row was dropped. To measure transfer from the
other families alone, leave `esci` out.

## Files

| File | Rows | Groups | Native tokens (Qwen3.5-0.8B-Base) |
| --- | ---: | ---: | ---: |
| `ib3.train.jsonl` (Noul) | 24,148 | 14,507 | 4,652,323 |
| `ib3.dev.jsonl` (split `select`; diagnostic only, never selection) | 1,762 | 1,109 | 329,456 |

The path also holds:

- per-row tokens (`*.tokens.jsonl`) and the freeze manifests;
- `build.json` (candidate build) and `final.json` (every drop, by reason);
- `stats.json` (per-family sizes, labels and cell balance) and `isolation.json`;
- `audits/` (count-only receipts);
- `license-registry-ib3.json`, `status.json` and `registry.json` (the SHA-256 of every file).

## Families, sources and licences

| Family | Target family | TRAIN | DEV | Source (pin) | Licence | Gold |
| --- | --- | ---: | ---: | --- | --- | --- |
| `wpd` | phishing link | 3,700 | 346 | Web page phishing detection (Hannousse and Yahiouche 2021), Mendeley Data `c2gw7fy2j4` v3, `dataset_B_05_2020.csv` (URL and status only) | CC BY 4.0 | phishing (PhishTank / OpenPhish) vs legitimate; balanced inside every URL-shape cell |
| `phiu` | phishing page | 1,720 | 170 | PhiUSIIL Phishing URL dataset (Prasad and Chandra 2024; UCI 967; URL, title and label only) | CC BY 4.0 | phishing vs legitimate; only `https://www.<host>` homepages (the shape both classes share), balanced per URL length |
| `esci` | product-search relevance (in-distribution) | 9,970 | 820 | Shopping Queries Dataset (Reddy et al. 2022), `amazon-science/esci-data@7916cdf6`, `split = train` | Apache-2.0 | Exact (yes) vs Irrelevant (no); one of each per query |
| `mqa` | maths MCQ, per-option check | 8,758 | 426 | MathQA (Amini et al. 2019) `train.json` (problems from AQuA-RAT) | Apache-2.0 | yes: the key option (verified by executing the annotated formula); no: another option more than 5% from the result; one of each per problem |

Attributions (full lines in `license-registry-ib3.json`):

- Hannousse and Yahiouche 2021 (Web page phishing detection, Mendeley Data, DOI 10.17632/c2gw7fy2j4.3);
- Prasad and Chandra 2024 (PhiUSIIL Phishing URL dataset, UCI Machine Learning Repository);
- Reddy et al. 2022 (Shopping Queries Dataset, Amazon Science);
- Amini et al. 2019 (MathQA) and Ling et al. 2017 (AQuA-RAT).

No IB3 row carries ShareAlike text. Instructions and options are English; `esci` content is English (6,582 rows),
Spanish (1,490) or Japanese (1,898).

**Built but not in IB3** (no row here):

- Grounding: FaithDial (`fdial`, response alone .743 vs a majority of .477) and HaluEval QA (`haluqa`, answer alone
  .601 vs .482) were removed by the shortcut gate; the passage-swap redesign (`fdial2`) passed the gate but the screen
  found 3 of 10 disagreements, so it was dropped. Grounding is therefore not covered.
- Contracts: MAUD (`maud`) kept 86 rows in 11 contract groups after the exclusions (its excerpts near-duplicate
  protected panel rows), too few to review; contracts are not covered.
- Phishing email: no legitimate-email corpus with a clear licence was found to pair with the CC BY 4.0 Nazario
  phishing corpus.

## Construction and audits (counts only)

- **Index rows (G0).** The text leaves of every suite row (selected and added rows) are compared with IB3's data
  leaves after NFKC normalisation, casefolding and `\w+` tokenisation (raw equality ≥ 20 characters, whole-leaf
  equality ≥ 6 tokens, sentence or leaf equality ≥ 8 tokens, any shared word 13-gram): 628 candidate groups dropped
  whole. Positive controls: 2,000 / 2,000 suite rows rendered as candidates and 2,000 / 2,000 perturbed copies flagged.
- **Index URLs and hosts (G0u).** Every URL and e-mail host in every suite row, normalized: 103 candidate groups whose
  URL or host occurs there were dropped (98 of them `wpd`). Positive controls: 500 / 500 suite URLs and 500 / 500
  perturbed copies (scheme, `www.`, case, trailing slash) flagged.
- **Held-out panels (G2).** Overlap scans against the protected inventory and a supplement (development panels,
  calibration rows, HR2 DEV, IB1 DEV of rounds 1 and 3, IB2 DEV) dropped every group with a quarantining hit (145),
  then 2 more on the first re-scan; the second re-scan and the final-file re-scan were clean.
- **C1 (G3).** No IB3 source is a C1 registry dataset, and the source-term scan found none in the raw files or
  candidates. The sealed directory was not read.
- **Names (G1).** No source is a protected panel or C1 source; the three hits in panel records name HotpotQA as an
  existing training source (lineage).
- **Shortcuts (G4).** Every family in IB3 is within 0.05 of the cross-validated majority on the state-removed,
  option-only and hypothesis-only views (`esci` product alone .515 vs .500; `mqa` proposed answer alone .543 vs .500).
  For `wpd` / `phiu` the URL-shape cell baseline is .432 / .466.
- **Balance (G5).** yes = no in every family and inside every declared cell.
- **Isolation (G7).** No shared id, group or input between IB3 TRAIN, IB3 DEV, IB1 DEV, IB2 DEV, HS1 dev, PN1 dev and
  HR2 DEV.
- **Leak guard.** URLs containing an IPv4 address were removed at build time; the final files and this tree are clean.

## Blind review

- **Screen:** 10 TRAIN rows per family (50) by one fresh reviewer: 5 disagreements (`fdial2` 3, `mqa` 1, `phiu` 1);
  `fdial2` dropped.
- **Decisive review:** a fresh sample of 54 rows per family (216; no screened row or group), answered blind by two
  fresh reviewers (two packets each, the second in another order) and a third on the 2 split items.
  - 29 errors: `phiu` 12 and `wpd` 8 (all labelled phishing, read as legitimate by both reviewers), `esci` 8 (6 Exact
    read as not exact, 2 Irrelevant read as exact), `mqa` 1 (a key both reviewers rejected).
- Limitation: the reviewers are one model family (AI adjudication, not human annotation).

## How to use

Not release-safe: use only for diagnostics or as a candidate for a further, preregistered clean-up round with a fresh
review. `mqa` (1 error in 54) is the family closest to the bar. The DEV slice is a diagnostic only. Any C1-scored model
trained on IB3 needs the custodian C1 content recheck first.
