# IB2: second licence-clean breadth block for the Decision Index task families (`m6/ib2/`)

> **Status: RELEASE-SAFE.** The decisive blind label review passed every preregistered bar:
>
> - 5 gold errors in 216 reviewed TRAIN rows: 2.31% (exact 95% CI 0.76–5.32%);
> - population-weighted 1.41% (bootstrap 95% CI 0.33–2.81%);
> - no family above the per-family bar (the most was 3 of 36, `ytspam`).
>
> The bars are ≤ 5.0% with an upper bound ≤ 8.0%, a weighted error ≤ 5.0%, and no family with an exact 95% lower bound
> above 5%. The 5 rows are removed from TRAIN. Every audit passed. Before a model trained on IB2 is scored on C1, the
> custodian C1 content recheck is required. See `status.json`.

Training rows for task families where Decision 2.0 is weakest and that IB1 could not cover:

- tool-call decisions: whether one of the available functions can carry out a request, and whether the request
  already holds every required argument;
- spam;
- stance;
- fact verification against Wikipedia passages, with more supported than unsupported claims;
- maths answer checking.

IB2 prefers labels a reader can check: function relevance from the schemas, argument presence by literal grounding,
curated spam labels, unanimous stance labels and numeric answers.

**No row of the Decision Index suite is included:** every candidate group with an exact, normalized or 13-gram match
to any suite row was dropped. The preregistration is `v2/data/records/ib2-prereg-2026-10-01.md` with amendments 1 and
2 (branch `xunzhuo/decision-2-training`).

**In-distribution disclosure.** `hover` (HoVer `train`) and `gsm2` (GSM8K `train`) are the train splits of datasets
whose evaluation rows the Index uses. The HoVer dev and test files were deleted unread, and every row matching an Index
row was dropped. To measure transfer from the other families alone, leave these two families out.

## Files

| File | Rows | Groups | Native tokens (Qwen3.5-0.8B-Base) |
| --- | ---: | ---: | ---: |
| `ib2.train.jsonl` (Choice 5,152, Noul 19,366) | 24,518 | 20,054 | 5,424,733 |
| `ib2.dev.jsonl` (split `select`; diagnostic only, never selection) | 1,272 | 1,212 | 216,180 |

The path also holds:

- per-row tokens (`*.tokens.jsonl`) and the freeze manifests;
- `build.json` (candidate build) and `final.json` (every drop, by reason);
- `stats.json` (per-family sizes, labels and balance) and `isolation.json`;
- `audits/` (count-only receipts);
- `license-registry-ib2.json`, `status.json` and `registry.json` (the SHA-256 of every file).

## Families, sources and licences

| Family | Target family | Type | TRAIN | DEV | Source (pin) | Licence | Gold |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| `fc_rel` | tool call or abstain | Noul | 5,732 | 0 | Glaive function-calling v2 `glaiveai/glaive-function-calling-v2@e7f4b645` | Apache-2.0 | yes: the first assistant turn calls a listed function (2,866); no: it refuses (150), or the list is replaced by schema-unrelated functions (2,716) |
| `fc_ready` | call or clarify | Noul | 2,582 | 0 | same | Apache-2.0 | yes: every required argument is literally in the request; no: one is supplied only in a later user turn |
| `ytspam` | spam | Noul | 1,450 | 92 | YouTube Spam Collection (UCI 380) | CC BY 4.0 | the authors' curated spam / ham labels |
| `argq` | stance | Choice (2) | 5,152 | 408 | IBM-ArgQ-Rank-30kArgs `ibm-research/argument_quality_ranking_30k@590726b3` `train` | **CC BY-SA 3.0** | stance where all 10 annotators agreed; support = oppose within every topic |
| `hover` | fact verification (in-distribution) | Noul | 4,020 | 200 | HoVer `hover-nlp/hover@39b84697` `train` (2- and 3-hop) with HoVer-wiki introductions | **CC BY-SA 4.0** | SUPPORTED / NOT_SUPPORTED; yes : no = 3 : 2 by design |
| `gsm2` | maths answer checking (in-distribution) | Noul | 5,582 | 572 | GSM8K `openai/gsm8k@740312ad` `main/train` | MIT | yes: the stated answer is the problem's answer; no: it is another problem's answer |

Attributions (full lines in `license-registry-ib2.json`):

- Glaive AI (glaive-function-calling-v2; synthetic conversations);
- Alberto, Lochter and Almeida 2015 (YouTube Spam Collection, UCI Machine Learning Repository);
- Gretz et al. 2020 (IBM-ArgQ-Rank-30kArgs; (c) IBM 2014, CC BY-SA 3.0);
- Jiang et al. 2020 (HoVer; evidence text from Wikipedia, by Wikipedia contributors, CC BY-SA);
- Cobbe et al. 2021 (GSM8K).

**ShareAlike.** `argq` and `hover` are CC BY-SA. They are credited here, and their rows can be left out by `family`
or `source`. ArgQ-30k, HoVer and GSM8K are already in earlier Decision 2.0 mixtures (disclosed lineage).

All text is English. DEV has no `fc_rel` or `fc_ready` rows: every DEV request group shares function schemas with a
TRAIN group, and the DEV-vs-TRAIN rule drops it.

**Built but not in IB2** (no row here):

- Removed by the shortcut gate (the text without the decisive part predicts the key above chance):
  - function selection (`fc_sel`, option-only .874 vs .246);
  - argument choice (`fc_args`, option-only .718 vs .325);
  - their frequency-matched redesigns (`fc_sel2` .472 vs .259; `fc_args2` .490 vs .329);
  - GSM8K with intermediate-result distractors (`gsm`, stated answer alone .564 vs .493);
  - QASC (option-only .321 vs .126) and ARC train (option-only .313 vs .250).
- Emptied by the Index-row and panel exclusion: ContractNLI train (shared NDA boilerplate).

Function selection and arguments, knowledge MCQ and contracts are therefore not covered. Phishing email is not
covered either: no legitimate-email corpus with a clear licence was found to pair with the CC BY 4.0 Nazario phishing
corpus.

## Construction and audits (counts only)

- **Index rows (G0).** The text leaves of every suite row (selected and added rows) are compared with IB2's data
  leaves after NFKC normalisation, casefolding and `\w+` tokenisation. A group is flagged on any of:
  - raw equality (≥ 20 characters);
  - whole-leaf equality (≥ 6 tokens);
  - sentence or leaf equality (≥ 8 tokens);
  - any shared word 13-gram.

  2,977 candidate groups were dropped whole. The positive controls flagged 2,000 / 2,000 suite rows rendered as
  candidates and 2,000 / 2,000 perturbed copies. Re-scans of the finalized files found no new Index match.
- **Held-out panels (G2).** Overlap scans against the protected inventory and a supplement (development panels,
  calibration rows, HR2 DEV and IB1 DEV) dropped every group with a quarantining hit, then a re-scan of the finalized
  files dropped 41 more; the second re-scan was clean, and one more group was dropped after the final re-scan.
- **C1 (G3).** No IB2 source is a C1 registry dataset, and the source-term scan found none in the raw files or
  candidates. The sealed directory was not read.
- **Names (G1).** No source is a protected panel or C1 source; the four hits in the panel records name ArgQ-30k and
  HoVer as existing training sources.
- **Shortcuts (G4).** Every family in IB2 is within 0.05 of the cross-validated majority on the state-removed,
  option-only and hypothesis-only views.
- **Balance (G5).** Noul families 50 / 50 (`hover` 60 / 40 by design); `argq` support = oppose in every topic.
- **Isolation (G7).** No shared id, group or input between IB2 TRAIN, IB2 DEV, IB1 DEV, HS1 dev, PN1 dev and HR2 DEV.
- **Leak guard.** One DEV row with an IPv4-like string was dropped; the final files and this tree are clean.

## Blind review

- **Screen:** 10 TRAIN rows per family (60) by one fresh reviewer: 0 disagreements; no family dropped.
- **Decisive review:** a fresh sample of 36 rows per family (216; no screened row or group), answered blind by two
  fresh reviewers (two packets each, the second in another order). They agreed on every item, so no third reviewer
  was needed.
  - 5 errors: `ytspam` 3 (comments labelled spam that both reviewers read as on-topic), `hover` 2;
  - 0 in `fc_rel`, `fc_ready`, `argq` and `gsm2`.
- Limitation: the reviewers are one model family (AI adjudication, not human annotation).

## How to use

Add `ib2.train.jsonl` as one block against a matched-token control (alone or with IB1), screened with HT-DEV v2. To
measure transfer from the other families alone, leave out the in-distribution families (`hover`, `gsm2`). To avoid
ShareAlike text, leave out `argq` and `hover`. The DEV slice is a diagnostic only.
