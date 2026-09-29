# HT-DEV v1: human-transfer development panel — preregistration (2026-09-29)

Owner: eval track (development-panel worker; branch `xunzhuo/decision-2-training-eval-devpanel`).
Commissioned by the coordinator on 2026-09-29 04:30 UTC+8: human transfer is the binding axis for the
9B and 4B release gates, and the three-task CSS pilot cannot see it. This file is committed and pushed
before any HT-DEV source is converted and before any model is scored on it.

HT-DEV is a DEVELOPMENT panel. It is never a release score, never enters the v3 composite, the rank
charts or a model card, and its numbers are labelled "development readout".

## 1. Purpose and target

- Target axis: the formal human-transfer score H of post-key same-panel JevArena v3 (CSS15; the median
  over the 15 tasks of task macro-F1, `REPORT.json` → `panels.css15.H`).
- Question: does a broader, dataset-isolated panel of the same kinds of social-text judgments track
  formal H within a tier better than the CSS pilot (`H_pilot`, median macro-F1 of semeval_stance,
  implicit_hate, discourse)?

## 2. Panel design

Fourteen English tasks with human gold from the source datasets, in native Choice / Noul / Score form.
The kinds follow CSS15 and the pilot; dialect, power/status and politeness have no permissive,
human-labelled, isolated source (research record: candidates vetted 2026-09-29; formality stands in for
register). Revisions are the Hugging Face `main` SHAs observed on 2026-09-29 and are re-pinned (full
SHA) at download.

| # | Task key | Kind | Source @ revision | Licence | Type | Template (identical for every item) | Gold rule |
|---|---|---|---|---|---|---|---|
| 1 | `humor/nycc_pairs` | humor | `jmhessel/newyorker_caption_contest` (ranking configs) @d81cbab7 | CC BY 4.0 | Choice | State: cartoon description fields + Caption A / Caption B. Q: "Which caption would New Yorker readers find funnier for this cartoon?" | source ranking label (crowd votes) |
| 2 | `implicature/circa` | implicature | `google-research-datasets/circa` @faa1b5a7 | CC BY 4.0 (card) | Choice | State: situation, X's question, Y's answer. Q: "How would X most likely interpret Y's answer?" Options: Yes / No / Yes, subject to some conditions / In the middle, neither yes nor no | relaxed gold label (`goldstandard2`); drop Other, N/A, not sure |
| 3 | `figurative/figqa` | figurative | `nightingal3/fig-qa` @b29ec2fa (validation) | MIT | Choice | State: the statement + interpretation 1 / 2. Q: "Which interpretation expresses what the figurative statement means?" | source label (crowd-authored pairs) |
| 4 | `persuasion/ukpconvarg` | convincingness | `UKPLab/acl2016-convincing-arguments` (UKPConvArg1Strict) @ca9d24e4 | CC BY 4.0 | Choice | State: topic + stance + Argument A / Argument B. Q: "Which argument is more convincing for this stance?" | source crowd majority |
| 5 | `misinfo/pubhealth` | misinformation | PUBHEALTH (`neemakot/Health-Fact-Checking` @02136d3e, or its HF mirror; sha256 pinned) | MIT | Choice | State: the claim only. Q: "What verdict would professional fact-checkers give this health claim?" Options: True / False / Mixture (partly true) | fact-checker verdict; drop `unproven`; never show `explanation` |
| 6 | `emotion/brighter` | emotion | `brighter-dataset/BRIGHTER-emotion-categories` (eng) @419566ac | CC BY 4.0 | Choice | State: the text. Q: "Which emotion does the speaker mainly express?" Options: anger / fear / joy / sadness / surprise / no clear emotion | items with exactly one annotated emotion, or none (→ no clear emotion) |
| 7 | `stance/claim_stance` | stance | `ibm-research/claim_stance` @ec4e2c2e (test topics) | CC BY 3.0 (card) | Noul | State: topic + claim. Criterion: "The claim argues in favour of the topic." | PRO = yes, CON = no |
| 8 | `ideology/hyperpartisan` | ideology | SemEval-2019 Task 4 by-article (Zenodo 10.5281/zenodo.5776081) | CC BY 4.0 | Noul | State: title + body (first 2,000 characters, cut at a sentence end). Criterion: "The article is hyperpartisan: it argues for one political side in an extreme, one-sided way." | source by-article label |
| 9 | `deception/diplomacy` | conversational outcome | `DenisPeskov/2020_acl_diplomacy` @c28c0870 (test) | CC BY 4.0 | Noul | State: up to 3 preceding messages + the target message. Criterion: "The sender intends this message to deceive the receiver." | sender's own truthful/lie annotation |
| 10 | `moral/scruples` | social norms | `allenai/scruples` anecdotes @c79697e9 (test, then dev) | Apache-2.0 | Noul | State: title + story (first 2,000 characters). Criterion: "According to most readers, the author of this story is in the wrong." | `binarized_label` (WRONG = yes); vote counts never shown |
| 11 | `formality/pavlick` | register | `osyvokon/pavlick-formality-scores` @904009d6 (news, blog, email; no Yahoo Answers) | CC BY 3.0 | Score (5) | State: the sentence. Q: "How formal is the writing style of this sentence?" Levels: very informal / somewhat informal / neutral / somewhat formal / very formal | mean human score on −3..3, cut at −1.8 / −0.6 / 0.6 / 1.8 |
| 12 | `empathy/empathic_reactions` | empathy | `wwbp/empathic_reactions` @3ddbf6b3 | CC BY 4.0 | Score (5) | State: the message the writer wrote after reading a news story. Q: "How much empathic concern did the writer feel for the people in the story?" Levels: none / little / moderate / strong / very strong | writer's own empathy score (1–7), cut at 2.2 / 3.4 / 4.6 / 5.8 |
| 13 | `hate/mhs` | hate | `ucberkeley-dlab/measuring-hate-speech` @5468f6e1 | CC BY 4.0 | Score (3) | State: the comment. Q: "How hateful is this comment toward the group it mentions?" Levels: supportive or counter-speech / neutral or ambiguous / hateful | per-comment `hate_speech_score`: < −1, −1 to 0.5, > 0.5 (dataset card) |
| 14 | `wordsense/wic_tsv` | word meaning | `semantic-web-company/wic-tsv` @7bddf326 | CC BY 4.0 | Noul | State: sentence, target word, sense definition + hypernyms. Criterion: "In this sentence the target word is used in the given sense." | source label |

Backups, in this order, replace a primary task only if it fails an admission check in §3 (never after
any model is scored): figurative → MAGPIE (`gsarti/magpie`, Noul "the expression is used
figuratively"); persuasion → IBM-ArgQ-9.1kPairs (Choice); misinformation → RumourEval-2019 veracity
(Choice); emotion → XED English (Choice), then EmoBank (Score, 5 levels of valence); stance →
RumourEval-2019 stance (Choice); deception → CaSiNo satisfaction (Score); moral → MFRC (Noul), then
Moral Stories (Choice); hate → HateXplain (Choice); word meaning → Swords (Noul). Humor, implicature,
ideology, formality and empathy have no backup: a failing task is dropped. **At least 10 admitted
tasks are required**; with fewer, the build stops and is recorded.

Sampling (label-blind conversion, then label-stratified selection):

- Split preference: the source's test split, then validation, then train only if needed to reach the
  floor (recorded per task).
- Cap 250 items per task, floor 150. One item per source group where the source allows it; otherwise a
  per-group cap (UKPConvArg and claim_stance ≤ 10 per topic, hyperpartisan ≤ 3 per publisher,
  Diplomacy ≤ 10 per game, others 1–4) with groups recorded for cluster bootstraps.
- Gold is balanced across classes / levels as far as the pool allows; selection is seeded by
  `sha256("ht-dev/1:<task>:<item id>")` order.
- Option display order is a salted hash of the item id (`schema.display_order`, salt `ht-dev-display`);
  states hold only the source text fields listed above (no source names, vote counts, annotator
  fields, URLs or explanations).
- Length cues: for Choice tasks the C1 builder's length balancing is applied, and the length-only
  baseline's macro-F1 must stay ≤ chance + 0.10, else the task is re-drawn with length-stratified
  balance (one re-draw, recorded).
- Target size: about 3,000–3,500 items. `python3 -m v2.eval.leak_audit audit` runs on the frozen panel.

## 3. Admission checks (isolation and licence)

A source is admitted only if all of these pass; results are recorded as counts, never text.

1. **Licence** re-verified at the pinned revision against the authors' own statement (permissive
   only: CC0, CC BY, CC BY-SA, MIT, Apache-2.0, ODC-BY/PDDL; the table shows the card's licence).
2. **Dataset-level names / provenance check** (`v2.eval.sealed.independence names`) over every
   training registry, manifest, readme and record of all track branches and every provenance field of
   every training row file: the full private training-data snapshot at HEAD `75e557f1` plus older
   versions, local pools (`m3a2`, `m3b` incl. `arms-v3/r1` and `gap/c2`, teachers-v2, arms-v1/v2, A7
   runs, per-track derived files) and node-B derived files; the Decision 1.0 pools as documented
   (incl. the Kai research mix with Natural Instructions). Hard exclusions also cover CSS15, the pilot
   and the rest of the SALT package, typed FINAL/DEV, SELECT/CAL, public 231, mlx-diag (MASSIVE,
   PAWS-X, XNLI), Decision Bench v4, and every JevArena-C1 registry dataset (all 25 candidates;
   the rejected list as soft). A confirmed hit rejects the source.
3. **Source-level lexical overlap:** every row of every split of the source is the protected set
   (`independence rows`); every training corpus file is scanned with `v2.eval.sealed.overlap`
   (8-token exact spans, containment). The source counts as "in training" and is rejected if ≥ 5
   training rows match at containment ≥ 0.8 or share an 8+-token exact span with matching provenance.
   Weaker matching source rows are removed from the pool.
4. **Item-level lexical overlap** of the pool against the protected evaluation panels (gold-free
   prompts of typed-final, css15, public231, typed-dev, css-pilot, mlx-diag, named explicitly; never
   the sealed directory): OVERLAP (containment ≥ 0.5 or exact) and REVIEW (0.2–0.5) items are dropped
   before sampling.
5. **Embedding overlap** (`v2/data/embed_scan.py`, Qwen3-Embedding-0.6B @97b0c614, 1,500 / 1,000
   character windows): pool items against the protected evaluation panels and against the training
   rows of the social-text-adjacent training sources (GoEmotions, FLUTE, TweetEval pools, SocialIQA,
   OASST1, ArgQ-30k, the stance / dialogue families built from CSS-pilot errors, SELECT/CAL and any
   source flagged in step 3); items with cosine ≥ 0.93 are dropped; 0.85–0.93 is counted.
6. **JevArena-C1:** dataset-level only (no registry dataset used); the sealed directory is never read.
7. Peer-aggregator presence (tasksource and the other aggregators) is recorded as a disclosure flag;
   it does not exclude a source.

## 4. Scoring

- Per task: macro-F1 over the task's gold labels (Choice options, Noul yes/no, Score levels); a
  missing or invalid answer counts as wrong (C1 scorer semantics).
- **Primary: `H_dev` = median over admitted tasks of task macro-F1** (the same functional as formal H).
  Secondary: task mean, per-type means, quadratic weighted kappa for Score tasks.
- Panel noise: item bootstrap within tasks (2,000 draws, seed 20260929) for each model's `H_dev` and
  for pair differences.

## 5. Validation (does HT-DEV track formal H better than the pilot?)

Models: every model below that has a post-key same-panel `REPORT.json` and can be re-run from its
formal run manifest (same adapter spec, input limit, image and, for FLA models, the persisted autotune
cache) on node A:

- 0.6B: Kai1, Lex1, Bosun, GLiNER2.5-Decide, DEV2.0-0.6B (T soup), V2 soup, X soup, Z soup.
- 0.8B: Eos1, Intern, Kev, JPT-0.8B, E8F soup (DEV2.0-0.8B lineage), E8F s1, B8F s1, E8V soup.
- 2B: Sol1, Decider 2B, This-That 1.2, Bosun 1.7B, S2T soup (DEV2.0-2B).
- 4B: Nox1, Decider 4B, Jet v6.2, Hopper-G, JPT-4B, N4LKr, N4L, N4T, N4J soups, X2.
- 9B: Lux1 (16K comparator run), JPT-9B, Nimble v2, L2, B-s1, DW.
- 27B only if the weights are already on node A and a tier pair exists (optional; not required).

Each model is collected once with `--panels ht-dev,css-pilot,typed-dev` in one job, so `H_dev` and
`H_pilot` (and `T_dev`) come from the same runtime at the formal input limit. Formal H and v3 come from
the stored `REPORT.json`; no v3 item is re-read and C1 is never touched.

Metrics (within a tier unless stated):

- **Primary: pairwise sign agreement** of ΔH_dev with ΔH_formal over within-tier pairs with
  |ΔH_formal| ≥ 0.02 ("decidable pairs"), compared with the same statistic for `H_pilot`.
  Also reported at all pairs, ≥ 0.01 and ≥ 0.03.
- Secondary: Pearson r of tier-demeaned `H_dev` vs tier-demeaned formal H (within-tier correlation);
  cross-tier Spearman and Pearson; the same for `H_pilot` and the pilot three-task mean; agreement with
  the formal CSS15 task-mean macro-F1 as a lower-noise target.
- Uncertainty: paired bootstrap over models within tiers (5,000 draws, seed 20260929; duplicate draws
  of one model form no pair), giving P(HT-DEV better) for each metric.

**Decision rule:** HT-DEV "tracks formal human transfer better than the pilot" if (i) its primary
agreement is higher with bootstrap P(better) ≥ 0.90 and (ii) its within-tier Pearson r is higher
(point estimate). Otherwise the result is reported as not better, plainly, and nothing is published as
a recommended proxy.

## 6. Proxy recommendation (only if §5 passes)

- Candidates: `P_HT = 100·√(T_dev·H_dev)` (primary) and `100·√(T_dev·H_dev,mean)`; baseline: proxy v2
  `P = 100·√(T_dev·H_pilot)`, all from the same-job readouts. Target: post-key v3.
- Metrics as in proxy v2: within-tier decision pairs with |Δv3| ≥ 2 in v3 order (primary), Spearman,
  LOO RMSE of the linear map, panel noise.
- `P_HT` is recommended as the development proxy if its within-tier decision-pair agreement is at
  least P's (point estimate) on the same models. If it is lower, P stays the v3 shortlist proxy and
  `H_dev` is recommended as a separate human-transfer screen.
- **Tie bands:** for `P_HT` vs v3, the larger of the normal-residual 10%-risk gap
  (1.2816·√2·σ/b from the LOO linear fit, rounded up to an integer) and the smallest |ΔP_HT| bin edge
  whose empirical reversal-by-≥ 2 rate is ≤ 10%. For `H_dev` vs formal H, the normal-residual
  10%-risk gap from a fit with tier fixed effects, rounded up to 0.005.

## 7. Compute, storage and rules

- Build and scans: node A CPU; the embedding scan uses one short shared-lease GPU job (≈ 0.2 GPU-h).
- Scoring: short shared-lease jobs (`run_same_panel.sh --shared-lease owner.eval-htdev`) on node A,
  preferring GPU5, then GPU0–1, then GPU6–7; each job ≤ 30 min; memory headroom checked before launch;
  no running job is paused or disturbed. Budget cap 8 GPU-h in total; stop and record if exceeded.
- Panel files: `/data/dev2/private/panels/{goldfree,gold}/ht-dev.*` (gold mode 600), registered by
  hash in `v2/eval/panels.py` (`DEVELOPMENT["ht-dev"]`), with a private HF backup in the eval-artifacts
  dataset. No items, gold or source text in git or the gist.
- Any deviation is a dated amendment committed before the affected step runs.
