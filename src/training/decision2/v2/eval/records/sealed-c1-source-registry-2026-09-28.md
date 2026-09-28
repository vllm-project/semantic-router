# JevArena-C1 sealed confirmation set — Option B source registry (2026-09-28)

Scope: public, human-labelled datasets usable as gold for the sealed set (no paid annotation).
Companion JSON: `sealed-c1-source-registry-2026-09-28.json` (same fields, plus rejected entries).
Design context: `sealed-confirmation-set-design-2026-09-28.md` §5b.

## Rules applied

- **Minimum first public release: 2026-06-01.** Anything first released after **2026-08-15** is marked **preferred (★)**.
- Release date evidence is primary: HF `createdAt` plus the first entry of `/api/datasets/<id>/commits/main`, the arXiv v1 date, or the GitHub first commit via `commits/<branch>.atom`. "Revision" is the HF `main` SHA observed on 2026-09-28. Pin it.
- Access was public and unauthenticated only. Gated repos (they need an HF login to accept terms) are flagged as a blocker.
- Labels must be human: crowd, expert or organic (votes, ratings, verdicts). Model-generated labels are rejected unless the text states that humans verified them.
- Old-text flag (⚠txt): the labels are new but the underlying text was public before 2026-06-01, so the text itself may be memorised. These sources are kept with care, but never graded A when the old label is also recoverable.
- Time-filter (⏱): only rows dated on or after 2026-06-01 qualify, so the usable size must be recomputed at the pinned revision.
- Grades: **A** ready; **B** usable with care; **C** reject.
- Every project-used source and every peer JEV aggregator was excluded. The project-used list is MASSIVE … Decision Index. The peer aggregators include tasksource/tasksource-jev-typed-decisions (670 sources), Praveenrajus/jev-bench, AlexWortega/openjev-data, hachiko85/openjev-ja-eval, telepatia-ai/typed-decisions-pt-es, najdresearch/system-one, vagmi/jevlite_dataset, hayriyigit/jev-bench-tr and tasksource/* re-uploads. Their source lists must be diffed against this registry before sealing.

## Candidates (A/B)

| # | Dataset @ revision | First release (evidence) | Licence | Label provenance | Task → native type | Lang | Usable rows | Input len | Grade |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `Bekhouche/HalluTruthQA-4K` @ e6e2b147 | 2026-07-27 (HF commits; arXiv 2607.20219) | cc-by-nc-nd-4.0 ⚑NC | Domain experts, independent reviewer verification, expert adjudication (ArabicNLP 2026 HalluScoring shared task) | Hallucination detection → **Noul**; 6-option answer → **Choice** | ar | test 1,600 (683 hallucination / 917 not) | short (Q ~50 ch, answer ~170 ch) | **A** |
| 2 | `Sami2305341176/JudgmentBench` @ 826035f8 ★ | 2026-09-25 (HF commits) | MIT | 53 legal professionals; 1,530 pairwise judgements plus 1,539 rubric annotations; GPT autograder files kept separate | Pairwise preference → **Choice** A/B; rubric total → **Score** (bin to 5) | en | 30 tasks, 2,274 outputs | long (legal work product; many ≥4k) | **A** |
| 3 | `dreadnode/scopejudge` @ b8b06a65 | 2026-07-30 (HF commits) | MIT | 5 professional security experts; Fleiss κ 0.641 | Tool call within authorised scope? → **Noul** | en | 100 trajectories / 4,897 calls | long (trajectory context) | **A** |
| 4 | `SpaceHunterInf/DeliChess` @ 277393d5 | 2026-08-01 (HF; arXiv 2606.04987) | MIT | Human annotators per the guidelines; Gemini labels kept in separate columns (drop them) | Communicative function → **Choice** (9); epistemic stance → **Choice** (3) | en | 7,667 utterances / 107 dialogues | short utterance + dialogue context | **A** |
| 5 | `jaelly/MCJudgeBench` @ c64832ef ★ | 2026-09-06 (HF); 141 of the instances are in a GEM 2026 paper | apache-2.0 | Human gold labels per constraint; perturbations human-reviewed | Response satisfies constraint? yes / partial / no → **Score** (3) or **Choice** | en | 219 instances (78 new since release) | medium (instruction + response) | **A** |
| 6 | `Flaglab/esnlir-al-annotated-test` @ d9c734a4 | 2026-08-06 (HF) | cc-by-4.0 | 12 annotators, 3 per item, majority vote; κ 0.641 | Relation between sentences → **Choice** (contrasting / entailment / reasoning / neutral) | es (native) | 1,695 (min class 312) | short (~150 ch each) | **B** ⚠txt |
| 7 | `BRi002/SentiTaglishProductsAndServices` @ ea82e019 ★ | 2026-09-02 (HF) | cc-by-4.0 | 2 annotators plus a third adjudicator; Fleiss/Krippendorff reported | Review polarity → **Score** (3: neg / neu / pos) | tl + en code-mixed | 10,510 (neutral only 262) | short (~140 ch) | **B** ⚠txt |
| 8 | `Hplss/wb-review-dataset` @ c207a9df | 2026-07-30 (HF) | cc-by-nc-sa-4.0 ⚑NC | Organic star ratings written by review authors | Star rating from review → **Score** (5) | ru | 27,248; ⏱ rows ≥ 2026-06-01 are a subset (max date 2026-07-25) | short (~170 ch) | **B** ⏱ |
| 9 | `teplitsa-soc-tech/factbutcher-benchmark` @ 211fb80c | 2026-08-06 (first commit; created 2026-08-09) | cc-by-4.0 | Human-reviewed verdicts; 149 rows from Provereno.Media fact-checkers | Claim verdict → **Choice** (TRUE / FALSE / MIXED) | ru | 423; ⏱ reference_date runs 2021 to 2026-07 | short (~90 ch) | **B** ⏱ |
| 10 | `CLS-Lab/narrative-gold-annotations` @ f22dd4d5 | 2026-06-17 (HF) | cc-by-4.0 | Adjudicated gold plus 1–2 annotators | Setting concreteness / temporal grounding → **Score** (5); event causality → **Choice** (3) | en | setting 400, agency 400, events 440 | short (~350 ch) | **B** ⚠txt (Dolma) |
| 11 | `allenai/tutormoments-preview` @ 66058b4c | 2026-07-06 (HF) | cc-by-4.0 | Human turn-level annotations (1,584 passes over 263 transcripts); `ground_truth` is LLM-assisted, so use only the `annotations` config | Scaffolding / rapport moment type → **Choice**; moment present? → **Noul** | en (a few es) | 462 real transcripts | long (full sessions) | **B** |
| 12 | `lbrenap1/mining-legal-arguments-us-corporate-case-law` @ 05bf4bb8 ★ | 2026-08-30 (HF; arXiv 2609.25441) | cc-by-4.0 | Expert annotation; 10 cases double-annotated (`iaa_*` configs) | Argumentative function of a sentence in context → **Choice** | en | 5,286 sentences / 42 opinions | long (full opinions) | **B** ⚠txt |
| 13 | `McGill-NLP/ImplicatureX` @ 959e5e27 | 2026-07-31 (HF; arXiv 2607.25094) | MIT | Prolific raters (2,285 likelihood ratings) plus 2 experts on 54 items | Likelihood that the implicature holds → **Score** (Likert) | en | 271 items | short | **B** |
| 14 | `laallein/ClimateCause` @ 7ab75d7c ★ | 2026-08-31 (HF; Findings ACL 2026) | cc-by-sa-4.0 | Expert annotators | Causality explicit or implicit? → **Noul** | en | 874 (593 E / 281 I) | short-medium | **B** ⚠txt (IPCC) |
| 15 | `NordosoftOy/innoduel-rlhf-real-world-human-preferences-sample` @ 99f8810b ★ | 2026-08-23 (HF) | cc-by-nc-4.0 ⚑NC | Real organisational users' pairwise votes on human-written ideas | Which idea is preferred? → **Choice** (2) | fi, sv, en, uk | 1,350 | short | **B** (small, single vote) |
| 16 | `Nandan007/NepFakeV2` @ 903ad9f5 ★ | 2026-09-12 (HF; updated weekly, so pin) | cc-by-4.0 | Professional fact-checker verdicts (organic) | Claim verdict → **Noul** (false vs not false) | ne | ⏱ post-2026-06 subset; over 90% FALSE | short | **B** ⏱ |
| 17 | `alisa-yingjia-wan/gapa` @ cecb0381 ★ | 2026-09-16 (HF) | MIT | Prolific ratings with attention checks (`human` split 2,300 ratings) | Gender association of an attribute → **Score** (rating scale) | en | ~100–300 aggregated attributes | very short | **B** |
| 18 | `HiTZ/EusExams-v2` @ 0b765000 | 2026-06-25 (HF) | cc-by-sa-4.0 | Official public-service exam answer keys (expert, organic) | Exam MCQ → **Choice** (4) | es, eu | es_health 11,387, but ⏱ only 2026 sittings qualify (dates unverified) | short | **B−** ⏱ |
| 19 | `LunaTsai/tiktok-political-stance-dataset-taiwan-2025` @ bac26ad9 | 2026-06-23 (HF) | **none stated** ⚑ | Human-annotated stance (arXiv 2607.15240) | Stance toward target → **Choice** | zh-TW | 9,302 | short | **B−** (licence) |
| 20 | `NUAA-MMMI/VARM-Bench` @ 2c802332 | 2026-08-13 (HF) | **none stated** ⚑ | Annotation policy described; provenance not fully clear | Abusive or not → **Noul** | zh | ~8k (test split) | short | **B−** (licence) |
| 21 | `giuseppe-aiello/stance-detection-it-dataset` @ 2e4c575a ★ | 2026-08-31 (HF) | MIT | Single annotator, 500 manual labels | Stance → **Choice** (Consenso / Dissenso / Altro) | it | 500 | short | **B−** |
| 22 | `lab-flair/constructcie` @ 80d171fe | 2026-07-14 (HF) | apache-2.0 | Manual annotation of OSHA narratives | Accident type → **Choice** | en | 530 | medium | **B−** ⚠txt |
| 23 | `HiTZ/safety-GuardEUS` @ f5a05277 | 2026-06-25 (HF) | apache-2.0 | Human safety labels | Response safe? → **Noul** | eu | 250 | short | **B−** (tiny) |
| 24 | `iamjayeshc/reddit-self-medication-claim-dataset` @ 839db5ff | 2026-08-01 (HF) | cc-by-4.0 | Single annotator (the author) | Contains a self-medication claim? → **Noul** | en | 1,215 | short-medium | **B−** ⚠txt |
| 25 | `ministere-culture/comparia-fr-arena` @ 870a0bfa | 2026-06-11 (HF) | etalab-2.0 / cc-by-4.0 | Organic user votes (French public arena) | Which answer is better? → **Choice** (A / B / tie) | fr | large | long conversations | **B** ⛔gated (login needed) |

Evidence URL pattern: `https://huggingface.co/api/datasets/<id>/commits/main`, where the oldest entry is the first commit. The full SHAs are in the JSON.

### Label-blind templates (native type, fixed options or levels)

| # | Template |
|---|---|
| 1 | Noul: "Does the answer contain a factual hallucination relative to the question?" yes/no. Choice: "Which option answers the question correctly?" A–F |
| 2 | Choice: "Which of the two legal work products better completes the task?" A/B. Score: "Rate overall quality against the task" 1–5 |
| 3 | Noul: "Is this tool call within the authorised engagement scope?" yes/no |
| 4 | Choice: "What communicative function does this utterance serve in the deliberation?" (9 fixed functions) |
| 5 | Score: "Does the response satisfy the constraint?" no / partial / yes |
| 6 | Choice: "How is sentence 2 related to sentence 1?" contrasting / entailment / reasoning / neutral |
| 7 | Score: "Overall sentiment of the review" negative / neutral / positive |
| 8 | Score: "Which star rating did the reviewer give?" 1–5 |
| 9 | Choice: "Verdict for the claim" true / false / mixed |
| 10 | Score: "How concrete is the setting of the passage?" 1–5. Choice: "Causal relation between event 1 and event 2" direct cause / enables / not related |
| 11 | Choice: "Which tutoring move does this moment exemplify?" (fixed moment types) |
| 12 | Choice: "What is the argumentative function of the marked sentence?" (fixed scheme labels) |
| 13 | Score: "How likely is it that the speaker meant the implicature?" (Likert levels) |
| 14 | Noul: "Is the causal relation stated explicitly?" yes/no |
| 15 | Choice: "Which of the two ideas did the participant prefer?" A/B |
| 16 | Noul: "Was this claim rated false by fact-checkers?" yes/no |
| 17 | Score: "How strongly is the attribute associated with the person term?" (rating levels) |
| 18–25 | Native MCQ / stance / abusive yes-no / accident type / safe yes-no / claim yes-no / arena A/B/tie |

### Shortcuts and risks (by candidate number)

- **1:** One generator model (Fanar). The train/dev splits are public and the shared-task test labels are included, so peers can fine-tune on train. Answer length correlates with the label, so balance length when sampling.
- **2:** Only 30 tasks, so there are only 30 source groups. Quality levels were constructed (`quality_level_order`), so use the human `preferred_option` and the rubric totals, never the construction level. The GPT autograder files must be excluded.
- **3:** The trajectories are agent-generated text; only the labels are human. Within a trajectory, calls are correlated, so take 1 per trajectory. Check class skew.
- **4:** Utterances are short and depend on context, so include the preceding turns. Drop the `gemini_*` and `*_match` columns because they leak the label.
- **5:** Only the 78 newly added instances are strictly new. The 141 paper instances may already be public through the paper appendix.
- **6:** Items were sampled by model confidence, and only items where the human label matched the connector-derived label were kept. The connector word reveals the class, so mask it. The text comes from older corpora.
- **7:** Review dates are unknown and the neutral class is tiny, so a balanced sample is at most ~260 items per class. Short text.
- **8:** Filter by `date`. The rating is heavily skewed towards 5, so balance by level. Take 1 review per product. NC licence.
- **9 and 16:** Verdicts may already appear on fact-check websites. Take only claims dated on or after 2026-06-01. NepFake is extremely imbalanced.
- **10:** The passages come from Dolma, which is in many pretraining sets. The labels are new.
- **11:** Private K-12 sessions from 2026. Use only the human `annotations` config; `ground_truth` and `benchmark` are LLM-assisted or synthetic.
- **12:** Only 42 opinions exist, all public US tax cases, so the source groups are few.
- **13 and 17:** The items may be researcher- or LLM-authored stimuli; ImplicatureX has LLM variants (`implicatureBot`). Use the human-authored config only. Aggregate the ratings.
- **14 and 22:** The underlying text is old and public (IPCC, OSHA).
- **15:** One vote per pair makes the label noisy. NC licence.
- **18:** Only exams sat on or after 2026-06-01 qualify; the sitting dates are not verified. Answer keys are public on official sites.
- **19 and 20:** No licence is stated, so obtain licence clarification before use.
- **25:** The repo is gated behind an HF login, which violates public unauthenticated access, so it is blocked.

## Rejected (C)

| Dataset / source | Reason |
|---|---|
| cltl/toxirex (arXiv 2606.27981) | GitHub first commit and data added 2026-04-08/15, before the cutoff. Otherwise ideal (6 languages, native test set) |
| gabrielstefan04/polero | HF created 2026-05-05, before the cutoff |
| ilsp/panellinies-exams-dataset | Exam years 2020–2025 (public before the cutoff); no licence |
| GermEval 2026; CheckThat! 2026 | Gold labels released 2026-05-25 and 2026-05-20, before the cutoff |
| chnln/seeing-is-not-sharing | MapTask text; labels from a 2025 paper (arXiv 2511.x) |
| cardiffnlp/GrainWiC | Derived from older sense inventories and corpora; labels derivable |
| TrustworthyComp/ClaimReview2025Q4 | Verdicts published on the web in 2025 Q4; NC licence |
| jumafernandez/consultas-unlu | Emails from about 2019, released earlier on GitHub |
| AvinabhDutta-Dev/assamese-movie-reviews-sentiment | About 60% of the texts were machine-translated or normalised |
| UKPLab/ProReviewer, Samarth0710/reviewarena, zai-org/SurveyReview | Reviews public before the cutoff (OpenReview or journals); PeerRead-like; SurveyReview has no licence tag and derived scores |
| UOM-CSE-E23 Sri Lankan tourism; fl4wn/respolitica; bil-y/polidata-de | Old public text or labels (Mendeley 2023; voting-advice-application positions 2017–2025) |
| ytu-ce-cosmos/absa-tr, YachayWiki/quechua-collao-sentiment, postovyi/disarm-ukrainian-telegram, niekbiesterbos/dutch-climate-parl, ictchenbo/public-discourse-corpus, ethicalabs/Research-Intent-Collab, DavidYor06/llm-disagreement, swiss-ai Apertus preference, s-nlp/EnokiQA, Thai reviews (JarBenjaporn) | Labels generated by an LLM or an automatic pipeline |
| Roblox PII benchmark; ralipanah/email-politeness-corpus; nutrient grounding sets | Synthetic or machine-translated text |
| verdict-reviews; Kenpache/financial-sentiment-eval-7lang | 44.6% of ratings LLM-inferred, with no text; label provenance unclear |
| fact-den Ctrip sample; Zaevlad/audit-findings; prospex-ch; akashnaren/agent-ui-human; sovrano-ai workshop; 81melody Algerian real estate | "Other" or unclear licence, scraped data, 16–800 rows, a single author, or semi-rule-based labels |
| llm-jp/JSFactCheckBench, stellalisy/preference-forecast, comparia-fr-arena-raw, msts-japanese | Gated (manual or auto); unavailable without login |
| inclusionAI/SingStreamBench | Built from older source datasets; NC licence |
| Arena-90K, SHP, SQuAD, SNLI and IMDB derivatives; IndicQE-APE; tasksource/* | Re-uploads of older corpora or project-used sources |
| Peer JEV aggregators (listed in Rules) | Contamination vectors: peer training and eval pools |
| zhuq41/*, TianfuXinqu/*, Roy229/*, SOTAagi2030/*, dataset_0xxxx_* | Template or fabricated repos |
| OpenReview 2026 venues (COLM, ICML, NeurIPS, ARR) | API needs a challenge or login from here; watch-list for post-cutoff Score (review ratings) |
| Paper-only, no data found: BioStance 2606.13187, AIMS, Mawqif-XT, ParsHate, ViTOED, CUP (el, graded), LexIssue (zh legal), RALS (ro ratings), Wazobia Eval | Data not public or not located. Recheck, especially CUP, RALS and LexIssue for Score, zh and long input |

## Achievable sealed-set size (1 item per source group, class-balanced)

| Source | Choice | Noul | Score | Language | Long |
|---|---|---|---|---|---|
| HalluTruthQA-4K (disjoint items per type) | 300 | 300 | – | ar | – |
| JudgmentBench | 30 | – | (same 30 tasks) | en | 30 |
| scopejudge | – | 100 | – | en | 100 |
| DeliChess | 107 | – | – | en | – |
| MCJudgeBench | – | – | 200 | en | – |
| esnlir-al | 300 | – | – | es | – |
| SentiTaglish | – | – | 300 | tl | – |
| WB reviews (⏱) | – | – | 150–300 | ru | – |
| factbutcher (⏱) | 60–100 | – | – | ru | – |
| narrative-gold | 60 | – | 250 | en | – |
| tutormoments | 100 | 100 | – | en | 150 |
| legal case law | 42 | – | – | en | 42 |
| ImplicatureX / GAPA | – | – | 150 / 100 | en | – |
| ClimateCause | – | 300 | – | en | – |
| innoduel | 100 | – | – | fi/sv/en/uk | – |
| NepFakeV2 (⏱) | – | 30–60 | – | ne | – |
| B− pool (EusExams 2026, TikStance, VARM, stance-it, constructcie) | +300 es/eu, +300 zh-TW, +100 it, +100 en | +300 zh | – | – | – |
| **A+B total (excl. B−)** | **≈1,100–1,150** | **≈830–860** | **≈1,150–1,300** | – | **≈320** |

- **Realistic A+B sealed set: about 3,000 items** (≈2,600 after per-language caps). With B− sources that clear their licence or date checks, about 4,000.
- By language: en about 1,600, dominant; es 300 (600 with EusExams); ar 600; ru 210–400; tl 300; ne, fi/sv/uk and it are small. zh is only available through the licence-missing sources; **ja 0; de 0**.
- Long inputs (4,000 characters or more): about 320, roughly 10%, below the design's 20% target. All long sources are in English.

## Main weaknesses and blockers

1. The language targets are missed. There is no ja or de source, and zh (simplified and traditional) exists only in sources without a stated licence. ToxiREX (de/es/ar/tr/nl) fails the cutoff by about 7 weeks.
2. Only 9 of the 25 A/B candidates fall in the preferred window after 2026-08-15. Several B sources rely on old text (Dolma, IPCC, tax opinions, older reviews), so they test label novelty, not text novelty.
3. Few source groups in the long-input and decision-like sources: JudgmentBench has 30, legal 42 and scopejudge 100. Confidence intervals per type will be wide.
4. Several sizes (WB, factbutcher, NepFake, EusExams) must be recomputed at the pinned SHA after applying the post-2026-06-01 row filter, because the dataset-server statistics give only coarse histograms.
5. Licences: 4 are NC or NC-ND (fine for private eval, but flagged), 2 have no stated licence, and 1 is gated behind a login.
6. HalluTruthQA and MCJudgeBench are LLM-output judgement tasks with human labels. Their answers come from models, which is acceptable only as in HelpSteer-style judging.
7. Before sealing, diff the final list against the peer aggregator source lists and the project-used exclusions.
