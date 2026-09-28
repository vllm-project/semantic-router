# Research & data Milestone 3b item 4 — dataset-level same-source audit against the CSS panel (2026-09-28)

Question: at 0.6B (Milestone 4) the development proxy favoured the mixture-v2 soup (V2) over the A7
soup (T) by +6.94 P, but post-key v3 ranked T first (43.54 against 39.13). The CSS pilot rose for V2
while CSS15 fell (.397 against .480). Did a training source that shares its origin with a CSS pilot
task inflate the proxy, and which sources share an origin with a CSS15 task? This record is
count-level and task-level only: no item text, no gold labels, no sealed path read, no GPU used.

**Answer.**

- No v1, v2, A7 or A0s-strict source shares a dataset, parent corpus or annotation project with any
  of the three CSS pilot tasks. **New quarantine list: empty.**
- One source is the same dataset as a CSS15 task: `css_flute_official_train` (FLUTE) for `flute`.
  A0s-strict, all six XL recipes and every M4 mixture already exclude it. Only the root A0 file
  contains it (120 rows). Keep it excluded.
- Four sources belong to the same task family as a CSS15 task but have a different origin. They
  should be disclosed, not quarantined: GoEmotions (emotion), ArgQ-30k (persuasion), OASST1 humour
  ratings (reddit_humor), and SentiMix/AfriSenti (emotion).
- The pilot "rise" for V2 is two tasks up (discourse, implicit hate) and one sharply down (stance).
  It is not significant, and the seed-mean pilot median is equal for the two arms. Typed DEV carries
  82% of the proxy gap. The misalignment is a proxy-composition effect, not a data-origin effect.

## 1. Method

- **Tasks.** The CSS pilot tasks are `PILOT_TASKS` and CSS15 is `EVALUATION_TASKS` in
  `transfer/build.py` (`css-transfer/1`). Upstream datasets come from SALT-NLP/LLMs_for_CSS at
  `55183a64`, which is the snapshot the panel is built from. Its `mappings.py` gives the ConvoKit
  corpus names, download URLs and column maps, and `data_loader.py` builds the tasks. Tasks loaded
  from local CSVs were identified from the paper's dataset section (Ziems et al., "Can Large
  Language Models Transform Computational Social Science?", arXiv 2305.03514) and the column maps.
- **Sources.** The universe is every `source` id in `license-registry-v1.json` (30),
  `license-registry-v2.json` (26 more) and `license-registry-a7-v1/v2.json` (14): 70 ids in total.
  A0s-strict's six sources are among them. For each id, the origin comes from its registry
  attribution and evidence URL plus the upstream paper or dataset card.
- **Specific checks:**
  - ArgQ-30k collection and the `sources/argq.py` projection.
  - The MLQE-PE Ru-En source table (Fomicheva et al., arXiv 2010.04480, Table 3).
  - The Coarse Discourse sampling notes (dataset README).
  - The EPITOME release notes.
  - GoEmotions subreddits and dates. Our 1,797 GoEmotions comment IDs (TRAIN 1,397, SELECT 200,
    CAL 200, all mapped) were joined to the public raw GoEmotions release, which has subreddit and
    `created_utc`. Only counts are kept.
  - A keyword scan of the 49 ArgQ TRAIN topics for SemEval-2016 targets.
- **Classes:**
  - SAME DATASET: the same release, or a split or derivative of it.
  - SAME PARENT CORPUS: both drawn from the same specific collection, so items can coincide by
    construction. Examples are one tweet collection, one Reddit thread sample, one IBM Debater
    argument set, or one ConvoKit corpus.
  - SAME TASK FAMILY ONLY: the same construct from an independent origin.
  - UNRELATED: everything else. Sharing only a platform (Reddit, Twitter, Wikipedia) or a domain
    (news), with independent sampling and no possible item coincidence, counts as UNRELATED. Links
    of this kind that the coordinator asked about are listed separately in §4b.
- **Magnitudes.** Row and token counts come from count-only scans on node A of the M4 mixture files
  (`m4-mix-t`, `m4-mix-v2`, `m4-mix-c`), the XL-full id manifest (native tokens), A0s-strict, and
  SELECT/CAL.

## 2. CSS tasks and their upstream datasets

| Role | Task | n | Upstream dataset | Text origin | Labels |
| --- | --- | ---: | --- | --- | --- |
| pilot | `semeval_stance` | 435 | SemEval-2016 Task 6 (Mohammad et al., 2016); SALT column `trump_stance` | English tweets, 2015, Donald Trump target test set (NRC collection) | favour / against / none (crowd) |
| pilot | `implicit_hate` | 498 | Latent Hatred (ElSherief et al., EMNLP 2021), `SALT-NLP/ImplicitHate` | Tweets from and around U.S. hate-group accounts | 6 implicit-hate classes, "other" dropped (crowd + expert) |
| pilot | `discourse` | 497 | Coarse Discourse (Zhang, Culbertson, Paritosh, ICWSM 2017), ConvoKit `reddit-coarse-discourse-corpus` | English Reddit threads sampled from all of Reddit up to May 2016 (non-English subreddits blacklisted); first post + direct reply | 7 discourse acts (crowd) |
| CSS15 | `emotion` | 498 | CARER (Saravia et al., EMNLP 2018) | English tweets | 6 classes from hashtag weak labels |
| CSS15 | `ibc` | 498 | Ideological Books Corpus (Gross et al. 2013) as used by Iyyer et al. (ACL 2014) | Sentences from partisan U.S. books and magazines | liberal / conservative / neutral |
| CSS15 | `media_ideology` | 498 | Article Bias Corpus (Baly et al., EMNLP 2020) | U.S. news articles from AllSides-rated outlets | left / centre / right |
| CSS15 | `indian_english_dialect` | 266 | Demszky et al. (NAACL 2021) | Indian English utterances | 22 features + none (expert) |
| CSS15 | `raop` | 399 | Yang et al. (NAACL 2019) labels on Random Acts of Pizza (Althoff et al., ICWSM 2014) | Reddit r/Random_Acts_Of_Pizza requests, 2010–2013 | 7 persuasion strategies |
| CSS15 | `talklife` | 498 | EPITOME explorations (Sharma et al., EMNLP 2020) | TalkLife according to the paper. The public EPITOME release is its Reddit portion (mental-health subreddits, from a 2018 dataset); which portion SALT used is unverified | strong / weak / no exploration |
| CSS15 | `wiki_politeness` | 498 | Stanford Politeness Corpus, Wikipedia part (Danescu-Niculescu-Mizil et al., ACL 2013), ConvoKit `wikipedia-politeness-corpus` | Requests on Wikipedia user-talk pages | polite / neutral / impolite (AMT) |
| CSS15 | `tempowic` | 344 | TempoWiC (Loureiro et al., COLING 2022; the paper cites WiC) | Tweets, 2019–2021 | same / different sense |
| CSS15 | `tropes` | 114 | CMU Movie Summary Corpus (Bamman et al., ACL 2013) with TV Tropes labels and IMDb quotes (Chu et al. 2018) | IMDb character quotes | 72 tropes |
| CSS15 | `flute` | 500 | FLUTE (Chakrabarty et al., EMNLP 2022); SALT samples its test items from `ColumbiaNLP/FLUTE` `train.jsonl` | Figurative NLI pairs built from earlier figurative datasets | idiom / metaphor / sarcasm / simile |
| CSS15 | `mrf` | 500 | Misinfo Reaction Frames (Gabriel et al., ACL 2022) | News headlines on COVID-19, climate and cancer | misinformation / trustworthy |
| CSS15 | `conv_go_awry` | 500 | Conversations Gone Awry (Zhang et al., ACL 2018), ConvoKit `conversations-gone-awry-corpus` | Wikipedia talk-page conversations (first two turns) | later personal attack (crowd) |
| CSS15 | `persuasion` | 434 | Winning Arguments (Tan et al., WWW 2016), ConvoKit `winning-args-corpus` | Reddit r/ChangeMyView, 2013–2015 | delta awarded |
| CSS15 | `reddit_humor` | 500 | Weller & Seppi (EMNLP 2019), `orionw/RedditHumorDetection` `reddit_full` test | Reddit r/Jokes submissions | funny / not funny (upvote threshold) |
| CSS15 | `wiki_corpus` | 500 | Echoes of Power (Danescu-Niculescu-Mizil et al., WWW 2012), ConvoKit `wiki-corpus` | Wikipedia talk pages (speaker history) | administrator or not |

Typed DEV, typed FINAL and SELECT/CAL are not CSS data. The typed panels are project-generated.
SELECT and CAL each hold 400 GoEmotions rows and 300 generated `decision2_rights_clean_oracle_holdout_v1`
rows.

## 3. What moved at 0.6B (V2 soup minus T soup)

CSS pilot macro-F1, from `06b/records/m4/readouts` and `m4/contrast-v2.json` (10,000 paired draws):

| Pilot task | T soup | V2 soup | Δ soup | T seeds (mean) | V2 seeds (mean) | Seed pairs with V2 > T |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| discourse | .298 | .342 | **+.043** | .284 / .316 / .259 (.286) | .332 / .309 / .307 (.316) | 2 / 3 |
| implicit_hate | .319 | .350 | **+.031** | .268 / .290 / .324 (.294) | .306 / .360 / .329 (.331) | 3 / 3 |
| semeval_stance | .539 | .312 | **−.227** | .486 / .526 / .540 (.517) | .271 / .274 / .271 (.272) | 0 / 3 |
| H_pilot (median) | .319 | .342 | +.023 [−.030, +.048] | mean .308 | mean .307 | — |
| Pilot micro accuracy (/1,430) | 575 | 556 | −.013 [−.034, +.007] | — | — | — |

Per-seed V2 − T H_pilot differences are +.022, −.007 and −.017, and every interval contains 0. The
control C has 4.6 times T's GoEmotions rows. Its seed values are discourse .256 / .318, implicit hate
.247 / .302 and stance .493 / .399.

The proxy is `P = 100·√(T_dev·H_pilot)`. T_dev moved from .336 to .461 and H_pilot from .319 to .342,
giving 32.74 → 39.68. In log terms, 82% of the gap comes from typed DEV and 18% from the pilot.
Holding H_pilot at T's value still gives P 38.32. On the formal panel, typed FINAL moved from .395 to
.385 and CSS15 H from .480 to .397; CSS15 carries 88% of the v3 gap.

CSS15 macro-F1 from the formal runs (T soup / V2 soup / Δ). **Twelve of fifteen tasks fell**:

- conv_go_awry .574 / .485 / −.090
- emotion .597 / .553 / −.044
- flute .344 / .315 / −.029
- media_ideology .260 / .188 / −.072
- mrf .460 / .359 / −.101
- persuasion .511 / .438 / −.072
- raop .581 / .572 / −.009
- reddit_humor .537 / .485 / −.052
- talklife .348 / .299 / −.049
- tempowic .480 / .346 / −.133
- wiki_corpus .484 / .397 / −.087
- wiki_politeness .539 / .484 / −.054

Two rose: ibc .423 / .471 / +.048 and indian_english_dialect .257 / .289 / +.032. Tropes is about 0 for
both (15 invalid each). The CSS15 median is tempowic for T (.480) and wiki_corpus for V2 (.397).

## 4. Source × task origin (dataset level)

Arm membership: T = A0s-r + A6 slice + A7 v3. V2 = A0s-r + `mx-v2-full-M` (H1, H3, H5, H6, E11, G2,
G6, G4h, V1S = A6g + A6h). XL-full = all pools in `m3b/mx-xl.manifest.json` (151.29M native tokens).

### 4a. Pairs that are not UNRELATED

| Source (`source`) | Pools | CSS task | Class | Evidence | Exposure |
| --- | --- | --- | --- | --- | --- |
| `css_flute_official_train` | root A0 only | flute | **SAME DATASET** | SALT `mappings.py` builds `flute` from FLUTE's official `train.jsonl`, the file this source is drawn from. The builder removed the 500 panel IDs and hash matches, but the dataset is the same. The registry already discloses it as same-task exposure | A0 (`61740be4…`) 120 rows. None in A0s / A0s-strict, the 6 XL recipes, or M4 T / V2 / C |
| `google_goemotions_official_train` | A0s-strict (every recipe); SELECT/CAL | emotion | SAME TASK FAMILY ONLY | The Choice family asks for the best emotion among 4 GoEmotions labels, and GoEmotions' 27 labels include all 6 CSS emotion labels. The text is Reddit comments (Demszky et al. 2020, January 2019) against English tweets with hashtag labels (Saravia et al. 2018) | 2,794 rows in T, V2 and XL-full (0.21% of XL tokens); 12,907 in C; 400 of 700 in SELECT and in CAL |
| `argq30k_train` | V1:A6h, H6 | persuasion | SAME TASK FAMILY ONLY | Argument quality ("recommend as is in a speech") against "would this reply convince you". Crowd-written original arguments on IBM Debater motions (Gretz et al., AAAI 2020) against r/ChangeMyView | T 830; V2 1,218; XL-full 2,577 rows (0.43%) |
| `argq30k_train` | V1:A6h, H6 | semeval_stance (pilot) | SAME TASK FAMILY ONLY | Pro/con arguments are stance-bearing, but stance fields never reach a row (`sources/argq.py`). None of the 49 TRAIN motions concerns Trump or any other SemEval-2016 target (keyword scan, 0 hits). The IBM crowd collection and the NRC 2015 tweet collection are independent | as above |
| `a7:oasst1_train` (family `oasst1_humor`) | A7q | reddit_humor | SAME TASK FAMILY ONLY | Human humour ratings of assistant replies written by OASST1 volunteers (2023) against r/Jokes upvotes | Not in M4. XL-full 2,046 humour rows (0.57%) of 22,401 OASST1 rows |
| `a7:sentimix_hinglish_train`, `a7:afrisenti_sw_train` | A7s | emotion | SAME TASK FAMILY ONLY | Tweet sentiment (SemEval-2020 Task 9 Hinglish; AfriSenti / SemEval-2023 Task 12 Swahili) against English emotion tweets; independent collections | Not in M4. XL-full 12,860 rows (1.43%) |
| same | A7s | semeval_stance (pilot) | SAME TASK FAMILY ONLY (weak: polarity, not stance) | Different SemEval tasks with independently collected tweets in other languages and years; not the SemEval-2016 collection | as above |

### 4b. Checked links that are platform-only or domain-only (UNRELATED)

| Source | CSS tasks | Why it is UNRELATED |
| --- | --- | --- |
| `google_goemotions_official_train` | discourse (pilot), persuasion, raop, reddit_humor, talklife | All 58,011 GoEmotions comments date from January 2019. None can be in Coarse Discourse (≤ May 2016), Winning Arguments (2013–2015), RAOP (2010–2013) or EPITOME's Reddit portion (2018 dataset). reddit_humor items are r/Jokes submissions, and GoEmotions holds comments only. Our 1,797 GoEmotions comments include 5 from r/changemyview, 6 from r/Jokes (5 TRAIN, 1 CAL) and none from r/Random_Acts_Of_Pizza, which is not in GoEmotions |
| `mlqepe_train` (`mlqepe_ruen`) | discourse (pilot), persuasion, raop, reddit_humor | 7,501 of the 10,000 Ru-En source sentences are Russian-language posts and comments from r/antireligious, r/PikabuPolitics, r/rupolitika and r/ru, pulled around 2020 with the Reddit API. The rest are WikiQuote proverbs. Coarse Discourse is English-only and ends in May 2016. Other MLQE-PE pairs are Wikipedia-article sentences. Rows: H6 1,583; M4 V2 178; XL-full 1,021 (0.19%) |
| `a7:sentimix_hinglish_train`, `a7:afrisenti_sw_train` | implicit_hate (pilot), tempowic | Twitter only; different collections, languages and years |
| `a7:oasst1_train` | all Reddit and Twitter tasks | Written by volunteers on the Open Assistant platform, not scraped. A7q's lexical screen already removed whole message trees close to CSS15 (61 groups) and the CSS pilot (5) |
| Wikipedia-article sources: `squad2_train`, `tydiqa_primary_train`, `hotpotqa_distractor_train`, `twowiki_train`, `musique_full_v1.0_train`, `quac_train_v0.2`, `dbpedia14_train`, `germanquad_train`, `piaf_train`, `sqac_train`, `drcd_train`, `cmrc2018_train`, `jsquad_v1.3_train`, `klue_mrc_train`, `dec10:squad2_train`, `legacy:squad2_answerability`, and non-Ru `mlqepe_train` | conv_go_awry, wiki_politeness, wiki_corpus, tropes | These are article-namespace text, while the three conversation tasks are talk-page corpora. Tropes' input is IMDb quotes, not the CMU corpus's Wikipedia plot summaries |
| Dialogue sources: `multiwoz22_train`, `taskmaster2_train`, `sgd_dstc8_train`, `abcd_v1.1_train`, `quac_train_v0.2`, `a7:oasst1_train` | discourse (pilot), conv_go_awry, wiki_politeness, wiki_corpus, persuasion, talklife | Crowd, Wizard-of-Oz, simulated or volunteer dialogues. Our families are intent, slot, answerability and reply quality, not discourse acts, toxicity, politeness, power or persuasion |
| News-domain sources: `onestop_english`, `dec10:multinli_nonfiction_train` (Slate and government genres), `klue_ynat_train` | media_ideology, ibc, mrf | Domain only. OneStopEnglish is Guardian articles rewritten for learners (2013–2016), and a same-publisher article in the AllSides-rated set is possible but unverified. MultiNLI Slate text is 1990s OANC. YNAT is Korean headlines |
| NLI sources: `dec10:snli_train`, `legacy:snli`, `dec10:multinli_nonfiction_train`, `jglue_jnli_v1.3_train` | flute | Format only. FLUTE pairs are NLI-shaped, but the CSS task labels the figurative type. No other source is a figurative-language dataset |

**All other pairs of the 70 source ids with the 18 tasks are UNRELATED.** They cover:

- Project-generated sources: `decision2_*`, `legacy:stage4-general-composition-v2`, `dec10:generated_*`,
  and the generated part of `legacy:stage3_replay`.
- Intent sets: BANKING77, CLINC150, MTOP, MASSIVE (which is ablation-only because it shares a source
  with `mlx-diag`).
- QA, commonsense, science and math sets: WinoGrande, GSM8K, CommonsenseQA, SciTail, ROPES, QuaRTz,
  QASC, ARC, OpenBookQA, AQuA-RAT, Cosmos QA.
- Korean and Japanese sets: KLUE STS/NLI/YNAT, JGLUE JSTS/JNLI/JCommonsenseQA.
- SAF.

`decision2_targeted_programmatic_v1` in A0s-strict holds only the interval-conjunction and
quantized-median families. The stance and dialogue-function families that the older pipeline built
from CSS pilot errors are in no current pool.

## 5. Quarantine decision

**Development proxy: no source to quarantine (list empty).**

- The two pilot tasks that rose for V2, discourse and implicit hate, have no same-dataset or
  same-parent-corpus source in any pool. The only Reddit-origin rows that V2 adds are 178 MLQE-PE
  Ru-En rows, and they cannot contain Coarse Discourse items. Neither arm has Twitter or hate-speech
  data.
- The sources closest to the pilot are present in both arms:
  - GoEmotions has 2,794 identical rows in T and V2. C has 12,907 and no higher discourse score.
  - ArgQ-30k has 830 rows in T and 1,218 in V2, and stance fell for V2 in every seed pair
    (−.215 to −.269).
- The rise itself is not a robust signal. H_pilot moved +.023 [−.030, +.048], the seed means are
  equal, and pilot micro accuracy fell. Most of the proxy gap is typed DEV plus a V2 soup effect:
  the V2 soup's T_dev is .461 against a seed mean of .379.
- No readout that a pilot-sharing source would inflate was found. SELECT and CAL contain no CSS-origin
  rows, and typed DEV is generated.

**Formal panel: keep `css_flute_official_train` out.** It is the same dataset as CSS15 `flute`.

- Its only home is root A0 (`rights_clean.train.jsonl`, 7,455 rows).
- 0.6B Milestones 1–2 trained on that file, so any CSS15 flute cell reported for those checkpoints
  is same-task supervised, as the v1 registry states. None of them is a release candidate.
- Milestones 3–4 and all XL recipes use A0s or A0s-strict and contain no FLUTE row.
- The older pipeline's Wikipedia-politeness train complement (`training/data/build_css_wiki_politeness`)
  is the same dataset as `wiki_politeness`. It is in no registry or pool and must stay out.

**Disclose, do not quarantine.** Flag emotion, persuasion and reddit_humor as tasks whose family was
seen in training, from a different origin:

- GoEmotions for emotion.
- ArgQ-30k for persuasion.
- OASST1 humour ratings for reddit_humor.
- SentiMix/AfriSenti for emotion.

For XL-trained models these tasks are not zero-shot at the task-family level. They do not explain
M4: every one of them fell for V2, or its exposure was equal in both arms.

## 6. Open uncertainties

- **talklife portion.** The paper says TalkLife, but SALT's column names differ from the public
  Reddit EPITOME CSV. Settling it would need the test file, which this audit did not read. No source
  comes from either portion.
- **Tasks loaded from local CSVs.** For emotion, ibc, media_ideology, indian_english_dialect,
  semeval_stance, tempowic, talklife, raop, mrf and tropes, the origin rests on the paper text and
  column maps. SALT ships no per-task provenance.
- **Details from the literature.** Tweet years, the Winning Arguments and RAOP windows, the Indian
  English source text, and the 2018 origin of EPITOME's Reddit portion come from the literature and
  were not re-verified here.
- **FLUTE's internal sources.** FLUTE builds on earlier figurative datasets, partly from social media.
  Those were not traced. No recipe source is a figurative-language dataset.
- **OneStopEnglish and media_ideology.** A same-publisher Guardian article is possible. Checking it
  needs test text.
- **Teacher lineage.** V2 distils own-Lux targets on 99.7% of its rows, against 24.5% for T. Lux
  1.0's own training corpus was not audited against CSS here. A teacher exposed to CSS-origin data
  could transfer pilot behaviour without any same-source training row.
- **Scope.** This audit is dataset-level only. Text-level overlap is covered by the existing lexical
  and embedding scans (PI-v3). Per-task pilot confidence intervals were not computed; they would need
  pilot gold, and the aggregate paired intervals of `m4/contrast-v2.json` are used instead.
