# Research & data Milestone 3b item 4 — dataset-level same-source audit against the CSS panel (2026-09-28)

Question: at 0.6B (Milestone 4) the development proxy favoured the mixture-v2 soup (V2) over the A7
soup (T) by +6.94 P, but post-key v3 ranked T first (43.54 against 39.13). The CSS pilot rose for V2
while CSS15 fell (.397 against .480). Did a training source that shares its origin with a CSS pilot
task inflate the proxy, and which sources share an origin with a CSS15 task? The record also covers
the training lineage of the own-Lux teacher (V2 distils it on 99.7% of its rows, T on 24.5%) and
gate 10 of amendment 2 (the H7 / H8 gap sources). It is count-level and task-level only: no item
text, no gold labels, no sealed path read, no GPU used.

**Answer.**

- No source shares a dataset, parent corpus or annotation project with any of the three CSS pilot
  tasks. That covers every v1, v2, A7, A0s-strict and H7 / H8 source, plus the teacher's own
  training pool. **New quarantine list: empty.** No recipe revision is needed (§5).
- **Teacher lineage: no shared origin (§4c).** The teacher is the unmodified 1.0 release
  `Decision-1.0-Lux-9B@bd45a30a`. Lux, like Eos, trained on the 1.0 natural 24k pool. That pool has
  8 sources and 30 families: Cosmos QA, SNLI, SQuAD 2.0, MultiNLI non-fiction, BANKING77, CLINC150
  and generated Stage1–4 rows. None comes from Twitter or Reddit, and none is a stance, discourse,
  dialogue or hate task.
  - The older 2.0 CSS-linked families postdate Lux and are in neither its pool nor any current
    pool. These are the TweetEval set and the stance and dialogue families built from CSS pilot
    errors.
  - T trains on 13,106 rows of the teacher's pool itself. V2 trains on none beyond the base that
    both arms share.
- **Gate 10 (H7 / H8): pass, nothing excluded (§4d).** HoVer, Natural Questions, TyDi QA, MIRACL
  es/fa/fr/hi/zh and JCommonsenseQA are UNRELATED to all 18 tasks. SentiMix Spanglish is SAME TASK
  FAMILY ONLY for emotion, and weakly for stance.
- One source is the same dataset as a CSS15 task: `css_flute_official_train` (FLUTE) for `flute`.
  A0s-strict, all six XL recipes and every M4 mixture already exclude it. Only the root A0 file
  contains it (120 rows). Keep it excluded.
- Four sources belong to the same task family as a CSS15 task but have a different origin. They
  should be disclosed, not quarantined: GoEmotions (emotion), ArgQ-30k (persuasion), OASST1 humour
  ratings (reddit_humor), and SentiMix/AfriSenti (emotion). SentiMix Spanglish (emotion) joins
  them if H8 enters a recipe.
- The pilot "rise" for V2 is two tasks up (discourse, implicit hate) and one sharply down (stance).
  It is not significant, and the seed-mean pilot median is equal for the two arms. Typed DEV carries
  82% of the proxy gap. The misalignment is a proxy-composition effect, not a data-origin effect.
  No development readout was inflated by a same-origin source or by the teacher lineage.

## 1. Method

- **Tasks.** The CSS pilot tasks are `PILOT_TASKS` and CSS15 is `EVALUATION_TASKS` in
  `transfer/build.py` (`css-transfer/1`). Upstream datasets come from SALT-NLP/LLMs_for_CSS at
  `55183a64`, which is the snapshot the panel is built from. Its `mappings.py` gives the ConvoKit
  corpus names, download URLs and column maps, and `data_loader.py` builds the tasks. Tasks loaded
  from local CSVs were identified from the paper's dataset section (Ziems et al., "Can Large
  Language Models Transform Computational Social Science?", arXiv 2305.03514) and the column maps.
- **Sources.** The universe is every `source` id in `license-registry-v1.json` (30),
  `license-registry-v2.json` (26 more) and `license-registry-a7-v1/v2.json` (14): 70 ids.
  `license-registry-m3b.json` adds 4 more for H7 / H8: HoVer, Natural Questions, MIRACL and SentiMix
  Spanglish. TyDi QA and JCommonsenseQA are already in the universe, which brings the total to 74.
  A0s-strict's six sources are among them. For each id, the origin comes from its registry
  attribution and evidence URL plus the upstream paper or dataset card.
- **Teacher lineage.** Lux's pool comes from three places: the A7 inventory
  (`a7/records/a7-inventory-2026-09-28.md`), the six 1.0 model cards, and the A7 v3 views
  `dec10-natural24k` and `dec10-semantic24k`. The views' members were joined to the A7 v3 TRAIN and
  AHO files on node A to count rows by source and family. The older pipeline's CSS-linked families
  come from `training/data/` (`build_targeted_candidate.py`, `build_tweeteval_human.py`,
  `build_rights_clean_v1.py`, the README, `targeted_open_evidence_v1.json`).
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
- **Magnitudes.** Row and token counts come from count-only scans on node A of:
  - the M4 mixture files (`m4-mix-t`, `m4-mix-v2`, `m4-mix-c`);
  - the XL-full id manifest (native tokens) and the 26 XL pool files;
  - root A0, A0s-strict and SELECT/CAL;
  - the A7 v3 files and views;
  - the H7 / H8 TRAIN files and their four-method overlap receipts (hit groups per CSS task).

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
Both arms start from `Qwen/Qwen3-0.6B-Base@da87bfb6`. The own-Lux teacher covers 6,430 of T's 26,203
rows and 47,759 of V2's 47,922.

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

**All other pairs of the 70 source ids with the 18 tasks are UNRELATED** (the four H7 / H8 ids are
in §4d). They cover:

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
from CSS pilot errors are in no current pool (§4c).

### 4c. Teacher lineage: own Lux 1.0 (and Eos)

**Teacher identity.** The teacher is `Decision-1.0-Lux-9B@bd45a30a`. Its weights are byte-identical
to Hub `main` `cdf4d3ef` (eval-track check), so it is the 1.0 release, not a 2.0 continuation.

- The 2.0 continuation `lux9b-human-structured8360` (BEST224) starts from these weights. Its TRAIN
  includes the 3,600 TweetEval rows below. It is on release HOLD and produced no target.
- Lux's pool, per its card and the A7 inventory, is the natural 24k `natural-treatment.jsonl`
  (`d56175ab`). That file is the natural 8k (Cosmos QA 4,000, SNLI 2,000, SQuAD 2.0 answerability
  2,000) plus 16,000 Stage4 v2 rows.
- Lux's final stage used the semantic variant `8f40cf03`: 21,329 rows are identical and 2,671 have
  re-described options.
- Eos trained on all 24,000 rows of the same pool; it is not an M4 teacher.
- Lux and Eos never trained on the full Stage1–4 curricula (Sol and Nox did). Of the curricula they
  saw only the 16,000 Stage4 v2 rows in the pool, which include 1,812 replayed Stage1–3 rows in the
  joined view.

The table gives the pool by source: 22,919 of 24,000 rows joined to A7 v3 on node A.

| Pool source | Rows | Families | Class for all three pilot tasks | Evidence |
| --- | ---: | --- | --- | --- |
| `dec10:generated_stage4_v2` | 10,201 | 8 `stage4_*` (arithmetic, automaton, boolean, dense table, ordinal, registers, relations, scope) | UNRELATED | Project-generated, no third-party text |
| `dec10:generated_stage1_3` | 1,812 | 15 replayed Stage1–3 families (authorization, evidence, high-k, mapping, policy, five arithmetic, evidence scope, three logic, transition set) | UNRELATED | as above |
| `dec10:cosmos_qa_train` | 3,984 | `natural_cosmos_qa` | UNRELATED | Personal-narrative blog posts (Spinn3r) with crowd questions |
| `dec10:snli_train` | 1,998 | `natural_snli` | UNRELATED | Flickr30k captions with crowd hypotheses |
| `dec10:squad2_train` | 1,514 | `natural_squad2_answerability` | UNRELATED | Wikipedia articles |
| `dec10:multinli_nonfiction_train` | 1,931 | `stage4_natural_nli` | UNRELATED | Government, Slate, travel and telephone genres. The telephone premises are Switchboard transcripts, the pool's only conversational text, labelled for entailment only |
| `dec10:banking77_train`, `dec10:clinc150_train` | 645 + 834 | intent replay (banking, clinc, natural high-k) | UNRELATED | Crowd-written intent utterances |

- The other 1,081 rows were removed by A7 admission (778) or requarantine (303). They come from the
  same file and sources. The semantic view (20,253 joined rows) has the same 8 sources.
- The A7 lexical screen flagged one CSS-pilot row in A7h and one in A7i. Both were method L only,
  meaning a short field was contained in a window of a long pilot field. Both were removed, so T has
  neither.
- **T itself.** The six A7 v3 sub-arms in T (A7g, A7h, A7i, A7m, A7o, A7p; 138,011 TRAIN rows) hold
  only these 8 sources.
  - T's A7 component is 14,179 rows: 13,106 from the `dec10-natural24k` view (13.69M tokens) plus
    1,073 fill rows.
  - A node-A family scan finds no stance, discourse, dialogue, hate or tweet family in `m4-mix-t`,
    `m4-mix-v2`, `m4-mix-c`, or any of the 26 XL pools (529,474 rows). The only keyword hits are the
    §4a families and the two targeted Noul / Score families.

**Older 2.0 CSS-linked sources.** These are in neither the teacher's pool nor any current pool, M4
mixture or XL recipe:

- **TweetEval, 3,600 rows (`build_tweeteval_human.py`).**
  - The stance part covers the five non-Trump SemEval-2016 targets (1,500 rows). It is the **SAME
    DATASET** as `semeval_stance`, with a different target, so it makes the stance pilot a
    cross-target test.
  - HatEval (500 rows) is SAME TASK FAMILY ONLY for `implicit_hate`.
  - SemEval-2018 emotion (400 rows) is SAME TASK FAMILY ONLY for `emotion`.
  - TweetEval reached only the 2.0 research mixes of 4,624, 5,824, 8,360 and 8,522 rows. The
    BEST224 continuation above trained on the 8,360-row mix. `build_rights_clean_v1` removed all
    3,600 rows from A0.
- **`targeted_attributed_stance` and `targeted_dialogue_function`, 500 rows each.**
  - Both come from source `decision2_targeted_programmatic_v1`. They were generated from aggregate
    CSS pilot errors of six models, Lux 1.0 among them (`targeted_open_evidence_v1.json`), so they
    postdate Lux.
  - They are SAME TASK FAMILY ONLY for `semeval_stance` and `discourse`, by design. The builder
    reports zero overlap with pilot text.
  - They exist only in targeted-2k (`59d40112…`) and targeted-anchor-3024 (`18714248…`).
  - Root A0, A0s-strict and every M4 and XL pool take only this source's interval-conjunction (450)
    and quantized-median (150) families. The node-A scan found 0 stance or dialogue rows.

**Could the lineage have inflated H_pilot (hence P) for V2?**

- Not through shared origin, because there is none.
- The direction also points the other way. T trains on the teacher's pool directly. V2 shares only
  the base's 2,903 legacy 1.0 rows. V2's extra teacher coverage is on its own v2 prompts, whose
  sources §4a and §4b classify.
- Distillation can still pass on general decision behaviour. Lux's own pilot median macro-F1 is
  .570 (its 1.0 observation in the 9B records), against .319 and .342 for the two soups. But a
  general effect is not specific to the pilot, and CSS15 fell for V2 on 12 of 15 tasks.
- M4 changes data and teacher coverage together (M4d), so its readouts cannot separate the two.

### 4d. Gap arms H7 / H8 (amendment 2, gate 10)

The sources are the six ids of `license-registry-m3b.json`, checked against all 18 tasks.

- HotpotQA only keys HoVer groups and is never rendered. §4b already classifies it as
  `hotpotqa_distractor_train`.
- Text-level hits are groups per CSS task, counted in the H7 / H8 four-method overlap receipts on
  node A (TRAIN + AHO + SHO).
- Every hit group was quarantined whole.

| Source (`source`) | Arm, families, TRAIN rows | Origin | Class | Text-level hits (quarantined groups) |
| --- | --- | --- | --- | --- |
| `hover_train_v1.1` | H7 `hover_answerable` + `hover_coverage`, 3,073 | Crowd-written claims over the intro paragraphs of HotpotQA's processed English Wikipedia (Jiang et al. 2020). Rows ask whether the paragraphs are enough to check the claim, or how many needed facts they state. The claim's truth never reaches a row | UNRELATED to all 18. Checked: mrf (fact-checking domain, but a different construct from headline credibility); wiki_politeness, conv_go_awry and wiki_corpus (talk pages, not articles); tropes (IMDb quotes) | CSS15: media_ideology 175, ibc 12, wiki_corpus 11, conv_go_awry 7, tropes 4, persuasion 2, wiki_politeness 1. Pilot: discourse 1 |
| `natural_questions_train` | H7 `nq_window_removal`, 1,858 | Real Google search queries over English Wikipedia article pages (Kwiatkowski et al. 2019) | UNRELATED to all 18 (article namespace; same checks as HoVer) | CSS15: media_ideology 86, conv_go_awry 6, wiki_corpus 5, persuasion 4, tropes 4, mrf 1. Pilot: discourse 3, implicit_hate 1 |
| `tydiqa_primary_train` (new items) | H8 `tydi_window_removal_<lang>`, 1,436 | The same dataset as the v2 source in §4b. New questions, disjoint by group from H5 and E11 | UNRELATED to all 18 | With MIRACL (shared `tydi-miracl` groups): CSS15 media_ideology 8, persuasion 1, wiki_corpus 1. Pilot: implicit_hate 1 |
| `miracl_v1.0_train` (es, fa, fr, hi, zh) | H8 `miracl_relevance_<lang>` + `miracl_pool_<lang>`, 7,040 | Native-speaker queries and relevance judgments over non-English Wikipedia passages (Zhang et al. 2023). MIRACL-English is excluded as HAGRID's parent | UNRELATED to all 18 | see TyDi QA |
| `jglue_jcommonsenseqa_v1.3_train` | H8 `jcqa`, 6,964 | Japanese crowd-written commonsense questions seeded from ConceptNet (JGLUE). The same dataset as v1 A5, with A5's items excluded | UNRELATED to all 18 | 0 |
| `sentimix_spanglish_train` | H8 `sentimix_spanglish`, 5,986 | Code-mixed Spanish–English tweets with sentence polarity (SemEval-2020 Task 9). The same Zenodo release as A7s Hinglish | emotion: SAME TASK FAMILY ONLY. semeval_stance (pilot): SAME TASK FAMILY ONLY, weak (polarity, not stance; SemEval is only the shared venue, and the 2016 Task 6 collection is separate). implicit_hate and tempowic: UNRELATED (Twitter only; different collections, languages and years) | 0 |

**Gate 10: PASS.**

- No H7 / H8 source is SAME DATASET or SAME PARENT CORPUS with a pilot or CSS15 task, so none is
  excluded.
- SentiMix Spanglish joins the disclosure list (§5) if H8 enters a recipe.
- The text-level hits are shared Wikipedia and news spans. 261 of H7's 307 CSS15-hit groups are
  `media_ideology`. They are a matter for gates 4 and 5 (quarantined; embedding scan pending), not an
  origin link.
- As of amendment 2, no H7 / H8 row had been published or trained on.

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
- **Teacher lineage.** Lux 1.0's pool shares no origin with discourse or implicit hate, or with
  stance (§4c). T trains on 13,106 rows of that pool. V2 trains only on the 2,903 legacy 1.0 rows
  of the base that both arms share. V2's extra teacher coverage (47,759 against 6,430 rows) is on its
  own v2 prompts. The teacher could not have carried same-origin pilot signal into V2.
- **Readouts: none inflated.**
  - Typed DEV is generated.
  - The CSS pilot has no same-origin training row or teacher lineage in T or V2.
  - SELECT700 (`32a4352d…`) and CAL700 (`3e34f6cb…`) each hold 400 GoEmotions and 300 generated
    oracle rows and no CSS row (node-A count).

**Recipes: no revision.**

- XL r1 (the six recipes of `m3b/mx-xl.manifest.json`) stays as it is, and so do the M2 D10
  recipes and controls.
- Gate 10 excludes no H7 / H8 source from the planned r2 (amendment 2 §4). r2 still depends on the
  other gates of amendment 2.

**Formal panel: keep `css_flute_official_train` out.** It is the same dataset as CSS15 `flute`.

- Its only home is root A0 (`rights_clean.train.jsonl`, 7,455 rows).
- 0.6B Milestones 1–2 trained on that file, so any CSS15 flute cell reported for those checkpoints
  is same-task supervised, as the v1 registry states. None of them is a release candidate.
- Milestones 3–4 and all XL recipes use A0s or A0s-strict and contain no FLUTE row.
- The older pipeline's Wikipedia-politeness train complement (`training/data/build_css_wiki_politeness`)
  is the same dataset as `wiki_politeness`. It is in no registry or pool and must stay out.
- The older pipeline's TweetEval stance rows are the same dataset as the `semeval_stance` pilot
  (§4c). They are in no registry or pool and must stay out. Any 2.0 research model trained on the
  4,624- to 8,522-row mixes must report its stance pilot cell as cross-target supervised.

**Disclose, do not quarantine.** Flag emotion, persuasion and reddit_humor as tasks whose family was
seen in training, from a different origin:

- GoEmotions for emotion.
- ArgQ-30k for persuasion.
- OASST1 humour ratings for reddit_humor.
- SentiMix/AfriSenti for emotion.
- SentiMix Spanglish for emotion (H8), only if it enters a recipe (r2).

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
- **Teacher lineage (resolved at dataset level in §4c; residuals).**
  - Lux's final-stage run manifest was not found on either node. Its pool rests on its card and the
    A7 inventory, and the 2,671 re-described semantic-24k options were not materialized.
  - Two groups of text-level hits were not joined to the Lux pool: the two 1.0 rows with a pilot hit
    (A7h 1, A7i 1) and the 1.0 Stage1 file's near-CSS15 contexts (27 RAOP, 1 Indian English; 2.0
    `audit_css_train_near`). They are a few rows at most, and only the two A7 rows concern the pilot.
  - Whether V2's stance drop in every seed pair reflects distilled Lux behaviour is untested. Lux maps
    None to Against on 89 of 435 stance items in the 2.0 open evidence. A test would need pilot gold.
  - Third-party pretraining of Lux's backbone and of `Qwen3-0.6B-Base` cannot be audited. The
    student start is identical in both arms.
  - AutoJev-27B's training data is undocumented and was not audited against CSS. M4 uses no AutoJev
    target, but an AutoJev-distilled XL candidate would need this check.
- **H7 / H8.** Gate 10 is dataset-level only. The text-level check is gates 4–5, and the embedding
  scan is still pending. H7's densest text contact is with media_ideology: 261 of its 307 CSS15-hit
  groups, all quarantined. SentiMix Spanglish's collection years and sampling come from the task
  paper and were not re-verified.
- **Scope.** This audit is dataset-level only. Text-level overlap is covered by the existing lexical
  and embedding scans (PI-v3). Per-task pilot confidence intervals were not computed; they would need
  pilot gold, and the aggregate paired intervals of `m4/contrast-v2.json` are used instead.
