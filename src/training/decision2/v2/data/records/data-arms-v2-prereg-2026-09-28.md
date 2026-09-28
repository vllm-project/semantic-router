# Data arms v2 (Milestone 2): preregistration (2026-09-28)

Signed before any v2 row is generated, selected or scanned. Rules of
`data-arms-v1-prereg-2026-09-28.md` and its amendments 1–2 apply unless changed here.
Frozen results go to `arms-v2-2026-09-28.md`; they may not change these rules. A rule
change is an amendment committed before the affected build.

Why: every v1 arm together is about 17M native tokens and 76% of A0 comes from one
programmatic source. The 9B study measured a 21.9-point loss for a head trained on the
current TRAIN against Lux's co-trained head on identical features, so data breadth and
volume are the binding limit. [A] Decider-4B stage 1 used 742M tokens.

Coordinator decisions in force: base = A0s; cross-tier teachers approved; Jev is never a
teacher and its results are never published; ConditionalQA excluded; a clean CAL revision
is cut by this track.

## 0. Preregistered targets (TRAIN partitions of new v2 arms only)

| Target | Floor | Stretch |
| --- | --- | --- |
| Admitted native tokens (Qwen3.5-0.8B-Base native `encode`) | ≥ 50M | ≥ 150M |
| Independent upstream sources over v1 + v2 admitted arms (a publisher release counts once; each generator arm counts once) | ≥ 30 | ≥ 40 |
| Score | every level count L = 2..10 present; generated grades exactly balanced within L; human ordinal levels ≤ 1.2× the rarest level within (source, L); Score ≥ 25% of v2 TRAIN rows | ≥ 1,000 rows at every L |
| Multilingual | ≥ 12 languages with ≥ 1,000 TRAIN rows; non-English ≥ 30% of v2 TRAIN rows | ≥ 16 languages |
| Long evidence | ≥ 20% of v2 TRAIN tokens in rows of ≥ 2,000 native tokens (cap 8,192) | ≥ 30% |

A shortfall is recorded, never filled by relaxing a rule.

## 1. Rule changes and additions

1. **Protected inventory PI-v3** = PI-v2 (39 roles) + the eval track's frozen multilingual
   diagnostic panel `mlx-diag` v1 (gold-free prompts) + every v1 arm held-out slice (AHO,
   quarantining) + v1 arm TRAIN files and A0 TRAIN (report-only roles, as amendment 1 §0.1).
   Built on a node from hash-verified files; manifest hash recorded before the first v2 scan.
2. **Source isolation additions.** Excluded as sources of any v2 row: the `mlx-diag`
   sources (MASSIVE, PAWS/PAWS-X, XNLI); every Decision Bench v4 record source, including
   HAGRID's parent MIRACL-English; the CSS15 and CSS pilot task datasets and their parents
   (e.g. Stanford Politeness, Conversations Gone Awry, SemEval-2016 stance, implicit hate,
   GoEmotions beyond A0, humour, ideology); v1 items of the same upstream source (MuSiQue,
   KLUE, JGLUE, SGD, ABCD) are excluded by source id so v1 and v2 arms stay independent.
3. **Cross-source group keys.** Sources that share items use one group namespace so a
   group lives in one partition: TyDi QA ↔ MIRACL (normalized question), SQuAD v1.1 ↔ v2
   (passage), MathQA ↔ AQuA-RAT (only AQuA-RAT is used), HotpotQA ↔ 2WikiMultihopQA
   ↔ MuSiQue (normalized question).
4. **Sealed held-out slice (SHO).** Besides AHO (`sha256(group_id) % 10 == 0`, diagnostic,
   read once at a selected checkpoint), every v2 arm reserves
   `sha256("sho-v2:" + group_id) % 50 == 0` among non-AHO groups (≈ 1.8%). SHO rows are
   never uploaded with the arm rows and never read by size tracks; they stay under
   `/data/dev2/private/sealed/` with hashes in the manifest and are handed to the eval track
   for one in-family readout per frozen release candidate.
5. **Instruction language.** New multilingual rows keep the native state and the native
   question text; fixed instruction scaffolding and option descriptions are English (as in
   `mlx-diag`). v1 A5/A6h keep supplying native-scaffold rows for ko/ja/de.
6. **Shortcut gate at scale.** Gates are evaluated per (source, family, task type) cell with
   ≥ 30 rows. A cell above 20,000 rows is audited on a fixed whole-group subsample
   (groups in `sha256("sc-v2:" + group_id)` order until 20,000 rows). A failing cell is
   dropped as built; the rest of the arm ships.
7. **Length-matched removal twins.** Every evidence-removal construction removes the same
   number of sentences (or paragraphs) from both twins, so length and "gap" cues carry no
   label information by construction. A state-length logistic baseline is reported per cell.
8. **Budget.** TRAIN rows ≤ 8,192 native tokens; rows above 1,024 Kai tokens are flagged
   T1a-ineligible (as v1).

All v1 gates stay: rights, item-group isolation (id / group / `input_sha256` across TRAIN,
AHO, SHO, SELECT, CAL and every panel), four-method lexical overlap (exact, short-leaf near,
long-leaf windowed near, rare n-gram) against PI-v3 with whole-group quarantine, embedding
scan (Qwen3-Embedding-0.6B@`97b0c614`, quarantine at cosine ≥ 0.93, 20-pair review sample in
[0.85, 0.93)), state-removed / option-only / hypothesis-only gates at majority + 5 points,
balance, oracles for generated rows, canonical freeze on a node from an exact mirror, and
byte-identical rebuild on the second node.

## 2. Pinned sources (read-only; publisher TRAIN splits only)

| Source | Pin | Licence (dataset / text) | Used by |
| --- | --- | --- | --- |
| HotpotQA (distractor) | HF `hotpotqa/hotpot_qa@1908d6afbbead072334abe2965f91bd2709910ab` | CC BY-SA 4.0 | H3, H6, E11 |
| 2WikiMultihopQA | HF `xanhho/2WikiMultihopQA@612bc5039a457880d9e7d84c3b0a4cf154b70e4f` (licence: GitHub `Alab-NII/2wikimultihop` Apache-2.0) | Apache-2.0 / Wikipedia CC BY-SA | H3, H6, E11 |
| MuSiQue-Full | as v1 (`bdsaglam/musique@22873a40…`, licence `StonyBrookNLP/musique@922ac98f…`) | CC BY 4.0 / Wikipedia CC BY-SA | H3, H6 |
| TyDi QA (primary + GoldP) | HF `google-research-datasets/tydiqa@da78f23f9119363459acbaf46bf89426ff26c259` | Apache-2.0 / Wikipedia CC BY-SA | H5, E11 |
| MIRACL (topics, qrels) + corpus | HF `miracl/miracl@5be20db9509754dadad47689368639fcec739c00`, `miracl/miracl-corpus@d921ec7e349ce0d28daf30b2da9da5ee698bef0d` (es, fa, fr, hi, zh only; wave 3, optional) | Apache-2.0 / Wikipedia CC BY-SA | H5 |
| SQuAD 2.0 | HF `rajpurkar/squad_v2@3ffb306f725f7d2ce8394bc1873b24868140c412` | CC BY-SA 4.0 | E11 |
| QuAC | official `train_v0.2.json` (S3), SHA-256 pinned at download; HF card `allenai/quac@2c1b73f0…` MIT | MIT / CC BY-SA 4.0 | E11 |
| JSQuAD v1.3 | JGLUE tarball as v1 (`yahoojapan/JGLUE@6f071c09…`) | CC BY-SA 4.0 | H5 |
| KLUE-MRC (answerable questions) | as v1 (`klue/klue@349481ec…`) | CC BY-SA 4.0 | H5 |
| CMRC 2018 | HF `hfl/cmrc2018@137f2c45a24275fb68f6961c4d357f46288886aa` | CC BY-SA 4.0 | H5 |
| DRCD | GitHub `DRCKnowledgeTeam/DRCD@b944790de5af02c5fbb7cd9cb1473d27d169eebf` | CC BY-SA 3.0 | H5 |
| GermanQuAD | HF `deepset/germanquad@fff05ceaf2ffbe5b65c7e0c57e678f7b7e1a0581` | CC BY 4.0 | H5 |
| PIAF | HF `etalab-ia/piaf@bda8c063bc7297180796cd835d1974c0bc71c521` | MIT | H5 |
| SQAC | HF `PlanTL-GOB-ES/SQAC@f9928e8819596a601b8887cc5f8598b15d589a82` | CC BY-SA 4.0 | H5 |
| MTOP | official `mtop.zip` (dl.fbaipublicfiles.com), SHA-256 pinned at download; admitted only if its bundled licence is permissive | CC BY-SA 4.0 (to verify) | H5 |
| MultiWOZ 2.2 | GitHub `budzianowski/multiwoz@fe0c8e65cfcd8462bd33c86e35f21addc84ca82b` | MIT | H1 |
| Taskmaster-2 | GitHub `google-research-datasets/Taskmaster@d92cb6af3005f1dc09c39e75e7daf4a04905e00b` | CC BY 4.0 | H1 |
| DBpedia-14 | HF `fancyzhx/dbpedia_14@9abd46cf7fc8b4c64290f26993c540b92aa145ac` | CC BY-SA 3.0 | H1 |
| WinoGrande (XL) | HF `allenai/winogrande@01e74176c63542e6b0bcb004dcdea22d94fb67b5` (licence GitHub `allenai/winogrande` Apache-2.0) | Apache-2.0 | H1 |
| CommonsenseQA | HF `tau/commonsense_qa@94630fe30dad47192a8546eb75f094926d47e155` | MIT | H1 |
| ARC (Easy + Challenge) | HF `allenai/ai2_arc@210d026faf9955653af8916fad021475a3f00453` | CC BY-SA 4.0 | H1 |
| OpenBookQA (main) | HF `allenai/openbookqa@388097ea7776314e93a529163e0fea805b8a6454` (licence GitHub `allenai/OpenBookQA` Apache-2.0) | Apache-2.0 | H1 |
| SciTail | HF `allenai/scitail@0cc4353235b289165dfde1c7c5d1be983f99ce44` (licence GitHub `allenai/scitail` Apache-2.0) | Apache-2.0 | H1 |
| AQuA-RAT | HF `deepmind/aqua_rat@33301c6a050c96af81f63cad5562cb5363e88971` | Apache-2.0 | H1 |
| GSM8K | HF `openai/gsm8k@740312add88f781978c0658806c59bc2815b9866` (human-written problems) | MIT | H1 |
| QuaRTz | HF `allenai/quartz@28c1dbb56caf81799296cb17892fa73402e23464` | CC BY 4.0 | H1 |
| ROPES | HF `allenai/ropes@d59f1e2ee2b423d7c6ba71edd47fceb4158b07dd` | CC BY 4.0 | H1 |
| MLQE-PE (direct assessments) | GitHub `sheffieldnlp/mlqe-pe@2a670a1140416cf80507b5a829659383c878feb8` | CC0-1.0 annotations / Wikipedia CC BY-SA | H6 |
| OneStopEnglish | GitHub `nishkalavallabhi/OneStopEnglishCorpus@37f8db3945cd2f3cc0caafe45674147b224349be` | CC BY-SA 4.0 | H6 |
| KLUE-STS, JSTS, ArgQ-30k | as v1 | as v1 | H6 (A6h2) |

Screened out (recorded): HellaSwag (upstream repository under a DMCA block for WikiHow
text since 2026-09-14); PIQA, Social IQa, SberQuAD, αNLI (no verifiable permissive licence
on a primary page); IIRC (licence not verifiable from a primary page; deferred); HoVer
(needs the HotpotQA Wikipedia abstract dump; deferred to v2.1); SciQ (CC BY-NC); WANLI and
any other set generated by a closed API; MS MARCO, TREC DL, RACE, QuAIL, ReClor, ConditionalQA
(non-commercial or research-only); civil_comments, ESCI, WikiTableQuestions, Spider and the
other Decision Bench sources; MASSIVE, PAWS/PAWS-X, XNLI; Natural Questions (size; deferred).

## 3. Constructions (shortcut resistance by construction)

- **C1 evidence-removal twins (Noul).** Extractive QA with answer spans. State = title +
  passage. False twin: remove every sentence containing a normalized gold answer string
  (and require the answer string absent from the state). True twin: remove the same number
  of random sentences that do not contain it. Both keep ≥ 3 sentences. Instructions: "Does
  the passage state the answer to this question? Question: <native question>". One group per
  (question, passage); exactly 50/50 per source × language.
- **C2 human-judged relevance.** TyDi primary: gold = annotated answer passage; negatives =
  other candidate passages of the same article that no annotator selected (≥ 200
  characters). *Choice* over 4 passages (1 gold + 3 negatives, rotation); *Noul twins*
  (question, gold) true / (question, negative) false. MIRACL (optional): qrels 1 vs judged 0.
- **C3 multi-hop answerability twins (Noul).** HotpotQA: 2 gold + 4 distractor paragraphs
  (true) vs 1 gold + 5 distractors (false); the true twin swaps one distractor for an unused
  one so both twins have one edit; paragraph order shuffled by seed. 2Wiki: same with its
  10 paragraphs. The gold answer string must be absent from a false state (bridge types).
  MuSiQue keeps its native answerable/unanswerable twins (v1 code, larger cap).
- **C4 evidence-coverage Score.** One counterfactual group per question with n supporting
  paragraphs (HotpotQA n = 2, 2Wiki n = 2–4, MuSiQue n = hops): L = n + 1 rows with exactly
  g ∈ {0..n} supporting paragraphs present, padded with the question's own distractors to a
  fixed count. Question, instructions and ordered level descriptions are byte-identical in
  the group and every grade appears once, so state-free baselines are at chance.
- **C5 abstention Choice (D11).** Comparison questions whose answer is one of two named
  entities: options = entity A, entity B, "The paragraphs do not give enough information";
  full-evidence twin (gold = entity) vs one compared entity's paragraphs removed (gold =
  abstain). Gold is abstain in exactly half of each source's rows; rotation by seed.
- **C6 class-balanced Choice.** Fixed label sets (DBpedia-14 classes, MultiWOZ intents of
  the active service, MTOP intents of the utterance's domain): equal rows per gold class,
  rotation by seed.
- **C7 twin multiple choice.** WinoGrande twin pairs share options and flip the gold; both
  twins in one group; dataset option order kept, so positions are exactly balanced.
- **C8 generic multiple choice** (CommonsenseQA, ARC, OpenBookQA, AQuA-RAT, QuaRTz,
  ROPES): rotation by seed; admitted only through the gates.
- **C9 numeric-candidate Noul twins** (GSM8K): "Is the final answer N?"; each problem's
  true answer is also the false candidate of exactly one other problem with the same digit
  count, so candidate numbers carry no label information.
- **C10 ordinal binning** (A6h2, MLQE-PE): L drawn per row by hash from 2..10; cut points =
  equal-frequency TRAIN quantiles per (source, L); rows within a guard band of a cut are
  dropped (STS 0.2 on 0–5; ArgQ 0.02; MLQE-PE 2.0 on the 0–100 mean); level balance ≤ 1.2×
  rarest within (source, L). OneStopEnglish: L = 3 reading levels, the three versions of
  an article form one group.
- **Dialogue** (MultiWOZ 2.2, Taskmaster-2): as v1 SGD — last ≤ 8 turns up to a user turn;
  *Choice* over the active service's intents (C6); *Noul* "In the latest message, does the
  user ask for / specify <slot>?" with true = an annotated slot, false = an unannotated slot
  of the same service, 50/50 per service.
- **Scaled generators** (unchanged code, new seed namespaces): G2 = A2 families
  (`decision2-g2-v1`, 3,000 groups/family), G6 = A6g families (`decision2-g6-v1`, 4,800 rows
  per level count), G4 = A4 v2 paired hard/random distractors (`decision2-g4-v1`, 1,260
  groups/family; files G4h and G4r keep the D4 pairing).

## 4. Arms (factor packaging; every row keeps source and family for sub-arm contrasts)

| Arm | Factor | Families (row caps, TRAIN+AHO+SHO) |
| --- | --- | --- |
| H1 | cross-domain human, short (en) | MultiWOZ intent Choice 4,000 + slot Noul 4,000; Taskmaster-2 slot Noul 4,000; DBpedia-14 4,200; WinoGrande twins 20,000; CommonsenseQA 9,000; ARC all; OpenBookQA all; SciTail (entails vs neutral, Noul, class-balanced) 10,000; AQuA-RAT 10,000; GSM8K twins all; QuaRTz all; ROPES 8,000 |
| H3 | long / multi-hop evidence (en) | MuSiQue twins 16,000; HotpotQA twins 16,000; 2Wiki twins 16,000 |
| H5 | multilingual native | TyDi GoldP C1 twins 4,000 per non-English language (10); TyDi primary C2 Choice 1,500 + twins 3,000 per non-English language; JSQuAD, KLUE-MRC, CMRC 2018, DRCD, GermanQuAD, PIAF, SQAC C1 twins 6,000 each; MTOP intent Choice 2,000 per non-English language; MIRACL C2 1,500 per language (optional wave 3) |
| H6 | human and human-derived Score | A6h2 (KLUE-STS, JSTS, ArgQ at L = 2..10) 1,800 per source; MLQE-PE 3,000 per language pair (7); OneStopEnglish all; C4 coverage: HotpotQA 9,000, 2Wiki 9,000, MuSiQue 6,000 |
| E11 | evidence removal and abstention (D11, en) | SQuAD 2.0 C1 twins 16,000; QuAC C1 twins 10,000; TyDi-English C1 twins 4,000; C5 abstention: HotpotQA 4,000, 2Wiki 4,000 |
| G2 / G4 / G6 | generated verifiable / hard negatives / Score | as §3 |

Caps keep whole groups in seed-hash order. Sources are processed in parallel on node CPUs.

## 5. Freeze, isolation, publication

As v1: canonical JSONL per partition (train, aho; sho private), manifest with counts by
type / language / source / family / Score L × grade / Choice positions / Noul balance /
native tokens under Qwen3-0.6B-Base@`da87bfb6`, Qwen3.5-0.8B-Base@`dc7cdfe2` and raw
Kai-0.6B@`7185f514`, licence table, receipt hashes, builder commit and tree. Isolation check
over SELECT, CAL, CAL-v2, every v1 and v2 partition. Admitted arms go to a new revision of
the private dataset `llm-semantic-router/decision-2.0-training-data` under `v2/arms/`;
restricted source archives and SHO rows stay under `/data/dev2/private/`. Each arm is
announced in gist `02-decision-2-research-data.md` as soon as it is frozen.

## 6. Clean CAL revision (CAL-v2)

CAL700 minus the one GoEmotions group (2 rows) flagged against a CSS15 item → CAL698,
same rows otherwise; content hash, manifest and isolation receipt; uploaded as `v2/cal/CAL698/`.
Release candidates fit final per-type temperatures on CAL-v2.

## 7. Teachers

**Policy.** A teacher is permitted when its weight licence allows using outputs to train
models (Apache-2.0, MIT; not NC, research-only or anti-distillation terms), it runs locally
from pinned weights (no hosted API), and it is never a weight origin. The teacher's
documented training-data provenance is recorded and disclosed. Jev, JPT (CC BY-NC) and
Hopper (research-only) are excluded. Candidates screened: own Lux1 (weights
`cdf4d3ef`, runtime-bearing `bd45a30a`), AutoJev 27B `6f5b557e` (Apache-2.0; its public
pipeline includes SFT rows generated with a closed OpenAI model — disclosed; see §7 note),
Decider 4B `eb5fbdfc` (Apache-2.0), own Nox1 (`0bb83350`).

**Screen TS-v2** (never-trained rows only): v1 AHO slices of A1, A2, A3, A4v2h, A5, A6g,
A6h, plus up to 300 rows per (arm, type) from the v2 AHO slices once frozen; CAL-v2 fits
per-type temperatures. Metrics per type and arm: valid rate, accuracy, mean gold
probability, NLL, Brier, top-label ECE (15 bins); Score expected-level MAE, RPS and
accuracy per L; raw and temperature-scaled. **Selection rule:** per type, the canonical
teacher is the permitted teacher with the lowest temperature-scaled Brier on the TS-v2 AHO
rows with valid rate ≥ 0.99 on in-cap rows; a difference below 0.005 prefers own Lux1.

**Production.** RP-v2 = fixed stratified pool from v2 TRAIN (never AHO/SHO): all
types, Score-weighted, ≤ 8,192 native tokens, about 150,000 rows (`sha256("rp-v2:" + id)`
order within (arm, source, type) quotas). Lux1 targets on all RP-v2 rows (native runtime,
attested revision, prompt digest per row, repeat-run check on 256 rows); a non-Lux teacher
that wins a type gets a second target file for that type's RP-v2 rows. Targets are stored as
`teacher_probs` replay rows with teacher id, revision, image, runtime, temperature and hashes.
The Lux TRAIN file `752b7c8f…` (decoder track; cross-checked by the 9B track) is registered
as the canonical own-Lux target file for A0 TRAIN.

**GPU estimate.** Measured Lux1 native throughput ≈ 20 prompts/s (1,517 RP-v1 prompts in
116 s incl. load). Screen ≈ 0.5 GPU-h (Lux 0.05, Nox 0.05, Decider 0.05, AutoJev 0.3); PI-v3
embedding scans of all v2 arms ≈ 1.0 GPU-h; Lux1 on RP-v2 ≈ 3–5 GPU-h; a second teacher on
the Score part of RP-v2 (≈ 60,000 rows) ≈ 2–3 GPU-h (27B). Total ≈ 7–10 GPU-h: node B GPU7
plus a request for three more GPUs for about four hours.

Note on AutoJev: the brief allows third-party decision models as teachers and its weights
are Apache-2.0; because its documented pipeline used closed-model-generated SFT rows (the
reason Jev itself is excluded is baking a competitor's outputs into our weights), AutoJev
targets, if selected, ship as a separate file that release candidates adopt only with a
recorded coordinator decision; Lux targets are complete on their own.

## 8. Mixture recipes (D10) — rules fixed now, manifests frozen after the arms

Every recipe = A0s in full (retention anchor; own-Lux targets available) + RP-v2 rows by
target token shares of the v2 portion: Score 30% (H6 12%, G6 12%, v1 A6g/A6h 6%), H3 long
evidence 15% (0% in T1a variants at the 1,024-token Kai cap), H5 multilingual 15%, H1 15%,
E11 10%, G2 + G4 15%; per-source cap 8% of recipe tokens; English ≤ 60% of recipe tokens;
optional own-1.0 R2 replay 10% for own-1.0 continuation (D7 guidance). Budgets per epoch:
S 8M, M 20M, L 40M native tokens; recommended M for 0.6B–4B, M or L for 9B, S or M for 27B.
Matched-token controls per recipe: (i) A0s ∪ A0s-resample (template S), (ii) A0s ∪ v1 arms,
(iii) the same recipe with hard labels only (no teacher targets). Sampling is deterministic
by `sha256("mx-v2:" + recipe + id)`; manifests list row ids, arms and hashes.

## 9. Waves and announcements

Wave A: CAL-v2, PI-v3, G2/G4/G6, A6h2 → announce. Wave B: H3, E11, H6 coverage + MLQE-PE →
announce. Wave C: H1, H5 → announce. Then the teacher screen, RP-v2 and Lux targets, the
recipes and the registry. Every admitted arm is announced with hash, HF revision and counts.
