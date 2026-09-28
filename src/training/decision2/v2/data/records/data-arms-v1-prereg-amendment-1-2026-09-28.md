# Data arms v1 — amendment 1: human-source arms A1, A3, A5, A6h (2026-09-28)

Committed before any row of these arms is built. Rules of
`data-arms-v1-prereg-2026-09-28.md` apply unchanged except where stated. Source
rights were screened from primary pages (licence files at the pinned commits, HF
cards at the pinned revisions); prior HOLD sources are not re-screened. Raw
source archives live under `/data/dev2/private/sources/` on node A.

## 0. Clarifications (apply to every arm, including A2/A4/A6)

1. **TRAIN-role overlap.** PI-v2 roles `rights_clean_train` (A0) is a training role,
   not an evaluation or selection role. New-arm hits against it are reported as
   cross-arm duplication and do **not** quarantine. Hits against every other role
   (SELECT, CAL, typed DEV/FINAL, CSS pilot/CSS15, JevBench, Decision Bench,
   multilingual and authored panels, zh bench) quarantine the whole group.
2. **A0p / RP-v1 errata.** FLUTE rows are excluded from A0p and RP-v1 (stricter than
   §3 of the preregistration; FLUTE is a CSS15 task).
3. **Teachers.** Jev outputs are not used as training targets: TypeSafe's Master
   Customer Agreement §2.3(b) forbids using outputs to train or distill models.
   R2 targets come only from our own Decision 1.0 models.

## 1. Pinned sources

| Arm | Source | Pin | Licence (dataset / text) |
| --- | --- | --- | --- |
| A1 | ABCD v1.1 | GitHub `asappresearch/abcd@6b8700ce67c6b37b062dd7a60abc76d7ef832a97`, tarball SHA-256 `60f6addf…ce97` | MIT / crowd-written chats |
| A1 | Schema-Guided Dialogue (SGD) | GitHub `google-research-datasets/dstc8-schema-guided-dialogue@e852981ae34990f4358979625854259302feaa78`, tarball `ff97a9ab…2188` | CC BY-SA 4.0 |
| A1 | QASC | HF `allenai/qasc@a34ba204eb9a33b919c10cc08f4f1c8dae5ec070` | CC BY 4.0 |
| A3 | MuSiQue-Full v1.0 | data HF `bdsaglam/musique@22873a405dd809893b22ada0b499299fb612d2df` (`musique_full_v1.0_train.jsonl`); licence GitHub `StonyBrookNLP/musique@922ac98f19a201998dbdae6d7f2887a5258dbdeb` | CC BY 4.0 / Wikipedia CC BY-SA |
| A5, A6h | KLUE | HF `klue/klue@349481ec73fff722f88e0453ca05c77a447d967c` (ynat, nli, mrc, sts) | CC BY-SA 4.0 |
| A5, A6h | JGLUE v1.3 | GitHub `yahoojapan/JGLUE@6f071c09316baae89c3d083a90985b4b1cb9968c`, tarball `10139619…0ff7` (jcommonsenseqa, jnli, jsts) | CC BY-SA 4.0 |
| A6h | IBM ArgQ-Rank-30k | HF `ibm-research/argument_quality_ranking_30k@590726b3765b1b90c5e53a17e3b1f77d92d3aa8a` | CC BY-SA 3.0 (stricter of card fields) |
| A6h | SAF (EN communication networks, DE micro-job) | HF `Short-Answer-Feedback/saf_communication_networks_english@9358d6f0f87371c0a5f150502b14cb16e382195f`, `…/saf_micro_job_german@30db3effcfc07b638291cfcff248b84dbd8013db` | CC BY 4.0 |

Only publisher TRAIN splits are read. No dev/test file is opened. Excluded from v1
after the screen: FEVER (sibling of held FEVEROUS/VitaminC; claim-only 61.7%),
STS-B (restricted MSR/EMM text), ConditionalQA ("only for NLP research" line),
Qasper, BoolQ, CaseHOLD/LEDGAR (LexGLUE family), HWU64 (MASSIVE lineage), SIB-200
(translated), SemRel-2024 (no licence), MIRACL and TyDi QA (deferred to v2 for size).

## 2. Projections (native System One rows)

Every projection keeps one group per original item group, renders option order by
a deterministic per-row rotation (`sha256("<arm>-v1:" + id)`), and never exposes
source label fields in the state.

**A1 — ABCD.** State = the conversation's customer and agent turns (speaker-labelled;
`action` turns removed because they name system operations). *Choice* `abcd_subflow`:
"Which request is the customer making?" over all subflows of the gold flow
(descriptions from the ABCD ontology/guidelines). *Noul* `abcd_flow`: "Is the
customer's issue about <flow description>?" with a 50/50 true/false split (false =
another flow, chosen by hash). Group = conversation id.

**A1 — SGD.** State = the last ≤ 8 turns up to a user turn where a new non-NONE
intent becomes active. *Choice* `sgd_intent`: the service's intents (schema
descriptions), services with ≥ 2 intents only. *Noul* `sgd_requested_slot`: "In the
latest message, does the user ask for <slot description>?" (true = a requested slot;
false = a non-requested slot of the same service; 50/50). Group = dialogue id.

**A1 — QASC.** State = the two annotated facts; *Choice* over the 8 answer options.
Group = normalized (fact1, fact2) pair.

**A3 — MuSiQue-Full.** State = the 20 paragraphs, each "Paragraph k — <title>:
<text>". *Noul* `musique_answerable`: "Do these paragraphs contain enough information
to answer: <question>?" — each answerable question and its unanswerable twin are one
group (balanced by construction). *Choice* `musique_final_support` (answerable twin
only): "Which paragraph states the fact needed for the final step of answering
<question>?" over the 20 paragraph titles; gold = the last decomposition step's
supporting paragraph.

**A5 — KLUE (ko).** *Choice* `klue_ynat` (7 topics, Korean descriptions; headline as
state); *Choice* `klue_nli` (entails / contradicts / neither, Korean descriptions);
*Noul* `klue_mrc_answerable` (passage + question; `is_impossible`; 50/50). Group =
guid for YNAT/NLI premise, passage hash for MRC.

**A5 — JGLUE (ja).** *Choice* `jglue_jcommonsenseqa` (5 options; question as state);
*Choice* `jglue_jnli` (3 labels, Japanese descriptions). Group = question id / premise.

**A6h — human ordinal Score.** Level counts are assigned per row by hash among the
allowed set; cut points are fitted on TRAIN only; rows within ±0.2 (STS scale 0–5) or
±0.02 (ArgQ 0–1) of a cut are dropped and counted; each level is downsampled to at
most 1.2× the rarest level within (source, L).

- `klue_sts` / `jglue_jsts`: mean rating 0–5; L ∈ {3, 4, 5, 6} (L=6 uses the raters'
  integer anchors; L=3–5 use TRAIN quantiles); level descriptions give the rating
  range in words. KLUE's `source` (round-trip vs sampled) is never shown and is a
  stratification key.
- `argq30k`: weighted-average quality (WA) binned by per-topic TRAIN quantiles into
  L ∈ {3, 4, 5}; kept only where the WA and MACE-P bins agree.
- `saf`: expert verdict Incorrect / Partially correct / Correct → L = 3.

## 3. Sizes, held-out slice, gates

Target TRAIN+AHO rows (caps; shortfalls are recorded, not filled): A1 ≈ 7,000
(ABCD 3,000, SGD 2,000, QASC 2,000); A3 ≈ 3,000 (1,000 twin pairs + 1,000 Choice);
A5 ≈ 6,000 (YNAT 1,500, KLUE-NLI 1,500, MRC 1,000, JCQA 1,000, JNLI 1,000); A6h ≈
6,000 (≤ 1,800 per source). AHO = `sha256(group_id) % 10 == 0`. Shortcut gates are
evaluated per source × task type; a failing source/task is dropped from the arm as
built and listed (the rest of the arm still ships). Overlap, rights, balance and
native-budget rules are as preregistered. A6 = A6g (generated) ∪ A6h (human) is
published as two separately ablatable sub-arms.
