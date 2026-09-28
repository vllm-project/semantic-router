# Eval M3 preregistration — amendment 2: JevArena-C1 build, seal and one-shot scoring (2026-09-28)

Committed after source verification and conversion, and before the salt exists, before
any selection, overlap exclusion or review of selected items. Design basis:
[`sealed-confirmation-set-design-2026-09-28.md`](sealed-confirmation-set-design-2026-09-28.md)
§5b (Option B, no paid annotation) and the 13:40 coordinator decision. Code:
`v2/eval/sealed/` (schema, per-source converters, candidates runner, overlap scanner,
build).

## 1. Sources (19 registry sources checked; 11 admitted)

The 19 = registry A/B sources #1–#18 plus stance-it (#21). Not in scope: the two sources
with no stated licence, the gated arena, and three weaker B− sources (kept as reserves).
Every source was downloaded at its pinned revision into node A private storage. Its
first public release was re-verified from the Hub commit history, GitHub history and
paper dates.

| Source | Outcome | Reason / caveat |
| --- | --- | --- |
| MCJudgeBench | admit (78 new instances only) | 141 paper instances public on arXiv 2026-05-05; responses model-written, labels human |
| ImplicatureX | admit | 7-level mean Prolific ratings; contexts are old public text |
| GAPA | admit | human split only; attention-check filter from the paper |
| HalluTruthQA-4K (test split) | admit, NC-ND | test released 2026-08-01; strong length cues (length balancing applies) |
| stance-it | admit | single annotator; comments public on Instagram before the cutoff (labels new) |
| WB reviews | admit, NC-SA | rows dated on or after 2026-06-01 only; organic star ratings |
| innoduel | admit, NC | single-vote pairwise labels (duplicate-pair agreement 55%) |
| DeliChess | admit | human columns only (Gemini and match columns never read) |
| tutormoments | admit | human annotation passes only; ASR transcripts |
| legal case law | admit | old public opinions (labels new); 42 source groups |
| narrative-gold | admit | adjudicated gold only; Dolma passages (labels new) |
| JudgmentBench | reject | byte-identical mirror of a 2026-05-07 release |
| ClimateCause | reject | same rows on GitHub since 2025-12 |
| esnlir | reject | gold public in a 2025-11 ESNLIR release; connector residue reveals the class |
| EusExams-v2 | reject | 2026 items come from question banks published March–April 2026 |
| SentiTaglish | reject | byte-identical copy of a 2024 dataset |
| factbutcher | reject | 32 post-cutoff claims (at most 12 balanced) |
| NepFakeV2 | reject | 40 post-cutoff items, 7 true; verdict wording in headlines |
| scopejudge | exclude | its offensive-security trajectories cannot pass the blind subagent review pipeline (policy filter), so the review requirement cannot be met; 92% one class |

Added label-blind tasks, to widen Noul beyond one Arabic source: tutormoments moments
split by the parity of the item-id hash into the Choice `moment_type` and the Noul
`is_rapport` (the same human classes); narrative-gold `span_is_event` (Noul, from the
human per-span event judgements).

## 2. Overlap screen (before selection)

`python3 -m v2.eval.sealed.overlap scan` over every admitted candidate against the
manifest of 15 corpora (training data at 5c0255ed, 39a120ca and the current main
revision including the data-v2 `m2/` arms; the protected panels; the eight peer JEV
aggregators), with `--exact-min-tokens 8`. A candidate is excluded if it is OVERLAP
(exact match of ≥ 8 tokens, or 8-gram containment ≥ 0.5), REVIEW with containment
≥ 0.2, or unscreenable (no shingle). REVIEW from a short exact match only (< 8 tokens,
for example a stock phrase in a chat) is kept. Data landing after the scanned revision is
rescanned before any scoring event (section 5).

## 3. Selection (`v2/eval/sealed/build.py`, config `v2/eval/sealed/c1-config.json`)

The salt is 32 random bytes (hex), created once on node A in private storage. Its
SHA-256 is committed in the seal record; it is never printed or committed. Per task, in
SHA-256(salt | task | item) order: drop excluded and duplicate states, then balance.

- `gold_length_rank` when the gold is the unique longest option ≥ 5 points above chance
  (per-item option sets only).
- `length` when a length-only baseline beats the better of majority and chance by
  ≥ 5 points (equal gold counts within length quintiles).
- Otherwise class balance (cap / classes; all of a smaller class).

Caps and group caps are those in the config. Item ids are opaque salted hashes.
Prompts are gold-free System One rows. The selected panel must pass the leak audit
(option surface CLEAN) and the per-task length checks, or the task is rebuilt once with
forced length balancing. A second failure drops the task.

## 4. Blind quality review (independent subagents)

A stratified sample of the selected items goes to three independent reviewer
subagents: 16 per task (8 for tasks with inputs ≥ 4,000 characters), the same sample for
all three. Reviewers see prompts only (no gold, no source name), answer each item and
flag: `ambiguous`, `not_answerable_from_state`, `template_mismatch`, `answer_hinted`,
`wrong_language_or_garbled`, `sensitive_content`. They work on node A private files over
SSH, and nothing is copied off the node.

Decisions are **task-level only**. Items are never dropped for disagreeing with the
human gold, so the gold stays purely human and no AI-agreement filter enters.

- A task is excluded (or its template fixed once and re-reviewed) if at least 25% of its
  sampled items are flagged by at least two of three reviewers.
- The same applies if the reviewers' majority agreement with the gold is not above chance.

Reported per task: reviewer–gold agreement (the AI-agreement estimate, disclosed, not
used for selection), inter-reviewer agreement and flag rates. There is no AI-adjudicated
tier in C1 v1.

## 5. Seal and one-shot scoring

- **Seal.** SHA-256 of prompts, gold, manifest, config, converter tree, candidate files,
  overlap receipt and hits, salt and review receipts, committed before any model sees a
  prompt. Prompts and gold are encrypted at rest (AES-256, key held only by the eval
  track in its local private config) on node A. An encrypted backup goes to the private
  eval-artifacts dataset. No plaintext stays at rest after sealing. Every decryption is
  logged in `ACCESS.log` on node A and in the eval gist.
- **Who is scored.** Only frozen, hash-pinned release candidates, plus the tier's
  Decision 1.0 model and at most two open same-tier peers, each exactly once. Use budget:
  one candidate per size and three scoring events in total. After that the set is
  declared post-key and a successor is built.
- **How.** Before an event, rescan training data that landed after the scanned revision.
  Collect natively with the frozen runner on the candidate's same-node runtime (8,192-token
  packaging; over-budget, missing or invalid answers are failures). Seal predictions
  before gold is decrypted, then score.
- **Metric.** C1 = 100 × mean over tasks of task macro-F1. Choice and Noul use the
  label set; Score uses its levels. Per-type means are also reported, plus Score
  quadratic-weighted kappa, long-input and non-English slices, NC-licensed sources
  shown separately, and paired bootstrap intervals by source group (5,000 draws).
- **No tuning.** No checkpoint selection, calibration or threshold may use C1. Results
  are reported whatever their direction, labelled "independent confirmation (JevArena-C1)",
  separately from post-key v3.
- **Protection.** Training tracks never access `/data/dev2/private/sealed/` or the key. No
  track may train on any dataset in the C1 source registry (all 25 candidates, admitted
  or not). Any breach retires the affected items.
