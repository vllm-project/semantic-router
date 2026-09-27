# ANLI TRAIN as a candidate native three-level Score source

**Prospective CPU-only screen.** This protocol is frozen before reading any
ANLI TRAIN labels or downloading the TRAIN files. It may produce a candidate
source pool for matched 0.8B, 2B, 4B, 9B and 27B data experiments, but it
does not authorize training, a release claim or a model evaluation. The
previously screened ANLI dev rounds are openly labeled **development** data.

## Pinned universe and semantic task

- Publisher: [`facebook/anli`](https://huggingface.co/datasets/facebook/anli/blob/main/README.md),
  revision `8e4813d81f46d313dac7892e1c28076917cfcdf9`, `plain_text`
  `train_r1`, `train_r2`, `train_r3`; publisher card declares 16,946, 45,460
  and 100,459 rows respectively. Read actual TRAIN bytes only after this
  protocol is signed. Obtain those bytes with HF CLI on an authorized private
  SSH node and record each file SHA-256.
- Publisher labels map to the native ordered Score rubric as
  contradiction→0 (“evidence contradicts”), neutral→1 (“evidence is
  insufficient”), entailment→2 (“evidence supports”). `reason` and publisher
  UID are never part of the native request. Map a whole original premise and
  hypothesis to the existing System One Score input, with the same three
  criteria as the prior ANLI dev screen. Source-label/rubric agreement,
  especially neutral, requires a separate blinded semantic review.
- Source license is CC BY-NC 4.0. Publisher paper describes HotpotQA-derived
  Wikipedia for R1/R2 and several additional original corpora for R3; these
  source conditions and any reuse risk must be recorded before candidate
  redistribution or model release. This screen does not infer weight rights
  from the dataset card.

## Whole-group candidate selection, fixed before label access

1. Group **all three TRAIN rounds and all dev rounds** by the NFKC,
   case-folded, whitespace-collapsed complete premise. A premise repeated in
   dev is excluded from every TRAIN candidate. A TRAIN group appearing in
   multiple rounds is excluded rather than assigned to a convenient round.
   Groups must never be split. Near-duplicate premises and suspicious
   original-corpus matches are quarantined at group level, not pair level.
2. Rank eligible groups independently within each round by SHA-256 of
   `decision2-anli-score-train-v1` + NUL + normalized premise, ascending; use
   normalized premise as the deterministic collision tie break. This ranking
   sees no label or model outcome. Within a group retain the source file
   order. Do not search seeds or reorder after observing class proportions.
3. Greedily include a complete group only if it fits all frozen ceilings:
   **4,000 rows per round, 12,000 rows overall, 3,600,000 complete native
   Score prompt tokens overall**, counted with the official Qwen3.8-27B
   tokenizer revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
   Reject a group with any request over **4,096 tokens**. If a group does not
   fit a ceiling, skip the whole group and continue down the fixed hash order.
   Other size tokenizers and context limits require their own later preflight.
4. Selection uses only input text, fixed group identity and token length. It
   cannot stratify by label. After selection, report class proportions and
   group counts. If any relation occupies under 15% of selected rows in a
   round, HOLD rather than resampling. The selected rows are candidate TRAIN
   only; existing rights-clean SELECT/CAL remain untouched.

## Admission gates and required aggregate evidence

- Verify source file hashes, schema, row counts, UID uniqueness, label domain,
  missing premises/hypotheses, group size and cross-round group reuse. Report
  by-round class counts and group counts; no raw text, IDs or per-row targets
  in public outputs. Compare all TRAIN premise groups with all ANLI dev
  premise groups; exact group overlap is a mandatory exclusion.
- Run exact raw, NFKC/whitespace-normalized and bounded near input-overlap
  against the pinned eight-role projected manifest SHA-256
  `26bbaf82eb1c30c0f2093c70d27718fab6731ea8c80e9a451547b03bd30897e1`.
  Check the whole source for exact duplicates and the selected candidate pool
  for bounded near matches; quarantine all matching premise groups. If a
  required role, source identity or complete projected input is unavailable,
  HOLD. A lexical zero cannot prove source-corpus or semantic disjointness.
- Audit full native Score prompt tokens for the selected pool, separately by
  round and class. The explicit criteria order is fixed 0/1/2; report the
  majority-label shortcut and original file-position modulo 3/9 label
  association by round. Report hypothesis-only character length buckets,
  negation, quantifier and date/number cue association with label using a
  fixed, simple cue inventory. Do not fit or run a model in this screen. If a
  single label exceeds 50%, position association exceeds the round majority
  by over 5 percentage points, or a cue with at least 50 examples has a
  conditional-label lift over 2.5, HOLD for shortcut review. These are
  screening triggers, not retrospective parameters.
- The publisher dev data remain public and may have been seen by generic
  foundation pretraining. A clean lexical screen only permits a later matched
  training proposal with open-dev diagnostics; it never makes ANLI dev an
  untouched test. No candidate pool is uploaded or used for training until
  source-level rights, neutral-rubric review, overlap and shortcut gates are
  independently cleared.

## Stop and reporting rule

If any gate fails or required evidence is unavailable, preserve aggregate
counts, hashes and reason code and mark **HOLD**. Do not change the hash seed,
ceilings, tokenizer, rubric, shortcut triggers or previously locked source
after observing labels. This source screen uses **0 GPU-hours** and does not
run an inference model.
