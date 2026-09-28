# Sol 2B ShARC source: narrow normalization diagnostic v2

**Prospective CPU metadata diagnostic only.** [V1](sol2b-sharc-human-policy-source-v1-hold-2026-09-28.md)
failed its exact-tuple feasibility gate: all 628 publisher tree IDs varied in
at least one exact rule/source/question field, yielding zero pairs. That
result and its private receipt remain unchanged. V2 asks whether harmless
case, Unicode, terminal punctuation or whitespace differences account for
the failure. It does not inspect model results, admit rows or authorize GPU.

- Reuse the byte-identical original publisher TRAIN archive whose SHA-256 is
  `72dca3f4f3ba73b1d796b40e952a80d53cd2011ef90b2168b8bcaa818f5edd1e`.
  Do not open DEV/TEST members or train/eval protected panels. Run only the
  signed v2 script on the authorized remote CPU, with exact code-mirror hash.
- Within each original `tree_id`, retain `source_url` exactly after outer
  whitespace strip. Normalize `snippet` and `question` independently with
  Unicode NFKC, casefold, collapsed whitespace and stripped terminal `.?!`
  only. Do not delete words, numbers, negation, named entities or internal
  punctuation. Count exact and normalized field variation without printing
  any raw text or free-text answer class.
- A possible Choice pair requires one exact `Yes` and one exact `No` TRAIN
  label, distinct utterance IDs and visible scenario/history, and the same
  original tree and normalized `(source_url, snippet, question)` tuple.
  Count at most one deterministic hash-selected pair per tree. Preserve the
  original pair/group IDs and the two possible option orders as one group.
- The unchanged feasibility floors are 100 independent trees, 50 URLs and
  100 normalized snippets. Failure remains HOLD. Passage-level semantic
  identity, native lengths, answer-blind evidence necessity, source rights,
  all train/protected overlap, shortcut and token-budget gates would still
  be necessary before any training data are admitted.

The v2 receipt may contain only aggregate counts, archive/script hashes and
feasibility status. It must not list publisher snippets, questions, scenarios,
histories, follow-up answers or raw candidate pairs.
