# Decision Index 0.2.1 independent reproduction: execution freeze

**Status:** prospective protocol, before any Decision 2.0 Index inference or
score. This is an external diagnostic; it does not replace JevArena FINAL,
cover Score, or create an official Space leaderboard entry. The
[0.2.1 audit](decision-index-v021-audit-2026-09-27.md) records the changes from
the public 0.2 kit and the published Decision 1.0 reference rows.

## Immutable reference and acceptance gates

- Reference Space revision:
  [`ed7d683e6626a4a5ace0086b8bdedd8b1cc58cf4`](https://huggingface.co/spaces/multimodalart/jev-decision-index/tree/ed7d683e6626a4a5ace0086b8bdedd8b1cc58cf4).
  `index-v0.2.1.json` SHA-256
  `5444deeacd9bd6ea9e8ccf008f99f1223fe1d43af6e40739259aec804284ec55`;
  `methodology-v0.2.1.json` SHA-256
  `235384612203690889a4d82ff22dab16fd3c65a6c8f5db0ef28b361f9ed9f665`.
- Public reproduction-kit base revision:
  [`19ad28ec9485493cc4f7fc07d91c178f948e6434`](https://github.com/apolinario/decision-index/tree/19ad28ec9485493cc4f7fc07d91c178f948e6434).
  Retain the kit's source and added-row hashes; record reconstructed artifact
  hashes in the private run manifest. An identical source hash is necessary,
  not sufficient, for 0.2.1 equivalence.
- Implement an explicit 0.2.1 edition, leaving 0.2 unchanged. Freeze the
  exact 120,340 scheduled / 119,898 scoreable **base** request manifest,
  30,419 added requests, and the complete 150,759 scheduled / 150,317
  scoreable request roster. Freeze 442 common base exclusions, 38 headline
  benchmark IDs, 13 gold benchmark IDs, five area weights, and the documented
  native metric/chance-baseline changes. The
  717 requests removed relative to 0.2 must be identified by stable row ID
  and a reproducible rule, not by model outputs.
- The documented Home duplicate rule determines 24 duplicate state pairs,
  but currently does not identify which member of each pair survives. Their
  questions and options differ. A first-in-source choice is provisional
  until an independent row identity or authoritative reference resolves it.
- Before a new model enters the run, test the scorer against all available
  published per-benchmark records and the six Decision 1.0 plus Jev
  area/headline rows. Match the Space's displayed rounding for every
  replayed value. Compare row-level records where published; aggregate
  parity alone does not prove per-request parity. If exact row selection or
  metrics cannot be reconstructed, label the port **partial** and do not
  issue a 0.2.1 score or rank.

## Native engine and run order

1. Freeze one exact Decision 2.0 package, calibration file, model/source
   revision, adapter code, prompts, and container image by SHA-256. The
   already packaged 27B development checkpoint is the first *diagnostic*
   candidate; its other release gates remain HOLD.
2. Call the packaged native Choice/Noul implementation through the kit's
   `Engine(state, questions)` contract. Answer every supplied option with a
   finite probability; report unsupported input as an explicit invalid
   response on the original denominator. Never truncate a request, drop an
   option, use a generic base-model logit in place of the trained head, or
   disclose gold to inference.
3. Run the kit's 86-request compatibility pass and inspect each response
   contract, probability sum, invalid reason, and packaged-versus-source
   parity. This pass is a format diagnostic, not an Index result.
4. Only after corpus, scorer, and native-engine gates pass, run the full
   0.2.1 panel once for the frozen candidate. Verify 150,759 scheduled IDs,
   150,317 scoreable IDs, all per-row outcomes, and every failed or missing
   response. Preserve original raw predictions and logs privately; create a
   public-safe hash/summary receipt. A short, failed, or altered-denominator
   run is **incomplete**, with no projected headline.
5. After a complete run, calculate per-benchmark metrics, five area scores,
   headline, calibration, latency and coverage. Compare against published
   same-edition rows only. For a new 2.0 row, give an *independent 0.2.1
   reproduction* rank and a parameter/score Pareto view using measured
   loaded parameters. Space inclusion or an official rank requires separate
   acceptance by the leaderboard maintainers.

The private run manifest records UTC start/end, GPU allocation and GPU-hours,
corpus/edition/scorer/engine/model/calibration/prompt/image hashes, command,
all failure counts, and the source of each external comparison row. Save the
manifest and a concise public-safe decision in the unified research gist.
No token, private host address, benchmark gold, restricted source row or raw
prediction belongs in a public artifact.

## Separation from the release claim

Published 1.0 Index 0.2.1 rows may be reused for this external comparison;
JevArena 2.0-versus-1.0 paired inference requires both model generations on
the same new frozen panel. The external Index tests only Choice/Noul typed
projections, so a high Index result cannot establish Score capability,
multilingual transfer, unrestricted tool execution, or JevArena release
qualification. Do not select a checkpoint on this public Index, then call
the result independent confirmation.
