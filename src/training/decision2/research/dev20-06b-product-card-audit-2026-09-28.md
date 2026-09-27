# DEV2.0-0.6B product-card and repository audit

Scope: read-only review of the checked-in 0.6B product renderer and the last
recorded exact-revision private Hub readback. No model, Hub repository, GPU
experiment, score, or collection was changed in this audit. The private Hub
revision was **not freshly queried** here; the last recorded authenticated
readback is `f9cbfb192e89c4cf48d3e1a61c959190ba684df0` in
`qwen3-06b-product-card-copy-v3-2026-09-28.md`. Recheck the exact revision
before any later visibility or package change.

## What already meets the product contract

| Requirement | Current evidence |
| --- | --- |
| Download-count query file | `full_bundle.py` puts `config.json` at repository root; `download_config.py` lists existing model, tokenizer, head and calibration paths. The last Hub readback recorded its hash as `74bc2bb167d94243f306e91e69ccdee07ee06c998c06bcb0d46d4cbf4ac157a8`. Hugging Face's default download-count query file is root `config.json`; this establishes the eligible layout, not a prediction of a private repository's download counter. |
| Clean repository | The builder has a strict product-content whitelist and manifest-bound model/runtime inventory. The last readback recorded 30 files: model, native loader, root config, calibration, README, notices, evaluation appendix and three charts. No training rows, optimizer state, raw predictions, private logs or temporary checkpoints were reported. |
| Family visual | The banner is a mosaic sticker owl with the exact `DEV2.0-0.6B` name. The card references it through a local `assets/` path. |
| Decision interface | The executable example accepts one `state` and a named `questions` map containing Choice, Noul and Score, then prints `answers`. The packaged `Decision2.system_one` checks structured state and uses the native decision head; this is a local library example, not a hosted HTTP or chat API. |
| Matched evidence | The first table uses the same v3 8,147 original items and separate JevBench v1.2 public 231 items for the three displayed models. It reports the overall paired interval and the Choice/Score regressions. Charts are JevArena rank, JevArena task matrix and JevBench public rank; the product package contains no Pareto chart or SOTA claim. |
| Source and license | The card names pinned official `Qwen/Qwen3-0.6B-Base` as the direct weight source and uses Apache-2.0 metadata. The package keeps the upstream license, NOTICE and attributions separately. |

## Changes worth making before a broader family template

1. **Put the decision product before its leaderboard.** The 1.0 Nox card
   introduces Choice/Noul/Score before measured capability. The 0.6B renderer
   currently places the full table and three figures before “Three ways to
   decide”. Move the three-type use-case table ahead of the results and make
   the lead example visible before the detailed charts. Keep the full measured
   table and figures available immediately below; do not remove caveats or
   turn this into a general chat-model card.
2. **Keep transport claims exact.** The Python `system_one` example has the
   System One request shape and a previously recorded successful exact-package
   three-type execution. It does not prove an HTTP `/v1/systemone` deployment.
   If an HTTP example is added, exercise it against the actual Decision
   serving runtime first; do not substitute a chat-completions example.
3. **Preserve the limitation next to the result.** The +2.582 v3 point estimate
   versus Kai 1.0 has paired 95% interval `[-2.032, +7.846]`; Choice and Score
   declined on this panel. Continue to call this a post-key, same-panel
   comparison requiring independent confirmation. It does not support a
   “major gain”, Pareto or SOTA headline for 0.6B.
4. **Repeat the final Hub readback on every new package.** Verify private
   status, exact revision, root `config.json`, strict file inventory, all local
   README links, actual downloaded model and native three-type output parity.
   The recorded earlier readback applies only to its exact files and cannot
   certify a newly regenerated family member.

Primary references: [Hugging Face download stats](https://huggingface.co/docs/hub/models-download-stats),
[Decision 1.0 Nox-4B card](https://huggingface.co/llm-semantic-router/Decision-1.0-Nox-4B),
[Decision 1.0 Kai-0.6B file tree](https://huggingface.co/llm-semantic-router/Decision-1.0-Kai-0.6B/tree/main),
and [System One API shape](https://docs.typesafe.ai/api).
