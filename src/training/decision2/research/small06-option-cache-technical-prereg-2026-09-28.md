# 0.6B independent-option prefix cache: technical preregistration

**Status: code-only, no GPU result or new model.** This experiment tests whether
the option-isolation prototype can reuse one Qwen3 prefix without changing
the final hidden state of each independently rendered candidate. It does not
train a decision head, evaluate a benchmark, alter the private first package,
or imply better Choice/Score accuracy. It addresses the measured 20.33× naive
token-work cost for one 255-option synthetic request.

## Fixed technical comparison

- Source: immutable official `Qwen/Qwen3-0.6B-Base` revision
  `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`; verify source files and
  installed Transformers/runtime before GPU model load. The frozen config and
  safetensors SHA-256 are `504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59`
  and `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`;
  the receipt records the tokenizer and all other source-file hashes. No
  other weights or teacher outputs are inputs.
- Requests: deterministic synthetic System One Choice with 2, 3, 10 and 255
  described options over one 512-word state; Noul with two semantic truth
  values; Score with three and ten ordinal levels. They contain no private
  answer labels. Render each option against the same state/question prefix,
  excluding every other option.
- Compare final candidate hidden states from independent full-prefix passes
  against one common-prefix cache followed by branch continuations in chunks
  of eight. For 255 options, compare all cached branches against eight
  deterministically selected full-prefix branches and measure the complete
  cached run. No truncation. The two paths use the same BF16 autocast, device,
  model and tokenizer.
- Technical parity gate: all eligible branches return finite vectors; max
  absolute element drift <= 0.01 and minimum cosine similarity >= 0.999 for
  each checked case. Both orderings of the 10-option case must merely permute
  aligned branch vectors at this untrained stage. A model output or cache API
  exception is a HOLD, never replaced by a different architecture.
- Resource gate: one verified idle authorized GPU, one bounded run, at most
  15 minutes wall time. Record per-case latency and peak allocated memory for
  both methods where run; the 255-option cached path must remain below 40 GiB
  peak and 120 seconds to qualify for a later *training design* review. This
  is a feasibility ceiling, not a claim of product throughput.

The frozen renderer, cache core and GPU probe file SHA-256 values are
`f2646a0c18e99f6c71b6ab5c93c52665a24c59bfabc957fc00d07106a5cba675`,
`c7510d552ce66b6442f3b17ec0b0329329704083396b3c4a2fedefac0cd6505d`
and `198ed4fafad82413603ce0feb4aa11c2c30f0ff18f9bc28b028fa1f4b0bef95f`.
The runtime must execute those exact mirrored bytes. One GPU visibility smoke
precedes model load and is part of the sole bounded run. A failure is retained
as a failed technical screen, not retried on a changed script or roster.

The implementation is intentionally separate from the current Decision 2.0
native inference adapter. Its outputs are hidden vectors with **no trained
scalar readout**, confidence or calibrated probabilities. A later arm would
need a newly preregistered student/control objective and equal token/FLOP
exposure, complete native Choice/Noul/Score scoreability, CPU overlap checks,
zero/one-step saved-model parity, and independent development transfer before
any formal model comparison. The real System One maximum of 255 options must
remain supported rather than silently dropped from evaluation.
