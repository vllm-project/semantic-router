# Lux 1.0 published Score example: batch-shape diagnostic lock

**Status before probe: HOLD.** The published own Lux 1.0 `usage` request has
three named questions. Its current qualified native output reproduces exactly
in both combined-request and fresh isolated-process runs, but the Score
middle-level probability differs from the published reference by
`0.0212842536`, beyond the frozen `0.02` release-example ceiling. The earlier
isolated-process test falsified process reuse. No formal JevArena predictions
or labels were used for this control.

## Frozen identity and one permitted intervention

- Published own package: `llm-semantic-router/Decision-1.0-Lux-9B` revision
  `bd45a30aee8c84032791c245c70f86dee5389cc8`, bundle manifest SHA-256
  `985ade73c509399291d60b5f98e8bbbbe99c0ee0efe611a5f84604f71420e0fd`,
  example SHA-256 `12d68a696b50851b3614c1ed9d5e73347780ca5ad487ee2ad35e746ab40bbb18`.
- The prior single-request native `usage` output SHA-256 is
  `923536bbf4d0f0a971e62e4aca4a9ae74828817e0b1ebd8aeeddae5155111a1c`.
  Its source input, model revision, runtime profile and three-answer IDs were
  previously verified. The script rechecks those exact bytes.
- The unmodified native collector SHA-256 is
  `b49054f1aef7a35c0a65b88dd5c1f1e5e252bb96dc210c7ef9e1d83942bb45ce`.
  The new gold-free prompt/analysis utility SHA-256 is
  `4a8c9aaa1113dc8b1e9175e4b3faeb358fcaeb4373304a17671c438fc0d311a9`.
  It selects the *same state and exact `severity` question* from the published
  request, dropping only `owner` and `active`. It does not alter weights,
  calibration, the Score rubric, option order, answer code or numerical gate.
- Use the exact locally cached qualified Lux image digest from the failed
  run, the same published bundle and native collector, one verified idle GPU
  on that node, and **one fresh process containing one Score-only request**.
  The other authorized node lacks that exact image and is excluded. Stop if
  source/image/model hashes, runtime qualification, prompt derivation or GPU
  isolation fails. Bound the launch to 120 seconds and at most 0.04 GPU-hour;
  no retry, no other question or model, no optimizer, and no formal panel.

The diagnostic computes the maximum absolute difference in Score class
probabilities and expected score between the fixed prior three-question
response, the one-question response and the published reference. The
predeclared interpretation uses `0.005` as a material *diagnostic* change:

1. If Score-only versus the prior three-question result differs by at least
   `0.005`, native batch size/padding shape is a contributing numerical factor.
   Record whether the one-question result moves toward or away from the
   published reference. This does **not** pass the published three-question
   request's unchanged `0.02` gate or prove the original release's batch shape.
2. If it differs by less than `0.005`, changing question batch shape is not a
   sufficient explanation. Continue HOLD and investigate original example
   runtime provenance below the recorded version/profile granularity.
3. Any crash, invalid answer, mismatched prompt/model/runtime, or over-budget
   request is a failed diagnostic. Preserve it and do not relax thresholds.

The eventual 9B JevArena control still needs a separate exact native package
gate and its own complete gold-free prediction freeze before any post-key
same-panel scoring. This probe cannot choose a model, change a release card,
or turn a post-key panel into an unseen test.
