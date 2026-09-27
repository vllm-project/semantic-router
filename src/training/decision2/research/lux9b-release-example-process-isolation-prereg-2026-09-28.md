# Lux 1.0 release example: process-isolation diagnostic lock

This is a gold-free diagnostic of the native package consistency gate, not a
JevArena score. The Lux 1.0 control remains **HOLD**. No formal panel or
answer key may be read in this experiment.

## Fixed evidence before the probe

The published own-model revision is
`llm-semantic-router/Decision-1.0-Lux-9B@bd45a30aee8c84032791c245c70f86dee5389cc8`.
Its bundle manifest SHA-256 is
`985ade73c509399291d60b5f98e8bbbbe99c0ee0efe611a5f84604f71420e0fd`;
the published five-answer example SHA-256 is
`12d68a696b50851b3614c1ed9d5e73347780ca5ad487ee2ad35e746ab40bbb18`.
The fixed release runtime image is
`sha256:ce895822fc48bb6864911d4488a3946f3a18fd3dd2ec90c8a0a49b259145f2fb`.
Both current request payloads match the published `state`/`questions` exactly,
including canonical input hashes. The package revision is attested, the
qualified runtime matches, and all five categories agree. The only failure
of the prospective 0.02 numerical gate is the `usage` Score class
probability: maximum absolute drift `0.02128425356890451`. The published
example records **isolated** README and usage examples, whereas our collector
loads one model and runs the two requests sequentially in one process.

## One bounded intervention

Run the **unchanged** native collector twice in separate fresh processes: one
process for the published `model_card` request and one for `usage`, each using
the exact same package, image, adapter source, input, and ROCm profile as the
failed combined-process check. Compare all five answers to the published
example with the existing `verify_lux9b_release_example.py` gate, after joining
the two prediction records. Keep the fixed 0.02 ceiling and zero category
changes. Retain original per-answer numbers, input fingerprints, image and
package hashes, elapsed GPU seconds, and stderr privately. Use one verified
free GPU and cap both launches together at 300 wall seconds. Do not rerun the
combined process; its immutable output is the already recorded comparator.
Do not edit weights, prompts, temperatures, package code, threshold or formal
scoring protocol. One intervention only; no retry if it fails.

Interpretation was fixed before the intervention:

- If both fresh processes pass the same five-answer gate, process reuse is a
  supported cause of the earlier consistency failure. This does **not** turn
  Lux into a v3 result. A later separately pre-registered same-panel control
  would need process-isolated native inference and a new gold-free freeze.
- If either process fails, process reuse alone is falsified. Lux remains HOLD
  while package/runtime numerical behavior is investigated by a new protocol.
- A crash, incomplete output or unqualified runtime is a failed diagnostic,
  not a reason to raise the threshold or count any benchmark score.

No model publication, HF mutation, training, or formal JevArena/JevBench
inference is authorized by this lock. Private machine and experiment paths
must not enter public code or research notes.
