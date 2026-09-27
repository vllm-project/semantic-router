# Sol 2B gold-free zero-step causal preflight

**Status before execution: prospective; no GPU run or result.** The existing
targeted160 → rights-clean-v2 BEST320, its historical SELECT700 zero-step
predictions, and both stopped soft-replay probes stay untouched. No trainer,
teacher construction, optimizer step, CAL fit, DEV/CSS scoring, public rank, or
sealed FINAL/CSS15 label access is part of this preflight.

## Why this is necessary

The historical control used the same FP32 materialized targeted160 source with
a fresh rank16/alpha32/dropout0.05 PEFT LoRA attached before baseline
inference. Its Qwen3.5 gated-delta backend was the installed FLA wrapper. The
subsequent deterministic zero-step probe used the **bare** materialized model
and patched gated-delta to the Transformers PyTorch reference. Source files,
SELECT identities/token hashes, batch mates, Torch/HIP versions, and all 32
categories matched; the largest probability difference was `0.0136505365`
against a frozen `1e-4` historical gate. This result cannot identify whether
the backend, the zero-init LoRA wrapper, or both caused the drift.

The materialized source model SHA-256 is
`2f4bb061e0881d2d5f29da1cedee8655655ce339bacfa4f6971bcd845add5ae9`;
receipt SHA-256 is
`eae632fba65bc3b208aa3698d2334a88cbd5a0a4b7521ade9398fa3be3aacd1c`.
SELECT700 SHA-256 is
`32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`.
The historical prediction file SHA-256 is
`ceda618a37069832af69375bc5905b28da952f8386be972c88dc6a30671c8cb3`.
The fixed 32-ID roster SHA-256 is
`4c247f30d6a879ce20d77acce1a0d9fdbd0d7c9b5a120f6e9b8febd499ad199a`.
These are the same source and sample as the failed probe; no checkpoint search.

## Locked 2 × 2 experiment

Run four fresh processes **sequentially on one exclusive physical GPU** in
this fixed order: `fla-bare`, `fla-lora`, `reference-bare`,
`reference-lora`. Each process verifies the same materialization and SELECT
hashes, uses the original batch-of-two mates and padding, maximum 8,192
tokens with no truncation, FP32 stored weights, BF16 autocast backbone, FP32
head, seed 20260926, eval mode and deterministic-algorithms flag. The LoRA
cells follow the historical attach-before-device order, require every freshly
initialized LoRA B matrix to be exactly zero, and enable the historical
gradient-checkpointing flag before eval. The FLA cells verify the installed
wrapper; reference cells verify its exact pinned PyTorch replacement before
model load. Each cell outputs only 32 input identities, native option
probabilities, category, backend, source hash and process metadata. Inputs
come from SELECT only; no gold field enters scoring.

The signed runner, its SHA-256, pinned container digest, exact code mirror,
private preregistration hash, commands, four output hashes and exit statuses
must be recorded before interpreting outputs. The container is network-off and
the source/data mounts are read-only; private outputs are exclusive-write.
One cell has a hard **45-second process timeout**. Four timeouts sum to at
most **180 GPU-seconds = 0.05 GPU-hours**. Any timeout, GPU conflict, wrong
file or runtime signature, invalid answer, prompt/token mismatch, or unsealed
cell stops the experiment; no alternate GPU, backend, checkpoint, sample or
threshold is substituted inside this preregistration.

## Frozen analysis

For each cell, compare its 32 native predictions with the historical control:
zero category changes and maximum absolute per-option probability drift
`≤ 1e-4` is the original historical gate. Also compare all four factorial
edges: FLA bare ↔ FLA LoRA and reference bare ↔ reference LoRA isolate the
wrapper; FLA bare ↔ reference bare and FLA LoRA ↔ reference LoRA isolate the
backend. Report maximum per-option drift, category changes and the number of
rows over `1e-4` for every edge. Do not average away the maximum.

* Attribute the observed drift to backend alone only if the historical
  FLA-LoRA cell passes, both within-backend LoRA edges pass at `1e-4`, and
  both cross-backend edges fail at `1e-4`.
* Attribute it to LoRA wrapping alone only if the historical FLA-LoRA cell
  passes, both cross-backend edges pass, and both within-backend LoRA edges
  fail at `1e-4`.
* If both types of edges fail, report both effects; if the historical FLA-LoRA
  cell fails, report the historical comparison **unresolved**, even if the
  factorial edges are informative. Interaction or mixed patterns remain
  unresolved. No probabilistic post-hoc relaxation is permitted.

This test only diagnoses zero-step inference. It cannot prove that a replay
training treatment beats its control. No training starts as an automatic
consequence of a pass. A later treatment needs a separately preregistered
common-runtime control and independent process repeatability `≤ 1e-6`, or a
fresh paired control run; the completed control is never relabeled.
