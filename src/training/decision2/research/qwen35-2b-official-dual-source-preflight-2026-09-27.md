# Official Qwen3.5 2B Base versus Posttrained: matched native source study

This prospective paired-source study follows the own-Sol-origin 2B
candidate's formal v3 HOLD. It does not use that candidate, Decider or any
other third-party Decision weights as an initializer. The own Sol1 control
and failed candidate keep their exact recorded scores; no formal v3,
public231 or CSS pilot labels select the source arm.

## Pinned source and shared inputs

| Arm | Official source revision | Config SHA-256 | Weight SHA-256 | Tokenizer SHA-256 |
| --- | --- | --- | --- | --- |
| Base | `Qwen/Qwen3.5-2B-Base@b1485b2fa6dfa1287294f269f5fb618e03d52d7c` | `ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4` | `928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d` | `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927` |
| Posttrained | `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc` | `ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4` | `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1` | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |

Both sources are official Qwen Apache-2.0 releases downloaded at pinned
revisions by the HF CLI. Use the same locally signed native segmented-option
prompt, Qwen3.5 text body, new 256-dimensional dynamic-option head and
seed 20260926. Rights-clean v2 TRAIN 7,455 SHA-256
`61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
SELECT 700 `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`,
and CAL 700 `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`
are identical between arms and isolated by source/group. A read-only full
native length audit admits all rows without truncation at 8,192 tokens for
both tokenizers: TRAIN 4,194,465 encoded tokens, maximum length 6,596;
SELECT/CAL also have zero overlength. Base audit SHA-256
`c18c92dfaeb8ffffb8d5b1aadd42ba242ba275d9388db23870b4577a1b839b38`;
Posttrained audit SHA-256
`1166c47d5fc87f87e1d30275684f4c19a26b66317a4bf7efb8e2723330e80493`.

## Ordered pre-optimizer admission

For each arm separately, on an exclusively reserved GPU:

1. Verify source/data SHA map again in the runtime. Make two independent
   fresh zero-step source starts and complete SELECT700 native inference.
   Require 700 valid each, same ordered input IDs, zero category changes,
   p99 probability drift ≤.005 and maximum ≤.02. Cap each arm's two
   starts at **0.30 GPU-hour** combined.
2. If step 1 passes, perform exactly one update from a *new* independent
   official source initialization. Fixed recipe: LoRA rank16, alpha32,
   dropout .05 plus head, CE+0.5 Brier, LoRA LR `1e-4`, head LR `2e-4`,
   microbatch1, accumulation16, clip1.0, AdamW decay .01, seed20260926,
   BF16 backbone computation and FP32 parameters/head. Require finite
   loss/gradient, saved checkpoint, then native same-batch first-32 SELECT
   reload with zero category changes and p99/max drift ≤.005/.02. Cap
   smoke plus reload at **0.15 GPU-hour** per arm.
3. A passed arm becomes eligible for a *separately signed* one-epoch
   matched-budget training protocol. If both pass, train both; do not choose
   one because its zero-step or one-update SELECT happens to be higher.

Stop an arm on missing source file, invalid native type, OOM, nonfinite
value, timeout or failed parity. Preserve its failed receipt and never
relax a threshold within this protocol. No optimizer beyond one step,
CAL tuning, DEV/CSS/public or formal comparison occurs during admission.
