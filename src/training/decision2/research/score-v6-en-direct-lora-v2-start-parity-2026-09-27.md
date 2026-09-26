# Score v6 English direct-LoRA v2: zero-step source gate

**Result: PASS for initialization parity only.** This receipt does not report
training gains, selector accuracy, transfer, multilingual ability, or a
release benchmark. The v1 merged-weight BF16 parity result remains failed;
the v2 path loads the frozen PEFT adapter and selected head directly, with no
weight merge and no second adapter.

The preregistered v2 protocol SHA-256 is
`28172bae9644625d653dd5bda99818f61f73d1c3d0fb08ef36e3402abf4325a2`.
The original adapter-plus-base fingerprint is
`d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2`.
The 32-row gold-free roster SHA-256 is
`193404fb2ed3905cbb9e34379400a2f33a40d86aaae971940454c6fe71163bc5`.
The exact signed source used for reference and both arm loaders was
`c2922dc47`; preflight and trainer file SHA-256s are respectively
`3314145dadbc1f7ceda9a993028e5daa3c2e1b3405e7e793dce7c565cfe61b39`
and
`b3417da5f70528671cd8f75608976c5c84dd39ed73a589c30a1f1fea423ef25f`.
All three were run sequentially in one immutable container image on one
physical GPU, with the same tokenizer, native one-question adapter, 1,024
token limit, BF16 backbone, FP32 head, temperature 1, and eval mode. The
private receipt binds the complete model/source-file hashes, adapter/head
hashes, per-input/token hashes, software, hardware, container, and prediction
files.

| Start | Valid / 32 | Same argmax / 32 | Maximum absolute option-probability drift | Frozen gate |
| --- | ---: | ---: | ---: | --- |
| Reference BEST368 | 32 | — | — | Source |
| Arm A direct LoRA | 32 | 32 | 0.0 | PASS |
| Arm B direct LoRA | 32 | 32 | 0.0 | PASS |

All runs had zero truncations. The frozen gate was 32/32 categorical agreement
and drift at most `1e-4` **for each arm**. The private combined PASS receipt
SHA-256 is
`f09802ee3da2921d66949796ab7efdab9da6c6b7dfb5fb48bb472a99e10441fa`.
Reference, A and B native prediction SHA-256s are, in that order,
`c8597d0965268b4614dd9e1ae5c6891399e05e7572402673590cd6115514cd57`,
`a882fa253f535c66608b785fb46c4983dddc89b659c0718acea0511c87aef91e`,
and
`3300ce75aa93f609d3bb21e1919d8db2d1a4aaf38f9083fc03f2c726ed5c39f4`.

The first container invocation set two overlapping ROCm visibility variables;
the runtime then could not initialize a GPU and exited before writing a
prediction. The retried launch used one visibility variable, still on the
same physical GPU and with the identical frozen source, image and evaluation
settings. No output from the failed launch entered the comparison.

No optimizer update or r2 selector-key access occurred before this PASS. The
two-arm trainer now checks this combined receipt's SHA-256 and status, both
arm starts, fixed source/roster/training hashes, and the requested arm before
loading data or constructing an optimizer. A later matched run must still
meet every preregistered training, retention and blind-comparison gate.
