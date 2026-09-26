# Score v6 English direct-LoRA v2: fixed two-arm run start

**Status: running; no result or selector-key use.** The v2 native BF16
zero-step start gate passed for both arms in the preceding signed receipt.
The private fixed-run plan SHA-256 is
`88e69065cd57abc43a57033c3f7cf108091a79adc1f84401c03b639f761b0a50`.
Its source is the exact adapter-plus-base fingerprint
`d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2`;
the combined PASS receipt is
`f09802ee3da2921d66949796ab7efdab9da6c6b7dfb5fb48bb472a99e10441fa`.
The training image ID and source-code hashes match those in the start-parity
note. Before either optimizer start, the source was re-fingerprinted on the
training host, with the expected 35 model and source files.

Both isolated fresh optimizer runs accepted their frozen arguments and wrote
provenance. Each admitted **2,777 TRAIN rows and exactly 474,124 native raw
tokens**, planned **174 updates**, and exposes **63,627,776 trainable
parameters** in the adapter and selected head. The only intended input
differences are the preregistered treatment/control TRAIN SHA-256s and the arm
identifier. Arm A provenance SHA-256 is
`38e21f17299d697c874cb6351effee3a32e1b9630f3beb37adc15fb132ad8f4b`;
Arm B provenance SHA-256 is
`5ffb86431edb4cf0380e5ca71e66cac53deef6e950d315460c3947121504485b`.
No alternate data, merged checkpoint, second adapter, previous optimizer
state, or selector key was loaded.

The fixed settings are one epoch, microbatch 1, accumulation 16, maximum 174
updates, max length 1,024 with no truncation, rank-8/alpha-16/dropout-.05
existing LoRA, cross entropy, fresh AdamW, LoRA LR `2e-5`, head LR `1e-5`,
weight decay .01, warmup .05, seed `20260927`, BF16 backbone/FP32 head, and
gradient checkpointing. Only step 174 is a scheduled save. The parent English
SELECT588 baseline and final retention checks are part of the run; CAL is
validated for lineage and split isolation but not evaluated by training.

This start receipt does **not** imply improvement, completion, independent
transfer, multilingual retention, package parity or release readiness. Both
final checkpoints and all gold-free r2 English native prediction files must
be sealed before opening the r2 key, followed by every preregistered advance
gate. A failure or mismatched step/token count stops the pair.
