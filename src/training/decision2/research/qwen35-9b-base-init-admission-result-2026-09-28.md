# Official Qwen3.5-9B Base initialization: technical admission

**Technical PASS; model quality and release remain unproved.** This completes
the zero-step and one-update gates frozen in the [prospective Base contrast](qwen35-9b-base-init-contrast-prereg-2026-09-28.md).
Both processes started independently from the pinned official Base revision;
neither inherited the earlier Posttrained control or the other Base process's
checkpoint. The same one-GPU, offline ROCm image and byte-identical seven-file
trainer used by the Posttrained control were verified before each start. The
source, TRAIN/SELECT/CAL and native tokenizer audit hashes matched the frozen
protocol. No formal, public or CAL labels were scored.

| Gate | Observation |
| --- | --- |
| Zero-step native SELECT | 700/700 valid; 254/700 correct; family macro .322101; prediction SHA-256 `44b5c3582e3415e34bb13fd215184645cc24f72e72c25872bb24b720b50dc23d` |
| Independent single update | 1/1 complete; 16 TRAIN rows, 13,328 tokens; finite loss 1.583261 and gradient norm 7.044865; peak GPU allocation 100.830 GiB |
| Single-update native SELECT | 700/700 valid; 239/700 correct; family macro .334758, Brier .493900; prediction SHA-256 `e8bc91313c51bff423b640093feaa949d35a0870cd02ccaf64e4675bfb38ec2e` |
| Fresh saved-checkpoint reload | 32/32 identity matched, Choice 13 / Noul 14 / Score 5, zero category changes, zero p99 and maximum probability drift; comparison SHA-256 `69e74e25634fd7129456e513e4b00e507c1ba7aabccd86dc2ca07aafaf9cb579` |
| GPU time | 55 seconds zero-step + 88 seconds one-update + 31 seconds reload = **174 single-GPU seconds, 0.04833 GPU-hour**, below the frozen 0.25 GPU-hour admission cap |

The single-update SELECT Brier is worse than zero-step, and the number correct
fell by 15. The preregistered gate required valid output, finite numerical
behavior and reload fidelity; it did not use this diagnostic to choose a
checkpoint. Technical PASS admits exactly one fresh official Base 458-update
arm, started from the original official source with the frozen SELECT-only
selection rule and a four one-GPU-hour cap. It does not imply development
transfer or release quality. Training at 4,096 native tokens also leaves
long-context gain unproven.
