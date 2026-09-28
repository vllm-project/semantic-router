# M3a — AutoJev-27B ROCm runtime qualification v2: PASS (2026-09-28)

Preregistration: [`m3a-prereg-v2-2026-09-28.md`](m3a-prereg-v2-2026-09-28.md) (committed at `e48564a47` before
any v2 process; all processes ran from its exact node-A mirror). Count-only report:
[`m3a/autojev-qualification-v2.report.json`](m3a/autojev-qualification-v2.report.json). v1 stays recorded
as failed ([v1 result](m3a-autojev-qualification-2026-09-28.md)).

| Gate | Result |
| --- | --- |
| G0 warm-up coverage | PASS — 106 answered warm-up prompts (48 W + 58 ladder; the 16 longest ladder prompts are over the native limit as intended), every 512-token bin below 7,680 covered, longest 8,096 input tokens |
| G1 identity | PASS — pinned model / revision (attested) / adapter / backend / decision config, image `f83b1d10…`, FLA path, 26,086,635,760 parameters, package `d0b1e161…` and source `550ccd85…` hashes identical in all five processes |
| G2 repeat determinism | PASS, bitwise — P1–P2, P1–P3, P2–P3, P1–P4: 910 / 910 answers identical, drift 0.0 |
| G3 frozen autotune | PASS — the warm-up froze 11 entries (6 `l2norm_fwd_kernel` buckets, digest `cff772eb…`); no process added or changed one |
| G4 spot check (eval node-A predictions) | PASS — argmax 99.61% (2 near-tie flips), p99 drift 0.0149, max 0.056; identical to v1 |

R ∪ S (`c1d6d922…`), sets (`02d91835…`) and W (`f9d958e8…`) were byte-identical to v1; warm-up W2
`b173a3c3…` (122 prompts). Processes: warm-up GPU2 123.2 s; P1 GPU2 134.9 s, P2 GPU3 136.6 s, P3 GPU4 131.9 s
(concurrent); P4 GPU2 122.5 s. **GPU-hours 0.180.** Production started at 12:31 UTC on GPU2–4.
