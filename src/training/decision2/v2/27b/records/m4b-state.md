# ~27B M4b state (resume file)

Updated: 2026-09-30 05:45 UTC+8 (M4b worker; **M4b COMPLETE — results `m4b-results-2026-09-30.md`**)
Branch: `xunzhuo/decision-2-training-27b-m4b` (worktree `/home/xunliu/code/vllm-sr-dev2-27b-m4b`; merged into
`xunzhuo/decision-2-training`). Gist file: `06b-decision-2-27b-m4b.md`. GPUs: node B GPU0–2, leases set idle,
nothing running. GPU-h 30.49 of 36.

## Outcome

- **F-b = A1-soup** (full-parameter FF, gold only): post-key v3 **71.684**, +4.47 [+0.19, +8.13] vs DEV2.0-27B;
  −0.45 [−3.68, +3.98] vs AutoJev-27B. Successor rule: PASS with item 4 on card-eligible mlx-diag (Choice + Noul
  −.012 [−.029, +.004]); FAIL if item 4 uses the full macro incl. XNLI (−.017 [−.031, −.003]). **Coordinator decision.**
- F-a (A2-soup, own-Lux soft targets) 70.420 and F-c (A3-s1, AutoJev twin) 68.315: fail item 1. No finalist beats
  AutoJev. The interpolation line was stopped by the prereg merge check (F1M agrees 84.4% < 99%: BF16 merge loses the
  LoRA delta).
- Artifacts (node B, FP32): `/data/dev2/runs/27b/m4b/A1-soup/checkpoint` (`5bcfa3d4…`), packages
  `m4b/F-{a,b,c}/package`, scored runs `m4b/F-{a,b,c}/formal`, verdicts `m4b/F-*/verdicts{,-mlx}.json`.

## Next (for the coordinator)

- Decide item 4's reading. If F-b is a successor, a release worker builds a `qwen-full` package from
  `m4b/F-b/package` (BF16 copy only on exact parity) and a successor gate profile.
- Combine with M4: full fine-tuning of the A20 mixture (M4's dose), gold only, two seeds (≈ 12 GPU-h per seed at
  1,721 tokens/s for 25M tokens), with a human-transfer guard (data, not soft targets).
