# Forward-token-budget runtime revisions — state (newest first)

Assignment: user release round 2026-10-01 18:08 UTC+8 (release worker 4c0a68cd), the queued IX1 hand-off
`dev2-runtime-forward-budget-2026-10-01.md`: runtime-only revisions of DEV2.0-0.8B, 2B, 9B and 27B carrying the
forward token budget. Worktree `vllm-sr-dev2-automap`, branch `xunzhuo/decision-2-automap`; node E GPU6–7.

## Now

- 13:20Z — **DONE.** Published 0.8B **`4afea305`** (decision `a9fc2777…`), 2B **`2973ad4a`** (`5a7a26e4…`), 9B
  **`5de3f9ed`** (`57ea2206…`), 27B **`09280791`** (`b24a2b54…`). Each has 0 of 11,053 answer changes before and after
  the download, AutoModel equal to native, the long-input regression passing (split exercised), Hub smoke 5.17 and
  5.18 passing, and post-checks ok. On the 27B, both private recorded requests return 32 / 32 valid answers equal to
  alone. Leases removed; about 4.8 GPU-h. Record: `dev2-budget-2026-10-01.md`.
- 11:15Z — The first 27B `--extra` failed the regression as written on one generated near-tie question (q24, P(true)
  0.4997 alone vs 0.5029 split). The diagnostic shows the same noise on the unchanged single-batch path; the
  regression gained `--tie-margin` (`d300e88a8`), and the 27B was re-staged and passed with q24 listed as a tie.
- 10:30Z — Chains launched (mirror `0ca621e0e`; regression with 48 questions on 0.8B / 2B so the split runs).
- 10:09Z — Leased node E GPU6–7; all four mains equal the auto_map revisions; private requests relayed node D → node E
  (private directory, hashes equal).
