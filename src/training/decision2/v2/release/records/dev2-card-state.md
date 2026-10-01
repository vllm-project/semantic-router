# Standard model-release cards — state (newest first)

Assignment: user card job 2026-10-01 21:58 UTC+8 (release worker 4c0a68cd): card-only revisions of all six DEV2.0
repositories with a standard, formal model-release card. Worktree `vllm-sr-dev2-automap`, branch
`xunzhuo/decision-2-automap`; node E GPU7.

## Now

- 16:20Z — **DONE.** Published 0.6B `38ac8e90`, 0.8B `db7a9259`, 2B `9f6ca45d`, 4B `dd60f0c2`, 9B `a12ef72e`, 27B
  `e7b4a372`; every receipt, gate re-evaluation, hub-links and card-http check passes, and the Hub fresh-cache
  Transformers smoke passes on all six. Published READMEs equal the previews. Leases removed (GPU7's lock directory
  was left empty after the owning track released; removed). Record: `dev2-card-2026-10-01.md`.
- 14:56Z — 4B published first (preview subject), then a sequential chain 0.6B → 0.8B → 2B → 9B → 27B that stops on
  the first failure.
- 14:50Z — Node build of the 4B card byte-identical to the preview; mirror `f85ea4e17`.
