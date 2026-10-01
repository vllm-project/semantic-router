# Standard model-release cards — state (newest first)

Assignment: user card job 2026-10-01 21:58 UTC+8 (release worker 4c0a68cd): card-only revisions of all six DEV2.0
repositories with a standard, formal model-release card. Worktree `vllm-sr-dev2-automap`, branch
`xunzhuo/decision-2-automap`; node E GPU7.

## Now

- 20:30Z (2026-10-01) — **Round 3 DONE** (user card fix 03:04 UTC+8): the Index charts use the board's
  served-parameter convention, and the footnote adds the row-level audit. Kai 0.6B `dcfb7d3e`, Eos 0.8B `9c7f3ea0`,
  Sol 2B `b42b6ff3`, Nox 4B `54b084f9`, Vega 27B `1efb5cbb`. Lux 9B is untouched, left to its successor release,
  which uses the fixed generator (`065c5bfe1` / `d1a8b5ab4`, `d152f93e5`). The collection is unchanged and no
  leases remain. Record: `dev2-card3-2026-10-02.md`.
- 18:50Z (2026-10-01) — **Round 2 DONE** (COORDINATION 2026-10-02 00:05): rename to `Decision-2.0-<codename>-<size>`
  and product cards. Kai 0.6B `35882d49`, Eos 0.8B `5c878c67`, Sol 2B `73bb148b`, Nox 4B `eff06485`, Lux 9B
  `586af779`, Vega 27B `d97928d3`. Every receipt, gate re-evaluation and Hub smoke passes; former IDs redirect; the
  collection lists all six (an edit at 17:01Z, before these uploads and not by them, ordered them largest first and dropped Route). No leases
  remain. Record: `dev2-card2-2026-10-02.md`.
- 16:20Z — **DONE.** Published 0.6B `38ac8e90`, 0.8B `db7a9259`, 2B `9f6ca45d`, 4B `dd60f0c2`, 9B `a12ef72e`, 27B
  `e7b4a372`; every receipt, gate re-evaluation, hub-links and card-http check passes, and the Hub fresh-cache
  Transformers smoke passes on all six. Published READMEs equal the previews. Leases removed (GPU7's lock directory
  was left empty after the owning track released; removed). Record: `dev2-card-2026-10-01.md`.
- 14:56Z — 4B published first (preview subject), then a sequential chain 0.6B → 0.8B → 2B → 9B → 27B that stops on
  the first failure.
- 14:50Z — Node build of the 4B card byte-identical to the preview; mirror `f85ea4e17`.
