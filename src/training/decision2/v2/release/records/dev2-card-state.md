# Standard model-release cards — state (newest first)

Assignment: user card job 2026-10-01 21:58 UTC+8 (release worker 4c0a68cd): card-only revisions of all six DEV2.0
repositories with a standard, formal model-release card. Worktree `vllm-sr-dev2-automap`, branch
`xunzhuo/decision-2-automap`; node E GPU7.

## Now

- 04:55Z (2026-10-02) — **Three-hour check done** (worker 4c0a68cd). Vega `b689ee66`, Lux `7195360d` and Sol
  `8ed41433` are fb5dd490's card-only revisions (entry below). Eos `3de61185` landed at 04:44Z after fb5dd490 skipped
  it for b49d1f36's release. All four carry banner A, byte-identical to the preview. Nox stays `54b084f9` and is left
  to 5e7b8132, its single publisher, which has a release pending (COORDINATION 12:30). Worker 4c0a68cd publishes no
  further revisions.
- 04:20Z (2026-10-02) — **Round 4 roll-out** (user request 10:35 UTC+8, banner A on every card now; card worker
  fb5dd490, worktree `vllm-sr-dev2-card4-rollout`). Card-only revisions, one at a time, each after a `main` check,
  a read of the owner's records and a check that no publisher runs on any node:
  - Vega 27B `1efb5cbb` → `b689ee66` (node E GPU7);
  - Lux 9B `259a4550` → `7195360d` (node A GPU1, from its K-a13IB card with that card's own Index input);
  - Sol 2B `b42b6ff3` → `8ed41433` (node E GPU7; 12:00 UTC+8 check: no 2B release running or imminent).

  Skipped, their releases carry banner A: Nox 4B (5e7b8132's `4b-LHA10SD` release under the 11:35 directive, in
  its last measurement; its driver pins `54b084f9`) and Eos 0.8B (b49d1f36's release imminent at the 12:00 check).
  Every Hub check and the verification pass; no lease remains. Record: `dev2-card4-2026-10-02.md` section 4.
- 02:10Z (2026-10-02) — **Round 4 (banner concept A):** default generator `cc7151c5b` / `032f33009`, merged at
  01:51Z. Kai 0.6B published as `479ea8d1`; verification passes, and the lease is removed. Eos, Sol, Nox, Lux and
  Vega are left to their pending releases. The three-hour check is due at 04:51Z. Record:
  `dev2-card4-2026-10-02.md`.
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
