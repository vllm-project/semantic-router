# Decision-2.0-Nox-4B Index-first successor: M17 `4b-LHS17SD` released as `main` `b285e7a1` (2026-10-02)

User rule 2026-10-02 09:55 UTC+8 (INDEX-FIRST) and the directive relayed at 12:25 UTC+8: release the highest-scoring
candidate directly, never a lower one once a higher one is measured. Release worker 5e7b8132 (worktree
`vllm-sr-dev2-4b-indexfirst`, branch `xunzhuo/decision-2-training-4b-indexfirst`), the single Nox-4B publisher.

- State record: [`dec-4b-indexfirst-state.md`](../../dec/records/dec-4b-indexfirst-state.md).
- M17 hand-over (integrity evidence): `dec-m17-handover-2026-10-02.md` on `xunzhuo/decision-2-training-dec-m17`.
- Ops: [`dev2-4b-indexfirst-2026-10-02/ops/`](dev2-4b-indexfirst-2026-10-02/ops/); node receipts under
  [`dev2-4b-indexfirst-2026-10-02/release/`](dev2-4b-indexfirst-2026-10-02/release/).

Index values stay in private files (node A `/data/dev2/private/release/4bif/S17/`, the node private directories and
the local private folder). Times are UTC. Everything is private.

## Result

- **Private `llm-semantic-router/Decision-2.0-Nox-4B@b285e7a17791a06058817680592484868dde9545`** (`main`, the only
  ref), 05:33:49Z.
  - `MODEL_MANIFEST.json` `35eba1817fc7f45183deba0cad0b12c0cad84177dd4937e4a6fd5c83019b0469` (36 uploaded files,
    37 on the Hub). Identity `74ec8b2f…`, the `v2.release.bf16_copy` of the FP32 soup `8995bd9d…`; 4,208,383,488
    loaded parameters; `qwen-full`, T = 1, 16,384 tokens, apache-2.0, the current revision's runtime, Transformers
    remote code and forward token budget; vendor sources from the formal run's mirror `d80933b6c`.
  - Supersedes `54b084f9` (LH, weights `6a555335…`; gate `6f03954a…`, decision `6f8eb8e9…`).
- **Choice:** the largest private Index lower bound vs the current release among the finished BF16 Index runs on the
  same panel (`6455d7be…`), seed 20261002 and 2,000 replicates: M17 `4b-LHS17SD`, ahead of M17 `4b-LHS10SD` and M13
  `4b-LHA10SD` (all three > 0). M16 `4b-LHA10SD-a75`, measured during the release, is lower. LHA10UP (node D) and
  a50 were not waited for. The earlier M13 choice was never published.
- **Final decision** [`Decision-2.0-Nox-4B.decision.4bif-S17.json`](dev2-4b-indexfirst-2026-10-02/Decision-2.0-Nox-4B.decision.4bif-S17.json)
  `18d60dac…` (profile `successor` with `index_first`), sealed in
  [`receipts/gate.json`](dev2-4b-indexfirst-2026-10-02/release/receipts/gate.json) `54a8c2ab…`.
  **`gate evaluate`: 10 of 10** (IF1, R3, IF3, references, items 2–7).

## Integrity checks

- **IF1** private paired bootstrap of exactly these weights vs the current revision's, lower bound > 0; both IX1
  receipts bound (candidate `c2c72651…`, LH `9d6158d5…`, same panel).
- **Package parity:** the IX1 86-request gate PASS (86 / 86, max |Δp| 0.0); release parity 0 answer changes and drift
  0.0 on typed-final 1,600 / css15 6,547 / public231 231 (formal cache) and mlx-diag 2,275 (mlx cache) against the
  formal runs' predictions, before the upload and on the real download; AutoModel vs native 0 changes.
- **Hub `trust_remote_code` smoke** under Transformers 5.17 and 5.18 (fresh cache): pass.
- **IF3** row-level Index audit `ad296959…` of the M17 TRAIN files: 0 item rows; planted control 200 / 200;
  `14bce13c…` (71,088 lines) 90 familiar-text rows (LH's own TRAIN: 114).
- **R3** formal typed-FINAL `m17-4b-LHS17SD`: choice / Noul / Score OK / OK / OK.

## References (not blockers; disclosed in the decision)

- Post-key JevArena v3 62.090 vs 67.345: −5.25 [−11.26, −1.67], significantly below the current revision.
- Card-eligible mlx-diag −.0279 [−.0395, −.0167]; human transfer [−.120, +.032]; public 231 174 vs 172 (p .791);
  vs adopted Nox 1.0 +5.62 [+0.46, +8.52]; vs Decider 4B +0.21 [−7.05, +3.43]; C1 not run.
- The training rows include the IB families matched to Index benchmarks (HoVer, When2Call, iSarcasmEval, GSM8K) and
  the BPoMP format; the transfer-only Index delta is private.

## Card, storage, collection

- Product card of the current revision with this candidate's reports, default `v2.release.card_assets` (banner
  concept A), and the Index input from `python -m v2.release.card_index` with the audited footnote. Base: the 0.8B
  Index-first release's input (the 0.8B main carries new weights); only the 4B point changed. Card HTTP and all 8
  links pass; anonymous access 401.
- Purge (`rewrite_history=False`, `hf_headroom.sh` first): the six LH weight blobs (9.70 GB; node copy = the LH
  release download) are gone, commits and refs unchanged, `main` served. Account 52.57 GB of 100 (Nox-4B 9.72 GB).
- Collection "🎲 Decision 2.0" unchanged (private; Vega-27B, Lux-9B, Nox-4B, Sol-2B, Eos-0.8B, Kai-0.6B); the
  driver's post checks passed (collection check fix `cd565a588` merged).
