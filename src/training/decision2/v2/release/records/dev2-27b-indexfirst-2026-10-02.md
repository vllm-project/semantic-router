# Decision-2.0-Vega-27B: M6-IB released as `main` under the Index-first rule (2026-10-02)

COORDINATOR INTERRUPT 12:40 UTC+8 item 1: release M6-IB (rank-128 soup of M6-IB s1 / s2, `a20ib1` mixture) to
`llm-semantic-router/Decision-2.0-Vega-27B` with banner A, on top of `b689ee66`. The spec, the final decision and the
release scripts are under [`dev2-27b-indexfirst-2026-10-02/`](dev2-27b-indexfirst-2026-10-02/). Index values are
private (private HF card and private records only). Everything is private.

## Result

- **Private `llm-semantic-router/Decision-2.0-Vega-27B@781b2b2431abccbdcfcbfa98b66c80b332725513`** (`main`),
  supersedes `b689ee66` (A20r, identity `2e074511`). Identity `e50fb4c1…`; `MODEL_MANIFEST.json` `4d1ca0f5…`.
- **27,497,508,864 loaded parameters:** pinned base text backbone 25,624,600,064 (not redistributed) + rank-128 LoRA
  1,867,644,928 + head 5,263,872.
- **Gate:** profile `index_first` (IF1 Index bootstrap with matching receipts, R3 types, IF3 audit with the planted
  control, 200 of 200 planted found, 0 item rows); decision
  [`Decision-2.0-Vega-27B.decision.27bif-M6-IB.json`](dev2-27b-indexfirst-2026-10-02/Decision-2.0-Vega-27B.decision.27bif-M6-IB.json)
  (decided 2026-10-02T05:53:36Z), sealed in `receipts/gate.json` `2851727b…`.
- **Card:** banner A, the card Index chained from the other tiers' Hub `main` identities (only the 27B point changed),
  rendered on the node.
- **Checks:** pre-upload examples, card, four-panel parity (with mlx-diag), AutoModel parity, `verify_bundle`; after a
  real Hub download with a fresh cache: examples, Transformers remote code, card example from the Hub, scored-panel
  parity, readback (`post.json` `e6cc3104…`, `parity-post.json` `0de71794…`, `download.json` `8943ec4f…`). Title
  unchanged, repo private, releases in order; anonymous model API, README and page return 401. `RELEASE-RECEIPT.json`
  `6a702fb3…`, `post_checks=ok`.
- **Storage:** 52.57 GB used before → 60.06 GB with the new revision → **58.17 GB** after `purge_superseded.py`
  removed A20r's superseded blobs (1.89 GB; commits and refs unchanged, old weights not served, `main` still served).
  Vega-27B 7.51 GB; headroom 41.83 GB.
- **GPU:** prerelease 05:57–07:02Z plus release 07:02–08:37Z (1.57 GPU-h) on node A GPU0, shared lease
  `owner.release-27b-27bif` (released).

## Incidents

- The first prerelease build stopped at the manifest privacy check ("node A" in `runtime_equivalence`); the public
  texts now say "the release node" (`make_27bif.py`, `69b52757d`).
- After that stop, the exit trap removed the GPU6 owner and eval-ix1 took GPU6 back; the release moved to the shared
  lease on GPU0.
