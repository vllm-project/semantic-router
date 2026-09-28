# A7 preregistration amendment 2: embedding scan and PI-v3 rescreen (2026-09-28)

Committed before any `a7-dec10-v3` file is written. Version `a7-dec10-v3` = the
admitted `a7-dec10-v2` files (HF revision `39a120ca`) minus the groups flagged
by the two scans below. Nothing else changes: no row is added, re-rendered or
moved between TRAIN and AHO, and views only lose removed members.

## Why

Prereg §2.4 left the GPU embedding scan pending. Since the v2 freeze the
research & data track published PI-v3 (PI-v2's 38 evaluation/development roles
plus the frozen multilingual diagnostic `mlx-diag` v1 and the eight v1 arm AHO
slices, which are quarantining roles in data-arms v2). A7 adopts PI-v3 so its
sub-arms meet the same bar as the v2 arms.

## Scans (receipts on node B under the v2 run directory)

1. **Embedding** (`v2.data.embed_scan`, the data track's settings: Qwen3-Embedding-0.6B
   snapshot `97b0c614`, 1,500/1,000-character windows (480/320 for CJK),
   quarantine at cosine ≥ 0.93, 20-pair review sample in [0.85, 0.93), seed
   `decision2-embed-scan-v1`) of all 12 v2 files against PI-v3 without TRAIN roles
   (`pi-v3-minus-train.json` `869aa57c…`, 47 roles; relocated on node B with
   identical per-file SHA-256 as `252d9a49…`). Driver `v2/data/a7/embed_nodeB.sh`
   at `f41025374` on node B GPU7 under an A7 lease owner file.
2. **Lexical rescreen** (`v2.data.overlap`, unchanged four methods) of the same
   files against the same 47-role manifest (`run_nodeB.sh rescreen` at `f9d756563`).

## Rule

Whole-group removal, from every sub-arm, of any group with a lexical hit in
either receipt or an embedding match at or above 0.93 (PI-v3 has no report-only
roles once TRAIN roles are left out). Review-band pairs are reported, not
removed, as in data-arms v1. Freeze, isolation, post-admission diagnostics and
upload follow prereg §2.9–§2.10 and §4 (`v2.data.a7.requarantine`, then
`run_nodeB.sh` stages `post`, `freeze`, `isolation`, `assemble`, `upload`).
Sub-arms without a removed group keep their v2 bytes and content hashes.
