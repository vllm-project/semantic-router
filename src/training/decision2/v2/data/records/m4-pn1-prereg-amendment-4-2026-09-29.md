# PN1 preregistration, amendment 4 — full-feature matching and node-A validity check (2026-09-29 ~21:10 UTC+8)

Committed before the re-finalize and before any validity GPU work. It amends
[the prereg](m4-pn1-prereg-2026-09-29.md) and [amendment 3](m4-pn1-prereg-amendment-3-2026-09-29.md).
Nothing else changes.

## A. Matching on every audit feature

- **Evidence:** the build `c3eb6072b0aa` came from the amendment-3 v2 re-judge, whose control probe passed:
  - identical pairs: median P(yes) .9998 (v1 .991);
  - coordination swaps: .991 (v1 .562);
  - seed fluency: 93% (v1 62%).

  It matched on 3 coordinates. On TRAIN, the preregistered A4 overlap learner (8 features) exceeds the
  majority + 0.05 rule, measured against the true class prior of .50 (the rows are exactly balanced):

  | Language | Accuracy (swap-group accuracy in brackets) |
  | --- | --- |
  | es | .672 (.745) |
  | fr | .643 (.681) |
  | ru | .602 (.702) |
  | ko | .593 (.643) |
  | zh | .542 (.626) |
  | ar | .541 (.599) |
  | de | .520 |
  | ja | .504 |

  The signal is concentrated in the swap group, where edit distance and word overlap separate adjacent
  coordination swaps and synonym twins from distant role swaps.
- **Change:** the 1:1 caliper matching of amendment 3 uses all features of the A4 overlap learner:
  - character-bigram Jaccard;
  - character-unigram-multiset Jaccard;
  - length ratio;
  - word Jaccard;
  - normalized edit distance;
  - containment, taken order-free as its minimum and maximum over the two directions, because the state order is
    drawn after matching.

  The same-multiset flag stays a stratum key. The caliper stays 0.03 on each coordinate. The strata, the order and
  the dev-first rules are unchanged.
- **Self-check fix:** the self-check's majority baseline is now the class prior of the evaluated rows. Before, it
  was the per-fold training majority, which on exactly balanced data scores below .50 and inflated the margins.
- **After the re-finalize the A4 rule applies unchanged,** with no further rebalancing. A language whose TRAIN still
  fails has its failing construction group (natural or swap, whichever fails on its own) dropped, and the record says
  so.

## B. Validity check moves to node A GPU1

- **Why:** coordinator decision (20:45 UTC+8): decoder Milestone 6 keeps node B GPU3–4. The remaining ≤ 0.555 GPU-h
  of the PN1 lend runs on node A GPU1 under a shared lease. The 0.6B release worker has GPU0.
- **Formal node-A settings,** matching N4XF's node-A mlx-diag run (`/data/dev2/runs/dec/formal/m4/m4-N4XF-soup-mlx`):
  - image `decision20-train-fast:host2` = `f83b1d10f14d`;
  - `--max-length 16384`, no truncation, offline container, gold-free prompt file only.
  - Runners: `v2.dec.infer_dec` for N4XF (node-A copy
    `m4/formal-candidates/staging-e8656221/m4/N4XF-soup`, identity by the recorded per-file list); `v2.dec.infer_1p0`
    for Nox 1.0 `@0bb833504965c0eabdb9630b7bbd385cb2fe5cd4` (the decoder's shared renderer).
  - Each run gets its own `cp -a` copy of N4XF's node-A mlx-diag Triton cache (`m4-N4XF-soup-triton`).
- **Preflight:** N4XF on the first 20 mlx-diag gold-free prompts must reproduce the answers of that run's
  predictions. Otherwise stop and record.
- **Criterion:** unchanged (prereg §7). Both models run on the same node, so the same-node rule holds; node B's
  frozen cache is not used.
- **Embedding scan (A1):** it runs on node A GPU1 if Qwen3-Embedding-0.6B@`97b0c614` is cached there. Otherwise it
  is recorded as not run, and the lexical and short-text scans stand alone.
