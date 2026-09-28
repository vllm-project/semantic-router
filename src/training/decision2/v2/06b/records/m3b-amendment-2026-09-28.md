# 0.6B Milestone 3 amendment (M3b-2): a gentler causal schedule as arm (a2)

Frozen 2026-09-28 before any (a2) run.

## Why

- All four bidirectional pilots stopped (`m3-padding-root-cause-2026-09-28.md` §3), so
  (b) and (d) do not run under the M3a-2 rule.
- Arm (a) (causal, Milestone 2 QCLMB1 recipe: one-row micro-batches, backbone LR 2e-5,
  5% warmup) split by seed on the Milestone 3 mixture: s1 learns (SELECT 227 → 310 at
  update 100, gradient norm 8–28), s2 collapses at the end of warmup (245 → 239 → 247 at
  updates 100/200, gradient norm 0.3–1.5). The recipe that trained on 4/4 Milestone 2
  seeds is a coin flip here; the stochastic early collapse at peak learning rate is the
  track's real blocker, and (a) cannot meet the two-seed finalist bar.
- Failed fine-tuning runs whose gradients vanish early are an optimization failure that a
  smaller peak learning rate with longer warmup usually removes (Mosbach et al., 2021,
  "On the Stability of Fine-tuning BERT"). The causal readout learns when it does not
  collapse, so a gentler schedule should keep it learning.

## Arm (a2)

`m3-a2-causal-s1|s2` = (a) with three changes, all declared: backbone learning rate
1e-5 (head stays 2e-4), warmup 10% (80 updates), and padded micro-batches (eight rows;
padding is exact, `m3-padding-root-cause-2026-09-28.md`, and faster). Same mixture,
teacher, loss, seeds, checkpoints, collapse stop (SELECT < 350 at update 300) and
finalist rule. It is a recipe candidate, not an attributable single-factor contrast.

## Queue and budget

(a) runs to its stop or end and is read out as preregistered. Then each GPU runs (a2)
before (c). Projected Milestone 3 total ≈ 3.7 GPU-hours, inside the 5.0 cap.
