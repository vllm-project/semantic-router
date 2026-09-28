# 0.6B Milestone 3 amendment (M3a-2): the freeze is not the fix; FP32 and low-LR pilots

Frozen 2026-09-28 before any P4/P5 run.

## Outcomes that triggered it (preregistered rules in `m3-prereg-2026-09-28.md` §2)

- **P1 (`m3a-qbw-s1`, QB s1 + backbone frozen for updates 1–47) hit the collapse stop:**
  SELECT 271 → 261 → 254 → 239 at update 175. The head alone learned nothing usable on
  frozen bidirectional-Qwen marker states (CE stayed near chance, gradient norm ~1);
  once the backbone was released its gradient norm rose to 11–31 and fell to 0.15 by
  update 80, the moment its own warmup reached the peak learning rate. **Fix not
  validated.**
- **P2 (`m3a-qbmb1-s1`, QB s1 with one-row micro-batches, no padding at all)
  collapses like padded QB:** gradient norm 12.0 at update 20, 0.30 at update 30,
  0.1–0.5 afterwards; SELECT 237 at update 58. Padding does not explain the
  bidirectional collapse.
- Both pilots passed the new preflight parity gate (FP32 loss difference ≤ 1e-7,
  gradient relative error ≤ 3e-5; BF16 padded-vs-FP32 cosine .983–.994).

## New pilots (Milestone 2 data and QB s1 recipe, padded, no freeze; collapse stop SELECT < 350 at update 175, else run to update 466)

| Pilot | Spec | Single change | Hypothesis |
| --- | --- | --- | --- |
| P4 | `m3a-qbf32-s1` | compute in FP32 (no BF16 autocast; `start.compute_dtype`) | BF16 gradient error in layers 1–7 (measured 5–77% relative) plus AdamW's per-parameter normalisation drives the collapse |
| P5 | `m3a-qblr-s1` | backbone learning rate 5e-6 instead of 2e-5, warmup 10% | the peak backbone learning rate itself destabilises these RMSNorm/SwiGLU backbones |

P3 (`m3a-qcw-s1`, causal control + freeze) is dropped: the freeze failed on P1.

Decision rules:

- A pilot **passes** if it clears the stop and reaches BEST SELECT ≥ 450/700.
- If exactly one passes, its change is the Milestone 3 fix for fresh-head bidirectional
  arms; if both pass, P5's (BF16, no extra compute); if neither passes, arms (b) and (d)
  are not run and the bidirectional collapse stays an open optimisation failure (not an
  implementation failure) in the records.
- Arm (a) (causal) no longer depends on the pilots: it uses the Milestone 2 QCLMB1
  recipe (one-row micro-batches, BF16, no freeze), which trained on both seeds, on the
  Milestone 3 mixture (`m3-a-causal-mb1-s1|s2`). The padded-with-freeze variants
  `m3-a-causal-s1|s2` are withdrawn unused. If P4 or P5 establishes a fix, (a) is not
  changed within this milestone (one factor per contrast against Milestone 2).
- Arm (c) (Kai lineage, own trained head) is unaffected.
