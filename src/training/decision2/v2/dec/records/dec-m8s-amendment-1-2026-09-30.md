# Decoder Milestone 8-small — amendment 1: GPUs after the 17:15 reclaim (2026-09-30)

Written ≈17:50 UTC+8, before any M8-small GPU job (only CPU prep and the part-1 data lock had run). Preregistration
[`dec-m8s-prereg-2026-09-30.md`](dec-m8s-prereg-2026-09-30.md) (`c99cf9b7f`); lock part 1 `16044feab`.

**Reason.** COORDINATION 2026-09-30 17:15 reclaimed node B GPU5 for the 27B M5 L128 arm ("never co-tenant a 27B
job") and assigned M8-small **node B GPU6–7 plus the spare GPU2**. At 09:44Z GPU5 held `d2-27b-M5-L128-s1-full`
(27B lease `status=running`).

**Change (operational only; no design, data, budget, rule or selection change):**

- GPUs: node B GPU2, GPU6, GPU7 (`m8s-lib.sh` no longer maps GPU5).
- A20r labels: the parity gate on GPU6, then **two** shards (GPU6, GPU7) instead of three.
- Chains: g2 = the C arms from the start (both tiers, with their seed-1 HT-DEV v2 collections and soups); g6 = parity,
  shard 0/2, the D1 arms (early rule, soups); g7 = shard 1/2, the D2 arms. References, lines and formal jobs use
  whichever of the three GPUs is free, as recorded co-tenants of this track's own jobs only.
- The 4B M8 preregistration was still not pushed at this time; the prereg's λ rule stands.
