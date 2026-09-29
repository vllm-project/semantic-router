# M3b preregistration — amendment 1: English cap and target reuse (2026-09-28)

Committed after the first XL build (not published) and before any XL file is published or any teacher run.

- **Finding (first build):** XL-full 152.5M tokens (Choice 37.7% / Noul 39.1% / Score 23.2%) and XL-short 125.4M
  both broke the 60% English cap (62.3%, 62.0%), and 255k XL-full rows had no own-Lux or AutoJev target: A7 rows,
  but also v2 rows, because the XL hash order picked other groups than the v2 recipes.
- **Pool targets changed** (millions of tokens): H1 10.5 → 9.0, A7i 7.9 → 7.0, H3 14 → 11, E11 6 → 4.5,
  H5 14 → 17.5, A7q 8 → 10. Everything else unchanged.
- **Group order:** within each pool, groups holding a row that already has own-Lux targets come first; for XL-short
  and the controls, groups holding an XL-full row also come first; each part keeps its own
  `sha256("mx-xl:<variant>:<pool>:" + group)` order. Content rules, caps and budgets are unchanged; this only
  reuses existing teacher targets and keeps XL-short and the controls close to XL-full.
- **Controls:** the caps bind before the controls reach XL's token total; they are published at their own totals
  and matched-token training repeats them, which the records state.
