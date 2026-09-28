# M3a preregistration — amendment 1: recipes nest only approximately (2026-09-28)

Committed after the first node-A wave rebuild failed its nesting assertion and before any
qualification or production process ran. Sections 1–4 (runtime, sets, protocol, gates) are
unchanged: W and R are still drawn from the RP-v2 pool.

## Finding

The D10 recipes are nested at the level of recipe budgets, not row by row (whole-group sampling at
each size). Over non-A0s rows (published `m2/mixtures/` at revision `002e5b42…`):

| Relation | Rows |
| --- | ---: |
| mx-v2-full-S rows in neither M recipe | 235 |
| mx-v2-full-M rows outside mx-v2-full-L and mx-v2-short-M | 46 (all also in full-S) |
| short-S within short-M | yes |
| S ∪ M ∪ L | 121,742 |

RP-v2 was built from mx-v2-full-L ∪ mx-v2-short-M (121,461 rows), so **281 recipe rows have no own-Lux
target**: 281 of mx-v2-full-S (261 v1 Score rows of A6g/A6h, 20 E11) and 46 of mx-v2-full-M (v1 Score).
The short recipes and full-L are fully covered. The Milestone 2 record said every S and M row was
covered; that holds for the short recipes only.

## Changes

- **Waves** (`v2.data.m3.waves`, with S, M, L = non-A0s ids of (full-S, short-S), (full-M, short-M),
  (full-L, short-M)): the published own-Lux waves are wave 1 = S ∩ L (28,501), wave 2 = M − S (55,976),
  wave 3 = L − S − M (36,984); the rebuild must reproduce their node-B prompt files byte for byte.
  **Own-Lux wave 4** = (S ∪ M) − L (281 prompts) is added and produced on node B GPU7 with the same
  runtime, image, mirror and autotune cache as waves 1–3 (one node per teacher file), converted with
  `v2.data.m2.targets` and published next to them.
- **AutoJev waves** in the coordinator's order "M-recipe prompts, then A0s, then the S/L remainder":
  **AJ-M** = M (84,523 = own-Lux waves 1 + 2 + the 46 M rows of wave 4), **AJ-A0s** (7,299),
  **AJ-SL** = (S ∪ L) − M (37,219 = own-Lux wave 3 + the 235 S-only rows of wave 4). Conversion rows
  come from `mx-v2-all.rows.jsonl` (every non-A0s row of S, M and L, 121,742 rows).
- The production repeat check of section 5 applies to AJ-SL on GPU4 as it did to AJ-L.
