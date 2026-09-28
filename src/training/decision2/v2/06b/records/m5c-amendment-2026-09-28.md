# 0.6B Milestone 5 amendment (M5-2): X's formal result; Z becomes the soup of every passing recipe

Written 2026-09-28 after X's formal run and **before any arm readout**. At that point every arm
(a), (b) seed was still training and (c) had not started. This amendment is a post-formal change
and is disclosed as such.

## X, formal post-key same-panel result (frozen runner, node A GPU1, package at 8,192 tokens)

| Model | v3 | T | H | Typed FINAL C / N / S | Public 231 (E/S/H) | mlx-diag (EN / non-EN) |
| --- | ---: | ---: | ---: | --- | --- | --- |
| **`m5-x-soup`** | **45.365** | .4350 | .4731 | **342** / 439 / 148 | 145 (48/56/41) | 59.8 (63.0 / 59.3) |
| `m4-t-a7-soup` (released) | 43.541 | .3953 | .4796 | 255 / 442 / 133 | 142 (48/54/40) | 61.5 (65.3 / 60.8) |
| `m4-v2-soup` | 39.130 | .3853 | .3974 | 289 / 435 / 93 | 151 (48/60/43) | 59.8 (63.6 / 59.2) |

**Paired v3 Δ for X** (5,000 draws, same node):

| Comparator | Δ v3 [95% CI] | Δ T [95% CI] | Δ H [95% CI] |
| --- | --- | --- | --- |
| Released T soup | **+1.82 [−0.62, +4.51]** | **+.040 [+.020, +.059]** | −.006 [−.051, +.048] |
| V2 soup | +6.23 [+2.48, +8.91] | | |
| Kai1 | +9.43 [+5.79, +12.85] | | |
| Kai1 at 8K | +9.40 [+5.44, +12.74] | | |
| GLiNER2.5-Decide | +2.84 [−0.31, +9.75] | | |
| Bosun 0.6B | +6.84 [−0.24, +10.07] | | |
| Lex | +14.34 [+4.75, +19.97] | | |
| Causal control | +6.84 [+0.67, +11.77] | | |

**X is not a successor:** the lower bound against the released candidate is below 0. What X does:

- It lifts typed reasoning significantly.
- Typed FINAL Choice rises to 342, above Kai1's 277. This removes the released card's disclosed
  Choice regression.
- It keeps human transfer level. V2's weights cost only .006 of H inside the soup, although V2
  alone was .082 below T.
- The pilot guard's reading (pilot mean .386 = T's) matched the level CSS15 H.
- **Disclose:** mlx-diag is 1.7 below the released candidate.

## Change: Z (part 1 §3)

**Old definition:** the three T seeds plus the three seeds of the best guard-passing arm.

**New definition:** **Z = the uniform soup of every seed of T, of V2, and of each M5 arm whose
artifact passes the part 1 §4 floors and transfer guard.** Each recipe contributes its three
same-init seeds, so every recipe has equal weight.

- **Why:** X showed that averaging recipes adds skills. V2 added typed reasoning without costing
  transfer, and every arm shares T's init and most of its data.
- **If no arm passes:** Z is not formed, since X already is T ⊕ V2.
- **What stays unchanged:** the guard and floors, the second formal slot (the best of A*, B*, C*
  and Z by part 1 §4, X excluded), the successor rule and the budget.
