# Data arms v1 — amendment 2: A4 repair and A5/A6h projection refinements (2026-09-28)

Committed after the wave-1 audits and **before** any A5, A6h or A4 v2 row is built.
Rules of the preregistration and amendment 1 otherwise apply unchanged.

## A4 v1: HOLD (not published)

Both A4 v1 files failed the preregistered shortcut gates (group-disjoint 5-fold,
accuracy ≤ majority + 0.05): A4h Choice option-only 0.286 vs 0.222 and Noul
option-only 0.587 vs 0.475; A4r Choice state-removed 0.295 / option-only 0.312 vs
0.222 and Noul option-only 0.587 vs 0.475. Diagnosed causes: Noul truth was not balanced
per (family, language), so a language-specific options prior beat the majority; in
counting scenarios the near-miss distractors all exceeded the gold count, so "pick the
smallest value" won; random distractors differed from gold in value range. A4 v1 stays
in the registry as a negative audit and is not used by any contrast.

**A4 v2** (new arm name `a4v2`, seed `decision2-a4-v2`, source ids
`decision2_verifiable_v2_a4v2h` / `…a4v2r`, group namespace disjoint from A2 and A4 v1):
Noul truth exactly 50/50 per (family, language) in both files; the gold's rank among
option values exactly uniform per (family, language) with near misses on both sides of
the gold; random distractors straddle the gold with the same rank distribution and a
comparable spread; entity options equally present in the state. The generator must keep
the frozen A2 and A6g bytes unchanged (regression test on their SHA-256 values). A4 v2
passes through the same overlap, embedding and shortcut gates; a failure is final for
v2.

## A5 and A6h refinements

1. **Class balance:** `klue_nli`, `jglue_jnli` and `klue_ynat` are downsampled
   deterministically to equal counts per gold class (options with fixed texts would
   otherwise carry a label prior); `jglue_jcommonsenseqa` is unchanged.
2. **NLI state form:** `{"premise": …, "hypothesis": …}` (English keys, native values),
   so the hypothesis-only baseline runs on both NLI sources.
3. **JGLUE grouping:** JNLI and JSTS group by caption image id (prefix of
   `yjcaptions_id`), keeping every sentence pair of an image in one group.
4. **KLUE-STS balance cell:** (source id, L), not (source id, L, KLUE stratum); the
   stratum mix per level is reported in the manifest. Stratifying by the never-shown
   round-trip/sampled field would leave only a few hundred rows.
5. **Rotation seed:** `sha256("<arm>-v1:" + row id)`, as for A1/A3.
