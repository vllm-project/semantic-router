# 0.6B Milestone 4 amendment (M4d): V2 trains with the scaled own-Lux targets

Frozen 2026-09-28 before any V2 run (M4c's V2 had not launched).

## Why

At 18:40 the research & data track published own-Lux RP-v2 targets for every row of the
S and M recipes (wave 1 `47049a6b…` at HF `7885baf6…`, wave 2 `2c6ab38d…` at HF
`6bd8eb4d…`; Lux `bd45a30a` native runtime; argmax vs gold Choice .79, Noul .78–.79,
Score .55–.56, informative Score targets). Milestone 4 item 3 says to adopt scaled Lux
targets as soon as they are announced, and the 18:45 coordinator note asks every tier to
adopt the v2 recipe now. With them, V2 is the D10 `mx-v2-full-M` recipe with own-Lux
distillation, which is what the recipe was designed for.

## Change

`m4-v2-s1|s2|s3` teacher = **`lux1-a0-rpv2m.merged.jsonl` `358a883d…`** (91,932 rows) =
the canonical A0 file `752b7c8f…` (as T and C) + RP-v2 waves 1 and 2, merged by
`v2.06b.teacher_merge` (hash-checked inputs, no repeated ids, valid distributions).
Coverage on the V2 mixture: 47,759 of 47,922 rows; the 117 renumbered A0s-r rows get no
teacher (input hash guard) and 46 generated A6g rows of the recipe's V1S pool have no
RP-v2 entry (gold only). Mean teacher max-probability on covered V2 rows: Choice .83,
Noul .86, Score .67 (A0 file .85 overall). KL weight 1.0 as in every Milestone 4 arm.
Everything else in M4c is unchanged.

## Disclosure

T and C keep the teacher on A0s-r rows only; V2 now has a teacher on 99.7% of its rows.
The T/V2 comparison is therefore a recipe comparison (data + distillation coverage), not
a single-factor contrast. A hard-label or Lux-only-on-A0s V2 control is not run in this
milestone. AutoJev RP-v2 targets are not published yet (coordinator 18:45: usable for
candidates after its repeatability check, each with a matched Lux control).
