# Data arms v2 — amendment 2: positional Choice keys (repair after the first gate run)

Committed after the first shortcut-gate run of the H1/H5/E11 builds at `b8804a3a5` and
before any v2 arm is frozen or published.

## What the gates caught

Rotated Choice rows kept the keys assigned before rotation (`o1..oN` in source order), so
a key identified the gold independently of its position: TyDi relevance Choice (gold always
built as `o1`) reached state-removed 0.95–0.998 and option-only 0.64–0.76 against majority
≈ 0.25 in all ten languages; abstention twins (abstain always `o3`) reached 0.40–0.47 against
0.32–0.34; ARC, OpenBookQA and AQuA-RAT (keys in source order with skewed source answer keys)
failed by 5–23 points. CommonsenseQA, QuaRTz, ROPES, DBpedia-14, MultiWOZ and MTOP passed.

## Repair (one version, re-audited once; a second failure is final)

`m2.common.make_row(rotate_choice=True)` renames default keys to `o1..oN` by displayed
position after rotation, so keys carry no label information. Affected families are rebuilt
with the same seeds, caps and sources and every arm is re-audited (overlap, shortcut cells,
embedding) on the rebuilt bytes. Families that still fail are dropped as built.

## Generator cell

G4h/G4r family `a2_calendar` fails the state-removed gate in G4h (0.277 vs 0.218, n = 740);
it is dropped from both G4h and G4r so the hard/random pairing is kept.
