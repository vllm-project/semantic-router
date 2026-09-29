# Rescreen overlap vs released scores — amendment 1 (eval & peers, 2026-09-29)

Written **after** the preregistered run (`overlap-effects.json` `1e69a257…`, code `e4976c5d2`), which reproduced all 17
stored paired files bit for bit and found no changed release conclusion. The preregistered analyses, rules and
numbers are unchanged; everything below is post hoc and labelled so in the record.

## Why

The preregistered contamination check pools all 82 flagged CSS15 items, but a model can only have benefited from
the flagged items that its own training rows touch. A row-level check of the three training files (group id, row id
and input hash of the 305 excluded groups' 727 rescreened rows, all three matching methods agreeing) found:

- DEV2.0-0.6B (`m4-mix-t`, `98e4e859…`): none of the 305 groups;
- DEV2.0-0.8B (`m2-full-a7-v1`, `d1dc33fc…`): 33 groups (V1-A3 29, A7m 4) → 11 scored items;
- 2B S2T (`m3-v2m-ret`, `13804ac6…`): 24 groups (H3 15, H6 4, E11 3, H1 2) → 19 scored items.

## Added (post hoc)

1. `v2.eval.overlap_effects exposure` reruns that row-level match from committed code on the node that holds each
   training file, against a payload of training-side ids only (no evaluation ids leave node A). The files'
   sha256 must equal the hashes above.
2. For each candidate with exposed items: the tier rescored without the candidate's own exposed items (every
   model), a **worst case** in which the candidate misses every exposed item it answered correctly (comparators
   unchanged), and the contamination check on the exposed items alone (reference: the same tasks' unflagged items;
   other flagged items on neither side). Same bootstrap (5,000 draws, seed 20260927); same change rules.
3. The descriptive gold-probability comparison becomes task-weighted (the first run pooled the unflagged items of the
   six tasks, which mixes tasks: 89% of the flagged items are `media_ideology`, 17% of that pool).

The final run writes to `/data/dev2/runs/eval/m5/overlap-effects/final/`; the first run stays on record.
