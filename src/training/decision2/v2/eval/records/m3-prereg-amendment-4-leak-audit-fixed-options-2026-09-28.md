# Eval M3 preregistration — amendment 4: fixed option sets in the leak audit (2026-09-28)

Committed after the leak audit of the C1 selection flagged narrative-gold
`setting_concreteness` (+10.0 [+0.7, +20.7] from `description_length`). Committed before
the M3.1 re-check (audit v3) and before the C1 build is re-run.

## Why

That task shows **one fixed option set**: the same five level descriptions for every
item. When every question in a group shows the same keys and descriptions in the same
order, every option-surface feature (position, key token, description length or
whitespace) is a fixed function of the option's identity. None of them can carry
item-level information.

The "gain" came from tie-breaking. Levels 1 and 2 are tied at 50 items each.

- The cross-validated label prior splits between the two tied levels and reaches 23.3%.
- The description-length tag happens to single out level 1 and reaches 33.3%.

This is the same class of reference artifact as amendment 1, not a leak.

## Change (`v2/eval/leak_audit.py`; test added)

- A group whose questions all show one option set (keys and descriptions, in display
  order) is CLEAN by construction for the option surface.
- The state cue, which can vary per item, is still assessed for such a group.
- Pooled rows test only questions from groups whose option sets vary.
- Every entry reports how many fixed-option-set questions it excluded.

## Effect on M3.1

In run v2 every fixed-option-set group was already CLEAN, with gains of 0 or below:
CSS15, CSS pilot, mlx-diag, SELECT/CAL fixed families, typed FINAL Score, and the fixed
Noul groups. Audit v3 re-runs all eight panels to confirm that no verdict changes.
