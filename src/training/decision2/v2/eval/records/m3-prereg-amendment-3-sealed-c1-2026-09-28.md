# Eval M3 preregistration — amendment 3: C1 selection gate (2026-09-28)

Committed after the first C1 selection (code `b4a699a3b`) and before the selection that
will be reviewed and sealed. The first selection was a dry run. No model and no reviewer
saw it. Its files stay in node A private storage as a record and are never used. Its
count-only manifest has SHA-256 `1e8fae64…`.

## What the dry run showed

- **GAPA.** 147 of 150 candidates are unscreenable, because the stimuli are shorter than 5
  tokens. Only 3 items are eligible, so the task cannot stand.
- **Length checks.** Six tasks failed the selected-set length check on point estimates:
  HalluTruthQA hallucination +14.0, legal +11.6, tutormoments Choice +11.0, narrative
  concreteness +7.9, narrative temporal grounding +8.9, and DeliChess stance +6.3.
- **Why the pool check missed them.** The pool-level check fired only for ImplicatureX.
  Class imbalance in the pools raises the majority reference, which hides a length cue
  that appears once classes are balanced.
- **Noise on small tasks.** At 40–85 items, a 5-fold length baseline has a standard
  error near 5 points, so a point-estimate threshold alone fires on noise.
- **Quota construction.** "Equal gold counts within each length quintile" (amendment 2)
  empties a whole task whenever one class is missing from one quintile. This is common
  for rare Score levels.

## Changes (build.py; tests added)

1. The length gate is LEAK-level in the pool and in the selected set. It fires when the
   gain over the better of majority and chance is ≥ 5 points and the group-bootstrap
   lower bound is > 0 (2,000 draws). This is the same evidence rule as the leak audit.
2. Length balancing gives each gold class the same count in every non-empty length
   quintile: the class's minimum over quintiles, capped at `cap / (5 × classes)`.
   Length is then independent of the gold. Classes need not be equal, and only a class
   missing from some quintile drops out.
3. The selected-set gate runs inside `build.py`. A failing task is rebuilt once with
   length balancing and dropped if it still fails. A task left with fewer than 30 items
   or a single gold class is dropped.

Nothing else changes: the salt (SHA-256 `43e6a3bc…`), candidates, overlap receipt,
config, caps and review, seal and scoring rules are as in amendment 2.
