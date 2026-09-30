# Decoder Milestone 8-small — amendment 3: the Score floor when the incumbent is itself flagged (2026-09-30)

Written ≈10:20Z (18:20 UTC+8), before any 0.8B candidate point was built or read and before any D-arm training job.
Amendment 2 (`60799a63f`) added the 4B M8 Score floor: Score5-typed-DEV check half without COLLAPSE, and without WARN
unless I's check half has WARN.

**Finding (reference readouts scored at 10:14Z, node A):** `2b-I` check half has no flag (top share .38), so the 2B
floor stays exactly as in amendment 2. **`08b-I` (the released DEV2.0-0.8B) is itself flagged COLLAPSE / NO-GAIN:** its
5-level accuracy is at chance (80 / 400 = .20, Wilson lower .164 ≤ .20) while its levels are spread (top share .31,
upper .357; level 0 rare). Read literally, "no COLLAPSE" would fail every 0.8B point, the released weights included,
so it would not be a floor.

**Change (0.8B, and any tier whose I check half is COLLAPSE):** the Score floor judges only a new level
concentration, with the eval's own top-share thresholds: a point fails if its check-half top share is ≥ 0.90, or its
top-share Wilson upper bound is ≥ 0.90 while I's is below 0.90. Where I's check half has no COLLAPSE (2B), amendment
2 applies unchanged. The accuracy-at-chance status of 0.8B Score is reported for every point and disclosed; it is
the released model's known limit (its card already reports that Score never predicts level 0).

Code: `m8s_rules.py score_floor` (+ test); nothing else changes.
