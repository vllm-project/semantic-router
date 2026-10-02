# Arm factory — amendment 7: two batch-3 variants on the S17 base (2026-10-02 ≈18:30Z)

Written before either arm's weights file was built or any of its GPU jobs ran; no factory 4B Index result exists.
COORDINATION 2026-10-03 02:18 (UTC+8): node C GPU3–4 are idle and go to the first lease-check. The factory takes them
for the S17-base counterparts of amendment 5's SDML-base variants.

| Arm | GPU | Data | Change | Seed |
| --- | --- | --- | --- | --- |
| `4b-LHS17IB4-e2` | C3 | M17's locked `4b-LHS17IB4` TRAIN (`dfed3944…`) | two epochs (`--epochs 2`); seed cap 3.5 GPU-h | s1 20260926 |
| `4b-LHS17IB4W2` | C4 | the same TRAIN + `af_weights.py`: IB4 p1 rows ×2 | IB4 dose ×2 on the S17 base | s1 20260926 |

- One seed each (each arm's artifact is that seed's BEST checkpoint, merged; disclosed); handed to the 4B owner.
- Node C gate 25 GPU-h. Node B GPU5 is left to the owners' Index lanes.
