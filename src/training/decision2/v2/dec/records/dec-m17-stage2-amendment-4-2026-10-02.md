# Decoder M17 stage 2, amendment 4: wave 5, wider cross-arm soups (2026-10-02)

Evidence (wave 4, amendment 3): every interpolation of two different soups measured above its arm, and the
cross-arm soup `4b-SDMLxALL` (SDML with six stage-2 soups) measured clearly above every other stage-2 point. Its
lower bound vs SDML is > 0, but it is still short of the 4B goal (JPT-4B). Averaging more of the trained 4B soups is
the strongest lever found, so wave 5 widens it.

## Points (CPU; `ops/m17/m17-soup.sh`; uniform FP32 averages, all on Qwen3.5-4B-Base `1001bb4d`)

| Point | Members |
| --- | --- |
| `4b-SDMLxALL9` | `4b-LHA10SDML` and every M17 arm: `4b-LHS10SD`, `4b-LHS17SD`, `4b-LHS17UP`, `4b-LHS23SD`, `4b-LHS17IB4-x3`, `4b-LHS17IB4X`, `4b-SDMLIB4-x3`, `4b-LHS17ML` |
| `4b-SDMLxALL15` | `4b-SDMLxALL9`'s nine, plus the earlier 4B soups: M10 `LH`, M11 `4b-LHB`, M12 `4b-LHA` and `4b-LHA10`, M13 `4b-LHA10SD` and `4b-LHA5` |

## Selection and integrity

- Each point gets one Index run on its BF16 release copy, compared with `IS-4b-LHA10SDML-bf16` (paired bootstrap).
  The rule is unchanged: a candidate is released only if its lower bound vs the current Nox is > 0, after the
  integrity checks.
- Choosing among a few soups by their Index results is a mild selection on the test panel. Ties within noise go to
  the point with fewer members. The formal-path readouts (typed DEV, CSS pilot) are a secondary check: a candidate
  must not drop on them vs SDML.
- Contamination: every member's TRAIN must be audited before release. Audits 1–3 cover M17's stage-1 and stage-2
  arms and M15's `4b-LHA10SDML`. `audit4` (`ops/m17/m17-index.sh`) covers wave 3's `4b-SDMLIB4` and `4b-LHS17ML`.
  `4b-SDMLxALL15` also needs its M10–M13 members' audit receipts. Without them it is reported but not released.
- Budget: CPU builds, then about 2.2 GPU-h of Index inference per point.
