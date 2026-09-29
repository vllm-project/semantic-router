# Decoder Milestone 4 — amendment 1: selection statistic after proxy v2 (2026-09-29)

Written and pushed while wave 1 (N4LR s1–s3, N4LR2 s1–s2) was still training: no M4 development readout,
formal result or seed checkpoint readout existed. Amends selection rules 3–4 of
[`dec-m4-prereg-2026-09-29.md`](dec-m4-prereg-2026-09-29.md).

## Trigger

The eval track published proxy v2 (`c3da006b8`,
`v2/eval/records/m5-proxy-v2-calibration-2026-09-29.md`):

- No candidate beats P, so proxy v2 is **P with the pilot median unchanged**, recalibrated to v3 ≈ 22.39 + 0.576·P.
- P_mean3 is statistically indistinguishable from P: the paired difference on the primary metric is
  [−0.09, +0.09], and its empirical tie band is 9.
- Within-tier tie band |ΔP| < 8. "No proxy separates close within-tier candidates reliably; ties go to the formal
  runner."

Rule 3 as written would therefore select on the CSS-pilot median. The coordinator's M4 instruction excludes that
signal ("human-transfer signals other than the single-task-dominated CSS-pilot median"): at 4B the median is
essentially `discourse`, which data v2 inflates, and v2 content is exactly what N4XA / N4LRQ vary against the
other arms.

## Amended rules

3. **R = P_mean3 = 100·√(T_dev·H_mean3)** for every arm, seed and soup, and for rule 1 (soup vs seed mean). P
   (median) and proxy v2's predicted v3 (22.39 + 0.576·P) are reported, never used.
4. **Tie band 9 P_mean3 units** (proxy v2's empirical band for this proxy). It is used for labeling ties and for
   the N4LKr-reference exclusion: an artifact is not sent if R < R(N4LKr soup) − 9. The two highest-R eligible
   artifacts (the cross-arm soup N4LX included) are the finalists. Any finalist difference inside the band is a
   tie that the formal runner decides.

   Slot 2 goes to the artifact with the higher min(C/460, N/228, S/378) only when the second and third differ by
   less than 1 R unit. That is about one per-model panel-noise SD; the old tie band would make slot 2 a
   typed-balance choice almost always.

Unchanged: eligibility (typed-DEV per-type floors 345 / 171 / 284, validity, no constant type), the cross-arm soup
rule, at most two finalists, the formal bar and every training rule.
