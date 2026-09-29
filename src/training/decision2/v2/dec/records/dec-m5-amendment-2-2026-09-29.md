# Decoder Milestone 5 — amendment 2 (N5B Lux sources for added rows)

Written and pushed before any N5B training and before any Milestone 5 development readout. N5N and N5BN are unaffected
(their block rows carry own-Nox targets; their other rows use N4XF's composed Lux file).

**Finding (N5B preflight 3, data lock part 2 `8ac7af18f`).** 1,054 of the 1,165 MTOP rows (H5 Choice) added to N5B had
no target in the sources the preregistration names for added rows ("the XL-r2 own-Lux waves"). All 1,054 are rows of
`mx-xl-full-r2` (pool H5 in its id list). All 1,054 are in the published own-Lux **RP-v2** waves, with the same id and
input hash: wave 1 (`7885baf6`) 241, wave 2 (`6bd8eb4d`) 523, wave 3 (`002e5b42`) 290. Those waves are among the
sources N4XF's own composed teacher used (M4 preregistration, precedence order), and research & data counts them in
the 100% XL-r2 coverage (`coverage-r2.json` `ecb6dc36…`). The prereg's short list omitted them.

**Amendment.** Lux targets for N5B's added rows come from the complete published own-Lux source list of the M4
composition, in the M4 precedence order: `A0-train.canonical` (`d8eae3e4`), RP-v2 waves 1–4, XL waves w1–w5 and
`c-w1`, then `h-w1` restricted to H8 rows. Rows outside the block keep N4XF's composed file unchanged. H7 stays
gold-only. No new labels are made. Preflight 3 (every row except H7 carries a target) is re-run on the recomposed
N5B teacher. The rows, quotas, arms and rules are otherwise unchanged.
