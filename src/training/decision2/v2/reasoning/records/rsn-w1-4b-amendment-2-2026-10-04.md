# Reasoning wave 1 (4B) — amendment 2: dev reads and the formal reads (written before any formal read)

Disclosed: written after the dev reads below, before any Index or JevArena read of a wave-1 model.

## Dev reads (`v2.reasoning.devread`, node F; family-macro accuracy)

| Point | RP-DEV finals | RP-DEV node views | SELECT |
| --- | ---: | ---: | ---: |
| release (Nox-4B) | .667 | .872 | .896 |
| `R4-TF` (α 1.0) | .924 | .940 | .866 |
| `R4-TF` α .75 | .920 | .939 | .874 |
| `R4-TF` α .5 | .910 | .938 | .883 |
| `R4-TFM` (α 1.0) | .925 | .956 | .893 |
| `R4-F0` (α 1.0) | .891 | .846 | .889 |

Paired bootstrap on RP-DEV finals (2,000 replicates over items): TF − F0 +.033 [+.016, +.052]; TFM − F0 +.034
[+.014, +.055]; TF − TFM −.001 [−.015, +.013]. Node views: TF − F0 +.094 [+.078, +.109]; TFM − TF +.017
[+.007, +.028].

**Reading.** Node supervision helps the final decisions beyond the same problems with final answers only (the
registered tree-signal test passes). Writing the parents' true conclusions into the node views adds nothing over
rewired conclusions; the gain comes from supervising the intermediate judgments themselves. SELECT moves only in two
small families (string composition, 40 rows; GoEmotions choice, 200 rows).

## Registered rule outcome and the change

The registered interpolation choice admits no `R4-TF` point: the SELECT floor is .886 (release − .01) and the
best `R4-TF` point reaches .883 (α .5). The registered stop rule then leaves wave 1 without a formal read, although
the tree signal is shown. Change (one, before any formal read): the three formal reads allowed by the prereg are the
three arms at the same α 1.0, which is the cleanest Index contrast of the registered arms:

1. `RS-R4-TFM-bf16` — passes the SELECT floor; release-eligible.
2. `RS-R4-F0-bf16` — passes the SELECT floor; release-eligible; the finals-only control.
3. `RS-R4-TF-bf16` — read for attribution only; not release-eligible (fails the SELECT floor).

Release candidate: among the eligible reads that pass the registered release gate, the one with the higher Index
(the program's rule when a tier has several candidates). Interpolation points were built with
`v2.reasoning.interpolate` (tensors matched by name; mathematically the registered soup multiplicities) because the
release soup and the trainer shard the 426 backbone tensors differently.
