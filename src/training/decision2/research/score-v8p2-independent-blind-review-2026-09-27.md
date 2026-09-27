# Score v8.2: independent gold-free blind review

**Decision: HOLD_OVERLAP.** This review used only the frozen TRAIN and SELECT
blind packets and the separately sealed near-pair packet from the
[v8.2 pilot handoff](score-v8p2-pilot-quality-result-2026-09-27.md). The
reviewer did not open generator keys, labeled source rows, model outputs, CAL,
or protected evaluation labels. No GPU or optimizer was used.

The packet SHA-256 values matched the handoff:

| Packet | Rows or pairs | SHA-256 |
| --- | ---: | --- |
| TRAIN blind | 45 items / 15 triplets | `21dbab7ad3b73423c5973acfb3cd92e3ca9134598b138d4a2b96ac51eb824260` |
| SELECT blind | 30 items / 10 triplets | `e9016fafe6fa57339e51e202db33810e2fd6251463dbc2b88b246f87c428bf5b` |
| Near-pair blind | 8 pairs | `b91645299b8afe4cd221af365e729cf017d76bfd1b8ba40a6472af0241c12003` |

All 75 rendered questions had a determinate answer under their stated rules.
The independent reviewer solved every item and inspected all 25 complete
triplets. Each triplet covered the three ordinal outcomes. In the five
evidence-sufficiency triplets, the dispatch observation alone and the meter
observation alone each left at least two outcomes indistinguishable. This
supports the intended two-source rule on this small pilot; it is not a model
result or a broad shortcut audit.

All eight flagged near pairs have material template overlap. Four pairs retain
essentially the same target decision after record or site renaming and minor
decoy changes. Four change the target evidence or outcome but retain the same
numeric-limit or scoped-exception question skeleton. This includes overlap
between TRAIN and SELECT as well as prior pilot versions. The
[frozen preregistration](score-v8p2-pilot-prereg-2026-09-27.md) holds the
entire pilot on a flagged near match; dropping only the unfavorable groups or
relabeling the candidate as clean would break that rule.

The five long-document triplets are answerable and vary setting, section
heading, and section order. Their rendered text still uses repeated
explanatory blocks and mechanically phrased facts about the original
schedule, signed amendment, unsigned draft, and delivery notice. They remain
synthetic document exercises, with limited evidence of realistic long-context
transfer. This quality concern reinforces the HOLD but is not needed to
trigger it.

The independent answer, per-group quality, and eight pair judgments were
sealed before any key comparison in a private review record with SHA-256
`a5793f53ce99f397ddb0d3961269e582242da03181796b2dd06866787ea71b81`.
The private record contains no protected evaluation labels. Its answers are
not published in this receipt.

Next step: version a new pilot with new numeric-limit and scoped-exception
renderers, fresh source IDs, and a new TRAIN/SELECT split; repair long-document
realism and rerun the gold-free overlap audit before another independent blind
review. This v8.2 pilot authorizes no 27B training, model selection, or release
claim.
