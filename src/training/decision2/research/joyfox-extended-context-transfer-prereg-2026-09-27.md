# Joyfox 0.8B context admission on independent development panels

**Prospective gold-free execution.** The public-231 context ablation found that
changing the native wrapper's no-truncation cap from 1,024 to 4,096 admitted
41 previously invalid long inputs and improved the public score by 10/231.
That result alone cannot tell whether the extra context helps typed decisions
or human-labeled transfer. This test evaluates the same unchanged model on
the existing typed DEV and CSS pilot panels. Neither panel is sealed FINAL.

Freeze Joyfox model revision
`ae7b7040aeff7802f6f2bcfdd27f08a72d5cd969`, native source revision
`2677b5a3714489847668175de793e2d92fe183f0`, and the checked
`decision_config.json` SHA-256
`859322c0adc3f2bb5a8b9b7448369ae2de3d216a3b7deff5ae2f8f13c786e0b2`.
The existing 1,024-token comparator predictions have SHA-256
`ec23b20f191aa03de65ad3a018f22eb74c1bec8f937ba0c48b8397b488e8825b`
(typed DEV) and
`f70333e8ba2d538f3ac63bbead9a1d326db269bf182645cec5a279ea5c461349`
(CSS pilot). Gold-free prompt SHA-256s are
`a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a`
(DEV 1,600) and
`598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda`
(CSS pilot 1,430).

Run exactly one new 4,096-token native adapter pass on each full panel, with
the existing collector's distinct `joyfox-native-extended-context-v1`
identity. No weight, tokenizer, decision head, candidate order, calibration,
scorer or prompt change is allowed. An overlength request remains explicitly
invalid; do not truncate. Store fresh outputs without overwriting the original
1,024-token receipts. Before labels, compare model/source/input identities,
answered IDs, previously admitted categorical answers and newly admitted
counts. Then score with the unchanged typed and CSS pilot scorers, reporting
per-type/per-task changes, invalid counts, calibration and long-item outcomes.

This is an **open development diagnostic** of an extended local wrapper.
It cannot qualify a Decision 2.0 weight package, claim the published Joyfox
contract has changed, or select from multiple context caps based on the
public-231 score. A promising result justifies a future same-data/compute
small-model training hypothesis and independent sealed evaluation.
