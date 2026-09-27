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

## Completed result

The full native inference passes returned 1,600/1,600 typed DEV and
1,430/1,430 CSS pilot prediction rows with the frozen model/source identity,
PyTorch `2.12.0+git6bbd260`, HIP `7.2.53211` and Transformers `5.17.0`.
The exact checked collector SHA-256 was
`ae976275cd58b4daa5504a9398f518d9403dbb342816ebc54fd39b2d1f6c49f7`;
the unchanged typed/CSS scorers were
`d02a3b2bbaa08ec45928fc354532b3c3b5aef80e0a5d8e9ed6348ad6d30e2bcc`
and `cfe199a1826bb89b27c9eb746f808d74f16b46ca6585ac7b0ff7e440d44eeaca`.

| Development panel | Native cap 1,024 | Local cap 4,096 | Interpretation |
| --- | ---: | ---: | --- |
| Typed DEV accuracy | 993/1,600 | 994/1,600 | Choice 616→618/800, Noul 205→204/400, Score 172→172/400. All rows valid at both caps. |
| Typed DEV Brier / ECE10 | .23653 / .07776 | .23645 / .07393 | Very small changes; not a validated gain. |
| CSS pilot valid | 1,407/1,430 | 1,428/1,430 | 21 previously overlength rows became valid; two still overflow at 4,096. |
| CSS pilot micro accuracy | 510/1,430 | 516/1,430 | Nine of the 21 newly admitted rows are correct; already admitted predictions also changed. |
| CSS pilot median task macro-F1 | .30119 | .29699 | Worse despite higher micro accuracy. Discourse .25572→.26862, implicit hate .30119→.29699, stance .47341→.47145. |
| CSS pilot median task Brier | .78193 | .78203 | No calibration benefit. |

The same IDs and input digests were checked before scoring. Changing the cap
also changed three categorical answers among all 1,600 previously admitted
DEV rows and ten among the 1,407 previously admitted CSS rows. On CSS, those
ten include one newly correct and four newly wrong answers, accounting for
the net six additional correct together with nine correct newly admitted
rows. This effect makes the comparison broader than pure length admission;
runtime/kernel sensitivity is not excluded. An initial CSS process with two
simultaneous GPU visibility masks failed at model load before writing any
prediction; the single-mask relaunch completed. The failure is retained as a
runtime setup event, not scored as a model answer.

The new prediction SHA-256s are
`f8126f3bc8bc1feac7e21c0adf58c540d97734bc955157c209b94ba8e8f2a380`
(DEV) and `b676ad67e7252582b716f50ca6c35a2b2a14981e14401c99d307a06dd6a57d54`
(CSS pilot); score SHA-256s are
`11e93f6ce1c1634fec5555d0009e4dff3a2d13e2d4ad626230f3fd514ffb1e4b`
and `d4d01d8623be2e8e60123e3537cbdc573ddf6adc315287cb5865d8e510db36e3`.
Summed per-request synchronized GPU latency was 111.76 seconds across the
two successful processes, a 0.031 GPU-hour lower bound before model loading
and setup. The result does **not** support a 0.8B release or a context-only
transfer improvement. Subsequent 0.8B training should target source-disjoint
human transfer and Score, with a new preregistered model/data arm.
