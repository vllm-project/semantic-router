# Score TRAIN v3: sealed blind verdict and ordered post-key aggregate

**Decision: HOLD / BLOCK_FOR_TRAINING.** Preserve the v3 corpus and anonymous
packet; neither is eligible for a GPU optimizer step or Hugging Face upload.

The independent reviewer received only the 144-row, 48-group gold-free packet
SHA-256 `3e63de13630ce6833417e5a6f23d0aa011cb8b3e3dd123faa07a7062334df9e0`
and public manifest SHA-256
`ff9f1655e49325b04ef08dbffb4c4670861c2615a490093336f7ba4925613c8c`.
It did not receive the builder, source group IDs, private alias join, salt,
gold, model output or another split. The row judgments, group judgments,
summary and final seal were committed by these SHA-256 values, respectively:

| Blind artifact | SHA-256 |
| --- | --- |
| Row judgments | `d200089e8a0890368b9878d550859682a23aa2527a1f5d25513ef6b5d336142c` |
| Group judgments | `07a9f9020c70cf672bd8abd3f2f7c0e1aa5eacf5b9704599f53f4121ad6c92bf` |
| Review summary | `1df22660630b31b8202720feab6578f5420248a9d70c8ef801bd9e234b40546d` |
| Final seal | `12faccf4209abdbd7a03be6685b9c95b7760e198ecdddf160a7d9efa8b0a59fb` |

The seal gives `2026-09-26T21:03:33+00:00`. Signed-off source commits
`4b9285b1e` and `b0bdc7827` provide a two-stage verifier, final source
SHA-256 `086f0a7b177e2cd2f4e449ea882152d08e98ad8aaf8dca80ff6e25f5f243bcc4`.
First, a CPU container mounted only the frozen public review, packet and
candidate directories, **without the private join or salt**, and verified
all SHA bindings, file and declared times, packet order, row and group
coverage, independent level counts and summary claims. It returned
`PUBLIC_SEAL_VERIFIED` for 144 rows and 48 complete groups. Only then did a
second run verify the private join/salt commitments and compare against TRAIN
labels. The private aggregate SHA-256 is
`ede46cba81a318ab4415c9515943a8b0a2036a37e2e3eebf76d5bd7e9f07c394`.
It contains no row text, row/source IDs or per-row labels.

The blind reviewer solved **144/144** rows correctly against the later-opened
key, with all 48 triplets coherent, no ambiguity, no ID leak and no cutoff
wording issue. Those are dataset answerability checks, **not model scores**.
Three independent shallow shortcuts nevertheless block training:

| Shortcut | Blind groups | Full 320-group candidate |
| --- | ---: | ---: |
| Accepted core-item count alone gives the obligation level | 12/12 | 80/80 |
| Adjacent on-time pair count alone gives the streak level | 12/12 | 80/80 |
| A single signal's mark strictly ranks all weighted levels | 4/12 | 24/80 |

The reviewer discovered these from model-visible inputs before seeing the
key; the signed post-key verifier independently reproduced the aggregate
counts. It also confirmed all blind inferred labels, aliases and group joins
against the frozen TRAIN candidate. The v3 weighted **unweighted-sum** shortcut
was repaired, but the new one-signal pattern remains. The fixed graph route
mechanism had no material shortcut in this blind review; that does not prove
all possible shortcuts absent. No held-out benchmark gold was opened, and no
claim of transfer or model improvement follows from this audit.
