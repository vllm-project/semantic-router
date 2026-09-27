# HelpSteer2 Score: independent blind review protocol

This protocol was committed before this reviewer opened any pilot text or
source-grade key. It covers the **existing** 12-prompt-group / 24-response
gold-free packet only. Packet SHA-256:
`cfc0df08d7bb12bcc67918f64d1fbc6f29199acb86a9e6e24a21b44505aa00af`.
The separately published key SHA-256 is
`d996c64cbfb68addaacbfbdaba5d3388bb7c7a9db2dda0c5e06a1a15cc65b097`;
its contents are not opened until a private blind review is sealed. No text,
source row ID, key, or prediction belongs in this repository or the gist.

The reviewer has read the upstream aggregate source audit: the packet balances
six source-grade pair bins and the direction of the response-length difference.
This reveals the pilot *design* but no response-level answer. The reviewer was
not the packet builder and has not seen individual source correctness grades.

## Judgment rules fixed before review

Grade the candidate response **for factual correctness and sufficient coverage
of the user request**, ignoring fluency, politeness and length except when a
missing requested fact makes the answer materially incomplete:

| Grade | Operational meaning |
| ---: | --- |
| 0 | Incorrect, unrelated, or does not answer. |
| 1 | Mostly incorrect or materially incomplete. |
| 2 | Partially correct with significant omissions or errors. |
| 3 | Mostly correct with only minor omissions or errors. |
| 4 | Correct and sufficiently complete. |

For each response independently, record one grade and zero or more private
flags: `FACT_UNVERIFIABLE`, `CONFLICTING_FACTS`, `MISSING_EVIDENCE`,
`MULTITURN_FORMAT`, `LENGTH_CUE`, `AMBIGUOUS_GRADE`, `CONSTRUCT_MISMATCH`,
`NONANSWER` or `OTHER`. A short private rationale may accompany each flag.
For the pair, also check whether a longer response alone appears favored when
the substance does not support it. Do not infer the publisher's source grade
from the known pair-bin design. A coherent response with unverifiable factual
claims receives an uncertainty flag rather than automatic grade 4.

Sealing order: verify exact packet SHA and schema, inspect text without opening
the key, write 24 grades and flags to a private immutable review file with
`gold_accessed=false`, record its SHA and UTC timestamp, then run the pinned
post-key comparator with exact hashes. Never revise the sealed opinions after
opening the key. Report only aggregate agreement and failure categories.

Candidate admission is **HOLD** if the packet has material factual uncertainty,
construct mismatch or unstable multi-turn mapping, or if blinded ordinal
agreement falls below 20/24 within one grade or 10/12 paired orderings. Those
thresholds are a pilot quality filter, not a statistical guarantee. Even a
pass does not admit source TRAIN: known protected overlap groups must be
quarantined, remaining rights and near-duplicate exposure checked, and an
independent transfer diagnostic established before any optimizer run.
