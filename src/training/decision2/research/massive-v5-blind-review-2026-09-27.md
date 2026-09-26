# MASSIVE v5 multilingual intent candidate: independent blind diagnostic

Status: **TRAIN HOLD**. This is a gold-free AI semantic audit of a candidate
training curriculum, not a native-speaker human review, a model evaluation, or
an approval to train. No model inference was run.

The review used two successive sealed packets. Stage 1 presented 84 localized
requests across seven locales, with a fixed set of four short intent
descriptions. I marked an intent only when one description **exactly** matched
the request; a merely related topic was insufficient. Stage 2 revealed the
English anchors and six localized requests for each of 12 opaque groups. It
checked action, entity, time, polarity, scope, intent, numeric details,
naturalness, and uncertainty without an answer key.

| Blind stage | Input seal (SHA-256) | Judgment seal (SHA-256) | Receipt seal (SHA-256) | Sealed UTC |
| --- | --- | --- | --- | --- |
| Stage 1 | `abd893f5dbde658d8a8f4602439b8ed34ec19adfcf650f18afca1ece093dee85` | `1bc0159c85b28de6efc46df4c57d2913aec17da20aa94c771662f40d613ba770` | `1dd13ad436d4f8ec2609384ca3d0740f0f628411e92cb34577d4a8d7a08e00f3` | 2026-09-26 23:13:14 |
| Stage 2 | `7fc9f976039d65ca58812233cdc2229dc68c0032d2f20308cf582d4fe43e3d2b` | locale: `5aec6d0779a934c2e80b4f8939f7677d6e0b4fabe62a6cc14cc7c55987d165a3`; group: `782e28ccb97b6cfce3a3c1f561cb5a1386d280c20f185b5a4d86a8d6ffc23918` | `9a29e46489b1d1d0380a8133a64219c446ee713b459309a0bb0f015291706f38` | 2026-09-26 23:16:50 |

The Stage 1 packet contained 12 requests in each of seven locales. Under
the strict option wording, **41/84** had one exact fit and **43/84** had no exact
fit. Naturalness judgments were 47 clear, 34 awkward, and 3 seriously
distorted. These are reviewer judgments, not agreement with a hidden key or
accuracy scores. They expose a likely mismatch between broad source intent
names and narrower displayed options; key adjudication must distinguish that
from labeling errors.

Stage 2 covered 12 English-anchored groups × six localized requests =
**72 comparisons**. The reviewer marked 26 exact, 26 minor, and **20/72
material semantic drifts**. A group passed the blind parallel-preservation
gate only when all six requests preserved the material request and were
natural and unambiguous: **1/12 passed; 11/12 were quarantined**. Recurring
problems included changed named entities, changed numeric thresholds,
translation errors, odd time expressions, and awkward phrasing. An intentional
locale adaptation can still be useful as its own item after separate review,
but it cannot be treated as an invariant parallel pair.

The first-stage judgments and receipt were sealed before the second-stage
comparison. Both stages excluded the private intent key and author-side
candidate/target files. The sealed row and group records remain private; no
source utterances, answers, credentials, or infrastructure details are in
this note.

**Release gate:** All seven locales remain **HOLD** until qualified human
reviewers independently confirm option semantics, translations, source labels,
and any intended localization policy. The post-key adjudication is a separate
step and must not retroactively alter these blind seals. No row from this
candidate should be promoted into the training corpus on the strength of this
AI diagnostic alone.
