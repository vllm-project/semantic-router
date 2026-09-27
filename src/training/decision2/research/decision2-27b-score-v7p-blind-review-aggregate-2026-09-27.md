# 27B Score v7p: independent blind quality review

**Decision: HOLD on the realism requirement.** This is a gold-free review of
the fixed v7p TRAIN and SELECT packets, not a model result or a release score.
No answer key, labeled source row, model prediction, typed FINAL label, or
CSS15 label was inspected. No GPU training was authorized by this review.

## Review boundary and immutable receipts

| Packet | Independent groups | Items | Blind-packet SHA-256 |
| --- | ---: | ---: | --- |
| TRAIN | 80 | 240 | `5c59e61ec2ba25e21f690a2a93050dfefc687a39686aa645691fbfc3be4d2f2d` |
| SELECT | 80 | 240 | `4945d21d0529a8a8bf3ee87946bc9091462cabf5d6c0448a98ae0ca0695398d2` |

An independently authored reviewer parsed the complete rendered prompt for
each of the 480 items, applied only the printed rule, and sealed a private
per-item answer, evidence-sufficiency and ambiguity receipt **before** any key
comparison. It also recorded a group-level source-necessity and shortcut note
for each of the 160 groups. The reviewer script SHA-256 is
`214caee01b23b20fa87d07689f45fc275b3a8732571209a6ea90c2d79cf6666a`;
the read-only private receipt SHA-256 is
`d66ac7e94e0c9556404cc1a3ed7bdd72ab9c495ea3dfd043e7e8a20c98c0db56`.
An additional manual semantic spot review checked 24 items from eight groups,
one per domain caption, covering both record orders. Its separately sealed
receipt SHA-256 is
`52477c69502d8a7a1c6e96f52411fae782edac28a60556d57f4638409605d470`.
The manual review covers these 24 items only; the other 456 have a deterministic
rendered-text review, not individual human adjudication.

## Aggregate findings

- All 480 prompts matched the declared three-option rule and were fully
  answerable from their rendered text under that rule; the independent parse
  found no malformed or ambiguous item. The sealed answer distribution is
  balanced at 160 items per Score level. The 24 manually reasoned answers
  agreed with the sealed independent answers.
- Every group contained three related variants and all three Score levels.
  The positive level requires the two named evidence documents; an applicable
  current scoped veto overrides the requirement status. The packet-level
  answer positions were approximately balanced. These are structural
  observations, not a claim that a learned model will use both sources.
- All eight domain captions reuse the same abstract A/B rule, opaque case and
  scope identifiers, short two-document form, and nearly identical lexical
  presentation. The captions change the setting but not the substantive
  decision semantics. This is a useful controlled source-joining mechanism
  test, yet it does not establish realistic cross-domain decision transfer or
  long-document ability. That limitation applies to all 160 groups.

The existing preregistration requires independent review of answerability,
ambiguity **and realism** before admitting v7p labels. The first two checks
pass the blind rendered-text review; realism does not. The fixed pilot remains
HOLD under that criterion. A later use as a narrowly described synthetic
mechanism ablation requires a prospective versioned decision rather than
silently treating this receipt as a quality pass. Any comparison with the
sealed key must occur after these receipt hashes are retained, and any mismatch
must be reported rather than repaired in place.
