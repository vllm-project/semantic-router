# ConTRoL native projection: second independent blind review

This is a CPU-only quality screen of a possible 4B training source, not a model
evaluation. A second AI reviewer judged the same frozen 30-item, independent
premise-group packet without seeing publisher labels, the first reviewer's
item-level decisions, or the post-reveal adjudication. The task brief disclosed
only the first screen's aggregate concern. The second judgments were sealed
before any of those files were opened. Restricted text, sample identities, and
item-level reasons remain in the private experiment record.

The [first review](qwen35-4b-control-native-blind-review-2026-09-28.md)
documents the publisher TRAIN revision, protected-set screen, deterministic
sampling, native question projection, and remaining long-input and rights
gates. This second pass uses the exact same packet; it does not increase the
independent sample size.

| Measure | Result |
| --- | ---: |
| Independent premise groups | 30 |
| First vs second three-way relation agreement | 27 / 30 |
| First vs second clear/ambiguous agreement | 27 / 30 |
| Both reviewers marked clear | 20 / 30 |
| Both reviewers marked ambiguous | 7 / 30 |
| Second reviewer relations matching publisher labels | 22 / 30 |
| First reviewer relations matching publisher labels | 21 / 30 |

The second review's publisher-label matches were 6/10 Choice, 9/10 support
Noul, and 7/10 contradiction Noul. All 30 second-review native answers matched
the reviewer's own three-way relation under the frozen Choice/Noul mapping.
The second reviewer marked 21 clear and 9 ambiguous. Seven of those nine
ambiguities overlap the prior post-reveal ambiguity set; two prior ambiguous
cases were judged clear and two prior clear cases were newly flagged. Neither
reviewer's judgment is itself a new gold label.

Both reviewers independently gave the same relation, different from the
publisher, on all three cases the first post-reveal audit called likely wrong.
They concern an obligation conflicting with a stated right, an unsupported
contradiction of an existential claim, and a policy consequence stated in the
passage. Another case had both reviewers judge neutral against a publisher
contradiction, even though the first post-reveal audit marked that case clear;
belief attribution versus a statement of actual origin warrants further
adjudication. This fourth concern is not silently added to the prior
three-case count. Source-stratified 30-item results do not estimate a
corpus-wide defect rate.

| Frozen private evidence | SHA-256 |
| --- | --- |
| Label-free packet | `b265d55eb211231bf6971e2875d46700697a9442a376c9c9cb5886d884b77cd2` |
| First pre-reveal judgments | `026c412187abc866a8d2a35cd4374f85d4748cd91c5c0882c21e323b599165c9` |
| Second pre-reveal judgments | `d0a8182913d637a8081675f08e328de7f5de69e8963f3489151003c0a0b9fa1c` |
| Publisher-label reveal | `2e910919e9197cd197ea87578588402c52a76c018f9e6397b188a0a861fb9242` |

**Decision: TRAIN HOLD.** No student TRAIN rows were admitted, no GPU training
started, and no formal evaluation keys were used. Before a matched 4B arm,
resolve at least the agreed likely wrong cases and rubric ambiguities,
complete full protected long-leaf overlap and source-rights checks, and
prespecify a Score-retention gate: this source supplies Choice/Noul but no
ordinal Score supervision. AI-to-AI agreement narrows the review queue but
does not substitute for an independent final label authority.
