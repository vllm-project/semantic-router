# Authored v2 r3: serialized Choice order invalidates blind-review allocation

**Status: r3 HOLD_BEFORE_REVIEW.** This corrects the earlier
[feasibility receipt](jev-arena-authored-release-scale-v2-feasibility-2026-09-27.md).
No reviewer had been assigned, no blind verdict was submitted, and no model
inference, protected FINAL access or HF publication occurred. All r3 private
snapshots and sealed packets remain immutable for audit; they must not be
distributed for editorial review or counted toward a release pool.

The v2 casebook committed a deliberate order of Choice criteria. Its private
builder created the intended ordered Python mapping, then wrote each prepared
JSONL row with `sort_keys=True`. That recursively reordered criterion keys.
The packet writer also sorted keys, but a read-only comparison confirms that
each of the three r3 packet files preserved its already reordered prepared
native inputs exactly (12/12 rows each). The fault therefore starts at the
**prepared prompt serialization** boundary, before the packet.

All four Choice originals have a visible criterion order different from the
casebook order. Their actual gold-option positions are 9 of 13, 1 of 4, 2 of
4, and 3 of 5, instead of the preregistered 1–4 position coverage. This
invalidates the claimed answer-position balance in the preflight and r3
feasibility receipt. Native option order and prompt hashes also differ from
the intended design. The other mechanical observations do not repair this
distribution failure, and no one should treat r3 as a review-ready pilot.

The source fix versions the candidate builder anew, writes the original
insertion order, and verifies every serialized original/paired native prompt
against its in-memory source and every Choice criterion order against the
casebook. The packet writer separately preserves and checks that order. A
nonalphabetic-key regression test reproduces the old failure. A corrected
private candidate must get **new** prompt/snapshot/packet hashes and repeat
the full native, domain, overlap, distribution and human-review gates; r3
remains a negative result. No post-hoc score or blind verdict from r3 is
eligible for reuse.
