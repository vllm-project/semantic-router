# Score v8.1: independent rendered-text blind review

**Decision: HOLD_SHORTCUT_REALISM.** No v8.1 training or formal-set inference is
authorized by this pilot. The reviewer was a separate Codex agent using only
the two gold-free rendered packets, without the generator, source labels,
sealed keys, model predictions, or author-side audit. A script assisted the
item-level reading; this was **not human annotation**.

The TRAIN and SELECT packet SHA-256 values were respectively
`0052e359b517e249b7ff5da96887b7fd6b62b7a72c7a60b6a6789931419aab8d`
and
`8ce1e30879337a7470547f6ba641b78d3cf898918b642e96dda97635d44ac5cc`.
The reviewer sealed all 75 answers, rationales, and 25 group-level quality
findings in a private receipt, SHA-256
`ccb43b31790715d142eed52e446e20edcde1bc1b6a915341b49f305f4c8bae79`,
**before** opening either key. Subsequent key comparison matched 45/45 TRAIN
and 30/30 SELECT answers. No ambiguity was identified in the rendered item
answers. The sealed key SHA-256 values were
`d58114d0764063e6b53815b5b7268518c2d85c57c112b179a93a19cb69a56c0f`
and
`fcc2b83f9866b647370e36a4a3645442ded78644a8af652df982162168788394`.

Two group-level problems remain despite the correct answers:

1. In all **five evidence-sufficiency groups** (15 rendered items), the
   target dispatch diary always says `outage`. The target meter trace alone
   therefore determines the correct level: `normal` → contradicted,
   `unavailable` → incomplete, `outage` → corroborated. The question describes
   two independent sources, but these items do not require using both. Wrong
   site records vary across levels; that distractor check does not remove the
   target one-source shortcut.
2. In all **five long-memo groups** (15 rendered items), the 2.2k-character
   dossier repeats much of the same background prose around one signed
   amendment and one delivery notice. After replacing record IDs and numbers,
   pairwise similarity of the first variant across those groups ranges from
   about 0.75 to 0.999. These are answerable, but the repeated filler gives
   weak evidence of realistic document diversity or long-distance transfer.

The other 15 groups are answerable synthetic probes for dated updates, numeric
limits, and scoped exceptions. The pilot's five mechanisms are a useful
starting inventory, but 75 correct blind answers cannot override the source
necessity and document-realism concerns. The author-side term, position, and
length checks detect different shortcuts; they are compatible with this
negative semantic finding.

A new version should vary **both** target sources across the corpus so that
neither the dispatch nor meter status alone predicts the label, while keeping
the frozen two-source decision rule. Its long dossiers should vary document
structure and place genuinely conflicting, scoped evidence in different
sections instead of lengthening common boilerplate. Freeze new group-disjoint
TRAIN and SELECT packets, repeat blind rendered review, and only then design
matched-budget A/B training. The v8.1 key is now opened and may be used only
for retrospective diagnosis, not as a fresh selector. This review consumed
zero GPU-hours and no JevArena v3 protected labels.
