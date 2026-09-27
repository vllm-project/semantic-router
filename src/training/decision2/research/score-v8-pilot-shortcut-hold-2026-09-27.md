# Score v8 pilot: shortcut HOLD before independent review

The 75-row v8 candidate was generated after the signed design at `7d80519a1`
using signed code `0f1f5e2c9`, on CPU only. TRAIN contains 15 complete triplets
and SELECT contains 10. Both roles have five mechanisms and equal 0/1/2
counts. The private candidate manifest has SHA-256
`3ac976925e8ae53dd5661e29a74614bf012fd8fa751fa36c4b469324dde91b87`;
the private gold-free review packets have SHA-256
`d648e88c5ae778fe601c0a379f98869f43217971dc15070f3d4f880c0b5f633a`
(TRAIN) and
`0d7d587dcd26ffe566f80da4c930c6ce123471ed17523fa180c5d70c5938138f`
(SELECT). Keys remain private and separate.

The CPU audit verified frozen parent TRAIN/SELECT/CAL identities, 28 protected
gold-free prompt inventories, previous v7p TRAIN and its blind SELECT packet,
all 75 rendered oracle answers, and normalized exact plus bounded near-text
overlap. It found zero matched comparisons. Its private receipt SHA-256 is
`c75208c52eb1bbaf26758a63f26a3ff71124c0c7d89fa16b80f96c3a4341415a`.
These checks do not establish realism or causal transfer.

An additional author-side shortcut inspection found decisive defects before
handing the packet to an independent reviewer:

- In all three TRAIN scoped-exception level-1 rows, `pending` occurs; in no
  level-0 or level-2 scoped-exception row does it occur. A word-presence rule
  can answer without checking site, activity, or date.
- The evidence-sufficiency distractor mirrors the target source status in each
  row. A model can infer the level without matching the claimed site.

The frozen candidate is **HOLD_SHORTCUT**. No reviewer was asked to spend time
on its known defects; no GPU, optimizer, CAL, formal labels, or release score
was used. Preserve its private artifacts and redesign in a new version rather
than substituting only the unfavorable rows.
