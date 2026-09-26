# Score TRAIN v4: frozen negative feasibility screen

**Disposition: HOLD_NO_CORPUS.** This is an in-memory, CPU-only feasibility
screen of the previously registered v4 method, not an admitted training set or
model result. Preserve the v4 code and this failed result unchanged. V1–v3
remain blocked for training after their respective independent reviews.

## Pinned method and ordered result

The corrected prospective v4 note was frozen at SHA-256
`aa7be2da837c31edac86bf787acc1c3d2b308c580b0e9996ec8c5301a94c2c61`
before any v4 construction. The signed v4 builder is commit
`fd039fd24cd8fcc480c72bfe7cce7b3742e57edc`, file SHA-256
`dd837928d213d27eb2fc82c9f5b5f8562a93d03733f1408241e8828918cf8062`.
Its focused CPU checks passed 4/4 in the pinned training container, digest
`sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1`.
The aggregate-only receipt script was signed in `dd9ee0747d3afed4dab4a8c17cc910eecff43ee7`
and its string-key parser was fixed in `c6bb57b25f6e1548a3b476a77e00029c0f3aab64`;
the final script SHA-256 is
`04f3f02eb6c1dad1a5607494fe89ca7b967b95192208788d6d9c8e04e6c9e4ae`.
The exact source mirror and pinned CPU container reproduced the failure. The
private aggregate receipt SHA-256 is
`8d807bc48f6c8a82d27ac6d32fe4f14e80e6b41f73bd4c90c4e4a474641b4328`.

The builder assembled 972 proposed rows in memory, including 243 weighted
rows, then failed its preregistered group-heldout single-signal classifier
gate. Best accuracy was **112/243 = 46.09%**; the 40% gate permits at most
**97/243**. The five fixed-position counts were **103, 108, 104, 112, 108**
correct. This was a genuine negative gate, not a source error. No v4 corpus,
gold-free editorial packet, private join, or HF dataset was emitted. No GPU
optimizer step ran. Because generation stopped at the first failing gate,
protected-roster overlap, tokenizer, rights-lineage, and independent blind
editorial gates were **not run** on v4; those checks must not be inferred from
earlier versions.

## What the failed gate does and does not establish

The source-group-heldout feature was one displayed signal's `(weight, mark)`
at one of five fixed positions. It learned label counts from the other groups,
predicted all three rows of the held-out group, and reported the best of the
five positions. The result is above the frozen threshold. The within-triplet
rule that no signal has three distinct marks can still hold while a recurring
two-state signal distinguishes one level from the other two across groups.

There is a separate mathematical limit: with fixed weights and fixed cutoffs,
if the **full `(weight, mark)` marginal** were identical across all three
labels for every position, mean weighted totals would be equal. A complete
triplet has strictly ordered integer totals, so exact equality is impossible.
This does **not** prove the weaker 40% classifier cap impossible. The present
screen establishes failure of the frozen v4 pairing selector on its specified
81 weighted groups. It supplies neither a global lower bound over every
allowed pairing nor permission to relax or retune v4 after observing the
result. The next attempt requires a new prospective design and an independent
quality gate.
