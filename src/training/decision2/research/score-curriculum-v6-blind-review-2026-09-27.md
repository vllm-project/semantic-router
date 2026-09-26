# Score TRAIN v6: independent blind review and seal-first key audit

The prospective two-source method and mechanical candidate audit are recorded
in `score-curriculum-v6-prereg-2026-09-27.md` and
`score-curriculum-v6-qa-2026-09-27.md`. This is TRAIN-only evidence; it is not
a model result or a JevArena release benchmark.

## Frozen candidate and reviewer custody

The mechanically gated candidate contains **969 rows / 323 complete groups**
after one whole-group near-clone quarantine, across four families (80/81/81/81
groups) and English 729 / Chinese 240 rows. Its private corpus SHA-256 is
`6aee966cc5499a87d2a77241676586c9c9f801b3c662a078daf025f001169f54`.
Twenty-two pinned protected prompt roles had zero exact or bounded near matches;
this does not prove semantic independence. Maximum pinned tokenizer length was
348/1,024. No single-source evidence baseline cleared the prospectively fixed
shortcut cap, and the source-necessity tests passed. The author did not run a
model or upload the corpus.

The separate gold-free reviewer packet has **144 rows / 48 complete groups**,
12 groups per family, English 108 / Chinese 36. Packet SHA-256
`7c0d65862598f0e90e4f7e24ff296a4c756967af578772756e0c2694fd61a21a`;
reviewer manifest SHA-256
`957e79084c31e17883c6589f245fc811c73f2fd003f60660487e62ef73c019fd`.
An agent independent of the v6 author derived 144 row judgments solely from
this packet, checked all 48 triplets and source-necessity patterns, and sealed
them at **2026-09-26 22:15:58 UTC**, after the packet freeze. Private review
seal SHA-256
`b487c9b7f4ab48d288e0b0b7803352b8630d87c4b33bf14745a4ac1c0720ae0c`;
judgment SHA-256
`f8253092bb6b9a2226e9bbc929338c00372a5c82162866db61f13e63c92ef716`.
Before sealing, the reviewer did not open the key, author source, private ID
join, oracle, model output or protected formal labels.

After validating packet/review hashes and chronology, a separate post-seal
check matched all 144 anonymized reviewer rows to byte-identical source inputs
and compared their sealed answers to the frozen candidate. Result:
**144/144 answer matches, zero input identity mismatches, zero row issues**.
Private aggregate receipt SHA-256
`8e9c31a9fd04f11e0542858973b984dcd9f354f57445a193a7c96f7b4224cdb9`.
The sealed judgments were not changed.

## Decision and limits

**Full v6 TRAIN remains on HOLD** pending qualified independent Chinese
editorial review. English structural reading passed, but an agent's provisional
Chinese reading is not certified native-language review. The v12 authored
abstract evidence-join similarity is disclosed in the candidate QA note; v12
cannot serve as independent validation. The separately authored three-level
Score SELECT r1 was blocked by a shortcut and must be replaced before the
prospectively specified matched training screen. No v6 GPU training, model
claim, HF dataset revision or FINAL scoring has occurred.
