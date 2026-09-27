# Score v8.4 pilot: independent blind review

Date: 2026-09-27

Disposition: **HOLD**

An independent reviewer answered all 90 items in 30 three-variant groups from a blind question packet, recorded a section citation and quality note for every item and group, and sealed the answers before opening the answer key. The answer file was sealed at 2026-09-27 13:36:44 UTC with SHA-256 `a0af1361266a9b13fb4a64f3eb36f0e48c8188b0cd29aef0ba207f4382dbe77b`; the seal receipt has SHA-256 `35ba9dede5e8143b98ba9f3dd57fa6ea492a88816e32298bb096d335a3a12667`. No model training or evaluation run was part of this review.

## Label correctness

The sealed answers match the key on **90/90 items and 30/30 groups**, with 30 matches in each of the 0, 1, and 2 classes. Under the explicit scoring instructions, the reviewer found no unresolved item-level contradiction or label ambiguity. This agreement verifies the pilot labels as written; it does not validate the intended reasoning difficulty.

## Quality and construct validity

- **Long context does not carry the answer.** In 24 groups (72 items), 12 or 24 archived background notes are identical across all three variants. The four operative sections already determine every label. The padding adds length without requiring retrieval or reasoning over it.
- **Six repeated scenario patterns allow shortcuts in all 30 groups.** The policy cases keep region and effective-date relationships fixed; service cases share the same weekday and holiday pattern; stock cases put the requested SKU first; workflow cases use the same two-ancestor chain; evidence cases expose literal outcome states; connection cases present one matching route. These regularities bypass much of the intended scope, source-joining, and date reasoning.
- **Copy and document realism need repair.** All 18 English core scenario groups contain missing spaces after punctuation. Several Chinese cases use unnatural transfer terminology or mixed punctuation. Some evidence cases attach a document identifier to a result described as unrecorded. These defects did not change the formal labels, but they weaken the credibility of the documents.
- **Blind packet metadata was too revealing.** Accompanying aggregate metadata disclosed partition sizes and class balance before the review. It contained no item-level answers, and the reviewer did not inspect source rows or per-group partition assignments. A future blind handoff should omit these aggregates until after the answer seal.

All 30 groups are quarantined from training and release pending a rewrite and another independent blind review. The next pilot should make the added context materially necessary, vary the currently fixed scenario dimensions and target placement, repair English and Chinese copy, and keep class and partition aggregates out of the blind packet.
