# JevArena authored scale v2 r4: corrected native Choice order

**Status: BLIND_PACKET_SEALED_REVIEW_PENDING, not release-qualified.** The
[r3 option-order audit](jev-arena-authored-release-scale-v2-option-order-hold-2026-09-27.md)
remains `HOLD_BEFORE_REVIEW`; its private files and packet hashes are retained
unchanged and must not be given to reviewers. R4 is a newly sealed private
development feasibility candidate under the prospective
[v2 protocol](jev-arena-authored-release-scale-v2-prereg-2026-09-27.md).
No model inference, protected FINAL access, reviewer assignment, human review
or Hugging Face publication occurred.

The signed option-order fix `dbf158d7a` was cherry-picked into the isolated
authoring branch as `ca5832be83b79f73e76b11e8128d1a6d3c5a49ed`. An exact
source mirror ran the private preparation, native tokenizer and overlap audit,
and packet sealing. The r4 casebook changes only the version marker: its
twelve structured cases, source facts, substitutions, proofs and typed answers
are semantically identical to r3. Serialization now preserves the intended
criterion insertion order in written native prompts and in sealed packets.

| Frozen private component | SHA-256 |
| --- | --- |
| R4 casebook | `8cc8508a8bec0a686ab4411ee42c56c9102b0b67984392487e02a355ee35f584` |
| Domain-witness audit | `9e605e90c9d8c64dfca539945276c66e18e0a446eff326d0b1a58e2cf73942d3` |
| Source/prompt/answer/proof preparation receipt | `3f86d59cc5e85c618c98914ab3874455c76d2bc7793366561ad018982ded948e` |
| Written original native prompts | `c997dc43ae94e3526fa2ef9c07488436f095221d7966e56ea07af7fc80940c7f` |
| Written paired native prompts | `544f65b6b5e84975a62f240bc5e56ffdfb074b06cdbdf82801babca4e1464a42` |
| Independent prepared-order audit | `067627149910b2bc9aae638f76e004c29000cf0c92a44fa0c5db49bf032b9866` |
| Prompt overlap and native-token preflight | `a14bfbfc10c6db4f6175c7646a6af4b70438e011f3bd9e328995cbde0d4248be` |
| Gold-free original packet A | `1332c1ca6262205ad0de0e14a56d28deec8a262581fc6605ac106b62c2c176d7` |
| Gold-free original packet B | `1ca98892c48e06a8622e6386ac0a8df803ef70f4d3387a5161d1a606d3d6597a` |
| Gold-free paired packet | `ed9f86f00b804dac2ea81ca97f4f650a294c05981cb19dc19698fac80f713b6b` |
| Packet seal receipt | `3599d79cc4958de1707f96b68ea224dfb44a0d377cbb82a39767d6d18fa40cef` |
| Independent packet-order audit | `0d625d9db9849dee6c0b8fb60ef64a5be03ae724bf1aef0c7bb5b1e0b6c89cf1` |

The private r4 candidate has **12 independent originals**, four each Choice,
Noul and Score, plus 12 paired substitutions that are **not** independent
originals. The typed oracle, two-source necessity proofs and domain screen
passed. Noul original targets are 2 true/2 false; Score levels 0/1/2 are
1/1/2; one Choice original correctly returns explicit joint-evidence HOLD.
Independent readback found **4/4 Choice originals and 4/4 paired views** have
exactly the casebook criterion order. The original Choice gold options are
visibly at **position 1 of 4, 2 of 4, 3 of 5 and 4 of 13**. Both separately
salted original packets and the paired packet preserve their prepared native
inputs and criterion order on every row, with no opaque-ID collisions or
unexpected fields.

The full-prompt Qwen3.5 0.8B Base tokenizer still places **9 originals short,
2 medium and 1 long**, spanning 228–2,665 tokens. This limited length mix is
a feasibility check, not evidence that the eventual release distribution is
achievable. Against 110 reference rosters and 159,030 prompt rows, the r4
preflight found zero normalized exact matches, zero three-gram near matches
at 0.7 and zero shared eight-word spans. Maximum roster three-gram Jaccard
was 0.008889; maximum across different originals was 0.019231. Lexical
screening does not establish semantic independence.

The packets are sealed **only for future independent blind editorial review**.
Two separate human reviewers must directly solve every original; another
reviewer must inspect paired views. Their review must assess the long case
paragraph by paragraph, alternative-source plausibility, both-source
necessity, ambiguity, rights and superficial shortcuts before any key is
opened. There are **zero assignments and zero completed human reviews**.
No r4 item qualifies for JevArena release scoring or model training. Even a
passed 12-original pilot would be far below the prospective 1,200–1,480
independent-original release requirement.
