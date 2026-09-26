# JevArena authored v11 DEV12: sealed blind and post-key audit

**Verdict: BLOCK_FOR_RELEASE_BENCH.** The independent reviewer solved and
sealed twelve original cases before opening the 39 source-deletion variants,
then sealed a second review before any private target or proof access. The
gold-free packet, both reviews and their summaries match the frozen SHA-256
commitments. Declared UTC seal times are 20:45:40.828164 for originals and
20:48:57.192259 for deletions on 2026-09-26; preserved file times corroborate
the order. These checks verify the handed-off artifacts and declared process,
not the reviewer's unobservable reading behavior.

| Frozen private artifact | SHA-256 |
| --- | --- |
| Original blind rows | `cbd8a6d905a5a6461eb5dd2c56e75a7ffa59ae39a337bddcd83b9ba8381a8233` |
| Original summary | `d69a53ea532648764a8c506c521b56beeb4b30353b897083268523724c996eaf` |
| Deletion blind rows | `42b26ad39c2f975644429199d7e2020130592c014fea0c6f4eabda132933fc7b` |
| Deletion summary | `0d31154df4f1b17debe018c74a7864835147c2ea29787d9b05ad1eb76a933e2d` |
| Aggregate-only private post-key report | `ff8da941e3b1d3cc47d183fe4033cef793820dc452ecfe92ff6188a4768bfe58` |

The sealed original answers matched private targets **12/12**: Choice 4/4,
Noul 4/4 and Score 4/4. The reviewer marked no original as materially
ambiguous or answer-label-leaking. All 39 deletions removed one entire source
and none made the original answer formally entailed by the remaining state.
Nevertheless, four deletions (36–39, parent ID
`3814c55e28494bb49b86`) are materially ambiguous: the unchanged rule did
not fix the four-stage universe or physical order, so three remaining
completed stages can naturally support Grade 3, whereas interpreting the
missing original stage as still required yields an unknown grade. That is a
substantive error in the deletion protocol, regardless of the 12/12 original
agreement or automated two-completion witnesses.

Deletion 32 also retains a conventional rubric cue, although it does not
prove an exact grade. Uniform removal instructions and repetitive qualifiers
in three long cases weaken the realism and information density. The reviewer
packet exposed parent/omitted-source linkage; those fields were kept private,
but they could cue a reviewer who saw the original stage. These findings
block admission of the entire v11 DEV pilot. No v11 item contributes to a
release score, training set, or model selection.

The signed verifier `jev_arena/authored_v11_postkey.py` checks source and
review hashes and seal order **before** opening targets, then checks typed
answers, proof/target agreement, row joins and blinded review aggregates.
The private report contains aggregate counts only. All frozen v11 artifacts
remain intact. A prospective v12 protocol will address the semantic and
reviewer-leakage faults; it will not relabel this result. No FINAL, model
scoring, GPU inference, training, Hugging Face upload or publication occurred.
