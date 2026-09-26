# JevArena authored v10 DEV12: sealed blind and post-key review

**Verdict: BLOCK_FOR_RELEASE_BENCH.** This is a private development-only
editorial pilot, not a release benchmark, model evaluation or training set.
The independent reviewer sealed original answers before opening the gold-free
source deletions, then sealed deletion judgments before any target or proof was
opened. The signed verifier in commit `de9c990fa` checked file hashes, both
seal contents, packet and review ordering, row joins, source-to-proof mappings,
and aggregate answers. File timestamps support the declared sequence; they
cannot independently prove what a reviewer saw off-host.

| Frozen artifact | SHA-256 |
| --- | --- |
| Gold-free reviewer manifest | `f204a36a44583f186b2539443b173c1907f7377c3ba07ecac51a51bbd91b27c2` |
| Original gold-free prompts | `79567ce204159eb9881316b89135a7b079c9cb6a98e5bea2e4f8b95f91d8f858` |
| Gold-free source deletions | `95002d7019b139c88eaf3131adf6e7f3f5ad77d5108bad82de3918d691996905` |
| Private targets | `564ba78e2dbb40164c62ef29742f0313e9e0b71ada7a50e995d3c8f2141e844e` |
| Private proof traces | `613aa973636bd946f63b746e92f59a73df0c0d78572b419fc05b22b62078780d` |
| Sealed original-item blind review | `17f8360bb49c93ac967b0112e4131b89560a6c5845a3d2e707c89874cf2d2ba7` |
| Sealed source-deletion blind review | `9201fc835336de286b8e2772095eb2eaf9c7199755b5f946a7f681eeeccd6a90` |
| Private post-key aggregate report | `8bc2f8412ac1c564c4892cf8b539c864bb3c826d33ea28bea56511dba6a7ec2e` |

The gold-free packet preceded its reviewer manifest. The original blind report
was sealed at 2026-09-26 20:10:04 UTC, and the deletion report at
20:13:34 UTC. Both sealed hashes match the retained JSONL reports and
summaries. The reviewer determined answers for **12/12** originals, and the
post-key comparison matched **12/12** targets: 4/4 Choice, 4/4 Noul and 4/4
Score. All **42/42** deletion variants lacked a uniquely provable original
answer under the literal structured-attestation rule. Their retained private
proof witnesses each admit distinct outputs after the nominated source is
removed.

The blind editorial gate still rejects the packet. Three variants of a
missing-source Choice case retain a lead that cues the hold decision. Three
status-source deletions retain natural-language hints about the removed
authorization or signature. Every deletion keeps the original introduction,
which sometimes refers to a document absent from the reduced packet. A
ballot-seal source is not answer-discriminating for its negative original
answer, and a wrong-scope distractor also fails to change the selection even
if mistakenly imported. Several long items add formulaic provenance text
around substantive multi-field calculations; the evidence format remains
highly uniform.

The 12-item result establishes answerability and a stronger mechanical
deletion property than v9, but it does not establish natural, shortcut-resistant
document reasoning. No v10 item enters the release-authored set. No FINAL,
model inference, training or Hugging Face publication was performed for this
pilot. The preregistered source is `e66576f2c`, the signed builder is
`dbc95ca94`, and the frozen-packet commitment is recorded in central gist
entry R128.
