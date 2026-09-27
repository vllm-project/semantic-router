# HelpSteer2 native Score: independent blind pilot result

**Decision: HOLD for training admission.** This is a CPU-only source-quality
pilot, not a model evaluation. It used the already frozen 12 prompt groups and
24 candidate responses. No optimizer, GPU, formal benchmark key, or model
prediction was involved. Raw prompts, responses, record IDs, grades by row,
and private file locations remain outside this repository and the research
gist.

The [review protocol](helpsteer2-independent-blind-review-protocol-2026-09-28.md)
was committed before the reviewer opened the blind packet. The reviewer was
not the packet builder. The independent review was sealed with
`gold_accessed=false` before the separate source-grade key was read. The
frozen blind packet, key, sealed review, pinned comparison program, and
aggregate-only receipt have SHA-256 identities:

| Artifact | SHA-256 |
| --- | --- |
| Gold-free packet | `cfc0df08d7bb12bcc67918f64d1fbc6f29199acb86a9e6e24a21b44505aa00af` |
| Separate publisher-grade key | `d996c64cbfb68addaacbfbdaba5d3388bb7c7a9db2dda0c5e06a1a15cc65b097` |
| Sealed independent review | `9dc519d5d9ad9b5c40c98bb4c4e2afef2778df86f5bb86694b03c14a4b6369da` |
| Comparison program | `bd9ebe17df0bba82f19d5180ba13da709f83a49ba6dace2fdfc9fb7a04f4f94e` |
| Private aggregate-only receipt | `a92ca009d6e33b68bcc9ad90fc88f0247e5937598a5e14bffedcf330fef28a45` |

The comparator verified the three input hashes, row and group identities,
and that the sealed review postdated the frozen packet and key. The grade
rubric remained the one stated in the protocol; no grades or thresholds were
changed after opening the key.

| Check | Result |
| --- | ---: |
| Exact five-level grade agreement | 7/24 |
| Within one level | 17/24 |
| Two or more levels apart | 7/24 |
| Mean absolute grade error | 1.125 |
| Mean reviewer minus publisher grade | -1.042 |
| Quadratic-weighted kappa | 0.50 |
| Paired ordering: concordant / tie / discordant | 11 / 1 / 0 of 12 |
| Ordering when publisher-higher response was longer | 5 concordant / 1 tie of 6 |
| Ordering when publisher-higher response was shorter | 6 concordant of 6 |
| Rows with at least one reviewer concern | 24/24 |

The paired ordering is promising as a *relative* quality signal and is not
explained solely by choosing the longer response in this balanced pilot. It
does not establish reliable **absolute** 0–4 correctness labels. The blind
review raised construct mismatch in 10 rows, conflicting factual claims in
11, unverifiable facts in 6, ambiguous grade boundaries in 5, and a
multi-turn presentation issue in 2; flags can overlap. These are reviewer
judgments from a small selected packet, not a prevalence estimate for the
full dataset. A single reviewer and limited external fact spot-checks also
constrain the result.

The preregistered admission bar required at least 20/24 grades within one
level, at least 10/12 correctly ordered pairs, and no material construct or
factual concerns. Only the pair criterion passed. The reviewer's construct
decision was sealed as `false` before the key comparison. Consequently this
source is **not admitted as direct native Score TRAIN, SELECT, CAL, or
independent transfer evidence**. No weak-label remapping or post hoc row
filter was attempted.

The next useful gate is a separately specified, blinded rubric-anchoring
study with multiple independent reviewers and source-disjoint examples,
followed by an independently sourced Score transfer diagnostic. If a future
pairwise arm is proposed, its task construct, rights, and source-group overlap
must be reaudited first; this small pilot alone does not authorize it. The
known protected overlap of two source prompt groups remains quarantined, and
the full source's unresolved near/semantic overlap remains a separate HOLD.
