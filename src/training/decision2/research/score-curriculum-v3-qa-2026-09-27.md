# Decision 2.0 Score TRAIN v3: pairing repair and blind handoff

**Status: editorial review pending; BLOCK_FOR_TRAINING.** This candidate is
TRAIN-only. No optimizer step, Hugging Face upload, held-out answer key, model
score, or release claim was produced. The independent v1 and v2 blind failures
remain in the unified research ledger; neither candidate is reused.

## Prospective intervention and source

The [unified preregistration](https://gist.github.com/Xunzhuo/cd90fce0fa548616d8a4f1b2d2398dea)
froze four equal mechanisms, three ordered levels per group, a fixed-multiset
weighted-pairing repair, independent oracles, count-only shortcut gates, and a
fresh opaque editorial packet. Signed builder commits are `f0ccfd10f` and
`437c0bb17`; the final builder SHA-256 is
`463c6d87780ceb339b4f2c08153ea7e4e91b2153540b0738ba86439c5a95e787`.
The packet builder and its tests are signed in `a70f49ad4`, builder SHA-256
`730adbe71e95d3b0154e341973524fca87e6ba12d27788b0583c08cf77e3669e`.

The parent TRAIN/SELECT/CAL partition hashes are bound in the code and
manifest. Added v3 examples are internally generated and contain no external
text. The parent rights-clean-v2 manifest records its source-level licenses,
including CC BY, CC BY-SA, AFL and internally generated material. This mixed
lineage remains private research data and is not cleared for redistribution.

## Protected-source correction

The first v3 build protected 14 gold-free source rosters. Before its editorial
packet, the authored v11 editor supplied the already-frozen final r3 roster.
The unified preregistration was amended at gist commit `f9f62ed` before a new
build. The 14-source artifact remains immutable but is not training-eligible.
The 15-source protected inventory SHA-256 is
`287bb2c602775dcd6070a15e8b8bdfb42943725b29a17bc166b3c9d1805e96b0`.
It includes the final r3 gold-free prompt SHA-256
`94b9ff5db3735ca59e0197700a7a4a2e16b9fd8d045cbfc87ffeb0a8ef6187d5`;
no private target or proof was read. Exact and approximate group-complete
checks found **zero matched rows and zero quarantined groups in every one of
the 15 gold-free rosters and parent TRAIN, SELECT and CAL partitions**. These
checks cannot prove full semantic independence.

## Frozen candidate and mechanical QA

| Artifact | SHA-256 |
| --- | --- |
| 15-source v3 manifest | `5ab30da57bd93f7e2045c68d68e607bd5d9fca62aa987d3594ae07f0311c89a7` |
| Candidate TRAIN | `c4ea3294247022fdf2359e6ed74c0abec0bc6de0295fad06af6271c37a9d0fb7` |
| Parent plus v3 TRAIN | `26f7aa77f5244be7b30040aa83f5c659fd9d032a31e79ae5e2a7602f8b1c0e59` |

The new corpus has 960 rows in 320 complete groups: 80 groups and 240 rows
per mechanism; every group contains levels 0, 1 and 2, yielding 320 rows per
level. English has 720 rows and Chinese 240. The revised weighted-points
triplets preserve the signal names/order, weights multiset, marks multiset,
unweighted sum, thresholds, options and strongest signal product while
changing mark-to-weight pairings. Independent arithmetic, BFS, ordered-streak
and precedence oracles passed. The preregistered shallow feature classifier
was at 80/240 for weighted points; direct unweighted-sum threshold and
group-held-out mark-sum lookup were also 80/240. Route link count and streak
on-time count each yielded 80/240. These gates remove the known v1/v2
shortcuts but do not rule out new position or phrasing shortcuts.

With the pinned Qwen3.8-27B tokenizer revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, rows span 210–319 tokens,
all below the 1,024-token cap. The added rows total 247,072 tokens. The code
passed `make impact` for the training domain, focused pinned CPU-container
tests (8 passed), and the repository `make test-training-contracts` gate.

## Gold-free editorial packet

The fresh packet takes 12 complete groups per mechanism, totaling 48 groups,
144 rows, and 36 rows per family. It has 108 English and 36 Chinese rows.
Public fields are only opaque review/group aliases, family, language, state,
instructions and options. A new private random salt HMAC-aliases both IDs;
the source join and key are outside the reviewer directory. Schema,
cardinality, SHA and private file permissions were checked before handoff.

| Packet artifact | SHA-256 |
| --- | --- |
| Gold-free packet | `3e63de13630ce6833417e5a6f23d0aa011cb8b3e3dd123faa07a7062334df9e0` |
| Public manifest | `ff9f1655e49325b04ef08dbffb4c4670861c2615a490093336f7ba4925613c8c` |
| Private alias join commitment | `49a8098abc076e5a583acb848d054305527fa51a14db26fc3f9f7be8ba579c7a` |
| Private salt commitment | `4e26b885bb8859e5bd9cd2b2494c116a3b7dc853a0927931d32e0f21cbe213f5` |

An independent reviewer must solve all rows and inspect triplet coherence,
Chinese cutoff wording, name/position features and shallow shortcuts before
sealing a verdict. The author must not compare a private key until that review
is sealed. Any material flaw holds the entire v3 candidate. A clean editorial
result would only justify a separately frozen group-disjoint three-level
selection gate and a proposed matched-control pilot, subject to project lead
approval. Synthetic arithmetic validity alone does not establish transfer.
