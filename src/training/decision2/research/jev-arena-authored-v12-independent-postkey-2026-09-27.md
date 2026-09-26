# JevArena authored v12 DEV8: sealed blind and post-key audit

**Verdict: HOLD_EDITORIAL_QA; BLOCK_FOR_RELEASE_BENCH.** Two independent
reviewers received separate gold-free packets from the [frozen v12
candidate](jev-arena-authored-v12-candidate-2026-09-27.md). Reviewer A sealed
eight original answers before reviewer B sealed eight variants. The private
post-key verifier checked packet, manifest, review, summary and seal hashes,
row rosters, declared UTC times and preserved file times **before opening any
target, proof, salt or join**. The declared chronology is candidate freeze
2026-09-26 21:27:34.001760 UTC, A seal 21:40:18.225197 UTC, then B seal
21:43:34 UTC. File times corroborate that order; they cannot prove a
reviewer's unobservable reading behavior.

| Frozen gold-free or review artifact | SHA-256 |
| --- | --- |
| A original packet / manifest | `124677770eeb3f36fae9914bf08cb863b62fa7b838b95f5378297c8406a24d25` / `55a07f3d7c0c35de50a4d72f2c35f24569e9e4a5e44f2fd960cefeb3015b91d7` |
| A rows / summary / seal | `19cc97c12279b40e494062270a167f28e4e093f024fa086108461c3affcb2c8d` / `1fa1a08f314e0610235962181e382d695e38772bd6b84dcd1884d5ca92e52089` / `0c10c0e7b8a50d84d09f420c658bc453bcb12691c8e97193dab32432050379cb` |
| B variant packet / manifest | `df3c5013153f6e9d47e7fae81b27e11179691a05e3603e9f0768e62e034bcb42` / `cd388d7bcbb8b61f48e421bf81993364f2d83a66ab184963943723655b0784fb` |
| B rows / aggregate / seal | `dcfa2abebe87d297cdebecebd655fad2818927fbffbbab73ee243035f880c498` / `7a131b2a9d55d93279c8154ca757f35fdb6ca49c0c689bd37aab5612cb663ef6` / `56883117ae024e51d497b8853c6d1a077b9e1bfb766fdc813fff06cd58e91883` |
| Gold-free preflight receipt | `6245383ece66dbe8119826170eed9ba50b963657f57fa5f103080e52b6beef04` |
| Private aggregate-only post-key report | `92750c52b33a4ce632109b67b834b0a9b510b13ea196aedd86a64447d8a18ee4` |
| Signed-source verifier file | `e43eabb7c4fd59990331e93e793e8eaf87232a8f182965a0dd7643fc6fcd1aff` |

After the preflight receipt was sealed, the verifier checked private source
and salt commitments, all target/proof/join hashes, typed row joins and all
sixteen source-withdrawal counterfactuals. Blind answers matched the frozen
targets **8/8 originals** (Choice 4/4, Noul 2/2, Score 2/2) and **8/8
variants**. The reviewers marked all original decisions directly solvable and
all variant answers derivable from their own prose. There were no typed-answer
or join mismatches. These are editorial answers, not model results.

The candidate still fails its editorial and native-task gates:

- Three blind *insufficient evidence* answers are absent from the supplied
  Noul or Score criteria. They cannot be admitted as native typed variants.
- Both correction variants make an earlier, fully superseded record
  unnecessary for the current answer. The reviewer identified the redundancy
  without the original packet or private join.
- Six of eight variants reduce to recognizing a missing or redacted fact.
  Reviewer A marked four originals with suspected shortcuts and noted the
  repeated rule wrapper, constructed arithmetic and possible residual cues.

The independent B reviewer sealed `HOLD_EDITORIAL_QA` before key access. The
post-key audit preserves that HOLD despite 16/16 mechanical answer-changing
source-withdrawal completions and perfect blind target agreement. The frozen
candidate and both reviews remain unchanged; repairing these faults requires
a new prospective version. No FINAL, GPU or model inference, training
admission, release score, Hugging Face upload or publication occurred.
