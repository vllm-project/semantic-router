# JevArena authored v13: independent post-key DEV audit

**Verdict: DEV_NATIVE_SOLVABILITY_PASS_RELEASE_HOLD.** The frozen v13 pilot
contains 12 independent originals, four each of Choice, Noul and Score, and
12 paired complete-source substitutions. The paired views do not increase the
independent count. No model ran on this pilot; it is neither a training source
nor a release benchmark.

The author froze the candidate at **2026-09-26 22:43:29 UTC**. Prompt-only
overlap and tokenizer preflight sealed at **22:44:42 UTC**. Reviewer A saw only
the salted original packet and sealed direct answers and editorial judgments
at **22:46:08 UTC**. A different reviewer saw only the independently salted
substitution packet B and sealed at **22:51:22 UTC**. Neither had access to the
private author targets, casebook, join, proofs, model outputs or protected
FINAL. These are independent blind editorial passes, not qualified human
review of a release set.

| Sealed component | SHA-256 |
| --- | --- |
| Preflight seal | `6279a7fc3cf578be814f172c902644a972ced7eeaaa8f6c39c07d1539abf4185` |
| Original packet A | `6b76e22cdbf5fbe8252c0acecbfa9bb961ce403197254a91da5eb105011dbe91` |
| A judgments | `2e5aba0b823ef3bb9b37bc4738f0c6069cf85721a78a7cb768c8c15a7c1acca0` |
| A seal | `d3d04c11a95402fd2fee995129316a8f9178d92e7cdfa65348db33b634b8be74` |
| Substitution packet B | `777ece895641d38f05c2e0dd3b305ce5aa463491789fc70f391aa980c7072c63` |
| B judgments | `75f0f8e5b0f5ed8bea0b418da5cf9e6223a1aa72d47c6c9b7b25e01fa2a27162` |
| B seal | `d243698dd3fedc53f33bb482e8a5687fb848ffc075a87098f81c7584b8c299cc` |
| Independent post-key aggregate receipt | `60885d9ffde77bad0cb1cc1791dd28993d302a16ac70737018fb2baf40068213` |

Only after both reviews sealed did a separate private audit open the frozen
targets. It rehashed both packets, manifests, reviews, seals, target files,
proofs, casebook and one-to-one join; checked freeze, preflight and review
chronology; checked typed answers against each row's native criteria; and
compared reviewer answers to the author oracle. **A matched 12/12, B matched
12/12, and all 12 matched original/substitution pairs changed their target
answer.** Both blind passes found no material ambiguity, missing native
answer or failed two-source-necessity check in their packet.

The limitations matter. All items are short, controlled, somewhat formulaic
two-record cases and establish no long-context or real-traffic coverage. In
one original, the first displayed source's top label happens to equal the
final Choice answer, although the second source remains necessary to certify
it. One B dependency chart has a minor arrow-notation clarity issue without
changing its answer. These observations make the pilot useful for native
validity but do not establish resistance to shallow model shortcuts.

**Next gate:** author and independently review a much larger, diverse and
separately frozen original pool, including long evidence, realistic documents
and difficult distractors. Keep this pilot in DEV. Do not turn its paired
variants into independent release samples or report model performance from
it as FINAL. Existing v9-v12 blocked versions remain blocked.
