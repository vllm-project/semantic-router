# JevArena authored v8 DEV12: editorial pilot, awaiting blind review

This is a **private development prototype**, not a release benchmark or a
model result. It responds to the independent v7 review: three v7 long-item
ablations retained an explicitly superseded same-case value, and several items
contained non-discriminating fields or decorative long prose. The v7 packet
remains frozen and blocked.

## Design and frozen evidence

- Twelve separately authored scenarios: four each of Choice, Noul, and Score.
  Each type has one long join, one missing source, one rule-precedence case, and
  one ordinary case. No v7 prompt is reused as a v8 item.
- Gold-free prompt SHA-256:
  `0cbbc2ed710c22b35280f33b0ab7d4a6204110888d826f382b590a2296b87904`.
  Private target SHA-256:
  `36890e64d673810bed6d1a055d06625aef4174e7c1789f4fb8f2c7541acc048f`.
  Private proof SHA-256:
  `2bc1da368de3be1a4097766314f60b73294c530cce006594b34e750922e8e6f9`.
- Private specification SHA-256:
  `24bf117c57b45d08d9d7d79077e6c10f3483db7f98e8baaef5cce26421efa108`.
  Private salt commitment SHA-256:
  `3becef4edad3cfb3e4d85ab3bee2ef7e7912f7e7eeb69af56b23bbb9e8c3530d`.
  Raw authored cases, salt, targets, and proofs are kept in the authorized
  private experiment workspace.
- Nine **gold-free** source ablations SHA-256:
  `ee4973a9730e0b18bca7e6bc55b597c1fb10494949ffe8c8a07da976736af5ed`.
  Private mechanical shortcut receipt SHA-256:
  `59b96bdaf0ac575320437e7ec88fd454f26240761577d1842be7fe6c936b0432`.

The three long dossiers contain 952, 966, and 1002 visible words. Each joins
three independently signed current target fields, separately marked **VOID**
same-file historical records, and adjacent-file evidence. They use at least
three document formats. The builder verifies that changing each current target
field to a domain-valid alternative changes the answer, and that removing its
source leaves that field absent even if a historical record survives. This
resolves v7's stale-value deletion ambiguity at the mechanical level. The
choice source and displayed-option positions are both one first and two fourth
among the three non-hold items. Maximum pairwise narrative five-gram Jaccard is
0.1402. These small-sample diagnostics do not prove absence of shortcuts.

The new Boolean archive-readiness rule has a separate reference oracle.
Programmatic proofs verify the exact-case/current-status selection, admissible
world aggregation, current/archived rule divergence where planned, and target
alignment. Missing-source items have outcomes that differ across their stated
admissible completions. No formal heldout labels, model predictions, or FINAL
set were used to construct this packet.

## Independent review protocol

An independent reviewer receives **only** the frozen prompt packet and its
SHA-256. Before opening any ablations, the reviewer records a literal answer,
the evidence needed to derive it, ambiguity, answer cues, source relevance,
scenario independence, and whether each long section contributes to the
decision. The reviewer seals that report with a hash. The reviewer then opens
only the gold-free ablation packet, checks its hash, and separately records for
each of nine rows whether the original answer remains provable. The ablation
instruction explicitly permits an underdetermined judgment; these are not
forced-answer scored items. The second report is sealed before the answer key
or proof trace is opened. A separate post-key check can then compare literal
answers and source claims without changing any frozen packet or blind report.

Approval requires more than matching twelve answers: a reviewer must find
substantive long-document reasoning, causally relevant sources, distinct
scenarios, unambiguous policy application, and no practical shortcut. If any
content or key changes, it becomes a newly versioned packet with new hashes.
Until that review passes, the status is **AUTOMATED_PROOF_ONLY** and the
authored release-set gate remains closed.
