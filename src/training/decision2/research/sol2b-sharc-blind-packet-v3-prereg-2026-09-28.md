# Sol 2B ShARC: prospective answer-blind source review v3

**Status: CPU source review only.** The [v2 inventory](sol2b-sharc-human-policy-source-v2-result-2026-09-28.md) found 589 possible opposite-label pairs in publisher TRAIN. It admitted zero training rows. This protocol freezes a *new* 24-pair semantic review packet before any further source inspection. It does not amend v1/v2, use protected evaluation gold, authorize GPU training or assert model improvement.

## Fixed input and selection

- Read only `sharc1-official/json/sharc_train.json` from the publisher archive with SHA-256 `72dca3f4f3ba73b1d796b40e952a80d53cd2011ef90b2168b8bcaa818f5edd1e` and TRAIN-member SHA-256 `d37d349758a69644fbf49827b1a0893f6099752b0ccc9c7e29bd315e5228429c`. Enumerating archive member names is permitted; opening DEV/TEST is not.
- Read both publisher exclusion lists, `sharc_negative_question_utterance_ids.txt` and `sharc_negative_scenario_utterance_ids.txt`; exclude every matching TRAIN utterance before pairing. Freeze their byte hashes at `4e185fb8fc82bc750597812b0458e0dc4288b37f54966ae5c00af9b27553b5e1` and `177b73465fbd241dd233a676e989cef97d5971d96bc1d5df3cc0cdda815e5220`. Reject a missing, duplicate or malformed list instead of silently retaining negatives.
- Validate identities and visible fields. Group within original `tree_id` by exact trimmed source URL and the [v2 normalization](sol2b-sharc-human-policy-source-v2-prereg-2026-09-28.md) of snippet and question. Eligible rows have exact publisher `Yes`/`No`, different utterance IDs and different visible scenario/history. Select the minimum SHA-256-ranked candidate pair per tree, then the minimum-ranked tree per source URL, and finally the minimum-ranked 24 distinct source URLs. The ranking salt is the fixed literal `decision2-sharc-blind-v3`; no text or model outcome enters ranking. If fewer than 24 URLs remain, stop without a packet.
- The packet orders states A/B by a deterministic, balanced 12/12 hidden-label assignment. The reviewer sees only the rule snippet, question, scenario and full visible follow-up history for each state. No publisher `answer`, `evidence`, utterance ID, source URL, tree ID, negative-list membership or alternative target field enters the review packet. Exact publisher source URL and tree/utterance IDs, plus exact and normalized snippet hashes for later source-level exclusion, stay in a separate private mapping without labels. Both private files use restrictive file permissions and cannot be committed or copied into a public gist.

## Blind review and decision boundary

An independent reviewer must answer for each pair, before seeing publisher targets:

1. Is the policy rule sufficiently clear for both states? Write the rule condition or exception that controls each state.
2. Does the **difference in visible scenario/history**, applied to the same rule and question, actually require opposite Yes/No outcomes? Mark both inferred outcomes and the minimal decisive evidence in each state. “Unclear” and missing facts are not Yes/No.
3. Can either answer be inferred without reading the rule (question wording, option position, stereotyped scenario, history token or source template)? Is the pair a negation/paraphrase shortcut rather than a policy-state contrast?
4. Are the two states semantically distinct and mutually consistent? Flag stale rules, missing context, conflicting history, ambiguity, or a fact outside the snippet.

The pair passes only if both reviewers independently mark the rule relevant, evidence necessary, outcomes opposite and unambiguous; a disagreement is adjudicated blind by a third reviewer. The previously frozen source-review floor remains **at least 18 of 24** correct, evidence-necessary pairs when publisher targets are unblinded by a separate adjudicator. The review result is a feasibility signal, not training admission.

Before any Sol 2B matched-budget training arm, additionally audit exact, near and semantic overlap by source URL, tree and snippet across TRAIN/SELECT/CAL and protected panels; verify rights/attribution, native token lengths, balanced option ordering, a state-removed shortcut control and zero-step source parity. Any failed gate remains HOLD. Preserve rejected pairs and reasons privately. Do not use these 24 pairs as a new model selection benchmark.

The publisher's [data schema](https://sharc-data.github.io/data.html) makes `snippet`, `question`, `scenario` and `history` input fields, while `answer` and `evidence` are targets. Its publisher negative-ID files are part of this archive, so exclusion is a required source-level check rather than a discretionary filter.
