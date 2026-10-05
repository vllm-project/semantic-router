# Pilot v0.1 — preliminary paired review

Local offline review, 2026-09-30, against integrated branch commit `60b841f31f445aa4c340ae8f1cab8b696bb658bd`. All three arms are present locally. This is inspection of captured evidence, not independent model inference, a jointly approved report or an adoption decision.

## Pinned evidence

- Jev: [captured records](jev-results.jsonl), SHA-256
  `45a176c8b1e4524197601b0d7868bdeb44d568fe2b3d3732aa0192d46c19014d`.
- Kai: [submission](https://github.com/subin9/semantic-router/tree/43ea6097d5e05a0f14ada9d0face2fdc5614d0bc/bench/jev/pilot-v0.1/kai),
  commit `43ea6097d5e05a0f14ada9d0face2fdc5614d0bc`.
- Vela: [supplementary PR](https://github.com/yuki-uix/semantic-router/pull/1),
  reviewed head `0132c13c6c1d75cbcef405f912dbf22f78bba237`.

## Results in these development cases

| Case | Reference | Jev | Kai | Vela |
| --- | --- | --- | --- | --- |
| pilot-001 | biology | biology | other | biology |
| pilot-002 | computer science | computer science | other | computer science |
| pilot-003 | math | math | other | math |
| pilot-004 | history | history | other | history |
| pilot-005 | health | health | health | biology |
| pilot-006 | diagnostic only | other | other | computer science |
| Matching scored references | 5 cases | 5/5 | 1/5 | 4/5 |
| Recorded contract-valid distributions | 6 cases | 6/6 | 6/6 | 6/6 |

The review independently recomputed exact 14-label membership, finite [0,1] bounds, probability sums, top-1 consistency and reference-label scoring for all 18 records. Each arm has six unique IDs in shared-input order, matching references and matching input/question hashes. Every record shows one attempt; Jev and Kai show HTTP 200. Case 006 has no correctness score. No discrepancy was found in those checks.

Jev and Kai requests match the shared text/question and contain no expected-label field. Response model IDs match requests. Kai's probabilities, choice and confidence match its raw HTTP responses. Vela's adapted inputs preserve the six texts and labels; every original raw-record field agrees with the enriched record, allowing the documented `case_id` to `id` rename. Its mapping/protocol hashes and runner-source hash match bundled files. Its ten-file checksum manifest passes. Jev's saved responses were also previously revalidated by the offline Go evidence test.

Maximum absolute probability-sum deviations are 0 for Jev, approximately 9.20e-8 for Kai, and 8.83e-8 for Vela, all within the 0.001 contract tolerance. No renormalization was performed. These checks used local JSON parsing, exact comparisons and SHA-256; they establish artifact consistency, not proof of execution or independently verified model provenance.

Kai's four misses all select `other`; its explanation is not established. Do not
rewrite candidate descriptions using these results and call the rerun held-out.
Vela's health/biology error shows that a contract-valid high probability is not a correctness guarantee: it assigns biology 0.8153 and the reference health 0.1412. Kai's diagnostic case assigns other 0.2904 and physics 0.2894, not an exact tie. Kai's separately recorded `confidence` is a top-two margin, not its top-1 probability. Case 006 is not scored for any arm.

## Execution differences to retain

| Condition | Jev | Kai | Vela |
| --- | --- | --- | --- |
| Interface | Hosted HTTP | Local runtime HTTP; entrypoint in a host Python 3.12.3 venv, not the `vllm-sr decision serve` container image | Local Transformers |
| Candidate descriptions | Supplied | Supplied | Not accepted by fixed head |
| Warmup | None | One separate non-pilot input | One call using first pilot input |
| Timeout | 10 seconds | 60 seconds | No enforced timeout reported |
| Failure policy | Stop and record remaining cases | Continue and record | Continue and record |
| Timing | Serialization + HTTP + validation | HTTP + JSON parse, excludes validation | Model forward; excludes tokenization and softmax |
| Observed range (ms) | 635.019–2695.384 | 204.171–208.365 | 14.835–17.581 (`latency_model_ms`) |
| Hardware | Hosted hardware unknown | Ryzen 7 H 255 CPU, float32, 8 threads | Apple M5 CPU; dtype/thread count unrecorded |

All measured cases succeeded at the call/contract level, so the different stop
policies were not exercised. These timings must not be ranked as equivalent
end-to-end measurements. Six cases do not support stable latency-tail or general
quality claims. Kai also reports one separate post-run sanity request, excluded
from pilot records. Shared metadata still says the protocol is not formally frozen.

## Reproduction status and corrections

Kai transport-read follow-up (2026-10-05): [subin's supplied commit](https://github.com/subin9/semantic-router/commit/f19e4c31a193770597a9b2847ae158ddc1222c8f) addresses the maintainer's remaining truncated-response finding. A local HTTP server regression verifies that truncated successful responses retain partial bytes, dropped connections produce failed records, and subsequent inputs still complete. The combined Kai/Vela suite now has 15 passing test methods; six report tests also pass. These are offline runner checks, not new model measurements. Historical captured records remain unchanged; maintainer re-review is still required.

Vela review fix (2026-10-02): [Xunzhuo's label-index finding](https://github.com/vllm-project/semantic-router/pull/4336#discussion_r4164707960) is addressed by lyy's supplementary PR #3 plus integration hardening for empty metadata and non-integer index types. The runner now checks complete index coverage and exact semantic label identity, retaining complete generic `LABEL_n` compatibility. Eleven offline regression tests pass. This improves future-run preflight validation; it does not retroactively prove the checkpoint configuration of the captured run. The historical records and 4/5 score are unchanged.

Kai review fix (2026-10-03): [subin's supplied commit](https://github.com/subin9/semantic-router/commit/3bf6b49a7b9774ea9aed930186ab92fcafd4ca6b) addresses [Xunzhuo's failure-preservation finding](https://github.com/vllm-project/semantic-router/pull/4336#discussion_r4164707953). Malformed response shapes and invalid probabilities are recorded with raw responses and contract errors; subsequent cases continue, and top-1 is computed only for valid distributions. Three offline test methods cover the malformed responses and a complete six-case simulated run. The combined Kai/Vela suite has 14 passing test methods. Historical captured records remain unchanged. Both runner review items have implementations and offline regressions ready for maintainer re-review; this does not imply maintainer approval or a research adoption decision.

Source-check follow-up (2026-10-01): Black 25.1.0 reformatted the Kai/Vela Python scripts and Jev report utilities. Ruff cleanup names existing constants and uses list unpacking in Kai, sorts Vela's standard-library imports, and documents intentional Chinese punctuation in the Jev report. The preparation/report utilities retain identical parsed ASTs; Vela's AST differs only in import order. Kai's saved-response and synthetic-failure checks agree with the original validator and request runner. The source-hash checks described above apply to the original snapshot at `efb0e7143fa52f715b3de31f84625c2031ceb6b2`; Kai/Vela run notes distinguish historical source hashes from current hashes. The Vela packaging manifest was refreshed, while historical run metadata and captured evidence remain unchanged. No experiment was rerun.

Kai's contribution was cherry-picked with authorship retained, and Vela supplementary PR #1 was merged. Their local records are [Kai](kai/kai-results.jsonl) and [Vela](vela/vela-results.jsonl), with reproduction instructions in [Kai run notes](kai/kai-run.md) and [Vela run notes](vela/run.md).

In the [2026-09-30 contributor review](https://github.com/vllm-project/semantic-router/pull/4336#issuecomment-5905046979), subin confirmed the Kai predictions, score, timing range, maximum sum deviation, pins, execution row and integrated file identity. The requested environment caveat is now explicit in the table: the runtime entrypoint ran in a host Python 3.12.3 virtual environment, not a container, because the source build had no default image. This matches the existing Kai run notes; no records or measurements changed. This acknowledgement covers the Kai evidence, not approval of the entire research conclusion.

Vela's missing baseline and mapping were supplied from the contributor's pinned commit `53526a3edfa7afaae6760670a68033692cc112cc`. Mapping bytes match the captured hash; baseline model/revision matches metadata. The original run did not record its YAML hash, so original YAML byte identity is unverified. The bundled mapping-only command passed without inference. This resolves the missing-file packaging gap, not independent full-run reproduction.

This review corrected two actual documentation errors in Vela's `run.md`: case 004's top-1 probability is 0.9678, and case 005's is 0.8153, not 0.9998 for both. Labels and 4/5 scoring were already correct. Captured records remain unchanged; the documentation checksum was updated.

Vela's saved P50 15.952 ms and P95 17.582 ms match the third and sixth sorted `latency_wall_ms` observations, consistent with nearest-rank percentiles. The ordinary median is 16.141 ms. The summary-generation method is not supplied in the runner. Both timing fields bracket the forward pass, not end-to-end inference. The original summary is preserved. Metadata also says “unrecorded warmup” despite the separate warmup capture; this historical wording is noted rather than silently changing metadata.

In the [Vela contributor review](https://github.com/vllm-project/semantic-router/pull/4336#issuecomment-5907015346), lyy confirmed the predictions, corrected probabilities, six valid distributions, 4/5 score and nearest-rank formula `ceil(q*n)`. The reported ordinary median is 16.141479 ms. The contributor also confirmed that enrichment/summary generation was a separate step not included in the runner, and checked the saved artifacts offline; that reproduction limitation remains. No inference rerun was needed.

On 2026-10-01, the unsigned Vela commit `0132c13c6c1d75cbcef405f912dbf22f78bba237` was replaced with the author's signed commit `384c7de11f9a93cdc3b78017746cbeca691dac9d`. Their Git trees are identical. Only the three descendant commits were rebuilt with new parent references, retaining their trees, authors and sign-offs. The repaired tip before this documentation update has the exact same tree as `026b1a5f322b15132202ed6632d935608de8d3c5`; earlier commit IDs in this report remain historical evidence references.

Checksums of the primary record files:

- Jev: `45a176c8b1e4524197601b0d7868bdeb44d568fe2b3d3732aa0192d46c19014d`.
- Kai: `889215a12db029a86aca66749b2a39418d914e118971e9c637eac17b4e924d63`.
- Vela: `96bf657ac18a59120922a6d2768eec790e8d47fefef5f9e8013df0455338e6a0`.

Recorded model pins are Jev `jev-1.13.0`; Kai `Decision-1.0-Kai-0.6B` revision `9d6872cde6950c2c2b5786d182ec9a06ca1bdd66`, runtime `58cd660b51ba19aff01ad6f89e66f2d07f13908b`; Vela `Vela-1.0-Encoder-307M-Domain` revision `f6354f54adcf38770f635ad903be2b00577f6c11`. This review did not reload any model to independently verify those declarations.

## What this pilot cannot establish

- General accuracy or calibration: only five reference classes out of 14, one example per class. Candidate descriptions were supplied to Jev/Kai but not the fixed-head Vela model.
- Injection resistance or abstention quality: case 006 is unscored, and the supplied `other` description explicitly mentions classifier-manipulation instructions.
- Production reliability: six valid Jev calls do not resolve the separately reported probability-sum failures. Those used a different setup and must not be pooled into this denominator or dismissed.
- Comparable speed/cost: clocks, hardware, network and warmup differ. Jev usage totals 4,164 input and 739 output tokens, but monetary cost is unknown.
- Fully reproduced execution: no models were rerun. Jev's exact process finish time/exit code is unavailable; Vela's region `local` is not a geographic location. Vela enrichment and summary generation are not bundled as a reproducible step.

The shared protocol's not-published/not-frozen fields are a hashed historical snapshot, not a live PR status tracker. Do not retroactively claim a pre-run freeze. Preserve these artifacts and distinguish this local synthesis from contributor approval.

## Next responsibilities

Proposed handoff, not new assignments:

| Who | Bounded next action | Done when |
| --- | --- | --- |
| Yuki | Synthesis and documentation corrections submitted; PR description updated | Initial handoff complete |
| subin | Kai evidence checked; requested host-venv caveat incorporated | Contributor acknowledgement linked above |
| lyy | Vela evidence checked; nearest-rank convention confirmed; signed replacement supplied | Contributor acknowledgement linked above; enrichment script remains unbundled |
| Yuki with contributors | Triage the separate Jev contract-failure evidence, then propose adopt/defer/reject against issue criteria | Linked evidence and bounded recommendation |
| Maintainer | Review recommendation and sufficient closure evidence | Explicit decision or specific remaining work |

Local evidence collection and cross-checking are complete, and both contributors have checked their respective arms. The research recommendation and maintainer decision remain open; these acknowledgements do not constitute approval of the entire conclusion. Do not automatically expand into a larger benchmark or production adapter.

No external comments, merges, pushes, paid calls or model downloads were made
as part of this review.
