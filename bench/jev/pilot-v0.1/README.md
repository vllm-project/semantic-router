# Shared pilot v0.1 — handoff package

Prepared locally for [research #3970](https://github.com/vllm-project/semantic-router/issues/3970).
Not published; do not interpret this package as a new collaborator approval.
This consolidates the confirmed task definition, with case 006 diagnostic-only.
The original rc1 files and original run remain unchanged.

## What to run

- [inputs.jsonl](inputs.jsonl): six English inputs, five scored and one diagnostic.
- [question.json](question.json): exact Choice instructions and 14 candidate descriptions.
- [protocol.json](protocol.json): label order, model/runtime pins, owners and hashes.
- Jev/Kai receive descriptions; Vela uses its fixed-label interface and only receives text.
- Keep expected labels out of requests. Preserve text, label meanings and source hashes
  when adapting field names. Jev's question key is `intent`.
- Sequential execution, no automatic retries. Record separate warmups and timeout.
- Compare native top-1 before thresholds/fallback. Preserve ties for review.
- Validate exact labels, finite probabilities in [0,1], sum within 0.001 of one.
  No renormalization. Provider confidence stays separate.
- Do not score case 006; preserve its prediction and complete probabilities.

Input and question bytes are identical to rc1. No new paid run is needed merely
because the package has a new name. Jev's existing run is supplied for paired review,
not relabeled as having run after a public protocol freeze.

## Failure handling: disclose, do not silently harmonize

A common stop/continue policy has not been confirmed. The Jev runner stops after
the first call or contract failure; this run had none. Each owner must state their
actual stop policy and preserve failed/unexecuted cases, with reasons and attempts.
Do not rerun away failures or count them as `other`. If an arm stops, review that
difference before claiming a complete paired pilot. This package does not adopt
the earlier unconfirmed continue-after-failure proposal.

## Handoff

| Owner | Next deliverable | Current evidence |
| --- | --- | --- |
| yuki-uix — Jev | Publish this package; assemble paired report | Six live records supplied here |
| subin9 — Kai | Run pinned model/runtime on these files; supply records and metadata | Configuration confirmed; no paired run supplied here |
| lyy26299 — Vela | Run pinned baseline on these inputs; supply records and metadata | Configuration confirmed; no paired run supplied here |
| All three | Review own rows and explain mismatches | Pending |

Native result schemas are welcome. Include case ID, native prediction, unchanged
full probabilities, contract result, errors, attempts and timing; identify raw HTTP
bodies versus SDK-parsed outputs. Include model/runner revisions, fixture/config
hashes, hardware, coarse region, timing boundary, timeout, retries and warmups.
Supply actual files and exact source, not hashes alone. Do not include credentials.

## Jev evidence and reproduction

[jev-results.jsonl](jev-results.jsonl) is an exact copy of the original run.
[jev-run.md](jev-run.md) describes provenance, limits and source reconstruction.
The reviewable runner and tests are in the parent directory. A local source
archive preserves the run's historical snapshot, but is ignored by Git and is
not a required checkout artifact. See the source provenance in `jev-run.md`.

From the repository root, inspect the saved results without API calls:

```bash
python3 bench/jev/report.py bench/jev/pilot-v0.1/jev-results.jsonl --mode live
```

The viewer trusts recorded contract outcomes; it is not a full contract revalidator.
Run `make test-jev-eval` to execute `TestPilotV01CapturedEvidence`, which checks
the captured file hashes, six-case coverage, exact requests, response validation
and scoring with no API calls. This verifies saved evidence, not service behavior today.

## Contributing through the research PR

Use this PR branch as the reference checkout. Add Kai/Vela artifacts under
`pilot-v0.1/kai/` or `pilot-v0.1/vela/`, with a run note and original JSONL.
These are proposed contribution locations, not new agreements on runner schemas.
Keep the shared inputs and question unchanged; discuss/version any correction.
Include reproduction commands and source revision for each arm.

With explicitly granted fork write access, push a normal commit to the shared
research branch; do not force-push collaborators' history. Otherwise send a
supplementary PR targeting that branch or share a commit for integration.
Neither route requires merging the research PR into upstream first.

## Completion boundary

**Pilot done:** all three arms have comparable records; missing/failed cases and
prediction/contract differences are reviewed. Perfect answers are not required.

**Research done:** a reproducible evaluation and documented mapping/error/abstention
semantics support an explicit adopt/defer/reject recommendation accepted by maintainers.
The six development cases do not establish general quality, calibration, P95 or cost.
A defer/reject is valid, but it must identify supporting evidence and uncompleted
evaluation requirements; maintainers must accept any reduced closure scope.

After the pilot, use the paired findings and reviewed supplementary contract failures
to decide the smallest remaining evaluation needed. No production adapter is required
to complete this research; #4311 is separate implementation work.
