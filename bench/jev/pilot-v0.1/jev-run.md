# Jev run supplied for pilot review

- Origin: user-executed rc1 local exploration on 2026-09-28, not a post-freeze run.
- Model requested/returned: `jev-1.13.0`.
- Source base: `2a95e988db9bd3f42838ee08fbc3092a8aba69d4`, with local changes.
- Exact revision field: `2a95e988db9bd3f42838ee08fbc3092a8aba69d4-dirty-rc1-local-5XrYUo`.
- Client build: Darwin arm64, Go 1.27.1. CPU model not recorded.
- Coarse location: China, operator-declared; provider hardware unknown.
- Sequential, no warmup, no automatic retries, timeout 10 seconds, six-case limit.
- Fail-fast after a call/contract failure; no such failure occurred.
- Timing includes request serialization, HTTP call and validation, excludes result
  writes; provider-only inference time unavailable.
- First start: 2026-09-28T09:39:34.128913Z.
- Last case start: 2026-09-28T09:39:39.550934Z.
- Exact process finish time and exit code not recorded.

## Result

Six records, six attempts, six HTTP 200 responses and six valid distributions.
Five scored cases match their references. Diagnostic case 006 returns
`other=0.97, physics=0.03`; correctness is omitted, not false.
Usage totals: 4,164 input and 739 output tokens. Actual cost unknown.
This small development run neither establishes overall accuracy nor contradicts
the probability-sum failures reported on other inputs/configurations.

## Artifact SHA-256

| File | SHA-256 |
| --- | --- |
| inputs.jsonl | 671c09f62dc9fc9b864efe54b0adfef0ec666f309f74b776dcec3d6d8cdd2ef6 |
| question.json | 33f7ef7c4826df98a4bc23c2e0fc2302ea3ab61e18b8b45d0515e51e64f69738 |
| jev-results.jsonl | 45a176c8b1e4524197601b0d7868bdeb44d568fe2b3d3732aa0192d46c19014d |
| source.tar.gz | b1a55bbfe7f617111b78c9105115d8603063426e42c312ee4fc2fdc2da46209c |

The local, Git-ignored source archive includes the run's Go harness/tests, report tool, rc1 JSON
fixtures, Go module files and Make registration. Overlay it on a separate checkout
of the base commit; it is not the whole repository. Review the archive listing first.
Build with `make build-jev-eval`; offline checks are `make test-jev-eval` and
`make vet-jev-eval` (loopback permissions and initial dependency access needed).
For PR review, use the Go source files in `bench/jev/` directly. The archive is
historical provenance, not a required file in a PR checkout. The current runner
implementation was compared byte-for-byte with the archived run sources during
local PR preparation; subsequent documentation/evidence tests are not run-time changes.
No executable or credential is distributed in the PR.
Live reproduction requires separate authorization and a key; do not rerun just to
rename the protocol. Remote responses and timings are not guaranteed deterministic.
