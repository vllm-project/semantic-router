# TD050: Benchmark Adapter Execution Attestation Gap

## Status

Open

## Owner Plan

[PL-0039: Evaluation Plane](../plans/pl-0039-evaluation-plane.md)

## Release Relevance

Exact source pins, safe normalized imports, first-party parsers, deterministic
replay, and live re-execution can ship as explicitly bounded evidence. An
import must remain exploratory and must not claim native benchmark or
leaderboard parity until its source-to-row transformation and native grader
have been independently attested.

## Scope

- native-to-normalized benchmark adapters;
- adapter binary, configuration, and transformation provenance;
- native metric and grader parity;
- redistribution, unsafe serialization, and hidden-label boundaries.

## Summary

The research inventory covers thirteen benchmark designs. Eleven pins expose a
safe data surface handled by maintained normalizers; RouteJudge/ORBIT and
RouterEval remain diagnostic-only because their pinned repositories do not
provide the required safe per-case export. Installed bundles are schema,
digest, identity, coverage, and visible/grading-boundary checked, but the
system does not execute arbitrary upstream code or prove that an upstream
generator or native scoring command produced caller-supplied bytes.

The current E0 and narrow live-reexecution labels are intentional and must not
be raised merely because a checkout and normalized artifact both validate.

## Evidence

- The plan and public benchmark atlas declare thirteen research descriptors,
  eleven executable safe normalizers, and two diagnostic-only sources.
- Suite installation validates exact pins, parser outputs, schemas, joins, and
  digests, but does not execute repository-owned native graders.
- Public reports explicitly keep normalized imports at E0 and disclaim native
  leaderboard reproduction.

## Why It Matters

Without adapter and native-metric attestation, a valid normalized bundle can
still be unrelated to the verified checkout or omit native slices, graders,
hidden labels, cascades, or dynamic agent execution. Treating it as leaderboard
parity would turn provenance plumbing into a false scientific claim.

## Desired End State

Every supportable source has a maintained, no-network adapter whose exact
binary, inputs, outputs, and native-parity result are sealed together. Sources
that cannot be handled safely remain explicit product limitations rather than
silently downgraded or executed as third-party code.

## Exit Criteria

- Maintain one versioned, sandboxed adapter for every benchmark whose license
  and safe export surface permit execution.
- Bind exact source and dataset pins, adapter image or binary, configuration,
  output objects, row counts, and transformation receipt in one independently
  anchored record.
- Compare normalized reducers with native benchmark results on maintained
  golden subsets and publish explicit tolerances.
- Keep licensing, hidden-label, unsafe-serialization, and unavailable-export
  blockers distinct from implementation readiness.
- Advertise native-parity discovery only for adapters whose source derivation,
  grader parity, and required artifacts all passed.
