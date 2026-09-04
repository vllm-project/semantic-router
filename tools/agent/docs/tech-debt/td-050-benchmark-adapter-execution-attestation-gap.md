# TD050: Benchmark Adapter Execution Attestation Gap

## Status

Open

## Owner Plan

[PL-0039: Evaluation Plane](../plans/pl-0039-evaluation-plane.md)

## Release Relevance

The exact-pin registry, source verifier, normalized suite contract, private
store, and data-only replay executor can ship. Imported suites remain E0 and
cannot substantiate an upstream or promotion claim until this gap is closed.

## Scope

- native-to-normalized adapters for the registered benchmark repositories
- adapter binary and configuration provenance
- native metric/grader parity
- redistribution and hidden-label handling

## Summary

The Evaluation Plane verifies all registered source and dataset revisions and
validates an operator-supplied normalized bundle. It does not yet execute a
repository-owned normalizer or cryptographically bind every normalized row to
the verified checkout. A caller could supply a schema-valid bundle unrelated to
that checkout, so source verification alone is not adapter attestation.

## Evidence

- `suite-install` reruns the system source verifier and ignores a caller-supplied
  receipt.
- The suite store validates content hashes, strict schemas, case joins,
  permissions, and immutable IDs.
- No adapter executable digest, transformation receipt, native grader parity
  result, or source-to-row derivation is currently required.

## Why It Matters

Without transformation attestation, an imported suite can prove contract
plumbing but not that RouterArena, ORBIT, CodeRouterBench, or another registered
benchmark was faithfully reproduced. Raising its evidence level would turn a
clean Git pin into a false scientific claim.

## Desired End State

Each benchmark has a maintained, sandboxed adapter that reads only its exact
pinned checkout, emits the normalized IR and a transformation receipt, and
passes parity tests against the benchmark's native splits, graders, and metric
reducers. Licensing and hidden labels remain separate from public artifacts.

## Exit Criteria

- Implement one versioned adapter package for every registered benchmark and
  dataset pin.
- Run adapters in a no-network, read-only-source sandbox with fixed dependencies,
  seed, and resource limits.
- Bind source, dataset, adapter image/binary, configuration, output objects, and
  row counts in one signed or independently anchored receipt.
- Compare normalized metrics with native benchmark outputs on maintained golden
  subsets and define accepted tolerances.
- Add Dashboard/operator suite discovery only for fully installed, attested
  suites; never execute upstream source from a browser request.
- Raise evidence above E0 only for the claims whose required native artifacts
  and parity checks passed.
