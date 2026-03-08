# TD006: Structural Rule Target Still Exceeds Reality in Key Legacy Hotspots

## Status

Open

## Scope

architecture ratchet versus current code

## Summary

The harness correctly ratchets the repo toward smaller modules, but several legacy hotspots still depend on explicit exceptions.

## Evidence

- [docs/agent/architecture-guardrails.md](../architecture-guardrails.md)
- [docs/agent/repo-map.md](../repo-map.md)
- [tools/agent/structure-rules.yaml](../../../tools/agent/structure-rules.yaml)
- [tools/agent/scripts/agent_doc_validation.py](../../../tools/agent/scripts/agent_doc_validation.py)

## Why It Matters

- The harness correctly states that large hotspot files are debt, not precedent, but several code areas still depend on hotspot-specific exceptions and ratchets.
- The harness-side validation layer still includes at least one oversized script that remains above the warning threshold even after related changes land.
- This is the right governance posture, but it remains a real code/spec gap until the worst hotspots no longer need special handling.

## Desired End State

- The global structure rules become the common case rather than something many hotspot directories can only approach gradually.

## Exit Criteria

- The highest-risk files no longer need special ratchet treatment to stay within the intended modularity envelope.
