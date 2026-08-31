# TD054: Typed Request Capability Eligibility Gap

## Status

Open.

## Owner Plan

[PL-0040: MoM Routing Hardening](../plans/pl-0040-mom-routing-hardening.md)

## Release Relevance

Built-in model cards declare tool, vision, structured-output, and long-context
traits, but runtime eligibility currently enforces only known context-window
capacity. Operators can therefore assign a model whose metadata is sufficient
for ranking but whose capabilities do not satisfy a request contract.

## Scope

- `src/semantic-router/pkg/services/classification_signal_types.go`
- `src/semantic-router/pkg/config/model_config_types.go`
- `src/semantic-router/pkg/selection/`
- `src/semantic-router/pkg/extproc/eval_model_selection.go`
- built-in model and assignment validation

## Summary

Request shape and protocol controls can imply hard capabilities such as tool
calling, image input, or structured output. Those requirements need a typed,
provider-neutral request envelope and one shared eligibility function. Ranking
algorithms must receive only eligible candidates and must not restore a known
ineligible model when the filtered set is empty.

## Evidence

- Request context is estimated from the complete prompt envelope and known
  insufficient candidates are removed before selection.
- `ModelParams.Capabilities` and catalog traits describe additional
  capabilities, but no equivalent request-to-model eligibility contract uses
  them for tools, images, or structured output.
- Unknown context metadata remains eligible for compatibility; capability
  metadata needs an equally explicit compatibility policy before enforcement.

## Why It Matters

Quality, latency, cost, load, session continuity, and learning are soft ranking
objectives. None can compensate for a backend that cannot honor the request.
Sending a known-incompatible request produces avoidable backend failures and
can contaminate learning feedback.

## Desired End State

The Router derives a content-free typed capability requirement from each
request, filters known-incompatible candidates once, and passes the same
eligible set to static, telemetry-aware, Looper, session-aware, Eval, and
learning paths. Compatibility behavior for missing metadata is explicit and
observable.

## Exit Criteria

- Define provider-neutral request requirements for context, tools, image input,
  and structured output.
- Normalize model capability metadata and validate assignment contracts.
- Apply one eligibility function before every selection algorithm, including
  session and learning overrides.
- Reject when all candidates are known incompatible; never fail open to the
  original pool.
- Add protocol-parity, unknown-metadata, mixed-pool, and all-ineligible tests
  plus built-in live probes.
