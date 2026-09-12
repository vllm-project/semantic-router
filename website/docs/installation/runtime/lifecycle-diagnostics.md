---
title: Lifecycle and Diagnostics
description: Prepare, admit, inspect, reload, and close Router model generations safely.
---

# Lifecycle and diagnostics

The Router publishes a configuration only after its required models and
consumers prepare successfully. Requests keep the generation they acquired;
a failed candidate leaves the previous generation available.

## Preparation and resource ownership

Preparation resolves default API consumers and reachable recipes, projects
each recipe's explicit model bindings, provisions their required artifacts,
then opens typed task handles. It validates actual architecture, graph/head,
label order, input limits, dimensions, layers, and modalities. Unreachable
module defaults are not substitute models for an overridden consumer.

Resource sharing follows actual execution identity. Compatible Candle modern
heads can share their encoder; independent ORT graphs are complete resources.
The pool owns shared native resources. Bindings own their task handles and
HTTP connectors; external model processes remain owned by their services.
Remote bindings share an admission reservation for the same operation, not
necessarily one connector object. Consumers own or borrow their task views. A final resource close waits for admitted native work to finish.
Cancellation can end the caller's wait without pretending a synchronous native
kernel has stopped or unloading the memory it still uses.

## Bound capacity

Admission can be keyed by a named deployment or an existing system model key:

```yaml
global:
  model_catalog:
    admission:
      local-pii:
        max_concurrency: 2
        max_queue: 8
        queue_timeout_ms: 1000
        on_overflow: shed
```

Declare `local-pii` as a deployment as shown in
[Models and bindings](models-and-bindings.md#declare-one-local-deployment).
Omitting admission retains the existing ungated behavior. `shed` rejects excess
work; `wait` waits for queue capacity and requires a nonzero queue bound;
`fail_open` bypasses the gate when full. Use the policy that matches the
consumer's failure handling. Shared physical resources also share capacity;
renaming a deployment does not create another copy of an HTTP endpoint.

A deadline covers queue wait and execution as one operation. Queue expiration,
request cancellation, unavailable capabilities, invalid outputs, and native
errors remain typed failures. They do not become successful zero scores.

## Reload and rollback

A reload prepares the candidate before switching the current generation.
Existing requests drain on their prior generation; new requests acquire the
published one. A failed model load, label mismatch, unsupported layer/device,
or capability error rolls back the candidate's owned resources.

Keep pinned model revisions in distinct artifact directories. Standard cache
metadata must agree with the requested revision before a populated directory
can be reused. A reload cannot overwrite files still used by an active or
retired generation. See [Artifact provisioning](models-and-bindings.md#provision-model-files).

Management updates can be **saved but pending** while preparation runs. KB
writes return `202` when that whole-generation publication is pending; poll
`GET /api/router/api/v1/config/hash` through the Dashboard proxy and compare the exact
`generated_runtime_hash` from the mutation with `active_runtime_hash` before reporting activation. A second mutation while the source is
pending returns `409` without replacing the first candidate.

KB document updates write a new asset version so the old generation can still
read its complete labels/documents. Deletion removes the candidate's config
reference rather than deleting a live generation's asset. Old asset versions
are currently retained; this change does not introduce automatic storage GC.
The [API contract](../../api/apiserver.md) defines the mutation responses.

## Read scores honestly

Distributions retain every declared label and their original score meaning.
Token spans retain actual scores and Unicode offsets. A categorical guard
verdict, an error-policy match, a disabled model, or an empty-input policy
default does not acquire a confidence merely because an API has such a field.

Intent/decision diagnostics use nullable confidence and explicit availability;
unavailable probability maps are omitted. The intent API's `probabilities`
is currently a signal summary containing only the selected category and its
score, not the full model vector. Validate complete distributions through the
typed model result. Replay carries the corresponding
availability flags. Error-policy matches remain visible through
`signal_error_matches`. Display “score unavailable” rather than replacing
`null` with zero or one. Fact-check/feedback empty-text policy defaults are
identified separately by `policy_default: empty_text`.

## Validate a deployment

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml --image-pull-policy never
curl -fsS http://localhost:8080/startup-status
```

Validation checks the canonical declaration. Preparation proves that the
configured image and files can produce the required task handles. Diagnostic
requests prove actual execution for those inputs. For GPU claims, retain the
execution-provider profile and verify that the required work did not fall back
to CPU. Keep model/runtime revisions alongside output tolerances and examples.

Test model changes with normal and boundary inputs, Unicode spans, unavailable
scores, malformed remote responses, admission saturation, and reload while
requests are active. Compare complete distributions or vectors, not only a
top label. Persistent indexes also need compatibility checks after an embedding
change.

## Native regression checks

From a configured source checkout:

```bash
make test-rust-ci
make test-owned-native CI=true
```

`test-rust-ci` includes the owned Candle instance tests and the early-layer
normalization tensor check. `test-owned-native` builds Candle and ORT and runs
real tiny ONNX instance tests with Go's race detector plus native classification
assembly tests for mapping rollback, two local rules, and projected startup.
It also checks the ROCm image default with implicit ORT and explicit Candle
forwards in one process, plus the matching model-download format contract.
Set `ORT_DYLIB_PATH` to an installed compatible runtime; otherwise the harness
installs the official CPU `onnxruntime==1.22.0` wheel in its environment.

The regular `make test` CI path includes these checks, and
`test-binding-minimal` includes the owned-native binding tests. Tiny fixtures
verify ownership and contracts; they do not replace maintained-checkpoint
parity, GPU execution profiles, or sustained local `vllm-sr serve` reload tests.
