---
sidebar_position: 5
description: Configure native Router model providers and understand their ownership and task limits.
---

# Native Backends

Semantic Router uses native providers for learned classifiers, token spans,
embeddings and other model tasks. A deployment selects the provider, device,
precision and input budget; its recipe binding selects the task contract and
adapter. See [Configure models](model-configuration#configure-models-used-by-router-tasks)
for the canonical YAML fields.

## Provider availability

`candle` and `ort` identify separate native providers. They can coexist in one
Router process when the build includes both bindings and their runtime
libraries. `http` connects to a named external service. A deployment cannot
gain a provider by changing its YAML alone: preparation must find a compatible
provider, adapter, model artifact and device in the running build.

| Deployment | Execution selection | Ownership |
| --- | --- | --- |
| `candle` | Local artifact, explicit device and precision | Router owns the native instance. |
| `ort` | Local ONNX artifact, execution provider and precision | Router owns the native session. |
| `http` | Named external model and task connector | Router owns calls and connections; the service owns its models. |

Available device names and accepted configuration are not a hardware support
report. Use results for the exact model revision, task, provider, device and
precision you deploy. A successful build or provider-registration message
does not demonstrate GPU execution. Unsupported combinations fail preparation;
the Router must not silently fall back to another device.

Native bindings require their corresponding native libraries. A non-CGo or
stub build does not perform local model inference. Keep using
`vllm-sr serve` with the matching local image when validating deployment
behavior, and verify real routed requests after startup.

## Task and input limits

A provider advertises capabilities for the actual task and adapter. Support
for embeddings does not establish support for a sequence classifier, token
classifier, NLI task or generative guard using the same model family.

The maintained mmBERT classification and token paths enforce a **512-token
limit**. Deployment input budgets can restrict this further. They do not
enable 32K classification because a checkpoint or embedding path carries a
32K name. An overflow policy must be supported by the adapter; token results
preserve source offsets and report truncation.

Task results retain their semantics: complete label probabilities when
available, raw scores in their own units and direction, source-aligned spans,
and embedding metadata. A generated label or span without a score is reported
with confidence unavailable.

## Instance lifecycle

Local deployments use owned instance handles. Preparation resolves recipe
bindings, checks compatibility and loads candidate resources before publishing
a generation. A failed candidate leaves the current generation available.
Requests that already acquired the old generation finish before its resources
are released.

Resource sharing requires provider compatibility and matching execution
identity; an equal path alone is insufficient. Closing one binding does not
close a shared resource still used by another binding. Shared instances use a
single total admission budget.

Cancellation may stop a caller waiting before a native call itself finishes.
The runtime retains ownership and admission capacity until that native call
actually returns. Remote service models are not unloaded by a Router reload.

When an instance change cannot safely run alongside the current generation,
preparation reports the failure or `restart-required`. Treat that as a
deployment operation and inspect the candidate error before replacing the
working configuration.

## Existing task support

Bindings fail during preparation when the requested task has no adapter. HTTP
adapters currently serve domain, PII, complexity, prompt guard, embeddings and
endpoint hallucination inputs. Fact check, feedback, modality and local NLI do
not acquire new HTTP adapters through a deployment declaration. ORT grounding
and NLI adapters are unavailable. A supported task contract does not imply that
every model architecture or execution device implements it.

Generic rules bind with `classifier.<rule name>` inside their own recipe. The
entire rule name is preserved, including dots. `local` and `sequence_classifier`
rules can select a local sequence deployment or an HTTP `http_classify`
deployment with `label_distribution.v1`. An `llm` rule retains its scored JSON
extraction instructions and requires HTTP `http_chat` with that same distribution
contract; it does not accept an unscored categorical chat result. The rule's
ordered `labels` remain its mapping, so `mapping_path` is rejected. Explicit
bindings replace obsolete `model` or `model_path` selectors; those selectors may
be omitted when a binding supplies them. Multiple local rules own independent
handles within one recipe.

The combined classification API evaluates each input through the prepared
recipe's actual intent, PII and security tasks. It does not imply a joint model
forward. The former Traditional unified initializer only produced aggregate
placeholder results and now returns an explicit capability error; use real
recipe bindings. Maintained merged LoRA classifiers retain separate task
execution and resource ownership. Adapter-only weights cannot stand in for a
merged checkpoint.

PII outside labels are removed by the PII consumer. Missing model scores remain
unavailable and cannot be used for a confidence threshold. The merged LoRA
security adapter recognizes `safe`, `benign` and `no_threat` as explicit negative
labels; `unsafe` is a positive verdict rather than matching a `safe` substring.

## Artifact preparation

Model preparation follows the default public API consumers and reachable recipes.
An explicit binding replaces that consumer's default artifact. Unused deployment
entries and embedding catalog paths do not trigger downloads. Separate Candle
head directories and mapping files are included when their paths have a
`mom_registry` entry; ORT heads identify complete graph files, relative to the
artifact directory unless absolute. Candle head paths identify directories
relative to the router working directory unless absolute.

`mom_registry` controls automatic Hugging Face provisioning. Custom mounted
artifact, head, and mapping paths do not require registry entries: their actual
provider checks files, shapes, labels, and capabilities before activation.

Native sequence and PII bindings verify the consumer's complete label order
against the loaded model. A reversed sidecar or an unknown PII label rejects
the candidate before activation; the current generation keeps serving. A model
whose entire vocabulary is the indexed `LABEL_0`, `LABEL_1`, ... convention uses
the explicitly configured index mapping. Semantic labels are never silently
renamed; existing task-specific aliases, such as feedback `SAT`/`satisfied`,
remain valid only for that task.
For a registered deployment, `revision` selects the Hugging Face snapshot.
A complete immutable commit snapshot whose standard HF cache metadata matches
is reusable offline. Populated directories with a different or unverifiable
revision are rejected before download; choose a separate empty directory.
Reload also refuses to resynchronize a directory used by the live generation.
This preserves models still serving requests while older generations drain.
Use separate local directories for simultaneous revisions. A head or mapping
from a different registered repository uses that repository's default revision;
the deployment revision pins its own artifact repository.

A process can need Candle weights and ORT graphs from the same registered
snapshot. Preparation retains both formats in that case and checks the external
tensor locations declared by ONNX graphs instead of assuming every graph uses
a `.data` file. Selected MMBERT embedding layers must actually be present;
classification remains limited to 512 tokens.

## Runtime diagnostics

Prepared bindings report `resolved`, `ready`, `failed` and `closed` lifecycle
events. Debug call events identify the recipe, binding, deployment, result
contract and adapter, along with the effective provider, device and precision.
A preparation failure does not invent effective execution facts.

Input diagnostics distinguish architectural capacity, the task limit and the
deployment budget. Actual tokenizer counts and truncation are included when
provided by the adapter; absence means unknown. Admission metrics use the
resolved deployment name, and invalid input or a foreign-recipe lookup does not
count as admitted model work. Binding diagnostics omit prompts, result content,
credentials and raw provider error bodies.

Intent and decision responses report `confidence_available: false` and
`confidence: null` when a rule or error policy provides a verdict without a
measured classifier score. Unavailable probabilities are omitted with
`probabilities_available: false`; `signal_error_matches` identifies matches
selected by an error policy. The Dashboard preserves these values and displays
“Score unavailable,” including score-free chat guard decisions.

Fact-check and feedback diagnostic responses also report `confidence: null`
with `confidence_available: false` when no score was produced. An empty-input
default preserves its verdict and reports `policy_default: empty_text`; its
confidence must not be treated as a measured probability.

Knowledge-base management writes also prepare a complete router generation.
A persisted candidate returns HTTP 202 with `activation_status: pending` and
`generated_runtime_hash` until that generation is published. Poll
`/api/v1/config/hash` for its `active_runtime_hash`; a failed candidate leaves
the existing router and classification APIs on their previous generation.
Standalone API servers close only the services they created, after accepted
requests finish; they do not close services borrowed from the router.
