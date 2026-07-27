# Mixture-of-Models Model Format and Lifecycle

Status: design proposal for the vLLM Semantic Router project.

The machine-readable v0alpha1 fixture exercises a thin bundle and an
existing-endpoint binding. Environment discovery and full
materialized/targeted closure are explicit H2 contracts; the fixture does not
present those unimplemented paths as complete.

## 1. Positioning

A Mixture-of-Models (MoM) is a virtual composite model, not a router
configuration and not a collection of agent glue. It presents one stable model
identity and interface while its engine admits and executes a finite trace over
independently versioned constituents.

The corresponding role for vLLM Semantic Router is:

> A lifecycle and execution engine for virtual composite models.

This role has five surfaces:

1. compile authored policy into a typed, bounded MoM intermediate
   representation;
2. build, import, export, verify, and distribute an immutable model bundle;
3. resolve the model against an environment and produce a binding lock;
4. serve the resolved MoM through an ordinary model API; and
5. train, evaluate, replay, compare, and promote the same identified object.

The engine remains below agents. Agents decide what task to perform and when to
invoke a model. The MoM engine decides which model-level traces are admissible,
which physical targets may realize them, what finite resources they may use,
and how every outcome is attributed.

## 2. The model format is a typed envelope, not a new tensor format

The MoM bundle sits above existing leaf-model formats. Constituent weights,
tokenizers, adapters, and executables retain their native representations such
as safetensors, ONNX, GGUF, or an immutable OCI/Hugging Face reference. A
closed model may remain an explicitly opaque provider reference. The MoM format
defines how those constituents become one model:

- public model identity and interface;
- behavior variant, allowed request-preference domain, and hard constraints;
- typed signals, projections, policy, and finite execution graph;
- logical constituent requirements and immutable references;
- input, output, and intermediate contracts;
- resource bounds and operator-set requirements; and
- evaluation protocols, without mutable evaluation results.

This makes a bundle analogous to a checkpoint package at the composite-model
level. It is loadable and exportable as one model, but it does not pretend that
all constituent parameters can be flattened into one tensor file.

## 3. Normative concepts

| Concept | Owner | Portable | Changes logical identity | Examples |
|---|---|---:|---:|---|
| Behavior variant | model author | yes | yes | accuracy-first, cost-first, grounding-first |
| Semantic core | model author | yes | yes | typed IR, contracts, policy, bounds |
| Bundle | publisher | yes | no, unless semantic core changes | manifest, source, model card, target templates |
| Realization profile | publisher or operator | yes | no | CUDA, ROCm, CPU, edge, hybrid requirements |
| Binding | environment owner | no | no | endpoints, placement, secret references, fleet policy |
| Resolution lock | resolver | no | no | exact revisions, images, runtimes, capability evidence |
| Run record | serving environment | no | no | realized trace, usage, checks, outcomes |
| Attestation | evaluator or publisher | independently portable | no | evaluation, provenance, SBOM, signature |

### 3.1 Behavior variants are models

Names such as `vllm-sr/mom-v1-ultra`, `vllm-sr/mom-v1-light`,
`vllm-sr/mom-v1-halu`, and `vllm-sr/mom-v1-secu` identify different logical
models when their default objectives, guards, policies, or graphs differ. A
request may still carry an allowed preference override, but it must not
silently turn one published model identity into another.

Several variants may share content-addressed assets and be grouped by a
non-normative family index. The unit of serving, evaluation, and promotion is
nevertheless one versioned model coordinate plus its semantic digest.

### 3.2 Realization profiles are not models

`cuda`, `rocm`, `cpu`, `edge`, and `cloud` describe compatible physical
realizations. They do not define the model's objectives or alter its graph.
The same immutable `mom-v1-ultra` can therefore be resolved under a CUDA
binding and a ROCm binding while retaining one semantic identity.

### 3.3 Binding and locking are distinct

A binding is environment policy: it lists admissible candidates and their
declared capabilities. A lock is the resolver's completed inventory of exact
revisions and runtime evidence. A run record states which locked candidates and
trace were actually used. This separation prevents desired state, resolved
state, and observed state from being conflated.

## 4. Portable bundle boundary

A canonical directory layout is:

```text
manifest.json
mom.ir.json
source/model.mom.yaml            # optional
contracts/
policies/
assets/
models/                          # immutable constituent references
eval/protocols/                  # protocols, never results
targets/                         # optional realization templates
MODEL_CARD.md                    # optional
```

The portable bundle may contain either references or vendored objects:

- **thin**: immutable references to external model artifacts and providers;
- **materialized**: redistributable weights, tokenizers, adapters, and other
  blobs are included; and
- **targeted**: a materialized export also carries runtime artifacts selected
  for one or more realization profiles.

Manifest roles distinguish constituent artifacts, runtimes, tokenizers,
adapters, and plugins. A logical identity can point at a bundled object through
a content-addressed `bundle://` URI; thin bundles instead retain immutable
external coordinates. The same typed IR therefore governs all three closures.

Materialization must fail for a closed provider or any dependency whose license
does not permit redistribution. A thin bundle remains valid when this is
reported explicitly.

The bundle must not contain secret values, concrete production endpoints,
device identifiers, live health, replica state, private traces, or online
learning counters. Those belong to the environment.

## 5. Identity model

The format separates human coordinates from four content identities:

1. `semanticDigest` commits to canonical typed IR and every semantic asset it
   references. It answers: *which logical MoM is this?*
2. `bundleDigest` commits to the exported manifest and object descriptors. It
   answers: *which distribution package was imported?*
3. `bindingDigest` commits to canonical normalized binding IR. It answers:
   *which environment policy was applied?*
4. `lockDigest` commits to canonical normalized resolution state. It answers:
   *which exact realization was available to execution?*

Formatting a YAML binding must not change its identity. Bindings and locks are
therefore parsed, defaulted, type-checked, and hashed as canonical JSON rather
than hashed as raw source bytes. Object descriptors follow the OCI pattern of
`mediaType`, `digest`, and byte `size`.

Evaluation results, signatures, SBOMs, and build provenance are independent
attestations whose subjects include the relevant digests. They are not mutable
fields inside the model manifest. This lets a new benchmark or signature be
published without changing the model being evaluated.

### 5.1 Constituent identity strength

The resolver records how strongly each constituent is identified:

- `content-addressed`: bytes are known by digest;
- `repository-revision`: an immutable repository commit is pinned;
- `provider-version`: a provider asserts a version identifier;
- `observed-opaque`: only an alias and time-stamped probe are available; and
- `floating`: no immutable evidence, allowed only in an explicit development
  mode.

Production locking should reject `floating` references. An opaque closed model
cannot honestly receive the same reproducibility claim as a content-addressed
checkpoint.

## 6. Backend abstraction

`backend` is too overloaded to be a useful schema field. Six axes are
independent:

| Axis | Examples |
|---|---|
| Wire transport | HTTP(S), gRPC, in-process |
| API adapter | OpenAI Chat/Responses, Anthropic Messages, custom typed interface |
| Model runtime | vLLM, SGLang, llama.cpp, managed provider |
| Leaf artifact | safetensors/HF snapshot, ONNX, GGUF, provider alias |
| Accelerator | CPU, CUDA, ROCm, NPU; vendor and architecture |
| Deployment driver | existing endpoint, local container, Kubernetes, provider API |

One MoM may use a ROCm vLLM generalist, a CUDA specialist, a CPU verifier, and
a managed cloud reasoner in the same admissible trace. A root-level
`backend: rocm` would therefore be both insufficient and incorrect.

Logical model requirements state API interfaces, modalities, interface
features, context limits, data policy, and required operator effects. A
realization profile may add wire-transport, API-adapter, runtime, artifact, or
accelerator requirements. Environment policy may
further restrict candidates but may never weaken a logical hard constraint:

```text
admissible candidate
  = semantic hard constraints
  ∩ realization-profile requirements
  ∩ environment policy.
```

An environment inventory declares discoverable endpoints and deployments. The
side-effect-free binder intersects that inventory with model constraints and a
realization profile, producing the binding plus rejection explanations. The
binding records concrete endpoints, deployment references, capacity, pricing
evidence, and secret references. The lock records exact runtime image
digests, accelerator/runtime versions, model and tokenizer revisions, protocol
adapters, and capability evidence.

## 7. Build, import, export, bind, serve, and evaluate

The lifecycle distinguishes compilation from registry operations:

```text
authoring source
    └─ build ─→ typed IR ─→ immutable bundle
                              ├─ export / push
external bundle ─ import ─────┤
                              └─ bind ─→ resolve + lock ─→ serve / evaluate
                                                            └─ attest / promote
```

Proposed commands are:

```text
vllm-sr mom build model.mom.yaml --output model.mombundle
vllm-sr mom build --from-config config.yaml --output model.mombundle

vllm-sr mom import model.mombundle
vllm-sr mom inspect vllm-sr/mom-v1-ultra:1.0.0
vllm-sr mom validate vllm-sr/mom-v1-ultra:1.0.0
vllm-sr mom diff old-ref new-ref

vllm-sr mom bind vllm-sr/mom-v1-ultra:1.0.0 \
  --realization rocm-mi300x --inventory fleet.inventory.yaml \
  --output deployment.binding.yaml
vllm-sr mom resolve vllm-sr/mom-v1-ultra:1.0.0 \
  --binding deployment.binding.yaml --output resolution.lock.json

vllm-sr serve --model vllm-sr/mom-v1-ultra:1.0.0 \
  --binding deployment.binding.yaml --lock resolution.lock.json
vllm-sr mom eval vllm-sr/mom-v1-ultra:1.0.0 \
  --binding deployment.binding.yaml --lock resolution.lock.json \
  --protocol accuracy

vllm-sr mom export vllm-sr/mom-v1-ultra:1.0.0 --format directory
vllm-sr mom push vllm-sr/mom-v1-ultra:1.0.0 oci://registry.example/mom
vllm-sr mom pull oci://registry.example/mom@sha256:...
```

`fleet.inventory.yaml` names the environment discovery contract consumed by
the binder. The v0alpha1 companion schemas the binding output; a
driver-neutral inventory schema and discovery API are Milestone 3 work because
they must cover existing endpoints, managed services, and provisionable local
targets without making any one orchestrator part of model identity.

- `build` expands source defaults, compiles typed IR, computes semantic
  closure, and creates an immutable bundle.
- `import` verifies an already-built bundle and registers it in a local
  content store. It does not reinterpret authoring source.
- `export` deterministically materializes a registered bundle as a directory,
  archive, or OCI artifact and rejects embedded secrets.
- `bind` is a side-effect-free constraint-solving step over an explicit,
  environment-owned inventory; it explains accepted and rejected candidates.
- `resolve` probes allowed candidates and freezes a lock.
- production `serve` requires a lock; development mode may create a visibly
  ephemeral lock.
- `eval` consumes the same model, binding, and lock as serving and emits a
  separate attestation.

Training follows the same identity discipline. An optimizer consumes
trace-linked data and emits a complete candidate semantic core, semantic asset,
or binding. Promotion creates a new model or binding version; it never mutates
the live model invisibly.

## 8. Lowering into the current vLLM Semantic Router runtime

The implementation path should preserve the current request hot path. A new
control plane compiles the model format into the canonical configuration that
the runtime already understands:

```text
DSL / canonical YAML / future frontend
                  ↓
          portable typed MoM IR
                  ↓
      capability-checked lowering
                  ↓
 canonical v0.3 serving realization
                  ↓
       existing router hot path
```

The split already present in the public configuration is a strong starting
point:

- [`routing.modelCards`, signals, projections, and decisions](https://github.com/vllm-project/semantic-router/blob/84e7440cf523452c92cb742fca14cb50d27598d3/src/semantic-router/pkg/config/canonical_config.go#L10-L27)
  supply much of the authored logical surface;
- [`providers.models[].backend_refs`](https://github.com/vllm-project/semantic-router/blob/84e7440cf523452c92cb742fca14cb50d27598d3/src/semantic-router/pkg/config/canonical_providers.go#L3-L45)
  are an initial logical-name-to-access-target binding;
- the canonical exporter already keeps the
  [routing-owned surface separate from deployment bindings](https://github.com/vllm-project/semantic-router/blob/84e7440cf523452c92cb742fca14cb50d27598d3/src/semantic-router/pkg/config/canonical_export.go#L40-L53);
- bounded workflows already declare
  [step, parallelism, token, timeout, and quorum limits](https://github.com/vllm-project/semantic-router/blob/84e7440cf523452c92cb742fca14cb50d27598d3/src/semantic-router/pkg/config/workflows_config.go#L25-L50); and
- output contracts already represent
  [schema, extraction, normalization, repair, and post-processing](https://github.com/vllm-project/semantic-router/blob/84e7440cf523452c92cb742fca14cb50d27598d3/src/semantic-router/pkg/config/output_contract_config.go#L21-L71).

The current DSL is an authoring frontend, not yet the complete model identity.
Its route syntax does not round-trip every runtime decision field. Until the
gap closes, typed IR must remain the semantic authority and unsupported
lowering must fail closed.

### 8.1 Configuration classification

When building a bundle from today's canonical configuration, fields must be
classified rather than copied wholesale:

| Current surface | Destination |
|---|---|
| routing model cards, signals, projections, decisions | semantic core |
| logical parts of model catalog, algorithms, contracts, plugins | semantic core or digested semantic assets |
| provider/backend `type`, provider family, API dialect, external model ID, weights, auth/header policy, pricing | wire transport, API adapter, deployment driver, and binding |
| reasoning family and behavior-affecting reasoning defaults | semantic core or explicit unsupported-lowering error |
| listeners, namespaces, storage, observability | runtime environment |
| endpoints, secret values, live health, counters | excluded from bundle |
| replay events and learning session state | run/evidence store |

Unknown behavior-affecting fields must fail export rather than be silently
dropped. A successful build should also emit an explicit lowering report.

### 8.2 What the current platform flag proves—and does not prove

The current [`serve --platform amd|nvidia`](https://github.com/vllm-project/semantic-router/blob/84e7440cf523452c92cb742fca14cb50d27598d3/src/vllm-sr/cli/commands/runtime.py#L210-L217)
selects the Semantic Router ROCm or CUDA image and adjusts router-internal model
placement; the CPU path is the default rather than a `--platform cpu` value.
The [image resolver](https://github.com/vllm-project/semantic-router/blob/84e7440cf523452c92cb742fca14cb50d27598d3/src/vllm-sr/cli/container_images.py#L43-L49)
does not provision constituent LLMs. Current provider bindings assume their
endpoints already exist. Constituent-level ROCm/CUDA/CPU deployment therefore
belongs to new deployment-driver and resolver work, not to a reinterpretation
of the existing flag.

### 8.3 Driver seam

Resolution and provisioning should be separate interfaces:

```text
TargetResolver
  probe(environment) -> typed capabilities
  resolve(requirement, profile, policy) -> ranked candidates + explanation

DeploymentDriver
  plan(candidate) -> immutable deployment plan
  materialize(plan) -> external resource references
  observe(plan) -> endpoint and revision evidence
```

The first implementation should support existing endpoints only, because that
matches current behavior. CUDA/ROCm vLLM deployment drivers, Kubernetes, edge,
NPU, and managed-provider drivers can then be added without changing the model
format.

## 9. Serving and evaluation share one identity

The standard `model` field selects a versioned MoM coordinate. To the caller,
this is indistinguishable from invoking an ordinary model: one request enters
and one response returns. Internally, the response and trace carry the four
digests plus the selected behavior variant and constituent revisions.

An evaluation protocol lives in the bundle because it defines how the model is
to be assessed. An evaluation result does not. A result attestation references:

```text
semanticDigest
bundleDigest
bindingDigest
lockDigest
protocolDigest
datasetDigest
harnessDigest
sampling and concurrency settings
metrics and confidence intervals
raw-result descriptor
```

This makes offline evaluation, online replay, and production serving comparable
without conflating the benchmark with the model. It also supports promotion
rules over accuracy, cost, latency, grounding, security, and sovereignty.

## 10. Distribution and format lineage

The format is transport-neutral: a canonical directory and deterministic
archive are sufficient for local exchange. OCI is the preferred registry
transport because descriptors provide media type, content digest, and size;
manifests provide a content-addressed package; indexes can group exported
variants; and referrers can attach signatures, SBOMs, provenance, and
evaluation attestations. CUDA and ROCm must remain MoM realization metadata,
not be misrepresented as standard OCI CPU-platform fields.

Other precedents inform specific boundaries:

- Hugging Face contributes the save/load experience, repository coordinate,
  immutable revision pinning, and separation of config, tokenizer, and weights.
- ONNX separates a typed, versioned model IR from capability-negotiated
  execution providers. MoM applies the same logical/physical boundary at the
  model-call level rather than the tensor-node level.
- MLflow contributes model signatures and a common identity across loading,
  serving, and evaluation, while opaque executable flavors are unsuitable as
  the MoM semantic authority.
- GGUF is useful as a self-contained edge constituent, not as a universal
  representation for remote and closed models.
- KServe demonstrates declarative serving graphs and runtime/resource
  separation, but its deployment resources do not define a portable MoM model
  identity.

## 11. Security and supply chain

- Import verifies normalized paths, descriptors, sizes, and digests before
  parsing objects; archives also require limits on expansion, files, links, and
  nesting.
- Source and model cards are never executable authority.
- Arbitrary Python, pickle, and implicit remote code are disabled by default.
- Extension operators declare a versioned interface, effects, sandbox needs,
  and content identity; deployment trust policy must approve them.
- Secret values are forbidden from bundles and locks; only secret references
  may appear in environment-owned bindings.
- Signatures prove publisher identity only when verified against an explicit
  trust policy; content digests alone prove integrity, not authorship.
- Sovereignty, security, and grounding constraints are fail-closed predicates
  over the complete trace, including every fallback and boundary crossing.

## 12. Terminology migration

The repository historically uses “MoM Model Family” and `MoMRegistry` for the
router's auxiliary embedding, classifier, safety, and grounding models. Those
objects are constituents of the engine, not the virtual composite model defined
here. They should migrate to “Semantic Router system models” and
`RouterModelRegistry`, with a deprecated compatibility alias for existing
configuration. `Mixture-of-Models` should thereafter name only the complete
versioned model exposed to callers.

## 13. H2 roadmap

### Milestone 1 — Format alpha

1. Publish an ADR for Bundle, IR, Behavior Variant, Request Preference,
   Realization Profile, Environment Inventory, Binding, Lock, Run Record, and
   Attestation.
2. Define language-neutral schemas, media types, canonicalization, negative
   fixtures, thin/materialized closure, and a deterministic directory/archive.
3. Rename the legacy auxiliary-model terminology with migration support.

### Milestone 2 — Build, import, and distribution

4. Implement a semantic-closure compiler from canonical v0.3 and the DSL;
   unknown semantics fail closed and every lowering decision is reported.
5. Add a local content store and immutable name/version/tag-to-digest registry.
6. Add build/import/export/inspect/diff plus OCI pull/push and referrer support.

### Milestone 3 — Binding, locking, and serving

7. Define engine and target capability descriptors with required/preferred
   negotiation and diagnostic explanations.
8. Implement canonical inventories, bindings, and locks with an
   existing-endpoint driver.
9. Lower a locked MoM into canonical v0.3 and expose its versioned coordinate
   through the standard model API.
10. Add vLLM CUDA, vLLM ROCm, and CPU deployment drivers, followed by edge,
    NPU, and managed-provider integrations.

### Milestone 4 — Evaluation and lifecycle optimization

11. Propagate all identities into response metadata, trace, replay, and
    learning records.
12. Make system evaluation consume a bundle and lock and emit a digest-bound
    attestation.
13. Add candidate promotion, rollback, and cross-environment conformance.

The H2 exit criterion is deliberately concrete:

> The same immutable MoM bundle can be imported, served, and evaluated once
> under a CUDA binding and once under a ROCm binding. The two runs retain the
> same semantic and bundle identities, receive different binding and lock
> identities, and make every response, trace, and evaluation attestation
> attributable to all four.
