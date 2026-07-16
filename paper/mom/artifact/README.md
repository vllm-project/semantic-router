# Proposed Mixture-of-Models Model Format

This directory is the machine-readable companion to the paper. The format is
v0alpha1: it is a design proposal, not an input format accepted by a current
vLLM Semantic Router release.

The bundle is intended to be the model artifact of a MoM. It fixes one virtual
model's identity, interface, constituent references, behavior variant, typed
execution semantics, contracts, and bounds. CUDA, ROCm, CPU, edge, and cloud
are realization choices for that model. Concrete endpoints, devices, and
images are environment-owned. A binding may contain secret references; secret
values remain in an external secret store and never enter the bundle or lock.

The design rationale, current-code alignment, lifecycle, driver boundary, and
H2 implementation roadmap are in [DESIGN.md](DESIGN.md).

## Schemas

- manifest.schema.json: portable bundle manifest and OCI-style descriptors;
- ir.schema.json: canonical semantic authority and bounded core operators;
- realization-profile.schema.json: endpoint-free runtime and accelerator
  compatibility templates;
- binding.schema.json: concrete environment candidates and policy;
- lock.schema.json: complete resolved candidate inventory;
- eval-protocol.schema.json: evaluation procedure stored with the model;
- eval-attestation.schema.json: external result linked to model and
  realization identities; and
- run-record.schema.json: one attributable realized trace.

The command design also names a future environment-inventory contract as bind
input. The v0alpha1 companion schemas the binding output; inventory discovery
is tracked as H2 resolver work rather than pretending that one deployment
system is already universal.

The example intentionally uses a small core operator set. A production format
would extend the opset and schemas to cover the complete decision, workflow,
fusion, plugin, and contract surface supported by the engine.

## Complete thin example

example/manifest.json defines the portable boundary. It references:

- mom.ir.json: canonical typed IR and sole semantic authority;
- source/mom-v1-ultra.mom.yaml: optional human-oriented source;
- assets/ and contracts/: content-addressed semantic objects;
- eval/protocols/accuracy.yaml: an evaluation protocol, not a result; and
- targets/hybrid-rocm-cloud.yaml and
  targets/hybrid-cuda-cloud.yaml: two realization templates for the same
model.

This fixture demonstrates a thin bundle resolved to existing endpoints. The
manifest schema reserves constituent, runtime, tokenizer, adapter, and plugin
roles, but inventory discovery and fully materialized/targeted closure checks
remain normative H2 contracts rather than simulated implementations.

The following colocated files are environment or evidence records and are not
members of the portable bundle:

- deployment.binding.yaml: a concrete ROCm-plus-cloud binding;
- resolution.lock.json: the complete resolved candidate inventory;
- resolution/: pinned and observed-opaque revision evidence;
- evidence/run-record.json: one identity-linked trace fixture;
- evidence/eval-attestation.json: a schema-linkage fixture, explicitly not a
  quality, cost, latency, or portability result; and
- evidence/build-provenance.json: external provenance.

Endpoints use the reserved .invalid domain. Model names, revisions, image
digests, capacities, prices, and metrics are structural fixture values, not
measurements, provider quotes, or released artifacts.

## Behavior variants and realization profiles

The example publishes vllm-sr/mom-v1-ultra:1.0.0 with an accuracy-first
behavior variant. Behavior affects policy and therefore semantic identity. A
different published objective such as cost-first, grounding-first, or
security-first should receive its own model identity or version.

The two hybrid-*-cloud files are realization profiles. They constrain the local
constituent to vLLM plus ROCm/AMD or CUDA/NVIDIA while allowing the reasoner to
remain a managed API. They contain no endpoint or secret and do not change
semantic identity. The binding selects one realization profile and supplies the
environment-specific candidates.

## Identity model

The format separates four content identities:

1. semanticDigest is SHA-256 over canonical JSON of the typed IR. Every
   semantic asset is referenced by digest from that IR.
2. bundleDigest is SHA-256 over canonical JSON of the manifest. The manifest
   contains mediaType, digest, and byte size for every distributed object.
3. bindingDigest is SHA-256 over canonical normalized binding IR, not raw YAML
   bytes.
4. lockDigest is SHA-256 over canonical normalized lock IR.

Formatting a source or binding cannot therefore change semantic or environment
identity. The v0alpha1 verifier is a Python reference prototype; the H2 format
ADR must fix cross-language number, Unicode, duplicate-key, defaulting, and
array-order rules before this canonicalization becomes a language-neutral
contract. No object contains its own digest, so the dependency order remains
acyclic: semantic objects, then manifest, then binding, then lock, then
evidence.

The fixture identities are:

    semanticDigest sha256:7896fd6d7c496c8afaf32907503d67f536fa12271a398020c342107fb3252c70
    bundleDigest   sha256:b9cde7d73f2591ab41b04a2652330ba540f0a47ac1726d3208f30a27eab61862
    bindingDigest  sha256:6b0c802858ff24889819da299b765e992487d6b69980d694eef6611ae6686265
    lockDigest     sha256:24f99ae82680efd39f1f6bff8d4bb6590307b624b6e720b6e6b7602d254ac579

Evaluation results, signatures, SBOMs, and provenance are independent
attestations whose subjects include these identities. Publishing new evidence
does not redefine the model.

## Build, import, and export

The lifecycle distinguishes source compilation from registry operations:

    source --build--> bundle --export/push--> distribution
                             <--import/pull--
    bundle + inventory --bind--> binding --resolve--> lock
    bundle + binding + lock --serve/eval--> trace and attestations

- build compiles source to typed IR and produces a bundle.
- import verifies an already-built artifact and stores it by digest; it never
  recompiles optional source.
- export deterministically materializes a thin, materialized, or targeted
  package and rejects secret values.
- bind intersects model constraints and a realization profile with an explicit
  environment inventory and emits an explained environment binding.
- resolve checks semantic requirements, realization-profile requirements, and
  environment policy before writing a lock.
- serving and evaluation consume the same model, binding, and lock identities.

## Verify

The verifier requires Python 3, jsonschema, and PyYAML:

    python -m pip install -r requirements.txt
    python verify.py
    python test_negative.py

It validates all eight schemas; normalized and traversal-safe object paths;
descriptor sizes and digests; semantic, bundle, binding, and lock identities;
manifest/IR closure; behavior variants and realization profiles; graph
reachability, universal termination, and explicit terminal paths; fail-closed hard constraints; constituent identity and
capability compatibility; complete locked candidate inventory; pinned versus
observed-opaque evidence; secret non-inclusion; and evidence linkage.
The negative suite confirms rejection of cyclic paths, missing entry targets,
partial locks, incompatible runtimes and images, unsupported opsets, and
out-of-domain request preferences.

A production importer must additionally enforce archive expansion limits,
hardlink and device-file rejection, license policy, operator trust, signature
verification, and cross-engine trace conformance.
