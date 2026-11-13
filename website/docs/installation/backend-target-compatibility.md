---
title: Backend Target Compatibility
description: See which backend target forms survive Docker, Helm, Operator, Dashboard, and recipe workflows.
---

# Backend Target Compatibility

Semantic Router uses `providers.models[].backend_refs[]` as the canonical
contract between a logical model name and one or more physical inference
targets.

This matrix covers **configuration production and preservation**. It does not
claim that an endpoint is reachable, healthy, or compatible with a particular
model protocol.

## Choose the right matrix

| Question | Source of truth |
| --- | --- |
| Which client endpoints and backend wire protocols work together? | [Protocol Compatibility](protocol-compatibility) |
| Which target fields survive CLI, Helm, Operator, Dashboard, and recipe workflows? | This page |
| Which deployment stacks and integrations does the project maintain? | [Deployment Support](support-matrix) |

The same canonical document can move through the CLI, Helm, and Dashboard. The
Operator instead accepts Kubernetes discovery inputs and translates its
supported subset into canonical backend references.

## Status meanings

- **Supported**: the surface accepts and preserves the canonical target form.
- **Adapter**: the surface accepts a narrower native input and generates a
  canonical backend reference.
- **Partial**: the form is accepted with the limitation stated below.
- **Not expressible**: the producer cannot represent that target form.

## Compatibility matrix

<!-- BEGIN BACKEND TARGET COMPATIBILITY MATRIX -->

| Target form | Canonical YAML | Docker / CLI | Helm | Operator | Dashboard | Maintained recipes |
| --- | --- | --- | --- | --- | --- | --- |
| Direct `endpoint` as `host[:port]` | Supported | Supported | Supported | Adapter | Supported | Supported |
| HTTP(S) `base_url`, including a path | Supported | Supported | Supported | Not expressible | Supported | Supported |
| Multiple weighted refs with shared route metadata | Supported | Supported | Supported | Adapter | Supported | Supported |
| Provider, API-version, path, and header metadata | Supported | Supported | Supported | Not expressible | Supported | Supported |
| Kubernetes Service DNS target | Supported | Supported | Supported | Adapter | Supported | Supported |
| KServe discovery | Not expressible | Not expressible | Not expressible | Partial | Not expressible | Not expressible |
| Label-selected Service discovery | Not expressible | Not expressible | Not expressible | Adapter | Not expressible | Not expressible |
| Different paths or request headers per weighted ref | Supported | Partial | Supported | Not expressible | Supported | Partial |

<!-- END BACKEND TARGET COMPATIBILITY MATRIX -->

The URL row is the full-URL target form: write it as a full URL in
`endpoint` ([Migrate model resources](model-resource-migration.md#request-backends)
shows the earlier spelling).

With `--gateway extproc`, the Docker / CLI path generates one Envoy route for
a logical model. Route-level headers, Host rewriting, paths, and TLS behavior
must remain compatible across its weighted refs. The default standalone
frontend builds upstream pools directly from the canonical references. In
both modes, use separate model aliases for different upstream contracts; see
[Custom models](custom-models#use-replicas-only-for-the-same-upstream-contract). Endpoint or replica selection
using live inference telemetry is a separate data-plane contract tracked in
[#2332](https://github.com/vllm-project/semantic-router/issues/2332).

## Portable target forms

Use `endpoint` for a direct host and optional port:

```yaml
providers:
  models:
    - name: local/general
      provider_model_id: Qwen/Qwen3-8B
      api_format: openai
      backend_refs:
        - name: primary
          endpoint: model-server.default.svc.cluster.local:8000
          protocol: http
          provider: vllm
          weight: 100
```

Write a full URL in `endpoint` when the upstream identity includes a scheme or
path. Keep credentials in environment references rather than committed YAML:

```yaml
providers:
  models:
    - name: hosted/reasoning
      provider_model_id: provider/model-id
      api_format: openai
      backend_refs:
        - name: hosted-primary
          endpoint: https://provider.example.com/v1
          provider: openai
          auth: {header: Authorization, prefix: Bearer, api_key_env: PROVIDER_API_KEY}
          headers:
            X-Tenant: production
          weight: 100
```

An `endpoint` of `host[:port]` takes its scheme from `protocol`; a full URL
carries its scheme and path, and its path is the complete API root. `auth`,
`headers` and `tls` have the same shape as a deployment's
[connection block](model-catalog.md#connection-block); a backend reads its key
from `auth.api_key_env`, and is always verified (see
[Trust a private CA](#trust-a-private-ca)).

`api_format` belongs to the model, not to an individual backend ref. It selects
the backend request and response codec; `protocol` on a backend ref selects the
HTTP transport. See [Protocol Compatibility](protocol-compatibility) before
choosing `api_format`.

## Trust a private CA

An https backend is verified against the system roots, under its endpoint's
host. A backend whose certificate comes from a private CA needs that CA named
in its `tls` block, or once for every https backend in
`providers.defaults.tls`:

```yaml
providers:
  defaults:
    tls:
      ca_file: /etc/vllm-sr/certs/example-corp-ca.pem
  models:
    - name: private-llm
      provider_model_id: private-llm
      backend_refs:
        - name: internal
          endpoint: https://llm.internal.example/v1
          provider: openai-compatible
          tls:
            ca_file: /etc/vllm-sr/certs/example-corp-ca.pem
            server_name: llm.internal.example
```

| Field | Meaning |
| --- | --- |
| `ca_file` | The PEM bundle this backend's certificate must chain to. It replaces the system roots for this backend, as Envoy's `trusted_ca` does. |
| `server_name` | The name sent as SNI and that the certificate must carry. Default: the endpoint's host. |
| `cert_file`, `key_file` | A client certificate for a backend that asks for one. Set both. |
| `providers.defaults.tls.ca_file` | The `ca_file` of every https backend whose own `tls` names none. |

The generated Envoy cluster and the Router's own calls to the backend use the
same settings, so both trust the same CAs. The Router's own calls are the
fallback attempt, the shadow copy, the context-window probe, Looper and
request-graph hops in either mode and, in standalone mode, every request.

- **Put the intermediate in the bundle.** A server that sends only its leaf
  certificate verifies only when the bundle holds the issuing intermediate
  and the root. With the root alone the handshake fails: OpenSSL reports
  `unable to verify the first certificate` (code 21), Go
  `certificate signed by unknown authority`. To check a bundle:

  ```bash
  openssl s_client -connect llm.internal.example:443 \
    -servername llm.internal.example \
    -CAfile /etc/vllm-sr/certs/example-corp-ca.pem </dev/null | grep 'Verify return code'
  ```

  `Verify return code: 0 (ok)` means the bundle is complete.
- **A CA file replaces the system roots.** A backend that names a `ca_file`
  no longer trusts the system roots. The default replaces them for every
  https backend without its own `ca_file`, public providers included. To keep
  trusting those, put the system bundle and your CA in one file:
  `cat /etc/ssl/certs/ca-certificates.crt example-corp-ca.pem > /etc/vllm-sr/certs/ca-bundle.pem`.
- **Validation.** Paths are absolute. `tls` on a plain-HTTP backend, a
  certificate without its key and `insecure_skip_verify` are rejected: a chat
  backend is always verified. A file that is present must be readable PEM. A
  file that is absent where the configuration is checked, as in the
  Dashboard's container, fails where it is read: when Envoy starts, when the
  standalone Router loads the configuration, or at the Router's call.
- **One `tls` per pool.** The backend refs of one model alias share one
  Envoy cluster, so they must share their `tls` settings.
- **Replacing a file.** Envoy reads the files when it starts, and the Router
  when it loads the configuration. Restart the stack after you replace a file
  in place.

### Mount the files

- **`vllm-sr serve`** mounts each file read-only at the same path into the
  Router container and, with `--gateway extproc`, the Envoy container. It
  refuses to start when a file is missing. Keep the bundle in its own file,
  such as under `/etc/vllm-sr/certs/`, rather than pointing at the host's
  system bundle: on SELinux hosts the mount relabels the file. Run
  `vllm-sr serve` again after you add a file.
- **Helm.** Put the bundle in a Secret or ConfigMap and mount it with
  `extraVolumes` and `extraVolumeMounts` at the path `ca_file` names:

  ```yaml
  extraVolumes:
    - name: backend-ca
      secret:
        secretName: example-corp-ca
  extraVolumeMounts:
    - name: backend-ca
      mountPath: /etc/vllm-sr/certs
      readOnly: true
  ```

  With `gateway.mode: extproc`, the gateway you run verifies the backend
  itself, through its own backend TLS settings; the Router's `tls` covers
  the Router's own calls.
- **Operator.** The `SemanticRouter` resource has no field for extra mounts
  yet. Deploy with Helm to reach a backend behind a private CA.

## Producer behavior

| Producer | Behavior and boundary |
| --- | --- |
| Docker / local CLI | Standalone builds upstream pools from canonical refs; `--gateway extproc` renders Envoy clusters and routes. Host, port, protocol, weight, compatible paths, authorization, and shared headers describe existing model services, not workloads to provision. |
| Helm | `configOverride` renders one complete canonical mapping without merging sample provider defaults. Reachability and model compatibility remain runtime checks. |
| Operator | `spec.vllmEndpoints[]` is a Kubernetes discovery adapter, not a copy of the full provider schema. It emits the supported canonical subset described below. |
| Dashboard | Reads and writes the supported canonical backend inventory, including provider identity, URL, auth metadata, API version, chat path, headers, and environment-key references. |

The Operator currently discovers:

- a named Kubernetes Service;
- a KServe `InferenceService`; and
- a label-selected Llama Stack Service.

Those adapters generate backend name, endpoint, protocol, and weight. Put a
complete external-provider target in canonical configuration supplied through
Helm or another canonical-config workflow. Broader CRD and Helm parity belongs
to [#2355](https://github.com/vllm-project/semantic-router/issues/2355).

KServe discovery currently assumes the conventional
`<InferenceService>-predictor.<namespace>.svc.cluster.local:8443` HTTPS target.
It does not resolve `status.url` or inspect the generated Service, so custom or
named-predictor service layouts require an explicit Service backend instead.

Unknown-field and supported-version parity across producers is tracked in
[#2469](https://github.com/vllm-project/semantic-router/issues/2469). Until that
work lands, do not rely on an unknown extension surviving a move between
producers.

## Validation boundary

The repository exercises this matrix at five layers:

- CLI tests validate direct targets, URL paths, TLS, weights, and headers in
  generated Envoy configuration.
- Helm validation renders a complete canonical override and checks that its
  backend fields survive without sample-default leakage.
- Operator tests validate Service discovery and the generated canonical ref.
- Dashboard frontend and backend tests validate the supported field inventory
  and persisted canonical output.
- The maintained-config contract parses its enumerated asset inventory and
  every recipe. The separate reference-config contract validates the shipped
  reference configuration, which exercises weighted direct and rich URL
  targets.

These checks prove configuration translation and preservation. They do not
replace a direct backend protocol check or a live request through the Router.
