# Security Hardening Guide

This document describes the RBAC-to-router integration and production-default
Envoy configuration shipped with the semantic router to close trust-boundary
gaps between the dashboard and the routing plane.

## Overview

The semantic router sits between external clients and upstream LLM backends.
It processes requests through Envoy's ext_proc filter and makes routing
decisions based on signal classification. This guide covers:

1. **Production-default Envoy config** — defense-in-depth header stripping at
   the proxy layer.
2. **Looper reentry authentication** — protecting recursive model calls and
   configuring trusted multi-replica deployments.
3. **Dashboard password lifecycle** — responding to compromised-credential
   warnings without weakening password-manager interoperability.
4. **Dashboard RBAC integration** — translating dashboard role/permission
   changes into router `role_bindings` and `ratelimit` rules.
5. **New dashboard permissions** — fine-grained RBAC permissions for
   feedback, replay, and security policy management.

## Architecture

```
 Client → Envoy → ext_proc (authenticate) → route header mutation → upstream LLM
                    │
 Dashboard ────────►│ (config fragment sync)
                    │
                    ├── Security Policy API
                    ├── Role-to-Model mappings
                    └── Rate-limit tier mappings
```

## 1. Production-Default Envoy Config

The Envoy template removes known internal headers during route processing
before upstream forwarding. This is defense in depth against metadata leakage
to model backends; it is not the Looper trust decision. The ext_proc filter can
observe the original request headers and must authenticate or reject them:

```yaml
request_headers_to_remove:
  - "x-vsr-looper-request"
  - "x-vsr-looper-secret"
  - "x-vsr-looper-decision"
  - "x-vsr-looper-iteration"
  - "x-vsr-fusion-depth"
  - "x-authz-user-id"
  - "x-authz-user-groups"
```

This is configured in both:

- `deploy/local/envoy.yaml` (local development)
- `src/vllm-sr/cli/templates/envoy.template.yaml` (production template)

Model-routing headers have a stricter availability boundary. Envoy route
matches consume `x-selected-model`, so every checked-in standalone, KServe,
OpenShift, Istio, Operator-generated, and CLI-generated ExtProc configuration
is explicitly fail closed (`failure_mode_allow: false`). If the router is
unavailable, Envoy returns an error and does not continue to a route selected
by client-supplied metadata. This trades availability during an ExtProc outage
for routing integrity; operators must restore the router rather than bypass it.

At the request-header boundary, the router treats every case variant and every
duplicate occurrence of `x-selected-model` as untrusted client input. It keeps
the value out of generic request context, removes all occurrences from Envoy's
live request, and invalidates any route Envoy may already have selected from
the forged value. After model selection, the router writes its trusted value
and unconditionally invalidates the route cache again so header-based routing
uses that value. `global.router.clear_route_cache` controls only optional
auxiliary mutations such as tool-selection body rewrites; setting it to
`false` cannot disable either selected-model provenance transition.

Do not treat virtual-host `request_headers_to_remove` as a pre-routing
sanitizer. Those removals protect the upstream boundary, but route selection
may already have observed a header. A deployment that intentionally changes
the fail-closed default needs a separate, proven pre-routing trust mechanism;
simply stripping `x-selected-model` at the virtual host is not sufficient.

Agentgateway uses its own `AgentgatewayPolicy` CRD instead of a raw Envoy
`ExternalProcessor` message. Its current ExtProc policy does not expose a
failure-mode override and is fail closed by contract, so the checked-in policy
documents that default rather than adding an unsupported field.

## 2. Looper Reentry Authentication

Looper algorithms make recursive OpenAI-compatible requests back through the
router. The originating router adds a runtime-owned 256-bit credential, and
the receiving ext_proc instance validates that credential before trusting any
Looper decision, iteration, or fusion metadata. Invalid internal metadata is
rejected with a generic `403` response before routing or plugin processing.

The entire `x-vsr-looper-*` namespace is reserved for this internal protocol;
`x-vsr-fusion-depth` is reserved as well. External clients and configured
Looper headers must not set these names. Unknown names under the reserved
prefix, duplicate metadata, partial credentials, and malformed iteration or
fusion-depth values all fail closed. Internal metadata is excluded from trace
extraction and generic request context, then removed before upstream dispatch.

### Single-Replica Default

When `VLLM_SR_LOOPER_SHARED_SECRET` is unset, the router creates a fresh random
credential at startup and retains it across configuration reloads. This default
is appropriate only when recursive Looper traffic is guaranteed to return to
the same router process, such as a single-replica local deployment.

### Multi-Replica Deployment

Deployments where Looper reentry can reach a different replica must set the
same `VLLM_SR_LOOPER_SHARED_SECRET` on every router replica. The value must be
exactly 64 hexadecimal characters (32 bytes), for example a value generated
with `openssl rand -hex 32`. A present but empty, malformed, or incorrectly
sized value causes router startup to fail instead of silently using a different
credential.

Keep this value in the deployment secret store, not in router YAML. With the
Helm chart, reference an existing Kubernetes Secret through `extraEnv`:

```yaml
extraEnv:
  - name: VLLM_SR_LOOPER_SHARED_SECRET
    valueFrom:
      secretKeyRef:
        name: semantic-router-internal-auth
        key: looper-shared-secret
```

For `vllm-sr serve --target k8s`, exporting the variable is optional. The CLI
generates one release-scoped 256-bit key when it is absent, stores it only in an
immutable Kubernetes Secret submitted through standard input, and reuses that
key from the current CLI-managed Secret on later deploys. Every router replica
therefore receives the same key, and CLI-managed revisions use `Recreate` so
different generations do not overlap. Export an explicit 64-character value
to rotate it. For the direct container target, export the variable in the
invoking environment; the CLI passes it only to the router and masks its value
in logs.

The configured Looper endpoint is a privileged trust boundary: it receives
model request content, forwarded model credentials, and the internal reentry
credential. Use only an operator-controlled internal endpoint, restrict it with
network policy, and use transport encryption when traffic crosses a host or
node boundary. The Looper client treats the configured URL as exact and does
not follow redirects. Its private pooled HTTP transport also ignores ambient
`HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY` settings so request content,
provider authorization, and the internal credential cannot be diverted to a
process-wide proxy. Do not point Looper at an untrusted or third-party
endpoint.

The shared value is a bearer credential for the lifetime of the process or
deployment generation; it is not a per-request proof. A party that can read the
router environment or observe plaintext east-west traffic can replay it until
rotation. TLS and network policy are therefore required trust boundaries, and
the key must be rotated after suspected disclosure. Deployments whose threat
model includes hostile workloads on the east-west path need request-bound
authentication or mTLS before enabling Looper reentry; broader Looper
hardening remains tracked by
[#2336](https://github.com/vllm-project/semantic-router/issues/2336).

## 3. Dashboard Password Lifecycle

### Chrome Compromised-Credential Warning

Chrome's [password warning guidance](https://support.google.com/chrome/answer/10311524)
explains that the entered username and password appeared in a known data
breach. The warning is generated by the browser; it is not a dashboard modal
and does not, by itself, identify which site leaked the credential. Treat the
credential as compromised regardless of its source:

1. Change it immediately at `/account/security` to a unique password generated
   by a password manager.
2. Update the browser's saved credential and change the password anywhere it
   was reused.
3. Review and revoke unexpected sessions. A successful dashboard password
   change revokes every prior dashboard session and issues one replacement
   session for the current browser.

If a reusable credential was ever committed to source control, copied into
public documentation, or shared between people, removal from the current file
is not remediation: rotate it, revoke every existing session, and keep the
replacement out of repository history and build artifacts. Public demos must
use per-user identities or short-lived, audience-scoped guest capabilities,
never a shared administrator password.

`make agent-validate` enforces a source-document regression across repository
Markdown, MDX, and HTML. It rejects nearby assigned `username`, `email`, or
`login` values paired with assigned `password` or `pass` values, including
list, key/value table, column table, and common HTML layouts. Explicit
environment/example placeholders and password-policy prose are not treated as
credentials; test fixtures and generated, dependency, or private working trees
are outside this source-doc gate. Findings contain only file and line
locations, never the candidate values.

This focused regression complements rather than replaces the supply-chain
security scanner. It does not inspect Git history, ignored or generated build
artifacts, binary files, lone secret values, or arbitrary credential formats.
Release and incident-response workflows must scan those surfaces separately.

Do not suppress this warning by disabling autofill or changing standard field
metadata. The sign-in, first-admin bootstrap, and password-change forms follow
Google's
[sign-in form guidance](https://web.dev/articles/sign-in-form-best-practices):
the account uses `autocomplete="username"`, the old password uses
`current-password`, and the replacement field uses `new-password`. The
password-change form does not duplicate the new-password input. Paste and
password-manager generation remain available. Every password field in these
three flows starts masked and has an explicit, non-submitting reveal control;
leaving and returning to the bootstrap password step masks the preserved value
again. The multi-step first-admin flow keeps a hidden, stable username input
beside its new-password field so the generated credential is saved against the
correct account. A bootstrap registration conflict clears that password before
rendering sign-in.

The dashboard also implements the
[change-password URL convention](https://web.dev/articles/change-password-url):

- `GET /.well-known/change-password` redirects to `/account/security`.
- Unknown `/.well-known/*` resources return a real `404`; they never receive
  the SPA fallback.

### Server-Side Password Controls

All bootstrap, administrator-create, administrator-reset, and self-service
change paths share one policy based on
[NIST SP 800-63B-4](https://pages.nist.gov/800-63-4/sp800-63b.html):

- at least 15 Unicode code points for a single-factor password, with a
  1,024-code-point maximum and NFC normalization;
- no character-class composition rule or forced periodic rotation;
- complete-value checks against common, compromised, service-specific, and
  account-specific values;
- versioned bcrypt-SHA-256 hashes at cost 12, so long Unicode passwords are
  not truncated by bcrypt's native input limit; legacy bcrypt rows migrate
  only after a successful compare-and-swap login;
- strict, bounded JSON bodies; uniform unknown, inactive, and wrong-password
  verification; rejection of invalid UTF-8 and unpaired JSON surrogate
  escapes before lossy decoding; bounded password work; and account-aware
  throttling;
- an exact HS256 session contract with expiration and server-owned session ID;
  password changes and resets atomically revoke prior sessions; changing an
  account to a non-active status revokes its sessions in the same transaction,
  so reactivation cannot revive a pre-deactivation token;
- a per-user ceiling of 16 active sessions and 64 retained recent session
  records, enforced in the same transaction that issues a token; excess live
  sessions are revoked before bounded inactive history is pruned;
- role/status mutations use a current-state compare-and-swap, while demotion,
  deactivation, and deletion enforce the last-active-`users.manage` invariant
  inside the same transaction;
- a monotonic auth generation rejects verified login or password-change work
  that spans any password or active/inactive ABA transition;
- account create, password reset, role/status update, and delete transactions
  revalidate the acting session and its live `users.manage` permission before
  committing. Other control-plane writes still require the route-bound
  reauthorization work tracked in
  [#2466](https://github.com/vllm-project/semantic-router/issues/2466).

The built-in blocklist is only a minimum safety net. Production password auth
must mount a reviewed newline-delimited corpus. With Helm, mount one key from an
existing ConfigMap:

```yaml
dashboard:
  passwordBlocklist:
    existingConfigMap: dashboard-password-blocklist
    key: passwords.txt
```

The chart mounts that key read-only and sets
`DASHBOARD_PASSWORD_BLOCKLIST_PATH`. Keep the ConfigMap within Kubernetes
object-size limits. A custom deployment may instead mount a larger approved
regular file and set the same environment variable or pass
`--password-blocklist`. The loader rejects missing, non-regular, malformed,
overlong, oversized, empty, or comments-only configured files and never logs
their entries. Do not
send raw submitted passwords to an external breach-check service. The corpus is
compared both as exact NFC-normalized complete values and as case/separator
variants. Exactly empty lines and lines beginning with `#` are metadata;
otherwise leading and trailing whitespace is retained so whitespace-bearing
complete values remain representable. The corpus is
loaded once at dashboard startup; after updating an existing ConfigMap, restart
the dashboard Deployment, wait for readiness, and test a newly blocked value
before treating the policy update as active.

The canonical local `vllm-sr serve` flow accepts
`DASHBOARD_PASSWORD_BLOCKLIST_PATH` as a host path. Before stopping containers,
creating networks, or writing runtime files, the CLI resolves it to an absolute
existing regular file. It mounts that file read-only at a fixed dashboard path
and passes only the container path to the dashboard. The same startup plan
scopes `DASHBOARD_JWT_SECRET`, `DASHBOARD_JWT_EXPIRY_HOURS`,
`DASHBOARD_ADMIN_EMAIL`, `DASHBOARD_ADMIN_PASSWORD`, `DASHBOARD_ADMIN_NAME`,
and `DASHBOARD_ALLOW_OPEN_BOOTSTRAP` to the dashboard container. JWT and
administrator-password values use name-only container arguments and are
available only in the dashboard launcher process environment; neither value is
rendered into command arguments or logs. Router, Envoy, observability, storage,
Fleet Sim, and Kubernetes router environment planning receive none of these
local dashboard-auth settings. Minimal mode starts no dashboard and therefore
ignores them, including a stale blocklist path.

### Deployment Requirements

- Terminate HTTPS before exposing login, bootstrap, or password-change routes.
- Keep split-runtime dashboard, router control/extproc/metrics, data-store, and
  observability host publications on their default `127.0.0.1` boundary. Put a
  TLS reverse proxy in front of the dashboard instead of changing that bind.
  Envoy gateway publication follows each configured listener address; use a
  wildcard only when public gateway access is intentional. The host-only
  `VLLM_SR_INTERNAL_BIND_ADDRESS` override accepts an IP literal and warns on
  non-loopback values; it is not a substitute for a firewall, authentication,
  or transport security.
- Emit HTTP Strict Transport Security at the public TLS terminator after the
  hostname is HTTPS-only. Add `includeSubDomains` only when every covered
  subdomain is also permanently HTTPS; do not make that assumption in a
  reusable dashboard configuration.
- Preserve the dashboard's `nosniff`, `no-referrer`
  referrer, and same-origin framing response headers at the edge. A reusable
  application-level Content Security Policy is intentionally not guessed:
  deployments must validate one against their configured proxy, iframe, and
  model-provider surfaces before enforcement.
- Preserve `Cache-Control: no-store` on every authenticated control-plane
  response, including authorization failures and streaming endpoints. Public
  fingerprinted static assets remain eligible for their immutable cache policy.
- Preserve the public dashboard `Host` and emit exactly one trusted
  `X-Forwarded-Proto: https` value at the TLS proxy. Every protected WebSocket,
  including embedded OpenClaw proxy paths, requires a browser `Origin` whose
  canonical scheme and authority match that effective request origin. Missing,
  `null`, sibling-domain, malformed, or ambiguous forwarded origins are rejected;
  do not work around a failed handshake by disabling origin checks.
- Keep the maintained browser cookie-only. It must not persist the dashboard
  JWT, add it to Authorization, or place it in iframe, EventSource, WebSocket,
  Referer, query data, or normal auth JSON. Maintained login, bootstrap, and
  password-change requests use `X-VSR-Auth-Mode: cookie`; omission preserves
  the deliberate non-browser bearer API. `/api/auth/me` hardens an old
  JavaScript cookie in place without minting a session. Unsafe cookie/query-authenticated HTTP requests, and
  browser logout even when no cookie arrives, must provide a canonical
  same-origin Origin or same-origin Fetch Metadata;
  credentialed CORS must never reflect a sibling origin. Embedded proxies must
  strip dashboard credentials, filter the dashboard cookie namespace, and use
  bounded upstream handshakes. Embedded responses must not widen Service
  Worker scope; every independent CSP policy must retain
  `frame-ancestors 'self'` and use `worker-src 'none'` as defense in depth.
  Bearer authentication remains for deliberate non-browser API clients.
- Keep the dashboard's bounded header/body-read and idle timeouts enabled;
  streaming WebSocket/SSE routes retain their transport-specific deadlines.
- Set `dashboard.jwtSecret.existingSecret` to a stable operator-managed Secret
  for production. `DASHBOARD_JWT_SECRET` must be at least 32 bytes, contain no
  control or surrounding whitespace, and be generated by a CSPRNG, for example
  `openssl rand -base64 32`. An unset value intentionally creates a random
  per-start key and invalidates sessions on restart; use that fallback only for
  local or disposable deployments.
- Apply client-source authentication rate limits at the trusted ingress. The
  dashboard never trusts forwarded client addresses without an explicit proxy
  trust contract; a private or loopback peer therefore disables the source
  bucket to avoid turning a shared proxy into a global account lockout. The
  single-process fallback still uses atomic account reservations, bounded
  overflow state, a global login admission budget, and a bcrypt slot reserved
  for authenticated password management. Those counters reset on process
  restart and are not a replacement for an edge denial-of-service policy.
- Configure and test the production blocklist before creating the first
  account. A configured blocklist or signing-key error prevents a healthy auth
  startup instead of silently weakening policy.
- Never publish or distribute a reusable dashboard administrator credential.
  Provision accountable per-user access and remove private recovery material
  after transferring it to an approved password manager.
- Monitor content-free credential audit-write warnings. Password and session
  state changes are atomic, but transactionally durable audit/outbox delivery
  remains tracked by
  [#2482](https://github.com/vllm-project/semantic-router/issues/2482).

The maintained browser's script-visible bearer and CSRF cleanup is closed by
the audit proof. Same-origin embedded upstream HTML still executes with ambient
browser authority unless deployed on a separate/capability-isolated origin;
that remaining browser boundary is tracked by
[#2465](https://github.com/vllm-project/semantic-router/issues/2465).

## 4. Dashboard RBAC Integration

### Security Policy API

The dashboard provides a Security Policy management page (`/security`),
accessible from the **Manager** dropdown in the top navigation bar. It
allows administrators to:

1. **Map RBAC roles to models** — Define which user groups or individual users
   can access which LLM models. Each mapping generates a router authz signal
   rule and a corresponding decision entry.
2. **Set rate-limit tiers** — Configure per-role/group rate limits that are
   translated into router `ratelimit.providers[].rules`.

### Auto-Apply on Save

When a security policy is saved via `PUT /api/security/policy`, the dashboard
automatically applies the generated fragment to the router's `config.yaml` and
triggers a hot-reload. The full pipeline is:

```
Dashboard UI → PUT /api/security/policy
  → GenerateRouterFragment()
  → toCanonicalYAML()          (maps to routing.signals.role_bindings,
                                 routing.decisions, and
                                 global.services.ratelimit)
  → mergeDeployPayload()       (deep-merges into existing config.yaml)
  → writeConfigAtomically()
  → applyWrittenConfig()       (Envoy restart + fsnotify router hot-reload)
```

The merge uses **replace semantics** for both `routing.decisions` and
`global.services.ratelimit`: the entire block is replaced by the security
policy fragment. Other global config fields (observability, authz, etc.) are
preserved. `routing.signals.role_bindings` are merged alongside other signals.

The API response includes an `"applied": true` field when the fragment was
successfully written and propagated to the runtime.

A `SecurityPolicyConfig` contains:

- `role_mappings[]` — each entry maps subjects (Users/Groups) to a router role
  and a set of allowed models
- `rate_tiers[]` — each entry maps a user/group to RPM and TPM limits

The `GenerateRouterFragment()` function converts this into:

- `role_bindings[]` — router-config-compatible subject-to-role bindings
- `decisions[]` — decision entries with authz rules referencing the mapped role
- `ratelimit.providers[].rules[]` — local-limiter rules with per-unit limits

### Example

Given this security policy:

```json
{
  "role_mappings": [
    {
      "name": "premium",
      "subjects": [{"kind": "Group", "name": "paying-customers"}],
      "role": "premium_tier",
      "model_refs": ["gpt-4", "claude-3"],
      "priority": 10
    }
  ],
  "rate_tiers": [
    {
      "name": "premium-rate",
      "group": "paying-customers",
      "rpm": 1000,
      "tpm": 100000
    }
  ]
}
```

The generated router config fragment is:

```json
{
  "role_bindings": [
    {
      "subjects": [{"kind": "Group", "name": "paying-customers"}],
      "role": "premium_tier"
    }
  ],
  "decisions": [
    {
      "name": "rbac-premium",
      "priority": 10,
      "rules": {"type": "authz", "name": "premium_tier"},
      "modelRefs": [{"model": "gpt-4"}, {"model": "claude-3"}]
    }
  ],
  "ratelimit": {
    "providers": [
      {
        "type": "local-limiter",
        "rules": [
          {
            "name": "premium-rate",
            "match": {"group": "paying-customers"},
            "requests_per_unit": 1000,
            "tokens_per_unit": 100000,
            "unit": "minute"
          }
        ]
      }
    ]
  }
}
```

### API Endpoints

| Endpoint | Method | Permission | Description |
|---|---|---|---|
| `/api/security/policy` | `GET` | `config.read` | Get current security policy |
| `/api/security/policy` | `PUT` | `security.manage` | Update policy, generate fragment, and auto-apply to router |
| `/api/security/policy/preview` | `POST` | `security.manage` | Preview fragment without saving |

### Validation Rules

The security policy is validated before processing:

- Role mapping `name` must be non-empty and unique
- Role mapping `role` must be non-empty
- At least one subject (User or Group) per mapping
- Subject `kind` must be `"User"` or `"Group"`
- Subject `name` must be non-empty
- Rate tier `name` must be non-empty
- At least one of `rpm` or `tpm` must be set per tier

## 5. Dashboard RBAC Permissions

Three new permissions were added to the dashboard RBAC system:

| Permission | Description | Admin | Write | Read |
|---|---|---|---|---|
| `feedback.submit` | Submit model selection feedback | Yes | Yes | Yes |
| `replay.read` | Access router replay API records | Yes | Yes | Yes |
| `security.manage` | Manage security policy (write) | Yes | No | No |

The `security.manage` permission is required for `PUT` and `POST` requests
to `/api/security/*` endpoints. `GET` requests only require `config.read`.

## Deployment Checklist

For production multi-user deployments:

- [ ] Verify Envoy config includes `request_headers_to_remove` for internal headers
- [ ] Configure rate-limit tiers via the Security Policy page or API
- [ ] Set `ratelimit.fail_open: false` for strict enforcement
- [ ] Configure role-to-model mappings to restrict model access by group
- [ ] Review dashboard user roles — only admins should have `security.manage`
- [ ] Treat all `x-vsr-looper-*` headers and `x-vsr-fusion-depth` as internal
- [ ] Use an operator-controlled internal Looper endpoint with network-level
      isolation and transport encryption when it crosses a host or node
- [ ] For multi-replica routing, inject the same 64-hex-character
      `VLLM_SR_LOOPER_SHARED_SECRET` into every router replica
- [ ] Rotate the Looper shared key after suspected process, environment, or
      east-west traffic disclosure
- [ ] If east-west workloads are not mutually trusted, require mTLS or a
      request-bound authentication design before enabling Looper reentry
- [ ] Serve dashboard authentication only over HTTPS
- [ ] Verify dashboard, router control/extproc/metrics, data, and observability
      host ports reject non-loopback traffic; expose only intentional Envoy
      listeners
- [ ] Emit HSTS at the public TLS edge after verifying the hostname is
      HTTPS-only
- [ ] Configure a stable CSPRNG-generated dashboard JWT signing Secret
- [ ] Verify browser storage and embedded URLs contain no dashboard JWT, unsafe
      sibling-origin HTTP/WebSocket requests fail, and embedded upstreams never
      receive or overwrite the dashboard session credential
- [ ] Mount and test a production password blocklist before account bootstrap
- [ ] Verify `/.well-known/change-password` redirects and the reserved unknown
      probe returns `404` at the public origin
- [ ] Rotate any credential reported by the browser and revoke old sessions
- [ ] Run `make agent-validate` to verify public source Markdown, MDX, and HTML
      contain no detectable reusable identity/password pair
- [ ] Scan Git history, generated release artifacts, binary files, and private
      deployment material with the organization's credential scanner
