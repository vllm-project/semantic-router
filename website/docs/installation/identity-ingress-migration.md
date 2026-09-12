---
sidebar_position: 1
---

# Migrate Identity Ingress Trust

This guide migrates deployments from implicit identity trust to the explicit
identity-ingress contract. The change separates two concerns that used to be
coupled:

- `global.services.authz.providers[].type: header-injection` resolves per-user
  model credentials;
- `global.services.authz.identity.ingress: header-injection` declares that a
  verified external authorization boundary may provide request identity.

The `header-injection` value is a Semantic Router configuration value, not a
standard HTTP header name and not an authentication mechanism. The actual
identity headers remain configurable with `user_id_header` and
`user_groups_header`.

## What changes after the upgrade

The Router creates one ingress-derived `RequestContext.TrustedIdentity`. Authz,
rate limits, cache partitioning, memory, Replay, Responses, and learning use
that typed snapshot. They must not reconstruct identity from `ctx.Headers`,
request metadata, query parameters, or request-body fields.

Without `authz.identity.ingress: header-injection`, configured identity headers
are treated as untrusted. They are ignored for identity derivation and removed
from the Router's semantic header view. This is intentional fail-closed
behavior.

## 1. Classify the deployment

Choose the path that matches the real ingress topology.

### Verified external authorization

Use this path when Envoy, Authorino, an API gateway, or another trusted
authorization component authenticates the caller and injects identity headers
after removing client-supplied copies.

The Router configuration must declare both contracts independently when both
features are used:

```yaml
global:
  services:
    authz:
      identity:
        user_id_header: x-jwt-sub
        user_groups_header: x-jwt-groups
        ingress: header-injection
      providers:
        - type: header-injection
          headers:
            openai: x-user-openai-key
```

Keep the `providers` entry when it is needed for per-user model credentials.
Adding the identity `ingress` entry does not make credential headers valid, and
keeping a credential provider does not authenticate identity.

The external authorization path must perform these operations in order:

```text
remove client-supplied identity headers
  -> authenticate and authorize the caller
  -> inject the configured identity headers
  -> Semantic Router ExtProc
```

The Router declaration records the expected trusted boundary; it cannot
authenticate a header by itself.

### Local or unauthenticated development

Use this path when the local Envoy fixture does not install external
authorization. Set the ingress explicitly to `none`, or omit it:

```yaml
global:
  services:
    authz:
      identity:
        user_id_header: x-user-id
        user_groups_header: x-user-groups
        ingress: none
```

Client-supplied identity headers are ignored and stripped. Identity-based RBAC
routes, per-user limits, and user-scoped memory therefore run as anonymous
unless a test harness adds identity after a trusted authentication step. Do
not use direct client headers as a substitute for that step.

### Static credentials without identity routing

If the deployment only needs static model credentials and does not use
identity-based routing, leave the identity ingress unset or set it to `none`.
`authz.providers` continues to control credential resolution independently.

## 2. Migrate an existing configuration

Find configurations that define identity headers or rely on authz role
bindings:

```bash
rg -n -C 3 \
  'user_id_header|user_groups_header|role_bindings|authz:|header-injection' \
  config e2e deploy
```

For each deployment, answer both questions:

1. Which component authenticates the caller and injects the identity headers?
2. Does `authz.providers` also need to inject model credentials?

Then apply the matching change.

### Old configuration: one provider used for both purposes

An older configuration may look like this:

```yaml
global:
  services:
    authz:
      identity:
        user_id_header: x-user-id
        user_groups_header: x-user-groups
      providers:
        - type: header-injection
          headers:
            openai: x-user-openai-key
```

If the ingress is genuinely authenticated, add the explicit identity
declaration while keeping the credential provider:

```yaml
global:
  services:
    authz:
      identity:
        user_id_header: x-user-id
        user_groups_header: x-user-groups
        ingress: header-injection
      providers:
        - type: header-injection
          headers:
            openai: x-user-openai-key
```

If the old `header-injection` provider existed only to make identity appear
trusted and no per-user model credentials are required, remove that provider
and add only `identity.ingress`.

### Configuration with no external authorization

Do not add `ingress: header-injection` just to restore an RBAC route in a local
environment. Use `ingress: none` and add a real authentication boundary before
enabling identity-based routing.

## 3. Roll out safely

This is a security-semantic change, so do not rely on mixed-version behavior as
the compatibility mechanism.

1. Update the gateway or Envoy configuration so client identity headers are
   removed before authentication output is injected.
2. Add `authz.identity.ingress` to the Router configuration for deployments
   with a verified identity boundary.
3. Validate the complete configuration and deploy the Router version that
   understands the new field.
4. Confirm positive identity routing through the authenticated ingress.
5. Confirm that direct client-supplied identity headers do not select an
   identity-based route.

Older Router versions may ignore the new `ingress` field and continue their
older provider-based behavior. Conversely, a new Router treats an old config
without `identity.ingress` as untrusted. During a rolling upgrade, expose old
instances only behind the same header-stripping and authentication boundary.

For rollback, restore the configuration and gateway behavior tested with the
older Router version. Do not run an older Router against a local configuration
that still contains a credential `header-injection` provider and exposes
custom identity headers; older versions may mistake that credential provider
for identity authentication.

## 4. Verify the migration

Validate each migrated config before serving it:

```bash
vllm-sr validate --config config.yaml
```

For the maintained authz-rbac profile, run the anti-spoofing assertion:

```bash
make e2e-test-specific \
  E2E_PROFILE=authz-rbac \
  E2E_TESTS=authz-header-spoofing
```

Also verify the following behavior through the actual ingress path:

- an authenticated request receives the configured identity headers only after
  the authentication step and matches the expected RBAC policy;
- a client request that supplies those headers directly does not become a
  trusted identity;
- user-scoped rate limits, cache keys, memory operations, Replay, Responses,
  and learning state follow the same trusted identity;
- per-user model credentials still resolve through `authz.providers`, and are
  removed before forwarding to a model provider.

If RBAC or user-scoped features become anonymous after upgrading, first check
that `identity.ingress` is present and that the configured header names exactly
match the headers injected by the authenticated ingress. Header lookup is
case-insensitive, but the ingress still needs to provide the configured
identity values after removing client input.

See [Security Hardening](./security-hardening) for the ongoing identity
consumer contract and [issue #1445](https://github.com/vllm-project/semantic-router/issues/1445)
for the accepted change that established this boundary.
