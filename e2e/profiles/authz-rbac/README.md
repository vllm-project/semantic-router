# Authorization and RBAC E2E profile

This profile verifies the safe boundary for identity-derived routing headers.
The Gateway removes client-supplied `x-authz-user-id` and
`x-authz-user-groups` before ExtProc, so a caller cannot claim a Router role by
setting those headers directly.

The current profile runs:

- `chat-completions-request` for the base inference path;
- `chat-completions-request-authz` for the stripped-header path;
- `authz-header-spoofing` for the explicit anti-spoofing assertion.

## What it does not prove

The checked-in Gateway resources do not install JWT validation or an external
authorization service. Therefore this profile does not prove positive routing
for a real admin, premium, or free-user identity. The policy in
[`values.yaml`](values.yaml) contains those example tiers, but validated claims
must be injected by a trusted authentication component after untrusted headers
are removed.

Do not send raw identity headers from a client as a substitute for that
component. A production flow must be ordered as:

```text
remove untrusted identity headers
  -> validate credential
  -> derive trusted identity headers
  -> Semantic Router ExtProc
```

The exact Envoy or Authorino configuration depends on the identity provider
and is intentionally not templated with placeholder issuer keys in this
profile.

## Run

```bash
make e2e-test E2E_PROFILE=authz-rbac
```

Run only the spoofing check while debugging the filter order:

```bash
make e2e-test-specific \
  E2E_PROFILE=authz-rbac \
  E2E_TESTS=authz-header-spoofing
```

Inspect the applied patch and gateway logs when the assertion fails:

```bash
kubectl get envoypatchpolicy ai-gateway-prepost-extproc-patch-policy \
  --namespace default -o yaml
kubectl logs --namespace envoy-gateway-system \
  -l gateway.envoyproxy.io/owning-gateway-name=semantic-router
```

`ratelimit-limitor` is not part of this profile because its direct client
identity headers conflict with the anti-spoofing filter. Test rate limits with
a trusted post-auth identity source or in the profile that owns that contract.
