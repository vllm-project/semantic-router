# Authz-RBAC Profile

This profile demonstrates user-level RBAC model routing where different users (admin, premium, free) are routed to different models (14B vs 7B) based on authenticated user identity.

## Security Requirements

**IMPORTANT**: Identity headers (`x-authz-user-id`, `x-authz-user-groups`) must **ONLY** come from validated auth backends (JWT, Authorino, etc.), **NOT** from client requests.

### Envoy Configuration

The Envoy configuration includes a Lua filter that **strips any client-supplied identity headers** before processing. This prevents header spoofing attacks where malicious clients could inject identity headers to gain unauthorized access to premium models.

The header removal filter runs **before** the JWT/auth filter and ext_proc filter, ensuring that:

1. Client-supplied identity headers are removed
2. JWT tokens are validated
3. Validated JWT claims are extracted into identity headers
4. The semantic router receives only validated identity headers

### Testing

For e2e tests:

- **DO NOT** send `x-authz-user-id` or `x-authz-user-groups` headers directly from the test client
- **DO** send JWT tokens in the `Authorization: Bearer <token>` header
- Envoy Gateway validates the JWT and extracts claims into identity headers
- The router receives only JWT-derived identity headers

### Production Deployment

When deploying this profile in production:

1. **Configure JWT Authentication**: Set up Envoy Gateway JWT filter or Authorino to validate JWT tokens
2. **Verify Header Stripping**: Ensure the EnvoyPatchPolicy in `gateway-resources/gwapi-resources.yaml` is applied to strip client-supplied identity headers
3. **Test Security**: Verify that sending identity headers directly from a client does NOT work (should be stripped)

#### JWT Configuration Example

To enable JWT validation in Envoy Gateway, you can use an `EnvoyPatchPolicy` to add a JWT filter. Here's an example configuration:

```yaml
apiVersion: gateway.envoyproxy.io/v1alpha1
kind: EnvoyPatchPolicy
metadata:
  name: jwt-authentication
  namespace: default
spec:
  jsonPatches:
  - name: default/semantic-router/http
    operation:
      op: add
      path: /default_filter_chain/filters/0/typed_config/http_filters/1
      value:
        name: envoy.filters.http.jwt_authn
        typedConfig:
          '@type': type.googleapis.com/envoy.extensions.filters.http.jwt_authn.v3.JwtAuthentication
          providers:
            jwt_provider:
              issuer: https://your-auth-server.com
              audiences:
                - your-audience
              remoteJwks:
                httpUri:
                  uri: https://your-auth-server.com/.well-known/jwks.json
                  cluster: jwt_provider_cluster
                  timeout: 5s
              forward: true
              fromHeaders:
                - name: Authorization
                  valuePrefix: "Bearer "
          rules:
            - match:
                prefix: /
              requires:
                providerName: jwt_provider
    type: type.googleapis.com/envoy.config.listener.v3.Listener
  - name: jwt_provider_cluster
    operation:
      op: add
      path: ''
      value:
        connect_timeout: 5s
        http2_protocol_options: {}
        lb_policy: ROUND_ROBIN
        load_assignment:
          cluster_name: jwt_provider_cluster
          endpoints:
            - lb_endpoints:
                - endpoint:
                    address:
                      socket_address:
                        address: your-auth-server.com
                        port_value: 443
        name: jwt_provider_cluster
        type: LOGICAL_DNS
    type: type.googleapis.com/envoy.config.cluster.v3.Cluster
  targetRef:
    group: gateway.networking.k8s.io
    kind: Gateway
    name: semantic-router
  type: JSONPatch
```

**Important Notes:**

- The JWT filter should be added **after** the Lua filter that strips client-supplied headers
- The JWT filter extracts claims from the validated token and sets them as headers (e.g., `x-authz-user-id`, `x-authz-user-groups`)
- Replace `your-auth-server.com` and `your-audience` with your actual authentication server details
- The JWT provider configuration must match your authentication server's JWT signing key and claims structure

**Alternative: Using Authorino**

Instead of configuring JWT directly in Envoy, you can use [Authorino](https://github.com/kuadrant/authorino) as an external authentication service:

1. Deploy Authorino in your cluster
2. Create an `AuthPolicy` resource that references Authorino
3. Authorino validates JWT tokens and injects identity headers
4. The Lua filter in `gwapi-resources.yaml` ensures client-supplied headers are still stripped

## RBAC Tiers

- **admin** → 14B (unrestricted, reasoning enabled)
- **premium_user** → 14B (complex queries) or 7B (simple queries)
- **free_user** → 7B only
- **(no match)** → 7B (default)

## Test Cases

The authz-rbac profile includes the following test cases:

- **chat-completions-request**: Basic functional test that validates end-to-end routing
- **chat-completions-request-authz**: Sends identity headers from the client; Lua strips them before ext_proc (same as spoofing defense). Asserts HTTP 200.
- **authz-header-spoofing**: Security test that verifies client-supplied identity headers are stripped and cannot be used for unauthorized access

**Rate limiting (`ratelimit-limitor`)** is intentionally **not** part of this profile: that test requires `x-authz-user-id` / `x-authz-user-groups` to reach the router from the HTTP client, which conflicts with the Lua filter that strips client-supplied identity headers (Issue #1447). Run `ratelimit-limitor` from a profile without that strip, or drive identity via JWT placed **after** the strip in the filter chain.

To run the security test:

```bash
make e2e-test E2E_PROFILE=authz-rbac E2E_TESTS=authz-header-spoofing
```
