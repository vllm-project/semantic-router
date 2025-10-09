# Dashboard Risks & Mitigations

## 1. Grafana iframe authentication / embedding issues
**Risk:** Grafana may block embedding (X-Frame-Options / login redirect).  
**Mitigation:** Set `GF_SECURITY_ALLOW_EMBEDDING=true` and ensure anonymous or proxy auth (future OIDC). Proxy strips frame-busting headers.

## 2. Open WebUI CSP restrictions
**Risk:** OWUI might set restrictive CSP preventing embedding.  
**Mitigation:** Proxy sanitizes CSP `frame-ancestors` → `'self'`; if OWUI sets strict CSP hashes, consider upstream config to relax embedding for dashboard origin.

## 3. Token leakage via iframes
**Risk:** Authorization headers forwarded to upstream frames could be exposed if scripts inside frames enumerate requests.  
**Mitigation:** Only forward auth to Router API and Open WebUI if required; never forward to Grafana/Prometheus until proper per-service auth is in place.

## 4. Clickjacking / framing external content
**Risk:** Allowing arbitrary path inputs to Grafana iframe could be abused.  
**Mitigation:** Constrain user-provided dashboard paths to begin with `/d/` (future validation) and sanitize input; keep `frame-ancestors 'self'` CSP.

## 5. CORS / mixed-origin surprises
**Risk:** Subresources inside iframes may attempt cross-origin calls failing silently.  
**Mitigation:** Keep reverse proxy paths stable; prefer embedding fully proxied origins instead of directly exposing upstream hostnames externally.

## 6. Service discovery drift across environments
**Risk:** Namespace or service name changes break baked-in K8s URLs.  
**Mitigation:** Move target URLs to a ConfigMap / Helm values; document override env vars clearly.

## 7. Performance overhead of naive reverse proxy
**Risk:** High-frequency panel refreshes add latency.  
**Mitigation:** Use `httputil.ReverseProxy` (already streaming); enable gzip (future), benchmark under load, optionally bypass proxy for static Grafana assets.

## 8. CSP collisions
**Risk:** Upstream CSP modifications may degrade dashboard security or functionality.  
**Mitigation:** Maintain allowlist for directives modified; log when CSP is rewritten (future logging hook) and optionally provide strict mode.

## 9. Metrics endpoint redirection
**Risk:** Simple redirect leaks upstream internal URL structure externally.  
**Mitigation:** Replace redirect with proxy stream (`/metrics/router` → direct fetch) in hardened mode; redact sensitive labels before exposing aggregated metrics.

## 10. Future OIDC / session fixation complexity
**Risk:** Introducing OIDC later can complicate existing reverse proxy assumptions.  
**Mitigation:** Design backend with middleware chain; isolate auth concerns so OIDC handler can wrap existing proxies without rewriting core logic.

## 11. Log noise & PII in headers
**Risk:** Proxy logging might capture sensitive headers or tokens.  
**Mitigation:** Add structured logger with header redaction list (Authorization, Cookie) before enabling detailed access logs.

## 12. Multi-tenant expansion
**Risk:** Scaling to multi-tenant contexts requires per-tenant config segregation.  
**Mitigation:** From start, keep config viewer read-only and plan `/api/session/context` endpoint to inject tenancy scope for future RBAC.

## 13. Version drift between dashboard and router API
**Risk:** Frontend expects endpoints not present in older router versions.  
**Mitigation:** Add a `/api/router/api/v1` capability handshake; feature-detect endpoints before rendering advanced UI elements.

## 14. Large config JSON rendering performance
**Risk:** Extremely large classification configs freeze UI.  
**Mitigation:** Add lazy chunking or collapsible sections; enforce size limit with warning (future enhancement).

## 15. Security scanning & supply chain
**Risk:** Distroless image reduces surface but still needs SBOM & vulnerability checks.  
**Mitigation:** Add CI step with `trivy` or `grype` scanning dashboard image.

---
This list will evolve as auth (OIDC), aggregation endpoints, and RBAC are implemented.
