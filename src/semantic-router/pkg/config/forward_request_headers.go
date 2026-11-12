package config

import (
	"fmt"
	"slices"
	"strings"

	"golang.org/x/net/http/httpguts"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// framingHeaders belong to one connection or describe the body of the
// request that carries them, so a call the Router builds sets its own and
// never copies a client's: the hop-by-hop headers (with every proxy-*
// header), the host, and the body and content-negotiation headers.
var framingHeaders = []string{
	"connection", "keep-alive", "te", "trailer", "transfer-encoding", "upgrade",
	"host", "content-length", "content-type", "content-encoding", "accept", "accept-encoding", "expect",
}

// validateForwardedHeaderName refuses an allowlist entry that no call may
// carry: an empty name, a pseudo-header and a known credential carrier.
// field names the allowlist in the error.
func validateForwardedHeaderName(field, name string) error {
	trimmed := strings.TrimSpace(name)
	if trimmed == "" {
		return fmt.Errorf("%s entries cannot be empty", field)
	}
	if strings.HasPrefix(trimmed, ":") {
		return fmt.Errorf("%s cannot include pseudo-header %q", field, trimmed)
	}
	if IsShadowCredentialHeader(trimmed) {
		return fmt.Errorf("%s cannot include credential header %q", field, trimmed)
	}
	return nil
}

// validateForwardRequestHeaders checks a forward_request_headers list, the
// client request headers a call the Router makes itself copies. It refuses
// what can never be forwarded: an empty or invalid name, a pseudo-header, a
// framing header, a credential (authHeaders are the auth headers of the
// upstreams the list applies to), a header in the Router's or Envoy's own
// namespace, an identity header (any x-authz- header, the authenticator's)
// the Router keeps from upstreams, and a name listed twice.
func validateForwardRequestHeaders(names []string, authHeaders ...string) error {
	seen := make(map[string]bool, len(names))
	for _, name := range names {
		if err := validateForwardedHeaderName("forward_request_headers", name); err != nil {
			return err
		}
		trimmed := strings.TrimSpace(name)
		lower := strings.ToLower(trimmed)
		switch {
		case !httpguts.ValidHeaderFieldName(trimmed):
			return fmt.Errorf("forward_request_headers lists %q, which is not an HTTP header name", trimmed)
		case forwardsCredential(lower, authHeaders):
			return fmt.Errorf("forward_request_headers cannot include credential header %q", trimmed)
		case slices.Contains(framingHeaders, lower) || strings.HasPrefix(lower, "proxy-"):
			return fmt.Errorf("forward_request_headers cannot include %q: the Router sets the framing and body headers of its own request", trimmed)
		case strings.HasPrefix(lower, headers.VSRPrefix) || strings.HasPrefix(lower, headers.EnvoyPrefix):
			return fmt.Errorf("forward_request_headers cannot include %q: x-vsr- and x-envoy- headers belong to the Router and Envoy", trimmed)
		case strings.HasPrefix(lower, headers.AuthzPrefix) || registrySensitivity(lower) == headers.SensitivityIdentity:
			return fmt.Errorf("forward_request_headers cannot include identity header %q, which the Router keeps from upstreams", trimmed)
		case seen[lower]:
			return fmt.Errorf("forward_request_headers lists %q twice", trimmed)
		}
		seen[lower] = true
	}
	return nil
}

// forwardsCredential reports whether a lower-case header name carries a
// credential: a known carrier, a header the registry records as a secret, or
// one of authHeaders.
func forwardsCredential(lower string, authHeaders []string) bool {
	if slices.Contains(credentialHeaders, lower) || registrySensitivity(lower) == headers.SensitivitySecret {
		return true
	}
	return slices.ContainsFunc(authHeaders, func(auth string) bool {
		return strings.EqualFold(strings.TrimSpace(auth), lower)
	})
}

func registrySensitivity(name string) headers.Sensitivity {
	if header, ok := headers.Lookup(name); ok {
		return header.Sensitivity
	}
	return headers.SensitivityNone
}

// validateFallbackForwardRequestHeaders checks the forward_request_headers of
// the global fallback policy and of each recipe's against the auth header of
// every backend, any of which a fallback attempt may reach.
func validateFallbackForwardRequestHeaders(cfg *RouterConfig) error {
	authHeaders := providerAuthHeaders(cfg)
	if policy := cfg.RoutingDefaults.Fallback; policy != nil {
		if err := validateForwardRequestHeaders(policy.ForwardRequestHeaders, authHeaders...); err != nil {
			return fmt.Errorf("global.router.fallback: %w", err)
		}
	}
	for _, recipe := range cfg.Recipes {
		if policy := recipe.Profile.Fallback; policy != nil {
			if err := validateForwardRequestHeaders(policy.ForwardRequestHeaders, authHeaders...); err != nil {
				return fmt.Errorf("%s.fallback: %w", recipePath(string(recipe.Name)), err)
			}
		}
	}
	return nil
}

// validateForwardedIdentityHeaders refuses a forward_request_headers entry
// that names a header the identity configuration reads the caller from. The
// Router keeps those from upstreams like the x-authz- set, so no call it
// makes may copy one.
func validateForwardedIdentityHeaders(cfg *RouterConfig) error {
	identity := cfg.Authz.Identity.HeaderNames()
	check := func(path string, names []string) error {
		for _, name := range names {
			if trimmed := strings.TrimSpace(name); slices.Contains(identity, strings.ToLower(trimmed)) {
				return fmt.Errorf("%s: forward_request_headers cannot include identity header %q, which the Router keeps from upstreams", path, trimmed)
			}
		}
		return nil
	}
	if policy := cfg.RoutingDefaults.Fallback; policy != nil {
		if err := check("global.router.fallback", policy.ForwardRequestHeaders); err != nil {
			return err
		}
	}
	for _, recipe := range cfg.Recipes {
		if policy := recipe.Profile.Fallback; policy != nil {
			if err := check(recipePath(string(recipe.Name))+".fallback", policy.ForwardRequestHeaders); err != nil {
				return err
			}
		}
	}
	for _, name := range sortedModelKeys(cfg.ModelDeployments) {
		if err := check("global.model_catalog.deployments."+name, cfg.ModelDeployments[name].ForwardRequestHeaders); err != nil {
			return err
		}
	}
	return nil
}

// providerAuthHeaders are the auth headers the provider profiles send their
// credential in: each backend's own, else its provider's.
func providerAuthHeaders(cfg *RouterConfig) []string {
	var names []string
	for _, name := range sortedModelKeys(cfg.ProviderProfiles) {
		profile := cfg.ProviderProfiles[name]
		if header, _, err := profile.ResolveAuthHeader(); err == nil && header != "" {
			names = append(names, header)
		}
	}
	return names
}

// DeploymentForwardedHeaders names, in lower case and once each, the client
// request headers some deployment forwards: the client's values a call made
// for a request may copy.
func (c *RouterConfig) DeploymentForwardedHeaders() []string {
	if c == nil {
		return nil
	}
	var names []string
	for _, deployment := range c.ModelDeployments {
		for _, name := range deployment.ForwardRequestHeaders {
			if lower := strings.ToLower(strings.TrimSpace(name)); !slices.Contains(names, lower) {
				names = append(names, lower)
			}
		}
	}
	return names
}
