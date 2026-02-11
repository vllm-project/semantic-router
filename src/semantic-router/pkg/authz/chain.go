package authz

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// CredentialResolver chains multiple Providers and resolves credentials
// using first-match semantics: the first provider that returns a non-empty key wins.
//
// Typical chain order:
//  1. HeaderInjectionProvider (keys from ext_authz / Authorino / Envoy Gateway)
//  2. StaticConfigProvider    (keys from router YAML config)
//
// Security modes:
//   - fail-closed (failOpen=false, default): KeyForProvider returns an error when
//     no provider can resolve a key. This prevents silent bypass from misconfig,
//     ext_authz downtime, or header spoofing.
//   - fail-open (failOpen=true): KeyForProvider returns ("", nil) when no key is
//     found. Use only for backends that don't require auth (e.g., local vLLM).
type CredentialResolver struct {
	providers []Provider
	failOpen  bool
}

// NewCredentialResolver creates a resolver with the given provider chain.
// Providers are tried in order; first non-empty result wins.
// By default the resolver is fail-closed (failOpen=false).
func NewCredentialResolver(providers ...Provider) *CredentialResolver {
	return &CredentialResolver{providers: providers, failOpen: false}
}

// SetFailOpen configures whether the resolver allows requests through when
// no provider can resolve a key.
func (r *CredentialResolver) SetFailOpen(failOpen bool) {
	if r != nil {
		r.failOpen = failOpen
	}
}

// FailOpen returns whether the resolver is in fail-open mode.
func (r *CredentialResolver) FailOpen() bool {
	if r == nil {
		return false
	}
	return r.failOpen
}

// KeyForProvider returns the API key for the given LLM provider and model.
// It tries each provider in chain order and returns the first non-empty key.
//
// Error behavior:
//   - Returns (key, nil) when a key is found.
//   - Returns ("", nil) when no key is found AND failOpen=true.
//   - Returns ("", error) when no key is found AND failOpen=false.
//     The error message lists every provider that was tried, making misconfig
//     immediately visible instead of silently bypassing.
//
// Logging:
//   - DEBUG: which provider resolved the key (happy path)
//   - DEBUG: each provider miss as the chain is walked
//   - WARN:  total miss in fail-open mode (allowed through but suspicious)
//   - ERROR: total miss in fail-closed mode (request will be rejected)
func (r *CredentialResolver) KeyForProvider(provider LLMProvider, model string, headers map[string]string) (string, error) {
	if r == nil {
		return "", fmt.Errorf("credential resolver is nil — cannot resolve key for %s (model=%s)", provider, model)
	}

	tried := make([]string, 0, len(r.providers))
	for _, p := range r.providers {
		if key := p.GetKey(provider, model, headers); key != "" {
			logging.Debugf("Credential resolved for %s (model=%s) via provider %q", provider, model, p.Name())
			return key, nil
		}
		tried = append(tried, p.Name())
		logging.Debugf("Provider %q returned no key for %s (model=%s), trying next", p.Name(), provider, model)
	}

	triedStr := strings.Join(tried, " → ")

	if r.failOpen {
		logging.Warnf("No credential found for %s (model=%s) after trying [%s] — fail_open=true, allowing request without key", provider, model, triedStr)
		return "", nil
	}

	err := fmt.Errorf("no credential found for provider %s (model=%s) after trying [%s] — all providers exhausted. "+
		"Check: (1) ext_authz/Authorino is running and injecting the expected headers, "+
		"(2) header names in authz.providers[].headers match what ext_authz injects, "+
		"(3) model_config has access_key set if using static-config fallback. "+
		"Set authz.fail_open=true only if this backend does not require auth",
		provider, model, triedStr)
	logging.Errorf("%v", err)
	return "", err
}

// HeadersToStrip returns the union of all headers that providers want stripped
// before forwarding upstream. Deduplicated.
func (r *CredentialResolver) HeadersToStrip() []string {
	if r == nil {
		return nil
	}
	seen := make(map[string]bool)
	var result []string
	for _, p := range r.providers {
		for _, h := range p.HeadersToStrip() {
			if !seen[h] {
				seen[h] = true
				result = append(result, h)
			}
		}
	}
	return result
}

// ProviderNames returns the names of all registered providers (for logging).
func (r *CredentialResolver) ProviderNames() []string {
	if r == nil {
		return nil
	}
	names := make([]string, len(r.providers))
	for i, p := range r.providers {
		names[i] = p.Name()
	}
	return names
}
