package authz

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// HeaderInjectionProvider reads per-user LLM credentials from request headers
// injected by an external authorization service.
//
// Supported ext_authz backends:
//   - Authorino (Kubernetes-native, gRPC ext_authz)
//   - Custom ext_authz HTTP/gRPC service
//   - Envoy Gateway SecurityPolicy (with or without bodyToExtAuth)
//
// All backends follow the same contract: on successful auth, inject provider
// key headers that this provider reads.
//
// Header names are fully configurable via YAML — no recompilation needed to
// support new providers or different header naming conventions.
type HeaderInjectionProvider struct {
	// headerMap maps LLMProvider → header name to read from.
	headerMap map[LLMProvider]string

	// stripHeaders lists all headers to remove before forwarding upstream.
	stripHeaders []string
}

// DefaultHeaderMap returns the standard header mapping used by Authorino and
// the custom ext_authz service. Used as the default when no YAML config is provided.
func DefaultHeaderMap() map[string]string {
	return map[string]string{
		string(ProviderOpenAI):    headers.UserOpenAIKey,
		string(ProviderAnthropic): headers.UserAnthropicKey,
	}
}

// NewHeaderInjectionProvider creates a provider that reads per-user keys
// from ext_authz-injected headers.
//
// headerMap maps LLM provider name → header name (e.g., "openai" → "x-user-openai-key").
// If headerMap is nil or empty, DefaultHeaderMap() is used.
func NewHeaderInjectionProvider(headerMap map[string]string) *HeaderInjectionProvider {
	if len(headerMap) == 0 {
		headerMap = DefaultHeaderMap()
	}

	hmap := make(map[LLMProvider]string, len(headerMap))
	strip := make([]string, 0, len(headerMap))
	for provider, header := range headerMap {
		hmap[LLMProvider(provider)] = header
		strip = append(strip, header)
	}

	return &HeaderInjectionProvider{
		headerMap:    hmap,
		stripHeaders: strip,
	}
}

func (p *HeaderInjectionProvider) Name() string {
	return "header-injection"
}

// GetKey reads the provider key from the corresponding injected header.
// The model parameter is ignored — header-based keys are provider-wide, not model-specific.
func (p *HeaderInjectionProvider) GetKey(provider LLMProvider, _ string, reqHeaders map[string]string) string {
	headerName, ok := p.headerMap[provider]
	if !ok {
		return ""
	}
	return reqHeaders[headerName]
}

// HeadersToStrip returns all injected header names that must be removed
// before forwarding the request upstream (prevents key leakage to LLM providers).
func (p *HeaderInjectionProvider) HeadersToStrip() []string {
	return p.stripHeaders
}
