// Package authz provides a pluggable credential resolution framework for
// external authorization in the semantic router.
//
// The router needs to obtain per-user API keys for upstream LLM providers
// (OpenAI, Anthropic, Gemini, etc.). These keys can come from multiple sources:
//
//   - Header injection: An ext_authz service (Authorino, custom, or Envoy Gateway
//     SecurityPolicy) validates the user's Bearer token and injects per-user
//     provider keys as request headers (e.g., x-user-openai-key).
//
//   - Static config: Keys defined in the router's YAML config per model.
//
// The CredentialResolver chains multiple providers and returns the first match,
// making the router agnostic to how credentials are supplied.
//
// Adding a new provider (e.g., Gemini):
//  1. Add a constant to LLMProvider (ProviderGemini)
//  2. Add a header constant to pkg/headers (UserGeminiKey)
//  3. Register the header in HeaderInjectionProvider
//  4. Use resolver.KeyForProvider(authz.ProviderGemini, ...) in the routing path
package authz

// LLMProvider identifies an upstream LLM provider.
type LLMProvider string

const (
	ProviderOpenAI    LLMProvider = "openai"
	ProviderAnthropic LLMProvider = "anthropic"
	// Future providers:
	// ProviderGemini    LLMProvider = "gemini"
	// ProviderMistral   LLMProvider = "mistral"
)

// Credentials holds per-user API keys for all known LLM providers.
// Fields are empty when no key is available from a given source.
type Credentials struct {
	// Keys maps provider name to API key.
	// e.g., {"openai": "sk-...", "anthropic": "sk-ant-..."}
	Keys map[LLMProvider]string
}

// Get returns the key for a provider, or "" if not set.
func (c *Credentials) Get(provider LLMProvider) string {
	if c == nil {
		return ""
	}
	return c.Keys[provider]
}

// IsEmpty returns true if no keys are set.
func (c *Credentials) IsEmpty() bool {
	if c == nil {
		return true
	}
	for _, v := range c.Keys {
		if v != "" {
			return false
		}
	}
	return true
}

// Provider is a source of per-user LLM credentials.
// Implementations extract keys from different sources (headers, config, etc.).
type Provider interface {
	// Name returns a human-readable name for logging (e.g., "header-injection", "static-config").
	Name() string

	// GetKey returns the API key for the given provider, or "" if not available from this source.
	// The model parameter allows model-aware providers to return different keys per model.
	GetKey(provider LLMProvider, model string, headers map[string]string) string

	// HeadersToStrip returns header names that should be removed before forwarding
	// the request upstream (to prevent credential leakage).
	HeadersToStrip() []string
}
