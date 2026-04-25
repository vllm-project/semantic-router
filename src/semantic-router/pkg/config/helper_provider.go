package config

import (
	"fmt"
	"net/url"
	"strings"
)

// ---------------------------------------------------------------------------
// Provider profile helpers
// ---------------------------------------------------------------------------

// providerTypeInfo holds the per-type defaults for a cloud provider.
// Every supported type MUST have an entry — no default/fallback branch.
type providerTypeInfo struct {
	AuthHeader string // HTTP header name for the API key
	AuthPrefix string // value prefix ("Bearer", "" etc.)
	ChatPath   string // path suffix appended after base_url path
}

// providerTypeRegistry is the single source of truth for type defaults.
// To add a new provider, add one entry here and a matching LLMProvider
// constant in pkg/authz/provider.go — nothing else needs a switch/default.
var providerTypeRegistry = map[string]providerTypeInfo{
	"openai":       {AuthHeader: "Authorization", AuthPrefix: "Bearer", ChatPath: "/chat/completions"},
	"anthropic":    {AuthHeader: "x-api-key", AuthPrefix: "", ChatPath: "/v1/messages"},
	"azure-openai": {AuthHeader: "api-key", AuthPrefix: "", ChatPath: "/chat/completions"},
	"bedrock":      {AuthHeader: "Authorization", AuthPrefix: "Bearer", ChatPath: "/chat/completions"},
	"gemini":       {AuthHeader: "Authorization", AuthPrefix: "Bearer", ChatPath: "/chat/completions"},
	"vertex-ai":    {AuthHeader: "Authorization", AuthPrefix: "Bearer", ChatPath: "/chat/completions"},
	"minimax":      {AuthHeader: "Authorization", AuthPrefix: "Bearer", ChatPath: "/v1/chat/completions"},
}

// ValidProviderTypes returns the set of recognised type strings (for error messages).
func ValidProviderTypes() []string {
	types := make([]string, 0, len(providerTypeRegistry))
	for t := range providerTypeRegistry {
		types = append(types, t)
	}
	return types
}

// GetProviderProfileForEndpoint resolves the ProviderProfile for a named endpoint.
//
// Returns (nil, nil) when the endpoint exists but has no provider_profile set
// (legacy address:port endpoint — this is not an error).
//
// Returns a non-nil error when:
//   - endpointName does not match any VLLMEndpoint
//   - the endpoint references a provider_profile name that does not exist in the map
func (c *RouterConfig) GetProviderProfileForEndpoint(endpointName string) (*ProviderProfile, error) {
	if endpointName == "" {
		return nil, nil // no endpoint selected (e.g., model has no preferred_endpoints)
	}
	ep, found := c.GetEndpointByName(endpointName)
	if !found {
		return nil, fmt.Errorf("endpoint %q not found in vllm_endpoints", endpointName)
	}
	if ep.ProviderProfileName == "" {
		return nil, nil // legacy endpoint, no profile — not an error
	}
	if c.ProviderProfiles == nil {
		return nil, fmt.Errorf("endpoint %q references provider_profile %q but no provider_profiles map is defined",
			endpointName, ep.ProviderProfileName)
	}
	profile, ok := c.ProviderProfiles[ep.ProviderProfileName]
	if !ok {
		return nil, fmt.Errorf("endpoint %q references provider_profile %q which does not exist in provider_profiles (have: %v)",
			endpointName, ep.ProviderProfileName, mapKeys(c.ProviderProfiles))
	}
	return &profile, nil
}

// mapKeys returns the keys of a map for diagnostic messages.
func mapKeys(m map[string]ProviderProfile) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

// ResolveAddress returns the host:port string for this endpoint.
//
// Two distinct modes — no silent fallback between them:
//   - provider_profile set → host:port is extracted from the profile's base_url.
//     Returns error if profile is missing, has no base_url, or base_url is unparsable.
//   - provider_profile NOT set → uses address:port fields directly.
func (ep *VLLMEndpoint) ResolveAddress(profiles map[string]ProviderProfile) (string, error) {
	if ep.ProviderProfileName == "" {
		// Legacy endpoint: address:port is the intended mode.
		return fmt.Sprintf("%s:%d", ep.Address, ep.Port), nil
	}

	// Profile-based endpoint: MUST resolve from base_url.
	if profiles == nil {
		return "", fmt.Errorf("endpoint %q has provider_profile %q but no provider_profiles map is defined",
			ep.Name, ep.ProviderProfileName)
	}
	profile, ok := profiles[ep.ProviderProfileName]
	if !ok {
		return "", fmt.Errorf("endpoint %q references provider_profile %q which does not exist",
			ep.Name, ep.ProviderProfileName)
	}
	if profile.BaseURL == "" {
		return "", fmt.Errorf("endpoint %q: provider_profile %q has no base_url",
			ep.Name, ep.ProviderProfileName)
	}

	u, err := url.Parse(profile.BaseURL)
	if err != nil {
		return "", fmt.Errorf("endpoint %q: cannot parse base_url %q: %w",
			ep.Name, profile.BaseURL, err)
	}
	if u.Host == "" {
		return "", fmt.Errorf("endpoint %q: base_url %q has no host",
			ep.Name, profile.BaseURL)
	}

	host := u.Host
	if !strings.Contains(host, ":") {
		switch u.Scheme {
		case "https":
			host += ":443"
		case "http":
			host += ":80"
		default:
			return "", fmt.Errorf("endpoint %q: base_url %q has unsupported scheme %q (expected http or https)",
				ep.Name, profile.BaseURL, u.Scheme)
		}
	}
	return host, nil
}

// ProviderType returns the provider type string, which matches authz.LLMProvider values.
// Returns an error if the type is empty or not in providerTypeRegistry.
func (p *ProviderProfile) ProviderType() (string, error) {
	if p == nil {
		return "", fmt.Errorf("provider profile is nil")
	}
	if p.Type == "" {
		return "", fmt.Errorf("provider profile has empty type")
	}
	if _, ok := providerTypeRegistry[p.Type]; !ok {
		return "", fmt.Errorf("unknown provider profile type %q (valid types: %v)", p.Type, ValidProviderTypes())
	}
	return p.Type, nil
}

// ResolveAuthHeader returns the (headerName, prefix) for the upstream auth header.
// Explicit AuthHeader/AuthPrefix fields override the type defaults.
// Returns error if the profile's type is not recognised.
func (p *ProviderProfile) ResolveAuthHeader() (string, string, error) {
	info, ok := providerTypeRegistry[p.Type]
	if !ok {
		return "", "", fmt.Errorf("unknown provider type %q — cannot determine auth header", p.Type)
	}
	headerName := info.AuthHeader
	prefix := info.AuthPrefix
	if p.AuthHeader != "" {
		headerName = p.AuthHeader
	}
	if p.AuthPrefix != "" {
		prefix = p.AuthPrefix
	}
	return headerName, prefix, nil
}

// ResolveChatPath returns the HTTP path for upstream requests.
//
// Resolution order (no silent fallback):
//  1. Explicit ChatPath field on the profile (used as-is, plus ?api-version for azure-openai).
//  2. base_url path + type-default suffix from providerTypeRegistry.
//  3. Type-default suffix alone if base_url has no path component.
//
// Returns error if the type is not recognised or base_url is unparsable.
func (p *ProviderProfile) ResolveChatPath() (string, error) {
	if p == nil {
		return "", fmt.Errorf("provider profile is nil")
	}

	info, ok := providerTypeRegistry[p.Type]
	if !ok {
		return "", fmt.Errorf("unknown provider type %q — cannot determine chat path", p.Type)
	}

	// Explicit override
	if p.ChatPath != "" {
		path := p.ChatPath
		if p.Type == "azure-openai" && p.APIVersion != "" {
			path += "?api-version=" + p.APIVersion
		}
		return path, nil
	}

	suffix := info.ChatPath
	if p.Type == "azure-openai" && p.APIVersion != "" {
		suffix += "?api-version=" + p.APIVersion
	}

	// Prepend base_url path component if present
	if p.BaseURL != "" {
		u, err := url.Parse(p.BaseURL)
		if err != nil {
			return "", fmt.Errorf("cannot parse base_url %q: %w", p.BaseURL, err)
		}
		if u.Path != "" && u.Path != "/" {
			return strings.TrimRight(u.Path, "/") + suffix, nil
		}
	}

	return suffix, nil
}
