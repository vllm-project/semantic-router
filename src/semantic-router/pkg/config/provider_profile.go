package config

import (
	"fmt"
	"net/url"
	"sort"
	"strings"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

// ValidProviderTypes is generated indirectly from the release-embedded model
// catalog. Native, compatible, and private-runtime providers share the same
// lookup contract; support tier remains visible in catalog presentation data.
func ValidProviderTypes() []string {
	registry, err := modelcatalog.BuiltIn()
	if err != nil {
		return nil
	}
	return registry.ProviderIDs()
}

// GetProviderProfileForEndpoint resolves the ProviderProfile for a named
// endpoint. Catalog-materialized v0.3 endpoints always carry a profile.
func (c *RouterConfig) GetProviderProfileForEndpoint(endpointName string) (*ProviderProfile, error) {
	if endpointName == "" {
		return nil, nil
	}
	ep, found := c.GetEndpointByName(endpointName)
	if !found {
		return nil, fmt.Errorf("endpoint %q not found in vllm_endpoints", endpointName)
	}
	if ep.ProviderProfileName == "" {
		return nil, nil
	}
	profile, ok := c.ProviderProfiles[ep.ProviderProfileName]
	if !ok {
		return nil, fmt.Errorf("endpoint %q references provider profile %q which does not exist (have: %v)", endpointName, ep.ProviderProfileName, providerProfileKeys(c.ProviderProfiles))
	}
	return &profile, nil
}

func providerProfileKeys(profiles map[string]ProviderProfile) []string {
	keys := make([]string, 0, len(profiles))
	for key := range profiles {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

// ResolveAddress returns the host:port resolved from a materialized provider
// profile, or from address/port for internal legacy runtime structures.
func (endpoint *VLLMEndpoint) ResolveAddress(profiles map[string]ProviderProfile) (string, error) {
	if endpoint.ProviderProfileName == "" {
		return fmt.Sprintf("%s:%d", endpoint.Address, endpoint.Port), nil
	}
	if profiles == nil {
		return "", fmt.Errorf("endpoint %q has provider profile %q but no provider_profiles map is defined", endpoint.Name, endpoint.ProviderProfileName)
	}
	profile, ok := profiles[endpoint.ProviderProfileName]
	if !ok {
		return "", fmt.Errorf("endpoint %q references provider profile %q which does not exist", endpoint.Name, endpoint.ProviderProfileName)
	}
	if strings.TrimSpace(profile.BaseURL) == "" {
		return "", fmt.Errorf("endpoint %q: provider profile %q has no base_url", endpoint.Name, endpoint.ProviderProfileName)
	}
	parsed, err := url.Parse(profile.BaseURL)
	if err != nil || parsed.Host == "" {
		return "", fmt.Errorf("endpoint %q has invalid provider base_url %q", endpoint.Name, profile.BaseURL)
	}
	host := parsed.Host
	if parsed.Port() == "" {
		switch parsed.Scheme {
		case "https":
			host += ":443"
		case "http":
			host += ":80"
		default:
			return "", fmt.Errorf("endpoint %q base_url %q has unsupported scheme %q", endpoint.Name, profile.BaseURL, parsed.Scheme)
		}
	}
	return host, nil
}

// ProviderType validates the catalog provider identity carried by a compiled
// profile. There is no separately maintained enum.
func (profile *ProviderProfile) ProviderType() (string, error) {
	definition, err := profile.catalogDefinition()
	if err != nil {
		return "", err
	}
	return definition.ID, nil
}

// ResolveAuthHeader combines catalog auth defaults with explicit profile
// overrides. An explicitly empty prefix is represented by the catalog value;
// Explicit v0.3 backend overrides take precedence over catalog defaults.
func (profile *ProviderProfile) ResolveAuthHeader() (string, string, error) {
	definition, err := profile.catalogDefinition()
	if err != nil {
		return "", "", err
	}
	header, prefix := definition.Auth.Header, definition.Auth.Prefix
	if profile.AuthHeader != "" {
		header = profile.AuthHeader
	}
	if profile.AuthPrefix != "" {
		prefix = profile.AuthPrefix
	}
	return header, prefix, nil
}

// ResolveReasoningTransport returns the catalog-owned request projection for
// reasoning controls. Compatible providers default to template kwargs; only
// providers that explicitly claim another wire shape opt into it.
func (profile *ProviderProfile) ResolveReasoningTransport() (modelcatalog.ReasoningTransport, error) {
	if profile != nil && profile.ReasoningTransport != "" {
		if !validReasoningTransport(profile.ReasoningTransport) {
			return "", fmt.Errorf("provider profile reasoning_transport %q is unsupported", profile.ReasoningTransport)
		}
		return profile.ReasoningTransport, nil
	}
	definition, err := profile.catalogDefinition()
	if err != nil {
		return "", err
	}
	if definition.ReasoningTransport == "" {
		return modelcatalog.ReasoningTransportChatTemplate, nil
	}
	return definition.ReasoningTransport, nil
}

func validReasoningTransport(transport modelcatalog.ReasoningTransport) bool {
	switch transport {
	case modelcatalog.ReasoningTransportChatTemplate,
		modelcatalog.ReasoningTransportTopLevelEffort,
		modelcatalog.ReasoningTransportTopLevelBoolean,
		modelcatalog.ReasoningTransportReasoningObject,
		modelcatalog.ReasoningTransportThinkingObject,
		modelcatalog.ReasoningTransportDeepSeekThinking:
		return true
	default:
		return false
	}
}

// ResolveCreatePath resolves the request-creation operation from the protocol
// registry, then applies provider-specific data overrides and a base URL path
// prefix. The selected protocol owns the operation path for Chat Completions,
// Responses, and Messages alike.
func (profile *ProviderProfile) ResolveCreatePath(protocolID string) (string, error) {
	definition, err := profile.catalogDefinition()
	if err != nil {
		return "", err
	}
	protocolID = profile.resolveProtocolID(protocolID, definition.DefaultProtocol)
	basePath, err := profile.operationBasePath(protocolID)
	if err != nil {
		return "", err
	}
	registry, err := modelcatalog.BuiltIn()
	if err != nil {
		return "", err
	}
	path, err := registry.ResolveOperationPath(definition.ID, protocolID, "create", basePath)
	if err != nil {
		return "", err
	}
	if profile.hasCustomChatPath(protocolID) {
		path = profile.ChatPath
	}
	return profile.appendAPIVersion(path, definition.APIVersionQuery), nil
}

func (profile *ProviderProfile) resolveProtocolID(requested, defaultProtocol string) string {
	if requested != "" {
		return requested
	}
	if profile.Protocol != "" {
		return profile.Protocol
	}
	return defaultProtocol
}

func (profile *ProviderProfile) operationBasePath(protocolID string) (string, error) {
	if profile.hasCustomChatPath(protocolID) || profile.BaseURL == "" {
		return "", nil
	}
	parsed, err := url.Parse(profile.BaseURL)
	if err != nil {
		return "", fmt.Errorf("cannot parse base_url %q: %w", profile.BaseURL, err)
	}
	return parsed.Path, nil
}

func (profile *ProviderProfile) hasCustomChatPath(protocolID string) bool {
	return profile.ChatPath != "" && protocolID == "openai/chat-completions@1"
}

func (profile *ProviderProfile) appendAPIVersion(path string, supported bool) string {
	if !supported || profile.APIVersion == "" {
		return path
	}
	separator := "?"
	if strings.Contains(path, "?") {
		separator = "&"
	}
	return path + separator + "api-version=" + url.QueryEscape(profile.APIVersion)
}

func (profile *ProviderProfile) catalogDefinition() (modelcatalog.ProviderDefinition, error) {
	if profile == nil {
		return modelcatalog.ProviderDefinition{}, fmt.Errorf("provider profile is nil")
	}
	if strings.TrimSpace(profile.Type) == "" {
		return modelcatalog.ProviderDefinition{}, fmt.Errorf("provider profile has empty type")
	}
	registry, err := modelcatalog.BuiltIn()
	if err != nil {
		return modelcatalog.ProviderDefinition{}, err
	}
	definition, ok := registry.Provider(profile.Type)
	if !ok {
		return modelcatalog.ProviderDefinition{}, fmt.Errorf("unknown provider ID %q (valid IDs: %v)", profile.Type, registry.ProviderIDs())
	}
	return definition, nil
}
