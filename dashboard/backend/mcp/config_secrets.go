package mcp

import (
	"fmt"
	"net/url"
	"strings"
)

// RedactedValue is the stable placeholder returned by the dashboard API for
// configured secrets. Clients may send the placeholder back during an update
// to retain the corresponding stored value.
const RedactedValue = "__VLLM_SR_REDACTED__"

// RedactedServerConfig returns a deep copy suitable for an API response. The
// manager-owned config is never modified, so active clients continue to use
// the original credentials.
func RedactedServerConfig(config *ServerConfig) *ServerConfig {
	if config == nil {
		return nil
	}

	redacted := cloneServerConfig(config)
	// Command arguments are an unstructured executable surface. A server may
	// carry a credential in any flag, header expression, environment assignment,
	// or positional value, so an allowlist of familiar credential flags cannot
	// make them safe for mcp.read responses. A single opaque marker tells editors
	// that arguments exist without exposing their count or structure.
	if len(redacted.Connection.Args) > 0 {
		redacted.Connection.Args = []string{RedactedValue}
	} else {
		redacted.Connection.Args = nil
	}
	redacted.Connection.Env = redactSecretMap(redacted.Connection.Env)
	redacted.Connection.Headers = redactSecretMap(redacted.Connection.Headers)
	redacted.Connection.URL = redactURLCredentials(redacted.Connection.URL)
	if redacted.Security != nil && redacted.Security.OAuth != nil {
		redacted.Security.OAuth.AuthorizationURL = redactURLCredentials(redacted.Security.OAuth.AuthorizationURL)
		redacted.Security.OAuth.TokenURL = redactURLCredentials(redacted.Security.OAuth.TokenURL)
		if redacted.Security.OAuth.ClientSecret != "" {
			redacted.Security.OAuth.ClientSecret = RedactedValue
		}
	}
	return redacted
}

// RedactedServerState returns a deep-copied API view of a runtime state.
func RedactedServerState(state *ServerState) *ServerState {
	if state == nil {
		return nil
	}

	redacted := *state
	redacted.Config = RedactedServerConfig(state.Config)
	redacted.Tools = append([]ToolDefinition(nil), state.Tools...)
	if state.Error != "" {
		redacted.Error = "Connection unavailable"
	}
	return &redacted
}

func cloneServerConfig(config *ServerConfig) *ServerConfig {
	if config == nil {
		return nil
	}

	cloned := *config
	cloned.Connection.Args = append([]string(nil), config.Connection.Args...)
	cloned.Connection.Env = cloneStringMap(config.Connection.Env)
	cloned.Connection.Headers = cloneStringMap(config.Connection.Headers)
	if config.Security != nil {
		security := *config.Security
		security.AllowedOrigins = append([]string(nil), config.Security.AllowedOrigins...)
		if config.Security.OAuth != nil {
			oauth := *config.Security.OAuth
			oauth.Scopes = append([]string(nil), config.Security.OAuth.Scopes...)
			security.OAuth = &oauth
		}
		cloned.Security = &security
	}
	if config.Options != nil {
		options := *config.Options
		cloned.Options = &options
	}
	return &cloned
}

func cloneStringMap(values map[string]string) map[string]string {
	if values == nil {
		return nil
	}
	cloned := make(map[string]string, len(values))
	for key, value := range values {
		cloned[key] = value
	}
	return cloned
}

func redactSecretMap(values map[string]string) map[string]string {
	redacted := cloneStringMap(values)
	for key := range redacted {
		redacted[key] = RedactedValue
	}
	return redacted
}

func redactURLCredentials(rawURL string) string {
	if rawURL == "" {
		return ""
	}
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return RedactedValue
	}
	if parsed.Host == "" || parsed.Opaque != "" {
		return RedactedValue
	}
	redacted := false
	if parsed.User != nil {
		parsed.User = url.UserPassword(RedactedValue, RedactedValue)
		redacted = true
	}
	// A path segment can itself be a bearer/capability token. Paths are not
	// structured enough to distinguish public routing labels from secrets, so
	// API views retain only the origin and whether a non-root path exists.
	if parsed.Path != "" && parsed.Path != "/" {
		parsed.Path = "/" + RedactedValue
		parsed.RawPath = ""
		redacted = true
	}
	if parsed.RawQuery != "" {
		// Query keys can themselves be opaque capability tokens, not only their
		// values. Hide the complete query while retaining an exact placeholder
		// that mergeRedactedURL can bind back to this stored endpoint.
		parsed.RawQuery = RedactedValue
		redacted = true
	}
	if parsed.Fragment != "" {
		parsed.Fragment = RedactedValue
		parsed.RawFragment = ""
		redacted = true
	}
	if !redacted {
		return rawURL
	}
	return parsed.String()
}

func mergeRedactedServerConfig(existing, updated *ServerConfig) (*ServerConfig, error) {
	if err := rejectSecretReuseAcrossEndpointChange(existing, updated); err != nil {
		return nil, err
	}
	merged := cloneServerConfig(updated)
	if merged == nil {
		return nil, fmt.Errorf("MCP server config is required")
	}

	var err error
	merged.Connection.Env, err = mergeSecretMap(existing.Connection.Env, updated.Connection.Env, "environment variable")
	if err != nil {
		return nil, err
	}
	merged.Connection.Headers, err = mergeSecretMap(existing.Connection.Headers, updated.Connection.Headers, "request header")
	if err != nil {
		return nil, err
	}
	merged.Connection.Args, err = mergeCredentialArguments(existing.Connection.Args, updated.Connection.Args)
	if err != nil {
		return nil, err
	}
	merged.Connection.URL, err = mergeRedactedURL(existing.Connection.URL, updated.Connection.URL)
	if err != nil {
		return nil, err
	}
	merged.Security, err = mergeSecurityConfig(existing.Security, updated.Security)
	if err != nil {
		return nil, err
	}
	return merged, nil
}

func rejectSecretReuseAcrossEndpointChange(existing, updated *ServerConfig) error {
	if existing == nil || updated == nil || !connectionEndpointChanged(existing, updated) {
		return nil
	}
	if len(existing.Connection.Args) > 0 && updated.Connection.Args == nil {
		return fmt.Errorf("command arguments must be replaced or cleared when the MCP endpoint changes")
	}
	if containsRedactedValue(updated.Connection.Args) {
		return fmt.Errorf("command argument placeholders cannot be reused for a different MCP endpoint")
	}
	if err := rejectMapSecretReuse(existing.Connection.Env, updated.Connection.Env, "environment variables"); err != nil {
		return err
	}
	if err := rejectMapSecretReuse(existing.Connection.Headers, updated.Connection.Headers, "request headers"); err != nil {
		return err
	}
	if strings.Contains(updated.Connection.URL, RedactedValue) {
		return fmt.Errorf("URL placeholders cannot be reused for a different MCP endpoint")
	}
	if updated.Security == nil {
		if securityContainsCredentials(existing.Security) {
			return fmt.Errorf("OAuth credentials must be replaced or cleared when the MCP endpoint changes")
		}
		return nil
	}
	if updated.Security.OAuth != nil && (updated.Security.OAuth.ClientSecret == RedactedValue ||
		strings.Contains(updated.Security.OAuth.AuthorizationURL, RedactedValue) ||
		strings.Contains(updated.Security.OAuth.TokenURL, RedactedValue)) {
		return fmt.Errorf("OAuth placeholders cannot be reused for a different MCP endpoint")
	}
	return nil
}

func connectionEndpointChanged(existing, updated *ServerConfig) bool {
	if existing.Transport != updated.Transport {
		return true
	}
	switch existing.Transport {
	case TransportStdio:
		return existing.Connection.Command != updated.Connection.Command ||
			existing.Connection.Cwd != updated.Connection.Cwd ||
			explicitCommandArgumentsChanged(existing.Connection.Args, updated.Connection.Args)
	case TransportStreamableHTTP:
		return !matchesStoredURL(existing.Connection.URL, updated.Connection.URL)
	default:
		return existing.Connection.Command != updated.Connection.Command ||
			existing.Connection.Cwd != updated.Connection.Cwd ||
			updated.Connection.URL != existing.Connection.URL
	}
}

func explicitCommandArgumentsChanged(existing, updated []string) bool {
	// A nil slice and the exact opaque marker both mean "preserve the stored
	// arguments". Any explicit argument list is part of a stdio endpoint's
	// identity: launchers such as npx and uvx select the actual server package
	// through their arguments.
	if updated == nil || isOpaqueArgumentMarker(updated) {
		return false
	}
	if len(existing) != len(updated) {
		return true
	}
	for index := range existing {
		if existing[index] != updated[index] {
			return true
		}
	}
	return false
}

func rejectMapSecretReuse(existing, updated map[string]string, fieldName string) error {
	if len(existing) > 0 && updated == nil {
		return fmt.Errorf("%s must be replaced or cleared when the MCP endpoint changes", fieldName)
	}
	for _, value := range updated {
		if value == RedactedValue || strings.Contains(value, RedactedValue) {
			return fmt.Errorf("%s placeholders cannot be reused for a different MCP endpoint", fieldName)
		}
	}
	return nil
}

func containsRedactedValue(values []string) bool {
	for _, value := range values {
		if strings.Contains(value, RedactedValue) {
			return true
		}
	}
	return false
}

func securityContainsCredentials(security *SecurityConfig) bool {
	if security == nil || security.OAuth == nil {
		return false
	}
	return security.OAuth.ClientSecret != "" ||
		security.OAuth.AuthorizationURL != "" ||
		security.OAuth.TokenURL != ""
}

// A nil map means that a client did not edit the field and preserves the
// existing values. A non-nil map is a complete replacement: omitted keys are
// deleted, an empty map clears the field, and non-placeholder values rotate a
// secret explicitly.
func mergeSecretMap(existing, updated map[string]string, fieldName string) (map[string]string, error) {
	if updated == nil {
		return cloneStringMap(existing), nil
	}

	merged := cloneStringMap(updated)
	for key, value := range merged {
		if value != RedactedValue {
			continue
		}
		existingValue, ok := existing[key]
		if !ok {
			return nil, fmt.Errorf("redacted %s %q has no stored value", fieldName, key)
		}
		merged[key] = existingValue
	}
	return merged, nil
}

func mergeRedactedURL(existing, updated string) (string, error) {
	if !strings.Contains(updated, RedactedValue) {
		return updated, nil
	}
	if updated == redactURLCredentials(existing) {
		return existing, nil
	}
	return "", fmt.Errorf("redacted MCP server URL does not match the stored URL")
}

func mergeSecurityConfig(existing, updated *SecurityConfig) (*SecurityConfig, error) {
	if updated == nil {
		if existing == nil {
			return nil, nil
		}
		return cloneServerConfig(&ServerConfig{Security: existing}).Security, nil
	}

	merged := cloneServerConfig(&ServerConfig{Security: updated}).Security
	if merged.OAuth == nil {
		return merged, nil
	}

	existingOAuth := &OAuthConfig{}
	if existing != nil && existing.OAuth != nil {
		existingOAuth = existing.OAuth
	}
	reusesClientSecret := merged.OAuth.ClientSecret == RedactedValue
	if reusesClientSecret && (existing == nil || existing.OAuth == nil) {
		return nil, fmt.Errorf("redacted OAuth client secret has no stored value")
	}
	if reusesClientSecret && (merged.OAuth.ClientID != existingOAuth.ClientID ||
		!matchesStoredURL(existingOAuth.TokenURL, merged.OAuth.TokenURL)) {
		return nil, fmt.Errorf("OAuth client secret cannot be reused for a different client or token endpoint")
	}
	var err error
	merged.OAuth.AuthorizationURL, err = mergeRedactedURL(
		existingOAuth.AuthorizationURL,
		merged.OAuth.AuthorizationURL,
	)
	if err != nil {
		return nil, fmt.Errorf("merge OAuth authorization URL: %w", err)
	}
	merged.OAuth.TokenURL, err = mergeRedactedURL(existingOAuth.TokenURL, merged.OAuth.TokenURL)
	if err != nil {
		return nil, fmt.Errorf("merge OAuth token URL: %w", err)
	}
	if !reusesClientSecret {
		return merged, nil
	}
	merged.OAuth.ClientSecret = existing.OAuth.ClientSecret
	return merged, nil
}

func matchesStoredURL(existing, updated string) bool {
	return updated == existing || updated == redactURLCredentials(existing)
}

func mergeCredentialArguments(existing, updated []string) ([]string, error) {
	if updated == nil {
		return append([]string(nil), existing...), nil
	}
	if isOpaqueArgumentMarker(updated) {
		if len(existing) == 0 {
			return nil, fmt.Errorf("redacted command arguments have no stored value")
		}
		return append([]string(nil), existing...), nil
	}
	for _, argument := range updated {
		if strings.Contains(argument, RedactedValue) {
			return nil, fmt.Errorf("redacted command arguments must use the exact opaque marker")
		}
	}
	return append([]string(nil), updated...), nil
}

func isOpaqueArgumentMarker(arguments []string) bool {
	return len(arguments) == 1 && arguments[0] == RedactedValue
}
