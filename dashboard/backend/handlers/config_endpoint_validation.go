package handlers

import (
	"fmt"
	"net"
	"net/url"
	"strings"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// validateEndpointAddress validates that an endpoint address is in a valid format.
// It allows:
// - IPv4 addresses (e.g., "192.168.1.1", "127.0.0.1")
// - IPv6 addresses (e.g., "::1", "2001:db8::1")
// - DNS names (e.g., "localhost", "example.com", "api.example.com")
// It rejects:
// - Protocol prefixes (e.g., "http://", "https://")
// - Paths (e.g., "/api/v1", "/health")
// - Ports in the address field (should use the 'port' field instead)
func validateEndpointAddress(address string) error {
	if address == "" {
		return fmt.Errorf("address cannot be empty")
	}

	// Reject protocol prefixes
	if strings.HasPrefix(address, "http://") || strings.HasPrefix(address, "https://") {
		return fmt.Errorf("protocol prefix not allowed in address (use 'port' field for port number)")
	}

	// Reject paths (contains '/')
	if strings.Contains(address, "/") {
		return fmt.Errorf("paths not allowed in address field")
	}

	// Reject ports (contains ':')
	// Note: IPv6 addresses contain ':' but we check for ':' that's not part of IPv6 format
	if strings.Contains(address, ":") {
		// Check if it's a valid IPv6 address (contains multiple colons or starts with '[')
		if net.ParseIP(address) == nil {
			// If it's not a valid IP, it might be an address with a port
			// Check if it looks like "host:port" format
			parts := strings.Split(address, ":")
			if len(parts) == 2 {
				// Could be IPv4:port or hostname:port
				// Try to parse the second part as a port number
				if len(parts[1]) > 0 && len(parts[1]) <= 5 {
					// Likely a port number, reject it
					return fmt.Errorf("port not allowed in address field (use 'port' field instead)")
				}
			}
		}
	}

	// Try to parse as IP address
	ip := net.ParseIP(address)
	if ip != nil {
		// Valid IP address
		return nil
	}

	// If not an IP, check if it's a valid DNS name
	// Basic DNS name validation: alphanumeric, dots, hyphens
	if len(address) > 253 {
		return fmt.Errorf("DNS name too long (max 253 characters)")
	}

	// Check for valid DNS name characters
	for _, char := range address {
		if (char < 'a' || char > 'z') &&
			(char < 'A' || char > 'Z') &&
			(char < '0' || char > '9') &&
			char != '.' && char != '-' {
			return fmt.Errorf("invalid character in DNS name: %c", char)
		}
	}

	// Basic DNS name format check
	if strings.HasPrefix(address, ".") || strings.HasSuffix(address, ".") ||
		strings.Contains(address, "..") {
		return fmt.Errorf("invalid DNS name format")
	}

	return nil
}

// validateCanonicalEndpointRefs checks each backend endpoint's shape as the
// Router reads it: host:port, path optional, with the backend's protocol, or a
// full http(s) URL, path included. A key never travels in the URL.
func validateCanonicalEndpointRefs(configData routerconfig.CanonicalConfig) error {
	for modelIndex, model := range configData.Providers.Models {
		for backendIndex, backend := range model.BackendRefs {
			endpoint := strings.TrimSpace(backend.Endpoint)
			if endpoint == "" || !strings.Contains(endpoint, "://") {
				continue
			}
			field := fmt.Sprintf("providers.models[%d].backend_refs[%d].endpoint %q", modelIndex, backendIndex, endpoint)
			parsed, err := url.Parse(endpoint)
			if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
				return fmt.Errorf("%s must be an http or https URL", field)
			}
			if parsed.User != nil {
				return fmt.Errorf("%s must not carry credentials; the key comes from auth.api_key_env", field)
			}
		}
	}

	return nil
}

func currentGlobalDefaults(configPath string) (*routerconfig.CanonicalGlobal, error) {
	defaults := routerconfig.DefaultCanonicalGlobal()
	configData, err := readPersistedDashboardConfig(configPath)
	if err != nil {
		return &defaults, nil
	}

	parsed, err := routerconfig.ParseYAMLBytesWithoutEnvExpansion(configData)
	if err != nil {
		return &defaults, nil
	}

	global := routerconfig.CanonicalGlobalFromRouterConfig(parsed)
	if global == nil {
		return &defaults, nil
	}

	return global, nil
}
