package handlers

import (
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ConfigHandler reads and serves the config as JSON from the local config file.
func ConfigHandler(configPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")

		configData, err := readCanonicalConfigFile(configPath)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to read config: %v", err), http.StatusInternalServerError)
			return
		}

		if err := writeYAMLTaggedJSON(w, configData); err != nil {
			log.Printf("Error encoding config to JSON: %v", err)
		}
	}
}

// ConfigYAMLHandler reads and serves the config as raw YAML text.
// This is used by the DSL Builder to load the current router config
// and decompile it into DSL via WASM.
func ConfigYAMLHandler(configPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		w.Header().Set("Content-Type", "text/yaml; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")

		data, err := os.ReadFile(configPath)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to read config: %v", err), http.StatusInternalServerError)
			return
		}

		_, _ = w.Write(data)
	}
}

// UpdateConfigHandler replaces config.yaml with a validated full config payload.
// The dashboard editor always sends the whole canonical document, so this path
// intentionally does not deep-merge into any legacy on-disk layout.
// After writing, it synchronously propagates the change to the managed runtime
// so Router and Envoy pick up the new config before the API returns success.
func UpdateConfigHandler(configPath string, readonlyMode bool, configDir string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost && r.Method != http.MethodPut {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Check read-only mode
		if readonlyMode {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusForbidden)
			if err := writeYAMLTaggedJSON(w, map[string]string{
				"error":   "readonly_mode",
				"message": "Dashboard is in read-only mode. Configuration editing is disabled.",
			}); err != nil {
				log.Printf("Error encoding readonly response: %v", err)
			}
			return
		}

		configData, err := decodeYAMLTaggedBody[routerconfig.CanonicalConfig](r.Body)
		if err != nil {
			http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
			return
		}
		if validationErr := validateCanonicalEndpointRefs(configData); validationErr != nil {
			http.Error(w, fmt.Sprintf("Config validation failed: %v", validationErr), http.StatusBadRequest)
			return
		}

		// Read existing config so runtime rollback can restore the previous file if needed.
		existingData, err := os.ReadFile(configPath)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to read existing config: %v", err), http.StatusInternalServerError)
			return
		}

		yamlData, err := marshalYAMLBytes(configData)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to convert to YAML: %v", err), http.StatusInternalServerError)
			return
		}

		parsedConfig, err := routerconfig.ParseYAMLBytes(yamlData)
		if err != nil {
			http.Error(w, fmt.Sprintf("Config validation failed: %v", err), http.StatusBadRequest)
			return
		}

		// Explicitly validate vLLM endpoints (Parse doesn't validate endpoints by default)
		if len(parsedConfig.VLLMEndpoints) > 0 {
			for _, endpoint := range parsedConfig.VLLMEndpoints {
				if endpoint.ProviderProfileName != "" && endpoint.Address == "" {
					continue
				}
				if err := validateEndpointAddress(endpoint.Address); err != nil {
					http.Error(w, fmt.Sprintf("Config validation failed: vLLM endpoint '%s' address validation failed: %v\n\nSupported formats:\n- IPv4: 192.168.1.1, 127.0.0.1\n- IPv6: ::1, 2001:db8::1\n- DNS names: localhost, example.com, api.example.com\n\nUnsupported formats:\n- Protocol prefixes: http://, https://\n- Paths: /api/v1, /health\n- Ports in address: use 'port' field instead", endpoint.Name, err), http.StatusBadRequest)
					return
				}
			}
		}

		if err := writeConfigAtomically(configPath, yamlData); err != nil {
			http.Error(w, fmt.Sprintf("Failed to write config: %v", err), http.StatusInternalServerError)
			return
		}

		if err := applyWrittenConfig(configPath, configDir, existingData, true); err != nil {
			http.Error(w, formatRuntimeApplyError("Failed to apply config to runtime", err), http.StatusInternalServerError)
			return
		}

		if err := writeYAMLTaggedJSON(w, map[string]string{"status": "success"}); err != nil {
			log.Printf("Error encoding response: %v", err)
		}
	}
}

// RouterDefaultsHandler returns effective canonical global config merged from
// router-owned defaults plus any current config.yaml global overrides.
func RouterDefaultsHandler(configDir string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		defaults, err := currentGlobalDefaults(configDir)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to load defaults: %v", err), http.StatusInternalServerError)
			return
		}

		if err := writeYAMLTaggedJSON(w, defaults); err != nil {
			log.Printf("Error encoding global defaults to JSON: %v", err)
		}
	}
}

// UpdateRouterDefaultsHandler updates the canonical global override block in config.yaml.
func UpdateRouterDefaultsHandler(configDir string, readonlyMode bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost && r.Method != http.MethodPut {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Check read-only mode
		if readonlyMode {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusForbidden)
			if err := writeYAMLTaggedJSON(w, map[string]string{
				"error":   "readonly_mode",
				"message": "Dashboard is in read-only mode. Configuration editing is disabled.",
			}); err != nil {
				log.Printf("Error encoding readonly response: %v", err)
			}
			return
		}

		rawPatch, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
			return
		}

		configPath := filepath.Join(configDir, "config.yaml")
		existingData, err := os.ReadFile(configPath)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to read config: %v", err), http.StatusInternalServerError)
			return
		}

		yamlData, err := mergeGlobalOverridePatchYAML(existingData, rawPatch)
		if err != nil {
			http.Error(w, fmt.Sprintf("Config validation failed: %v", err), http.StatusBadRequest)
			return
		}

		if _, err := routerconfig.ParseYAMLBytes(yamlData); err != nil {
			http.Error(w, fmt.Sprintf("Config validation failed: %v", err), http.StatusBadRequest)
			return
		}

		if err := writeConfigAtomically(configPath, yamlData); err != nil {
			http.Error(w, fmt.Sprintf("Failed to write config: %v", err), http.StatusInternalServerError)
			return
		}

		if err := applyWrittenConfig(configPath, configDir, existingData, false); err != nil {
			http.Error(w, formatRuntimeApplyError("Failed to apply config to runtime", err), http.StatusInternalServerError)
			return
		}

		if err := writeYAMLTaggedJSON(w, map[string]string{"status": "success"}); err != nil {
			log.Printf("Error encoding response: %v", err)
		}
	}
}

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

func validateCanonicalEndpointRefs(configData routerconfig.CanonicalConfig) error {
	for modelIndex, model := range configData.Providers.Models {
		for backendIndex, backend := range model.BackendRefs {
			endpoint := strings.TrimSpace(backend.Endpoint)
			if endpoint == "" {
				continue
			}
			if strings.Contains(endpoint, "://") {
				return fmt.Errorf("providers.models[%d].backend_refs[%d].endpoint %q must not include a protocol prefix", modelIndex, backendIndex, endpoint)
			}
			if strings.Contains(endpoint, "/") {
				return fmt.Errorf("providers.models[%d].backend_refs[%d].endpoint %q must not include a path", modelIndex, backendIndex, endpoint)
			}
		}
	}

	return nil
}

func currentGlobalDefaults(configDir string) (*routerconfig.CanonicalGlobal, error) {
	defaults := routerconfig.DefaultCanonicalGlobal()
	configPath := filepath.Join(configDir, "config.yaml")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return &defaults, nil
	}

	parsed, err := routerconfig.ParseYAMLBytes(configData)
	if err != nil {
		return &defaults, nil
	}

	global := routerconfig.CanonicalGlobalFromRouterConfig(parsed)
	if global == nil {
		return &defaults, nil
	}

	return global, nil
}
