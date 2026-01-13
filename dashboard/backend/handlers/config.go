package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ConfigHandler reads and serves the config.yaml file as JSON
func ConfigHandler(configPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		data, err := os.ReadFile(configPath)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to read config: %v", err), http.StatusInternalServerError)
			return
		}

		var config interface{}
		if err := yaml.Unmarshal(data, &config); err != nil {
			http.Error(w, fmt.Sprintf("Failed to parse config: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(config); err != nil {
			log.Printf("Error encoding config to JSON: %v", err)
		}
	}
}

// UpdateConfigHandler updates the config.yaml file with validation
func UpdateConfigHandler(configPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost && r.Method != http.MethodPut {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var configData map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&configData); err != nil {
			http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
			return
		}

		// Read existing config and merge with updates
		existingData, err := os.ReadFile(configPath)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to read existing config: %v", err), http.StatusInternalServerError)
			return
		}

		existingMap := make(map[string]interface{})
		if err = yaml.Unmarshal(existingData, &existingMap); err != nil {
			http.Error(w, fmt.Sprintf("Failed to parse existing config: %v", err), http.StatusInternalServerError)
			return
		}

		// Store original key count for validation
		originalKeyCount := len(existingMap)

		// Merge updates into existing config (deep merge for nested maps)
		for key, value := range configData {
			if existingValue, exists := existingMap[key]; exists {
				// If both are maps, merge them recursively
				if existingMapValue, ok := existingValue.(map[string]interface{}); ok {
					if newMapValue, ok := value.(map[string]interface{}); ok {
						// Deep merge nested maps
						mergedMap := make(map[string]interface{})
						// Copy existing values
						for k, v := range existingMapValue {
							mergedMap[k] = v
						}
						// Override with new values
						for k, v := range newMapValue {
							mergedMap[k] = v
						}
						existingMap[key] = mergedMap
						continue
					}
				}
			}
			// For non-map values or new keys, just set the value
			existingMap[key] = value
		}

		// Safety check: merged config should have at least as many keys as original
		// (it might have more if new keys were added, but should never have fewer)
		if len(existingMap) < originalKeyCount {
			http.Error(w, fmt.Sprintf("Merge would result in data loss: original had %d keys, merged has %d keys. This indicates a bug. File: %s", originalKeyCount, len(existingMap), configPath), http.StatusInternalServerError)
			return
		}

		// Convert merged config to YAML
		yamlData, err := yaml.Marshal(existingMap)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to convert to YAML: %v", err), http.StatusInternalServerError)
			return
		}

		// Validate using router's config parser
		tempFile := filepath.Join(os.TempDir(), "config_validate.yaml")
		if err = os.WriteFile(tempFile, yamlData, 0o644); err != nil {
			http.Error(w, fmt.Sprintf("Failed to validate: %v", err), http.StatusInternalServerError)
			return
		}
		defer func() {
			if removeErr := os.Remove(tempFile); removeErr != nil {
				log.Printf("Warning: failed to remove temp file: %v", removeErr)
			}
		}()

		parsedConfig, err := routerconfig.Parse(tempFile)
		if err != nil {
			http.Error(w, fmt.Sprintf("Config validation failed: %v", err), http.StatusBadRequest)
			return
		}

		// Explicitly validate vLLM endpoints (Parse doesn't validate endpoints by default)
		if len(parsedConfig.VLLMEndpoints) > 0 {
			for _, endpoint := range parsedConfig.VLLMEndpoints {
				if err := validateEndpointAddress(endpoint.Address); err != nil {
					http.Error(w, fmt.Sprintf("Config validation failed: vLLM endpoint '%s' address validation failed: %v\n\nSupported formats:\n- IPv4: 192.168.1.1, 127.0.0.1\n- IPv6: ::1, 2001:db8::1\n- DNS names: localhost, example.com, api.example.com\n\nUnsupported formats:\n- Protocol prefixes: http://, https://\n- Paths: /api/v1, /health\n- Ports in address: use 'port' field instead", endpoint.Name, err), http.StatusBadRequest)
					return
				}
			}
		}

		if err := os.WriteFile(configPath, yamlData, 0o644); err != nil {
			http.Error(w, fmt.Sprintf("Failed to write config: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]string{"status": "success"}); err != nil {
			log.Printf("Error encoding response: %v", err)
		}
	}
}

// RouterDefaultsHandler reads and serves the router-defaults.yaml file as JSON
// This file is located in .vllm-sr/router-defaults.yaml relative to config directory
func RouterDefaultsHandler(configDir string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// router-defaults.yaml is in .vllm-sr directory relative to config
		routerDefaultsPath := filepath.Join(configDir, ".vllm-sr", "router-defaults.yaml")

		data, err := os.ReadFile(routerDefaultsPath)
		if err != nil {
			// If file doesn't exist, return empty config
			if os.IsNotExist(err) {
				w.Header().Set("Content-Type", "application/json")
				if encErr := json.NewEncoder(w).Encode(map[string]interface{}{}); encErr != nil {
					log.Printf("Error encoding empty response: %v", encErr)
				}
				return
			}
			http.Error(w, fmt.Sprintf("Failed to read router-defaults: %v", err), http.StatusInternalServerError)
			return
		}

		var config interface{}
		if err := yaml.Unmarshal(data, &config); err != nil {
			http.Error(w, fmt.Sprintf("Failed to parse router-defaults: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(config); err != nil {
			log.Printf("Error encoding router-defaults to JSON: %v", err)
		}
	}
}

// UpdateRouterDefaultsHandler updates the router-defaults.yaml file
func UpdateRouterDefaultsHandler(configDir string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost && r.Method != http.MethodPut {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var configData map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&configData); err != nil {
			http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
			return
		}

		routerDefaultsPath := filepath.Join(configDir, ".vllm-sr", "router-defaults.yaml")

		// Read existing config and merge with updates
		existingMap := make(map[string]interface{})
		existingData, err := os.ReadFile(routerDefaultsPath)
		if err == nil {
			if unmarshalErr := yaml.Unmarshal(existingData, &existingMap); unmarshalErr != nil {
				log.Printf("Warning: failed to parse existing router-defaults, starting fresh: %v", unmarshalErr)
			}
		}

		// Merge updates into existing config (deep merge for nested maps)
		for key, value := range configData {
			if existingValue, exists := existingMap[key]; exists {
				if existingMapValue, ok := existingValue.(map[string]interface{}); ok {
					if newMapValue, ok := value.(map[string]interface{}); ok {
						mergedMap := make(map[string]interface{})
						for k, v := range existingMapValue {
							mergedMap[k] = v
						}
						for k, v := range newMapValue {
							mergedMap[k] = v
						}
						existingMap[key] = mergedMap
						continue
					}
				}
			}
			existingMap[key] = value
		}

		// Convert to YAML
		yamlData, err := yaml.Marshal(existingMap)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to convert to YAML: %v", err), http.StatusInternalServerError)
			return
		}

		// Ensure .vllm-sr directory exists
		vllmSrDir := filepath.Join(configDir, ".vllm-sr")
		if mkdirErr := os.MkdirAll(vllmSrDir, 0o755); mkdirErr != nil {
			http.Error(w, fmt.Sprintf("Failed to create .vllm-sr directory: %v", mkdirErr), http.StatusInternalServerError)
			return
		}

		if err := os.WriteFile(routerDefaultsPath, yamlData, 0o644); err != nil {
			http.Error(w, fmt.Sprintf("Failed to write router-defaults: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]string{"status": "success"}); err != nil {
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
