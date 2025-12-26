package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var (
	protocolRegex = regexp.MustCompile(`^https?://`)
	pathRegex     = regexp.MustCompile(`/`)
	ipv4PortRegex = regexp.MustCompile(`^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+$`)
	ipv6PortRegex = regexp.MustCompile(`^\[.*\]:\d+$`)
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

		var existingMap map[string]interface{}
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

		// Create backup before making changes
		backupPath := configPath + ".backup"
		if err = os.WriteFile(backupPath, existingData, 0o644); err != nil {
			log.Printf("Warning: failed to create backup: %v", err)
			// Continue anyway, but log the warning
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

		// Parse and validate the config
		parsedConfig, err := routerconfig.Parse(tempFile)
		if err != nil {
			log.Printf("Config validation failed: %v", err)
			http.Error(w, fmt.Sprintf("Config validation failed: %v", err), http.StatusBadRequest)
			return
		}

		// Explicitly validate vLLM endpoints (Parse doesn't validate endpoints by default)
		if len(parsedConfig.VLLMEndpoints) > 0 {
			for _, endpoint := range parsedConfig.VLLMEndpoints {
				if err = validateEndpointAddress(endpoint.Address); err != nil {
					log.Printf("Config validation failed: invalid address in endpoint '%s': %s", endpoint.Name, endpoint.Address)
					http.Error(w, fmt.Sprintf("Config validation failed: vLLM endpoint '%s' address validation failed: %v\n\nSupported formats:\n- IPv4: 192.168.1.1, 127.0.0.1\n- IPv6: ::1, 2001:db8::1\n- DNS names: localhost, example.com, api.example.com\n\nUnsupported formats:\n- Protocol prefixes: http://, https://\n- Paths: /api/v1, /health\n- Ports in address: use 'port' field instead", endpoint.Name, err), http.StatusBadRequest)
					return
				}
			}
		}

		// Write the validated config
		if err = os.WriteFile(configPath, yamlData, 0o644); err != nil {
			// If write fails, try to restore from backup
			if backupData, backupErr := os.ReadFile(backupPath); backupErr == nil {
				if restoreErr := os.WriteFile(configPath, backupData, 0o644); restoreErr == nil {
					log.Printf("Restored config from backup after write failure")
				}
			}
			http.Error(w, fmt.Sprintf("Failed to write config: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err = json.NewEncoder(w).Encode(map[string]string{"status": "success"}); err != nil {
			log.Printf("Error encoding response: %v", err)
		}
	}
}

// validateEndpointAddress validates endpoint address format
// Supports both IP addresses (IPv4/IPv6) and DNS names (domain names, localhost, etc.)
func validateEndpointAddress(address string) error {
	trimmed := strings.TrimSpace(address)
	if trimmed == "" {
		return fmt.Errorf("address cannot be empty")
	}
	if protocolRegex.MatchString(trimmed) {
		return fmt.Errorf("protocol prefixes (http://, https://) are not supported, got: %s", address)
	}
	if pathRegex.MatchString(trimmed) {
		return fmt.Errorf("paths are not supported, got: %s", address)
	}
	if ipv4PortRegex.MatchString(trimmed) || ipv6PortRegex.MatchString(trimmed) {
		return fmt.Errorf("port numbers in address are not supported, use 'port' field instead, got: %s", address)
	}
	// Try to parse as IP address first
	ip := net.ParseIP(trimmed)
	if ip != nil {
		// Valid IP address
		return nil
	}
	// If not an IP, validate as DNS name (domain name)
	// Basic DNS name validation: alphanumeric, dots, hyphens, underscores
	// Must start and end with alphanumeric character
	if len(trimmed) > 253 {
		return fmt.Errorf("DNS name too long (max 253 characters), got: %s", address)
	}
	// Simple validation: check for valid DNS name characters
	// Allow: letters, numbers, dots, hyphens, underscores
	dnsRegex := regexp.MustCompile(`^[a-zA-Z0-9]([a-zA-Z0-9\-_\.]*[a-zA-Z0-9])?$`)
	if !dnsRegex.MatchString(trimmed) {
		return fmt.Errorf("invalid address format (must be IP address or valid DNS name), got: %s", address)
	}
	return nil
}
