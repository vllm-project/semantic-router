package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ConfigHandler reads and serves the config as JSON from the local config file.
func ConfigHandler(configPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")

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

		if err := json.NewEncoder(w).Encode(config); err != nil {
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

		w.Write(data)
	}
}

// UpdateConfigHandler updates the config.yaml file with validation.
// After writing, it triggers regeneration of the Router's flattened config
// (router-config.yaml) so the Router picks up changes via fsnotify.
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
			if err := json.NewEncoder(w).Encode(map[string]string{
				"error":   "readonly_mode",
				"message": "Dashboard is in read-only mode. Configuration editing is disabled.",
			}); err != nil {
				log.Printf("Error encoding readonly response: %v", err)
			}
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

		// Merge updates into existing config (recursive deep merge for nested maps)
		existingMap = deepMerge(existingMap, configData)

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

		// Trigger async regeneration of router-config.yaml so the Router
		// picks up the change via fsnotify. This calls the Python CLI's
		// generate_router_config() which converts nested user format
		// (providers.models, signals.keywords) to flat Router format
		// (vllm_endpoints, keyword_rules) and writes to .vllm-sr/router-config.yaml.
		go regenerateRouterConfig(configPath, configDir)

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
			if err := json.NewEncoder(w).Encode(map[string]string{
				"error":   "readonly_mode",
				"message": "Dashboard is in read-only mode. Configuration editing is disabled.",
			}); err != nil {
				log.Printf("Error encoding readonly response: %v", err)
			}
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

// regenerateRouterConfig calls the Python CLI to regenerate .vllm-sr/router-config.yaml
// from the user-facing config.yaml. This bridges the gap between the Dashboard
// (which edits the nested Python CLI format) and the Router (which reads the flat format).
// The Router's fsnotify watcher will detect the change and hot-reload automatically.
//
// The function is designed to be called asynchronously (via goroutine) so it does
// not block the HTTP response. Errors are logged but not propagated to the caller.
func regenerateRouterConfig(configPath string, configDir string) {
	outputDir := filepath.Join(configDir, ".vllm-sr")

	// Check if output directory exists; if not, this is likely a dev environment
	// without the Python CLI setup, so skip silently.
	if _, err := os.Stat(outputDir); os.IsNotExist(err) {
		log.Printf("Config propagation: .vllm-sr directory not found at %s, skipping router config regeneration (dev mode?)", outputDir)
		return
	}

	// Python script that calls generate_router_config from the CLI
	pythonScript := fmt.Sprintf(`
import sys
sys.path.insert(0, '/app')
try:
    from cli.commands.serve import generate_router_config
    result = generate_router_config(%q, %q, force=True)
    print(f"Regenerated router config: {result}")
except ImportError:
    # Python CLI not available (e.g. dev environment without vllm-sr)
    print("SKIP: Python CLI not available, skipping router config regeneration")
except Exception as e:
    print(f"ERROR: Failed to regenerate router config: {e}", file=sys.stderr)
    sys.exit(1)
`, configPath, outputDir)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "python3", "-c", pythonScript)
	cmd.Dir = configDir
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("Config propagation: failed to regenerate router config: %v\nOutput: %s", err, string(output))
		return
	}

	log.Printf("Config propagation: %s", strings.TrimSpace(string(output)))
}
