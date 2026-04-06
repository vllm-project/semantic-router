package handlers

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	routercontract "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routercontract"
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

		configData, err := decodeYAMLTaggedBody[routercontract.CanonicalConfig](r.Body)
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

		if _, err := routercontract.ParseYAMLBytes(yamlData); err != nil {
			http.Error(w, fmt.Sprintf("Config validation failed: %v", err), http.StatusBadRequest)
			return
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

		if _, err := routercontract.ParseYAMLBytes(yamlData); err != nil {
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

func validateCanonicalEndpointRefs(configData routercontract.CanonicalConfig) error {
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

func currentGlobalDefaults(configDir string) (*routercontract.CanonicalGlobal, error) {
	defaults := routercontract.DefaultCanonicalGlobal()
	configPath := filepath.Join(configDir, "config.yaml")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return &defaults, nil
	}

	parsed, err := routercontract.ParseYAMLBytes(configData)
	if err != nil {
		return &defaults, nil
	}
	if parsed.Global == nil {
		return &defaults, nil
	}

	return parsed.Global, nil
}
