package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
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

		config, err := loadDashboardConfig(configPath)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to load config: %v", err), http.StatusInternalServerError)
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

		_, _ = w.Write(data)
	}
}

// UpdateConfigHandler updates the config.yaml file with validation.
// After writing, it synchronously propagates the change to the managed runtime
// so Router and Envoy pick up the new config before the API returns success.
func UpdateConfigHandler(configPath string, readonlyMode bool, configDir string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost && r.Method != http.MethodPut {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if !ensureReadonlyDisabled(w, readonlyMode) {
			return
		}

		configData, err := decodeConfigUpdateBody(r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		existingData, existingMap, err := loadRequiredYAMLMap(configPath)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		yamlData, err := mergeAndValidateConfigUpdate(configPath, existingMap, configData)
		if err != nil {
			http.Error(w, formatConfigUpdateError(err), http.StatusBadRequest)
			return
		}
		if err := writeConfigAtomically(configPath, yamlData); err != nil {
			http.Error(w, fmt.Sprintf("Failed to write config: %v", err), http.StatusInternalServerError)
			return
		}
		if err := propagateConfigToRuntime(configPath, configDir); err != nil {
			if restoreErr := restorePreviousRuntimeConfig(configPath, configDir, existingData); restoreErr != nil {
				http.Error(w, fmt.Sprintf("Failed to apply config to runtime: %v. Failed to restore previous config: %v", err, restoreErr), http.StatusInternalServerError)
				return
			}
			http.Error(w, fmt.Sprintf("Failed to apply config to runtime: %v. Previous config restored.", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]string{"status": "success"}); err != nil {
			log.Printf("Error encoding response: %v", err)
		}
	}
}

// RouterDefaultsHandler reads and serves the router-defaults.yaml file as JSON.
func RouterDefaultsHandler(configDir string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		routerDefaultsPath := filepath.Join(configDir, ".vllm-sr", "router-defaults.yaml")
		data, err := os.ReadFile(routerDefaultsPath)
		if err != nil {
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

// UpdateRouterDefaultsHandler updates the router-defaults.yaml file.
func UpdateRouterDefaultsHandler(configDir string, readonlyMode bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost && r.Method != http.MethodPut {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if !ensureReadonlyDisabled(w, readonlyMode) {
			return
		}

		configData, err := decodeConfigUpdateBody(r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		routerDefaultsPath := filepath.Join(configDir, ".vllm-sr", "router-defaults.yaml")
		yamlData, err := mergeRouterDefaultsConfig(loadOptionalYAMLMap(routerDefaultsPath), configData)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		vllmSrDir := filepath.Join(configDir, ".vllm-sr")
		if err := os.MkdirAll(vllmSrDir, 0o755); err != nil {
			http.Error(w, fmt.Sprintf("Failed to create .vllm-sr directory: %v", err), http.StatusInternalServerError)
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
