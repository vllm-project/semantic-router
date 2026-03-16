package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"

	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// GlobalConfigYAMLHandler returns the canonical config.yaml global override block
// as raw YAML. The response body contains only the contents that live under
// config.yaml `global:`, not the full config document.
func GlobalConfigYAMLHandler(configPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		globalYAML, err := readRawGlobalOverrideYAML(configPath)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to read global config: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "text/yaml; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
		_, _ = w.Write(globalYAML)
	}
}

// UpdateGlobalConfigYAMLHandler replaces the config.yaml global override block
// using a raw YAML payload that represents the contents nested under `global:`.
func UpdateGlobalConfigYAMLHandler(configPath string, readonlyMode bool, configDir string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost && r.Method != http.MethodPut {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

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

		existingData, err := os.ReadFile(configPath)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to read config: %v", err), http.StatusInternalServerError)
			return
		}

		rawBody, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to read request body: %v", err), http.StatusBadRequest)
			return
		}

		updatedYAML, err := replaceGlobalOverrideYAML(existingData, rawBody)
		if err != nil {
			http.Error(w, fmt.Sprintf("Global config validation failed: %v", err), http.StatusBadRequest)
			return
		}

		if err := writeConfigAtomically(configPath, updatedYAML); err != nil {
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

func readRawGlobalOverrideYAML(configPath string) ([]byte, error) {
	existingMap, err := readCanonicalConfigMap(configPath)
	if err != nil {
		return nil, err
	}

	globalValue, ok := existingMap["global"]
	if !ok || globalValue == nil {
		return []byte("{}\n"), nil
	}

	globalYAML, err := yaml.Marshal(globalValue)
	if err != nil {
		return nil, fmt.Errorf("marshal global override: %w", err)
	}
	return globalYAML, nil
}

func replaceGlobalOverrideYAML(existingData, rawGlobalYAML []byte) ([]byte, error) {
	existingMap := make(map[string]interface{})
	if err := yaml.Unmarshal(existingData, &existingMap); err != nil {
		return nil, fmt.Errorf("parse config.yaml: %w", err)
	}

	globalOverride, hasOverride, err := parseRawGlobalOverride(rawGlobalYAML)
	if err != nil {
		return nil, err
	}

	if hasOverride {
		existingMap["global"] = globalOverride
	} else {
		delete(existingMap, "global")
	}

	updatedYAML, err := yaml.Marshal(existingMap)
	if err != nil {
		return nil, fmt.Errorf("marshal updated config: %w", err)
	}

	if _, err := routerconfig.ParseYAMLBytes(updatedYAML); err != nil {
		return nil, err
	}

	return updatedYAML, nil
}

func parseRawGlobalOverride(raw []byte) (map[string]interface{}, bool, error) {
	if strings.TrimSpace(string(raw)) == "" {
		return nil, false, nil
	}

	var parsed interface{}
	if err := yaml.Unmarshal(raw, &parsed); err != nil {
		return nil, false, fmt.Errorf("invalid YAML: %w", err)
	}

	if parsed == nil {
		return nil, false, nil
	}

	globalOverride, ok := toStringKeyMap(parsed)
	if !ok {
		return nil, false, fmt.Errorf("global override must be a YAML mapping")
	}

	return globalOverride, true, nil
}

func readCanonicalConfigMap(configPath string) (map[string]interface{}, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, err
	}

	configMap := make(map[string]interface{})
	if err := yaml.Unmarshal(data, &configMap); err != nil {
		return nil, err
	}

	return configMap, nil
}
