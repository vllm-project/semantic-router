package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"

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

		yamlData, err := yaml.Marshal(configData)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to convert to YAML: %v", err), http.StatusInternalServerError)
			return
		}

		// Validate using router's config parser
		tempFile := filepath.Join(os.TempDir(), "config_validate.yaml")
		if err := os.WriteFile(tempFile, yamlData, 0o644); err != nil {
			http.Error(w, fmt.Sprintf("Failed to validate: %v", err), http.StatusInternalServerError)
			return
		}
		defer func() {
			if err := os.Remove(tempFile); err != nil {
				log.Printf("Warning: failed to remove temp file: %v", err)
			}
		}()

		if _, err := routerconfig.Parse(tempFile); err != nil {
			log.Printf("Config validation failed: %v", err)
			http.Error(w, fmt.Sprintf("Config validation failed: %v", err), http.StatusBadRequest)
			return
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
