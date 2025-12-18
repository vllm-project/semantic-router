package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"

	yaml "gopkg.in/yaml.v3"
)

// ConfigHandler reads and serves the config.yaml file as JSON
func ConfigHandler(configPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Only allow GET requests
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Read the config file
		data, err := os.ReadFile(configPath)
		if err != nil {
			log.Printf("Error reading config file: %v", err)
			http.Error(w, fmt.Sprintf("Failed to read config file: %v", err), http.StatusInternalServerError)
			return
		}

		// Parse YAML
		var config interface{}
		if err := yaml.Unmarshal(data, &config); err != nil {
			log.Printf("Error parsing config YAML: %v", err)
			http.Error(w, fmt.Sprintf("Failed to parse config: %v", err), http.StatusInternalServerError)
			return
		}

		// Convert to JSON and send response
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(config); err != nil {
			log.Printf("Error encoding config to JSON: %v", err)
			http.Error(w, fmt.Sprintf("Failed to encode config: %v", err), http.StatusInternalServerError)
			return
		}
	}
}

// UpdateConfigHandler updates the config.yaml file
func UpdateConfigHandler(configPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Only allow POST/PUT requests
		if r.Method != http.MethodPost && r.Method != http.MethodPut {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Read the request body
		var configData map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&configData); err != nil {
			log.Printf("Error decoding request body: %v", err)
			http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
			return
		}

		// Convert to YAML
		yamlData, err := yaml.Marshal(configData)
		if err != nil {
			log.Printf("Error marshaling config to YAML: %v", err)
			http.Error(w, fmt.Sprintf("Failed to convert config to YAML: %v", err), http.StatusInternalServerError)
			return
		}

		// Write to file
		if err := os.WriteFile(configPath, yamlData, 0o644); err != nil {
			log.Printf("Error writing config file: %v", err)
			http.Error(w, fmt.Sprintf("Failed to write config file: %v", err), http.StatusInternalServerError)
			return
		}

		log.Printf("Configuration updated successfully")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if err := json.NewEncoder(w).Encode(map[string]string{"status": "success", "message": "Configuration updated successfully"}); err != nil {
			log.Printf("Error encoding response: %v", err)
		}
	}
}
