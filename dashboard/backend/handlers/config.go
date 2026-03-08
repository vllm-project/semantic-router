package handlers

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/configlifecycle"
)

// ConfigHandler reads and serves the config as JSON from the local config file.
func ConfigHandler(configPath string) http.HandlerFunc {
	return ConfigHandlerWithService(configlifecycle.New(configPath, ""))
}

func ConfigHandlerWithService(service *configlifecycle.Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		config, err := service.ConfigJSON()
		if err != nil {
			writeLifecycleError(w, err)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
		if err := json.NewEncoder(w).Encode(config); err != nil {
			log.Printf("Error encoding config to JSON: %v", err)
		}
	}
}

// ConfigYAMLHandler reads and serves the config as raw YAML text.
func ConfigYAMLHandler(configPath string) http.HandlerFunc {
	return ConfigYAMLHandlerWithService(configlifecycle.New(configPath, ""))
}

func ConfigYAMLHandlerWithService(service *configlifecycle.Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		data, err := service.ConfigYAML()
		if err != nil {
			writeLifecycleError(w, err)
			return
		}

		w.Header().Set("Content-Type", "text/yaml; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
		_, _ = w.Write(data)
	}
}

// UpdateConfigHandler updates the config.yaml file with validation.
func UpdateConfigHandler(configPath string, readonlyMode bool, configDir string) http.HandlerFunc {
	return UpdateConfigHandlerWithService(configlifecycle.New(configPath, configDir), readonlyMode)
}

func UpdateConfigHandlerWithService(service *configlifecycle.Service, readonlyMode bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost && r.Method != http.MethodPut {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if readonlyMode {
			writeReadonlyResponse(w, "Dashboard is in read-only mode. Configuration editing is disabled.")
			return
		}

		var configData map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&configData); err != nil {
			http.Error(w, "Invalid request body: "+err.Error(), http.StatusBadRequest)
			return
		}
		if err := service.UpdateConfig(configData); err != nil {
			writeLifecycleError(w, err)
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
	return RouterDefaultsHandlerWithService(configlifecycle.New("", configDir))
}

func RouterDefaultsHandlerWithService(service *configlifecycle.Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		config, err := service.RouterDefaults()
		if err != nil {
			writeLifecycleError(w, err)
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
	return UpdateRouterDefaultsHandlerWithService(configlifecycle.New("", configDir), readonlyMode)
}

func UpdateRouterDefaultsHandlerWithService(service *configlifecycle.Service, readonlyMode bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost && r.Method != http.MethodPut {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if readonlyMode {
			writeReadonlyResponse(w, "Dashboard is in read-only mode. Configuration editing is disabled.")
			return
		}

		var configData map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&configData); err != nil {
			http.Error(w, "Invalid request body: "+err.Error(), http.StatusBadRequest)
			return
		}
		if err := service.UpdateRouterDefaults(configData); err != nil {
			writeLifecycleError(w, err)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]string{"status": "success"}); err != nil {
			log.Printf("Error encoding response: %v", err)
		}
	}
}
