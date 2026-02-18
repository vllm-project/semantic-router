package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

// SettingsResponse represents the dashboard settings returned to frontend
type SettingsResponse struct {
	ReadonlyMode bool   `json:"readonlyMode"`
	Platform     string `json:"platform"`
	EnvoyURL     string `json:"envoyUrl"` // Envoy proxy URL for evaluation endpoint
}

// SettingsHandler returns dashboard settings for frontend consumption
func SettingsHandler(cfg *config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		response := SettingsResponse{
			ReadonlyMode: cfg.ReadonlyMode,
			Platform:     cfg.Platform,
			EnvoyURL:     cfg.EnvoyURL,
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		}
	}
}
