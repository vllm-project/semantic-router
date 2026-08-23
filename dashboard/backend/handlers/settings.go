package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

// SettingsResponse represents the dashboard settings returned to frontend
type SettingsResponse struct {
	ReadonlyMode    bool   `json:"readonlyMode"`
	ServerReadonly  bool   `json:"serverReadonly"`
	Platform        string `json:"platform"`
	EnvoyURL        string `json:"envoyUrl"` // Envoy proxy URL for evaluation endpoint
	RouterPublicURL string `json:"routerPublicUrl"`
	RouterEvalURL   string `json:"routerEvalEndpoint"`
	FleetSimEnabled bool   `json:"fleetSimEnabled"`
}

// SettingsHandler returns dashboard settings for frontend consumption
func SettingsHandler(cfg *config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		readOnlyMode := cfg.ReadonlyMode
		if !readOnlyMode {
			if ac, ok := auth.AuthFromContext(r); ok && !ac.Perms[auth.PermConfigWrite] {
				readOnlyMode = true
			}
		}

		response := SettingsResponse{
			ReadonlyMode:    readOnlyMode,
			ServerReadonly:  cfg.ReadonlyMode,
			Platform:        cfg.Platform,
			EnvoyURL:        cfg.EnvoyURL,
			RouterPublicURL: cfg.RouterPublicURL,
			RouterEvalURL:   defaultRouterEvalEndpoint(cfg.RouterAPIURL),
			FleetSimEnabled: cfg.FleetSimURL != "",
		}

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		}
	}
}
