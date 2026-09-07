package handlers

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

const fallbackRouterEvalEndpoint = "http://localhost:8080/api/v1/eval"

func defaultRouterEvalEndpoint(routerAPIURL string) string {
	if routerAPIURL == "" {
		return fallbackRouterEvalEndpoint
	}
	return strings.TrimSuffix(routerAPIURL, "/") + "/api/v1/eval"
}

// SettingsResponse represents the dashboard settings returned to frontend
type SettingsResponse struct {
	ReadonlyMode                bool   `json:"readonlyMode"`
	ServerReadonly              bool   `json:"serverReadonly"`
	RuntimeConfigWritable       bool   `json:"runtimeConfigWritable"`
	RecipeStoreWritable         bool   `json:"recipeStoreWritable"`
	SetupMode                   bool   `json:"setupMode"`
	Platform                    string `json:"platform"`
	EnvoyURL                    string `json:"envoyUrl"` // Envoy proxy URL for evaluation endpoint
	RouterEvalURL               string `json:"routerEvalEndpoint"`
	EvaluationAvailable         bool   `json:"evaluationAvailable"`
	EvaluationUnavailableReason string `json:"evaluationUnavailableReason"`
}

// SettingsHandler returns dashboard settings for frontend consumption.
//
// SetupMode comes from setupResolver, not cfg. cfg.SetupMode is the legacy flag
// frozen at startup, and this endpoint must agree with /api/setup/state and the
// bootstrap gate.
func SettingsHandler(cfg *config.Config, setupResolver *setupmode.Resolver) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		readOnlyMode := cfg.ReadonlyMode || !cfg.RuntimeConfigWritable
		if !readOnlyMode {
			if ac, ok := auth.AuthFromContext(r); ok && !ac.Perms[auth.PermConfigWrite] {
				readOnlyMode = true
			}
		}

		response := SettingsResponse{
			ReadonlyMode:                readOnlyMode,
			ServerReadonly:              cfg.ReadonlyMode,
			RuntimeConfigWritable:       cfg.RuntimeConfigWritable,
			RecipeStoreWritable:         cfg.RecipeStoreWritable,
			SetupMode:                   setupResolver.Active(),
			Platform:                    cfg.Platform,
			EnvoyURL:                    cfg.EnvoyURL,
			RouterEvalURL:               defaultRouterEvalEndpoint(cfg.RouterAPIURL),
			EvaluationAvailable:         cfg.EvaluationAvailable,
			EvaluationUnavailableReason: cfg.EvaluationUnavailableReason,
		}

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		}
	}
}
