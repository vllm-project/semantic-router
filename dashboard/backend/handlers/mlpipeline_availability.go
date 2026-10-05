package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

type mlPipelineAvailabilityResponse struct {
	MLPipelineAvailable         bool   `json:"mlPipelineAvailable"`
	MLPipelineUnavailableReason string `json:"mlPipelineUnavailableReason"`
}

// MLPipelineAvailabilityHandler reports the ML pipeline availability flag on the
// ML-authorized surface. General settings require config.read, which can be
// granted independently of mlpipeline.manage, so the ML Setup gate needs a
// source it can read with its own permission.
func MLPipelineAvailabilityHandler(cfg *config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(mlPipelineAvailabilityResponse{
			MLPipelineAvailable:         cfg.MLPipelineAvailable,
			MLPipelineUnavailableReason: cfg.MLPipelineUnavailableReason,
		})
	}
}
