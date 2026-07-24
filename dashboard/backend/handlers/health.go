package handlers

import (
	"encoding/json"
	"net/http"
)

type healthCheckResponse struct {
	Status          string                 `json:"status"`
	Service         string                 `json:"service"`
	ProjectionDrift *ConfigProjectionDrift `json:"projection_drift,omitempty"`
}

// HealthCheck handles health check endpoint
func HealthCheck(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	resp := healthCheckResponse{
		Status:          "healthy",
		Service:         "semantic-router-dashboard",
		ProjectionDrift: currentConfigProjectionDrift(),
	}
	_ = json.NewEncoder(w).Encode(resp)
}
