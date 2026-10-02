package handlers

import (
	"encoding/json"
	"net/http"
)

// HealthResponse is the liveness body served at /healthz.
type HealthResponse struct {
	Status  string `json:"status"`
	Service string `json:"service"`
}

var healthBody, _ = json.Marshal(HealthResponse{Status: "healthy", Service: "semantic-router-dashboard"})

// HealthCheck handles health check endpoint
func HealthCheck(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(healthBody)
}
