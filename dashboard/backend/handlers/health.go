package handlers

import (
	"net/http"
)

// HealthCheck handles health check endpoint
func HealthCheck(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"status":"healthy","service":"semantic-router-dashboard"}`))
}
