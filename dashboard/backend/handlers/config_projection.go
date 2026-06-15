package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"
)

// ConfigDeploymentsHandler lists persisted deployment projections.
// GET /api/router/config/deployments
func ConfigDeploymentsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if configProjectionStore == nil {
			http.Error(w, "Config projection store is not initialized", http.StatusServiceUnavailable)
			return
		}

		deployments, err := configProjectionStore.ListDeployments()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(deployments); err != nil {
			log.Printf("Error encoding config deployments: %v", err)
		}
	}
}

// ConfigDeploymentDetailHandler returns one deployment projection by version.
// GET /api/router/config/deployments/{version}
func ConfigDeploymentDetailHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if configProjectionStore == nil {
			http.Error(w, "Config projection store is not initialized", http.StatusServiceUnavailable)
			return
		}

		version := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/router/config/deployments/"), "/")
		if version == "" {
			http.Error(w, "version is required", http.StatusBadRequest)
			return
		}

		deployment, err := configProjectionStore.GetDeployment(version)
		if err != nil {
			if strings.Contains(err.Error(), "not found") {
				http.Error(w, err.Error(), http.StatusNotFound)
				return
			}
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(deployment); err != nil {
			log.Printf("Error encoding config deployment detail: %v", err)
		}
	}
}

// ActiveConfigProjectionHandler returns the active deployment projection.
// GET /api/router/config/active-projection
func ActiveConfigProjectionHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if configProjectionStore == nil {
			http.Error(w, "Config projection store is not initialized", http.StatusServiceUnavailable)
			return
		}

		projection, err := configProjectionStore.GetActiveProjection()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(projection); err != nil {
			log.Printf("Error encoding active config projection: %v", err)
		}
	}
}
