package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/configprojection"
)

// ConfigDeploymentsHandler lists persisted deployment projections.
// GET /api/router/config/deployments
func ConfigDeploymentsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if configProjectionStore == nil {
			http.Error(w, "Config projection store is not initialized", http.StatusServiceUnavailable)
			return
		}

		deployments, err := configProjectionStore.ListDeployments()
		if err != nil {
			log.Printf("failed to list deployments: %v", err)
			http.Error(w, "failed to list deployments", http.StatusInternalServerError)
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
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
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
				http.Error(w, "deployment not found", http.StatusNotFound)
				return
			}
			log.Printf("failed to get deployment: %v", err)
			http.Error(w, "failed to get deployment", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := writeConfigDeploymentView(w, r, deployment); err != nil {
			log.Printf("Error encoding config deployment detail: %v", err)
		}
	}
}

// ActiveConfigProjectionHandler returns the active deployment projection.
// GET /api/router/config/active-projection
func ActiveConfigProjectionHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if configProjectionStore == nil {
			http.Error(w, "Config projection store is not initialized", http.StatusServiceUnavailable)
			return
		}

		projection, err := configProjectionStore.GetActiveProjection()
		if err != nil {
			log.Printf("failed to get active projection: %v", err)
			http.Error(w, "failed to get active projection", http.StatusInternalServerError)
			return
		}

		if !callerCanViewConfigSecrets(r) {
			projection.LastError = ""
			if projection.Deployment != nil {
				projection.Deployment.DSLSnapshot = ""
				projection.Deployment.Validation.Diagnostics = nil
			}
		}
		payload, err := json.Marshal(projection)
		if err == nil {
			err = writeConfigViewPayloadJSON(w, r, payload)
		}
		if err != nil {
			log.Printf("Error encoding active config projection: %v", err)
		}
	}
}

func writeConfigDeploymentView(w http.ResponseWriter, r *http.Request, deployment *configprojection.DeploymentRecord) error {
	if !callerCanViewConfigSecrets(r) {
		deployment.DSLSnapshot = ""
		deployment.Validation.Diagnostics = nil
	}
	payload, err := json.Marshal(deployment)
	if err != nil {
		return err
	}
	return writeConfigViewPayloadJSON(w, r, payload)
}
