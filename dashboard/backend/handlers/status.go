package handlers

import (
	"encoding/json"
	"net/http"
)

// ServiceStatus represents the status of a single service
type ServiceStatus struct {
	Name      string `json:"name"`
	Status    string `json:"status"`
	Healthy   bool   `json:"healthy"`
	Message   string `json:"message,omitempty"`
	Component string `json:"component,omitempty"`
}

// RouterRuntimeStatus captures router startup progress beyond process-level health.
type RouterRuntimeStatus struct {
	Phase            string   `json:"phase"`
	Ready            bool     `json:"ready"`
	Message          string   `json:"message,omitempty"`
	DownloadingModel string   `json:"downloading_model,omitempty"`
	PendingModels    []string `json:"pending_models,omitempty"`
	ReadyModels      int      `json:"ready_models,omitempty"`
	TotalModels      int      `json:"total_models,omitempty"`
}

// SystemStatus represents the overall system status
type SystemStatus struct {
	Overall        string               `json:"overall"`
	DeploymentType string               `json:"deployment_type"`
	Services       []ServiceStatus      `json:"services"`
	RouterRuntime  *RouterRuntimeStatus `json:"router_runtime,omitempty"`
	Models         *RouterModelsInfo    `json:"models,omitempty"`
	Endpoints      []string             `json:"endpoints,omitempty"`
	Version        string               `json:"version,omitempty"`
}

// vllmSrContainerName is the container name used by the Python vllm-sr CLI
const vllmSrContainerName = "vllm-sr-container"

// StatusHandler returns the status of vLLM-SR services
// Aligns with the vllm-sr Python CLI by using the same Docker-based detection
func StatusHandler(routerAPIURL, configDir string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		status := detectSystemStatus(routerAPIURL, configDir)

		if err := json.NewEncoder(w).Encode(status); err != nil {
			http.Error(w, `{"error":"Failed to encode response"}`, http.StatusInternalServerError)
			return
		}
	}
}
