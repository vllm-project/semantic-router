package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
	"github.com/vllm-project/semantic-router/dashboard/backend/statusstore"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
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
	Phase             string                                 `json:"phase"`
	Ready             bool                                   `json:"ready"`
	Message           string                                 `json:"message,omitempty"`
	DownloadingModel  string                                 `json:"downloading_model,omitempty"`
	PendingModels     []string                               `json:"pending_models,omitempty"`
	ReadyModels       int                                    `json:"ready_models,omitempty"`
	TotalModels       int                                    `json:"total_models,omitempty"`
	EmbeddingProvider *startupstatus.EmbeddingProviderStatus `json:"embedding_provider,omitempty"`
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
	History        statusstore.History  `json:"history"`
}

// StatusHandler returns the status of vLLM-SR services
// Aligns with the vllm-sr Python CLI by using the same Docker-based detection
func StatusHandler(routerAPIURL, envoyURL, configDir string, credentialProvider ...routerauth.CredentialProvider) http.HandlerFunc {
	return NewStatusMonitor(routerAPIURL, envoyURL, configDir, nil, credentialProvider...).Handler()
}

// Handler serves the live snapshot together with server-observed hourly history.
func (m *StatusMonitor) Handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		status := detectSystemStatus(m.routerAPIURL, m.envoyURL, m.configDir, m.credentialProvider...)
		status.History = readStatusHistory(r, m.historyStore, status.Services)

		if err := json.NewEncoder(w).Encode(status); err != nil {
			http.Error(w, `{"error":"Failed to encode response"}`, http.StatusInternalServerError)
			return
		}
	}
}

func readStatusHistory(r *http.Request, store *statusstore.Store, services []ServiceStatus) statusstore.History {
	names := make([]string, 0, len(services))
	for _, service := range services {
		names = append(names, service.Name)
	}
	if store != nil {
		history, err := store.Read(r.Context(), names)
		if err == nil {
			return history
		}
		log.Printf("status history read failed: %v", err)
	}
	return statusstore.UnknownHistory(time.Now(), names)
}
