package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/statusstore"
)

// ServiceStatus is the public availability of one product component. Keep this
// contract free of internal addresses, runtime topology, and diagnostic text.
type ServiceStatus struct {
	Name    string `json:"name"`
	Status  string `json:"status"`
	Healthy bool   `json:"healthy"`
}

// SystemStatus is intentionally limited to public product availability.
// Authorized routing, model, usage, and diagnostic data belongs to the Router
// Management API and must not be projected through this endpoint.
type SystemStatus struct {
	Overall  string              `json:"overall"`
	Services []ServiceStatus     `json:"services"`
	History  statusstore.History `json:"history"`
}

// StatusHandler returns a credential-free public health summary. Detailed
// operational state is available only through authorized Management APIs.
func StatusHandler(routerAPIURL string, historyStore *statusstore.Store) http.HandlerFunc {
	return NewStatusMonitor(routerAPIURL, historyStore).Handler()
}

// Handler returns a credential-free public health summary. Detailed
// operational state is available only through authorized Management APIs.
func (m *StatusMonitor) Handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		status := detectSystemStatus(m.routerAPIURL)
		status.History = readStatusHistory(r, m.historyStore, status.Services)

		if err := json.NewEncoder(w).Encode(status); err != nil {
			http.Error(w, `{"error":"Failed to encode response"}`, http.StatusInternalServerError)
			return
		}
	}
}

func readStatusHistory(
	r *http.Request,
	historyStore *statusstore.Store,
	services []ServiceStatus,
) statusstore.History {
	serviceNames := make([]string, 0, len(services))
	for _, service := range services {
		serviceNames = append(serviceNames, service.Name)
	}
	if historyStore != nil {
		history, err := historyStore.Read(r.Context(), serviceNames)
		if err == nil {
			return history
		}
		log.Printf("status history read failed: %v", err)
	}
	return statusstore.UnknownHistory(time.Now(), serviceNames)
}
