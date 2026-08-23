package handlers

import (
	"encoding/json"
	"net/http"
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
	Overall  string          `json:"overall"`
	Services []ServiceStatus `json:"services"`
}

// StatusHandler returns a credential-free public health summary. Detailed
// operational state is available only through authorized Management APIs.
func StatusHandler(routerAPIURL string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		status := detectSystemStatus(routerAPIURL)

		if err := json.NewEncoder(w).Encode(status); err != nil {
			http.Error(w, `{"error":"Failed to encode response"}`, http.StatusInternalServerError)
			return
		}
	}
}
