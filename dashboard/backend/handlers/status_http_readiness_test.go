package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// Exercise the public status response across a Router startup transition. The
// Router's process health remains good throughout; only its ability to serve
// requests changes. This catches a status handler that reports process liveness
// as routing readiness, even when the collector's own tests pass.
func TestStatusHTTPTracksRouterReadinessAndRecovery(t *testing.T) {
	setRunningManagedStatusContainers(t)

	var phase atomic.Int32
	var modelRequests atomic.Int32
	routerServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health", "/v1/models":
			w.WriteHeader(http.StatusOK)
		case "/startup-status", "/ready", "/api/v1/inventory/models":
			if r.Header.Get("Authorization") != "Bearer status-e2e-token" {
				w.WriteHeader(http.StatusUnauthorized)
				return
			}
			switch r.URL.Path {
			case "/startup-status":
				w.Header().Set("Content-Type", "application/json")
				switch phase.Load() {
				case 0:
					w.WriteHeader(http.StatusServiceUnavailable)
					_, _ = w.Write([]byte(`{"phase":"waiting_for_config","ready":false}`))
				case 1:
					_, _ = w.Write([]byte(`{"phase":"ready","ready":true}`))
				default:
					_, _ = w.Write([]byte(`{"phase":"error","ready":true,"message":"Candidate failed; serving previous configuration"}`))
				}
			case "/ready":
				if phase.Load() == 0 {
					w.WriteHeader(http.StatusServiceUnavailable)
				}
			case "/api/v1/inventory/models":
				modelRequests.Add(1)
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"models":[],"summary":{"ready":true,"phase":"ready"}}`))
			}
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer routerServer.Close()

	dashboardServer := httptest.NewServer(StatusHandler(
		routerServer.URL,
		routerServer.URL,
		t.TempDir(),
		statusCredentialProvider{token: "status-e2e-token"},
	))
	defer dashboardServer.Close()

	assertStatus := func(wantOverall, wantPhase string, wantReady bool, wantModelRequests int32) {
		t.Helper()
		response, err := dashboardServer.Client().Get(dashboardServer.URL + "/api/status")
		if err != nil {
			t.Fatalf("GET /api/status: %v", err)
		}
		defer func() { _ = response.Body.Close() }()
		if response.StatusCode != http.StatusOK {
			t.Fatalf("GET /api/status returned %d, want 200", response.StatusCode)
		}

		var status SystemStatus
		if err := json.NewDecoder(response.Body).Decode(&status); err != nil {
			t.Fatalf("decode /api/status: %v", err)
		}
		if status.Overall != wantOverall {
			t.Errorf("overall = %q, want %q", status.Overall, wantOverall)
		}
		if status.RouterRuntime == nil || status.RouterRuntime.Phase != wantPhase || status.RouterRuntime.Ready != wantReady {
			t.Errorf("router runtime = %+v, want phase=%q ready=%v", status.RouterRuntime, wantPhase, wantReady)
		}
		services := make(map[string]ServiceStatus, len(status.Services))
		for _, service := range status.Services {
			services[service.Name] = service
		}
		for _, name := range []string{"Router", "Dashboard", "Envoy"} {
			if service, ok := services[name]; !ok || !service.Healthy {
				t.Errorf("%s process/service must stay healthy: %+v", name, service)
			}
		}
		if access, ok := services["Routing access"]; !ok || access.Healthy != wantReady {
			t.Errorf("routing access = %+v, want healthy=%v", access, wantReady)
		}
		if (status.Models != nil) != wantReady {
			t.Errorf("model inventory = %+v, want present=%v", status.Models, wantReady)
		}
		if got := modelRequests.Load(); got != wantModelRequests {
			t.Errorf("model inventory requests = %d, want %d", got, wantModelRequests)
		}
	}

	assertStatus("degraded", "waiting_for_config", false, 0)
	phase.Store(1)
	assertStatus("healthy", "ready", true, 1)
	phase.Store(2)
	assertStatus("healthy", "error", true, 2)
}
