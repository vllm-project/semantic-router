package handlers

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
)

func setRunningManagedStatusContainers(t *testing.T) {
	t.Helper()
	docker := writeFakeStatusDockerCLI(t)
	t.Setenv("PATH", filepath.Dir(docker)+string(os.PathListSeparator)+os.Getenv("PATH"))
	for _, service := range []struct{ name, variable string }{
		{"ROUTER", routerContainerNameEnv}, {"ENVOY", envoyContainerNameEnv}, {"DASHBOARD", dashboardContainerNameEnv},
	} {
		name := "status-test-" + service.name
		t.Setenv(service.variable, name)
		t.Setenv("TEST_"+service.name+"_CONTAINER", name)
		t.Setenv("TEST_"+service.name+"_STATUS", "running")
	}
}

func TestStatusCollectorsRequireRoutingReadiness(t *testing.T) {
	setRunningManagedStatusContainers(t)
	for _, mode := range []string{"direct", "managed"} {
		t.Run(mode, func(t *testing.T) {
			for _, test := range []struct {
				name, startup, suffix                string
				startupCode, readyCode, gatewayCode  int
				badCredential, wantReady, wantModels bool
				wantReadyRequests                    int32
			}{
				{name: "waiting", startup: `{"phase":"waiting_for_config","ready":false}`, startupCode: 503, readyCode: 200, gatewayCode: 200},
				{name: "serving", startup: `{"phase":"ready","ready":true}`, startupCode: 200, readyCode: 200, gatewayCode: 200, wantReady: true, wantModels: true},
				{name: "failed_candidate_still_serving", startup: `{"phase":"error","ready":true,"message":"Candidate failed; serving previous configuration"}`, startupCode: 200, readyCode: 200, gatewayCode: 200, wantReady: true, wantModels: true},
				{name: "missing_not_ready", startupCode: 404, readyCode: 503, gatewayCode: 200, wantReadyRequests: 1},
				{name: "missing_ready", startupCode: 404, readyCode: 200, gatewayCode: 200, wantReady: true, wantModels: true, wantReadyRequests: 1},
				{name: "empty_unavailable_observation", startupCode: 503, readyCode: 200, gatewayCode: 200, wantReady: true, wantModels: true, wantReadyRequests: 1},
				{name: "unauthorized_observation", startupCode: 401, readyCode: 503, gatewayCode: 200, wantReadyRequests: 1},
				{name: "ready_auth_denied", startupCode: 404, readyCode: 200, gatewayCode: 200, badCredential: true, wantReadyRequests: 1},
				{name: "gateway_unavailable", startup: `{"phase":"ready","ready":true}`, startupCode: 200, readyCode: 200, gatewayCode: 503, wantModels: true},
				{name: "normalized_management_url", startupCode: 404, readyCode: 200, gatewayCode: 200, suffix: "//", wantReady: true, wantModels: true, wantReadyRequests: 1},
			} {
				t.Run(test.name, func(t *testing.T) {
					var readyRequests, modelRequests atomic.Int32
					token := "dashboard-status-token"
					if test.badCredential {
						token = "invalid-status-token"
					}
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
						switch request.URL.Path {
						case "/health":
							w.WriteHeader(http.StatusOK)
							return
						case "/v1/models":
							w.WriteHeader(test.gatewayCode)
							return
						case "/ready":
							readyRequests.Add(1)
						case "/api/v1/inventory/models":
							modelRequests.Add(1)
						case "/startup-status":
						default:
							w.WriteHeader(http.StatusNotFound)
							return
						}
						if got := request.Header.Get("Authorization"); got != "Bearer "+token {
							t.Errorf("%s Authorization = %q; management credential was not forwarded", request.URL.Path, got)
						}
						if request.Header.Get("Authorization") != "Bearer dashboard-status-token" {
							w.WriteHeader(http.StatusUnauthorized)
							return
						}
						switch request.URL.Path {
						case "/startup-status":
							w.WriteHeader(test.startupCode)
							_, _ = fmt.Fprint(w, test.startup)
						case "/ready":
							w.WriteHeader(test.readyCode)
						case "/api/v1/inventory/models":
							_, _ = fmt.Fprint(w, `{"models":[],"summary":{}}`)
						}
					}))
					defer server.Close()
					runtimePath := filepath.Join(t.TempDir(), "state", "runtime.json")
					provider := statusCredentialProvider{token: token}
					var status SystemStatus
					if mode == "managed" {
						status = collectManagedDockerStatus(runtimePath, server.URL+test.suffix, server.URL, provider)
					} else {
						var observed bool
						status, observed = collectDirectStatus(runtimePath, server.URL+test.suffix, server.URL, provider)
						if !observed {
							t.Fatal("live Router process was not observed")
						}
					}
					for _, service := range status.Services {
						switch service.Name {
						case "Router":
							if !service.Healthy || service.Status != "running" {
								t.Errorf("readiness hid process liveness: %+v", service)
							}
						case "Routing access":
							if service.Healthy != test.wantReady || (service.Message == "Ready") != test.wantReady {
								t.Errorf("routing = %+v, want ready=%v", service, test.wantReady)
							}
						}
					}
					wantOverall := "degraded"
					if test.wantReady {
						wantOverall = "healthy"
					}
					if status.Overall != wantOverall {
						t.Errorf("overall=%q, want %q", status.Overall, wantOverall)
					}
					if (status.Models != nil) != test.wantModels || (modelRequests.Load() > 0) != test.wantModels {
						t.Errorf("model fetch before readiness: models=%+v requests=%d want=%v", status.Models, modelRequests.Load(), test.wantModels)
					}
					if got := readyRequests.Load(); got != test.wantReadyRequests {
						t.Errorf("ready fallback requests=%d, want %d", got, test.wantReadyRequests)
					}
				})
			}
		})
	}
}

func TestManagedStatusWithoutManagementURLPreservesObservations(t *testing.T) {
	setRunningManagedStatusContainers(t)
	envoy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) }))
	defer envoy.Close()
	for _, test := range []struct {
		name, state string
		ready       bool
	}{
		{"legacy_container_only", "", true},
		{"local_waiting", `{"phase":"waiting_for_config","ready":false}`, false},
		{"local_serving", `{"phase":"ready","ready":true}`, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "runtime.json")
			if test.state != "" {
				if err := os.WriteFile(path, []byte(test.state), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			status := collectManagedDockerStatus(path, "", envoy.URL)
			if got := status.Services[0]; got.Healthy != test.ready {
				t.Errorf("routing=%+v, want ready=%v", got, test.ready)
			}
			if got := status.Services[1]; !got.Healthy || got.Status != "running" {
				t.Errorf("process liveness lost: %+v", got)
			}
		})
	}
}
