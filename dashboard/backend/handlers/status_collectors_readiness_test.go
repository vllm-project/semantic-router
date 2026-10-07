package handlers

import (
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
)

func TestStatusCollectorsRequireRoutingReadiness(t *testing.T) {
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
						status = collectManagedStackStatus(runtimePath, StackState{}, server.URL+test.suffix, server.URL, provider)
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

// The Router's startup file outlives the process that wrote it: of a Router
// that does not answer, it still tells only why startup failed.
func TestManagedStatusTrustsTheStartupFileOfAStoppedRouterOnlyForAFailure(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	stoppedRouter := "http://" + listener.Addr().String()
	if closeErr := listener.Close(); closeErr != nil {
		t.Fatal(closeErr)
	}
	for _, test := range []struct {
		name, state, message string
		runtime              bool
	}{
		{"was_ready", `{"phase":"ready","ready":true}`, "Not running", false},
		{"was_downloading", `{"phase":"downloading_models","ready":false}`, "Not running", false},
		{"failed", `{"phase":"error","ready":false,"message":"Router startup failed: models missing"}`, "Router startup failed: models missing", true},
	} {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "runtime.json")
			if err := os.WriteFile(path, []byte(test.state), 0o600); err != nil {
				t.Fatal(err)
			}
			status := collectManagedStackStatus(path, StackState{}, stoppedRouter, stoppedRouter)
			router := status.Services[1]
			if router.Name != "Router" || router.Healthy || router.Status != "not running" || router.Message != test.message {
				t.Errorf("router = %+v, want not running with message %q", router, test.message)
			}
			if (status.RouterRuntime != nil) != test.runtime {
				t.Errorf("router runtime = %+v, want present=%v", status.RouterRuntime, test.runtime)
			}
		})
	}
}
