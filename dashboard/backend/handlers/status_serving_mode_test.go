package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/statusstore"
)

const engineReadyHealth = `{"service":"classification-api","status":"healthy","serving_mode":"engine"}`

func TestServingModeUsesActiveFrontendIdentity(t *testing.T) {
	for _, test := range []struct {
		name, health, mode string
		status             int
		healthy            bool
	}{
		{"router", `{"service":"classification-api","status":"healthy","serving_mode":"router"}`, servingModeRouter, 200, true},
		{"engine", engineReadyHealth, servingModeEngine, 200, true},
		{"missing active mode", `{"service":"classification-api","status":"healthy"}`, servingModeUnknown, 200, true},
		{"unknown active mode", `{"service":"classification-api","status":"healthy","serving_mode":"unknown"}`, servingModeUnknown, 200, true},
		{"bare worker is not frontend", `{"api_version":"2.1.0","status":"ready","reason":null,"model":"vela"}`, servingModeUnknown, 200, true},
		{"generic healthy", `{}`, servingModeUnknown, 200, true},
		{"oversized health", strings.Repeat(" ", servingHealthLimit) + engineReadyHealth, servingModeUnknown, 200, true},
		{"failed health", engineReadyHealth, servingModeUnknown, 503, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			requests := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests++
				if r.URL.Path != "/health" || r.Header.Get("Authorization") != "Bearer serving-mode-token" {
					t.Errorf("unexpected identity request: %s", r.URL.Path)
				}
				w.WriteHeader(test.status)
				_, _ = w.Write([]byte(test.health))
			}))
			defer server.Close()
			probe := probeServingHealth(server.URL, statusCredentialProvider{token: "serving-mode-token"})
			if probe.mode != test.mode || probe.healthy != test.healthy || requests != 1 {
				t.Fatalf("probe=%+v requests=%d", probe, requests)
			}
		})
	}
}

func TestServingModeDoesNotFollowCredentialRedirects(t *testing.T) {
	for _, redirectPath := range []string{"/health"} {
		t.Run(redirectPath, func(t *testing.T) {
			var redirectedRequests atomic.Int32
			otherPort := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				redirectedRequests.Add(1)
				_, _ = w.Write([]byte(engineReadyHealth))
			}))
			defer otherPort.Close()
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("Authorization") != "Bearer private-management-token" {
					t.Error("configured origin did not receive its credential")
				}
				if r.URL.Path == redirectPath {
					http.Redirect(w, r, otherPort.URL+redirectPath, http.StatusTemporaryRedirect)
					return
				}
				_, _ = w.Write([]byte(engineReadyHealth))
			}))
			defer upstream.Close()
			probe := probeServingHealth(upstream.URL, statusCredentialProvider{token: "private-management-token"})
			if probe.mode != servingModeUnknown || redirectedRequests.Load() != 0 {
				t.Fatalf("redirected identity probe: mode=%s redirected requests=%d", probe.mode, redirectedRequests.Load())
			}
		})
	}
}

func TestServingModeHealthTimeoutIsBounded(t *testing.T) {
	canceled := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
		close(canceled)
	}))
	defer server.Close()
	probe := probeServingHealth(server.URL)
	if probe.mode != servingModeUnknown || probe.healthy {
		t.Fatalf("timed-out probe = %+v", probe)
	}
	select {
	case <-canceled:
	case <-time.After(time.Second):
		t.Fatal("health timeout did not cancel the upstream request")
	}
}

func TestStatusHTTPReportsServingModeAndEngineReadiness(t *testing.T) {
	for _, test := range []struct {
		name, health, mode string
		code               int
		ready              bool
	}{
		{name: "router", health: `{"service":"classification-api","status":"healthy","serving_mode":"router"}`, mode: servingModeRouter, code: 200, ready: true},
		{name: "unknown healthy API", health: `{}`, mode: servingModeUnknown, code: 200, ready: true},
		{name: "engine ready", health: engineReadyHealth, mode: servingModeEngine, code: 200, ready: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			var healthRequests, routerOnlyRequests atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/health":
					healthRequests.Add(1)
					w.WriteHeader(test.code)
					_, _ = w.Write([]byte(test.health))
				case "/startup-status":
					routerOnlyRequests.Add(1)
					_, _ = w.Write([]byte(`{"phase":"ready","ready":true}`))
				case "/api/v1/inventory/models":
					routerOnlyRequests.Add(1)
					_, _ = w.Write([]byte(`{"models":[],"summary":{"ready":true}}`))
				default:
					w.WriteHeader(http.StatusOK)
				}
			}))
			defer upstream.Close()
			for _, collector := range []string{"managed", "direct", "handler"} {
				t.Run(collector, func(t *testing.T) {
					healthRequests.Store(0)
					routerOnlyRequests.Store(0)
					var status SystemStatus
					switch collector {
					case "managed":
						status = collectManagedStackStatus("", StackState{}, upstream.URL, upstream.URL)
					case "direct":
						var present bool
						status, present = collectDirectStatus("", upstream.URL, upstream.URL)
						if !present {
							t.Fatal("known engine must remain observable while warming")
						}
					default:
						recorder := httptest.NewRecorder()
						StatusHandler(upstream.URL, upstream.URL, t.TempDir(), StackState{}).ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/api/status", nil))
						if recorder.Code != http.StatusOK {
							t.Fatalf("status response = %d", recorder.Code)
						}
						if err := json.Unmarshal(recorder.Body.Bytes(), &status); err != nil {
							t.Fatal(err)
						}
					}
					if status.ServingMode != test.mode || healthRequests.Load() != 1 {
						t.Fatalf("mode=%s, health requests=%d, want %s and 1", status.ServingMode, healthRequests.Load(), test.mode)
					}
					if test.mode == servingModeEngine {
						if len(status.Services) != 2 || status.Services[0].Name != "Model Engine" || status.Services[0].Healthy != test.ready || status.RouterRuntime != nil || status.Models != nil || routerOnlyRequests.Load() != 0 {
							t.Fatalf("engine status inherited Router semantics: %+v", status)
						}
						wantOverall := "degraded"
						if test.ready {
							wantOverall = "healthy"
						}
						if status.Overall != wantOverall {
							t.Fatalf("overall=%s, want %s", status.Overall, wantOverall)
						}
						if !test.ready && statusObservations(status.Services)[0].State != statusstore.StateStarting {
							t.Fatal("warming Engine must be recorded as starting, not an outage")
						}
					}
				})
			}
		})
	}
}

func TestUnavailableStatusRetainsExplicitUnknownMode(t *testing.T) {
	recorder := httptest.NewRecorder()
	StatusHandler("", "", t.TempDir(), StackState{}).ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/api/status", nil))
	var response map[string]any
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response["serving_mode"] != servingModeUnknown {
		t.Fatalf("unavailable serving_mode = %#v", response["serving_mode"])
	}
}
