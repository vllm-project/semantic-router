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

const engineStatusContract = `openapi: 3.0.3
info:
  title: vLLM Semantic Router Model Runtime
  version: 2.1.0
paths:
  /health:
    get: {}
  /v1/models:
    get: {}
  /v1/decisions:
    post: {}
`

const engineReadyHealth = `{"api_version":"2.1.0","status":"ready","reason":null,"model":"Vela-2.0-4B"}`

func TestServingModeUsesUpstreamIdentity(t *testing.T) {
	for _, test := range []struct {
		name, health, contract, mode string
		status, contractStatus       int
		healthy                      bool
		contractRequests             int32
	}{
		{name: "router", health: `{"service":"classification-api","status":"healthy"}`, status: 200, mode: servingModeRouter, healthy: true},
		{name: "generic healthy service", health: `{"status":"healthy"}`, status: 200, mode: servingModeUnknown, healthy: true},
		{name: "empty healthy service", status: 200, mode: servingModeUnknown, healthy: true},
		{name: "engine ready", health: engineReadyHealth, contract: engineStatusContract, status: 200, contractStatus: 200, mode: servingModeEngine, healthy: true, contractRequests: 1},
		{name: "engine loading", health: `{"api_version":"2.1.0","status":"loading","reason":null,"model":null,"models":{}}`, contract: engineStatusContract, status: 503, contractStatus: 200, mode: servingModeEngine, contractRequests: 1},
		{name: "generic API with runtime health shape", health: engineReadyHealth, contract: strings.Replace(engineStatusContract, "vLLM Semantic Router Model Runtime", "Other API", 1), status: 200, contractStatus: 200, mode: servingModeUnknown, healthy: true, contractRequests: 1},
		{name: "engine version mismatch", health: engineReadyHealth, contract: strings.Replace(engineStatusContract, "2.1.0", "2.2.0", 1), status: 200, contractStatus: 200, mode: servingModeUnknown, healthy: true, contractRequests: 1},
		{name: "missing runtime operation", health: engineReadyHealth, contract: strings.Replace(engineStatusContract, "/v1/decisions", "/unrelated", 1), status: 200, contractStatus: 200, mode: servingModeUnknown, healthy: true, contractRequests: 1},
		{name: "missing model health field", health: `{"api_version":"2.1.0","status":"ready","reason":null}`, status: 200, mode: servingModeUnknown, healthy: true},
		{name: "unknown engine status", health: `{"api_version":"2.1.0","status":"healthy","reason":null,"model":"model"}`, status: 200, mode: servingModeUnknown, healthy: true},
		{name: "missing runtime contract", health: engineReadyHealth, status: 200, contractStatus: 404, mode: servingModeUnknown, healthy: true, contractRequests: 1},
		{name: "oversized health", health: strings.Repeat(" ", servingHealthLimit) + engineReadyHealth, status: 200, mode: servingModeUnknown, healthy: true},
		{name: "oversized contract", health: engineReadyHealth, contract: engineStatusContract + strings.Repeat(" ", servingOpenAPILimit), status: 200, contractStatus: 200, mode: servingModeUnknown, healthy: true, contractRequests: 1},
		{name: "failed health", health: engineReadyHealth, status: 500, mode: servingModeUnknown},
	} {
		t.Run(test.name, func(t *testing.T) {
			var healthRequests, contractRequests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("Authorization") != "Bearer serving-mode-token" {
					t.Error("management request did not use the scoped credential")
				}
				switch r.URL.Path {
				case "/health":
					healthRequests.Add(1)
					w.WriteHeader(test.status)
					_, _ = w.Write([]byte(test.health))
				case "/openapi.yaml":
					contractRequests.Add(1)
					w.WriteHeader(test.contractStatus)
					_, _ = w.Write([]byte(test.contract))
				default:
					t.Errorf("unexpected expensive discovery request: %s", r.URL.Path)
				}
			}))
			defer server.Close()
			probe := probeServingHealth(server.URL, statusCredentialProvider{token: "serving-mode-token"})
			if probe.mode != test.mode || probe.healthy != test.healthy {
				t.Fatalf("probe = %+v, want mode=%s healthy=%v", probe, test.mode, test.healthy)
			}
			if healthRequests.Load() != 1 || contractRequests.Load() != test.contractRequests {
				t.Fatalf("requests: health=%d contract=%d, want 1 and %d", healthRequests.Load(), contractRequests.Load(), test.contractRequests)
			}
		})
	}
}

func TestServingModeDoesNotFollowCredentialRedirects(t *testing.T) {
	for _, redirectPath := range []string{"/health", "/openapi.yaml"} {
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
		{name: "router", health: `{"service":"classification-api","status":"healthy"}`, mode: servingModeRouter, code: 200, ready: true},
		{name: "unknown healthy API", health: `{}`, mode: servingModeUnknown, code: 200, ready: true},
		{name: "engine ready", health: engineReadyHealth, mode: servingModeEngine, code: 200, ready: true},
		{name: "engine warming", health: strings.Replace(engineReadyHealth, "ready", "warming", 1), mode: servingModeEngine, code: 503},
	} {
		t.Run(test.name, func(t *testing.T) {
			var healthRequests, routerOnlyRequests atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/health":
					healthRequests.Add(1)
					w.WriteHeader(test.code)
					_, _ = w.Write([]byte(test.health))
				case "/openapi.yaml":
					_, _ = w.Write([]byte(engineStatusContract))
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
