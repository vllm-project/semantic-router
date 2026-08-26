package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/statusstore"
)

func TestStatusHandlerExposesOnlyPublicAvailability(t *testing.T) {
	originalContainerDetection := isRunningInContainer
	isRunningInContainer = func() bool { return false }
	t.Cleanup(func() { isRunningInContainer = originalContainerDetection })

	// Avoid container-runtime probing so this test isolates the outbound Router
	// contract. The public status endpoint may call health and nothing else.
	t.Setenv(routerContainerNameEnv, "status-contract")
	t.Setenv(envoyContainerNameEnv, "status-contract")
	t.Setenv(dashboardContainerNameEnv, "status-contract")
	t.Setenv("TARGET_ENVOY_URL", "http://127.0.0.1:1")

	var requestedPaths []string
	router := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestedPaths = append(requestedPaths, r.URL.Path)
		if r.URL.Path != "/health" {
			http.Error(w, `{"secret":"must-not-be-exposed"}`, http.StatusUnauthorized)
			return
		}
		w.Header().Set("X-Internal-Endpoint", "https://private.invalid")
		w.WriteHeader(http.StatusNoContent)
	}))
	t.Cleanup(router.Close)
	historyStore, err := statusstore.Open(filepath.Join(t.TempDir(), "status.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = historyStore.Close() })
	monitor := NewStatusMonitor(router.URL, historyStore)
	monitor.sample()

	recorder := httptest.NewRecorder()
	monitor.Handler().ServeHTTP(
		recorder,
		httptest.NewRequest(http.MethodGet, "/api/status", nil),
	)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body=%s", recorder.Code, http.StatusOK, recorder.Body.String())
	}
	if got := recorder.Header().Get("Cache-Control"); got != "no-store" {
		t.Fatalf("Cache-Control = %q, want no-store", got)
	}
	if !reflect.DeepEqual(requestedPaths, []string{"/health", "/health"}) {
		t.Fatalf("Router paths = %v, want only sampled and live /health probes", requestedPaths)
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(recorder.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode status response: %v", err)
	}
	if got, want := sortedJSONKeys(payload), []string{"history", "overall", "services"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("top-level fields = %v, want %v", got, want)
	}

	var services []map[string]json.RawMessage
	if err := json.Unmarshal(payload["services"], &services); err != nil {
		t.Fatalf("decode services: %v", err)
	}
	for index, service := range services {
		if got, want := sortedJSONKeys(service), []string{"healthy", "name", "status"}; !reflect.DeepEqual(got, want) {
			t.Fatalf("service %d fields = %v, want %v", index, got, want)
		}
	}

	var history statusstore.History
	if err := json.Unmarshal(payload["history"], &history); err != nil {
		t.Fatalf("decode status history: %v", err)
	}
	if history.WindowHours != statusstore.RetentionHours {
		t.Fatalf("history window = %d, want %d", history.WindowHours, statusstore.RetentionHours)
	}
	if len(history.Services) == 0 || len(history.Services[0].Hours) != statusstore.RetentionHours {
		t.Fatalf("status history is not a dense %d-hour service series: %#v", statusstore.RetentionHours, history)
	}
	for _, service := range history.Services {
		if got := service.Hours[len(service.Hours)-1].Status; got == statusstore.StateUnknown {
			t.Fatalf("current %s observation was not persisted", service.Name)
		}
	}

	for _, forbidden := range []string{
		"deployment_type", "endpoint", "router_runtime", "models", "credential", "version",
		"api_key", "environment", "path", "private.invalid", "must-not-be-exposed",
	} {
		if strings.Contains(strings.ToLower(recorder.Body.String()), forbidden) {
			t.Fatalf("public status leaked forbidden value %q: %s", forbidden, recorder.Body.String())
		}
	}
}

func TestStatusHandlerKeepsLiveStatusAvailableWithoutHistoryStore(t *testing.T) {
	originalContainerDetection := isRunningInContainer
	isRunningInContainer = func() bool { return false }
	t.Cleanup(func() { isRunningInContainer = originalContainerDetection })
	t.Setenv(routerContainerNameEnv, "missing-status-router")
	t.Setenv(envoyContainerNameEnv, "missing-status-envoy")
	t.Setenv(dashboardContainerNameEnv, "missing-status-dashboard")
	t.Setenv("TARGET_ENVOY_URL", "http://127.0.0.1:1")

	recorder := httptest.NewRecorder()
	StatusHandler("", nil).ServeHTTP(
		recorder,
		httptest.NewRequest(http.MethodGet, "/api/status", nil),
	)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body=%s", recorder.Code, http.StatusOK, recorder.Body.String())
	}
	var status SystemStatus
	if err := json.Unmarshal(recorder.Body.Bytes(), &status); err != nil {
		t.Fatal(err)
	}
	if len(status.Services) == 0 || len(status.History.Services) == 0 {
		t.Fatalf("live status or honest history fallback missing: %#v", status)
	}
	for _, service := range status.History.Services {
		for _, hour := range service.Hours {
			if hour.Status != statusstore.StateUnknown {
				t.Fatalf("fallback history invented %q for %s at %s", hour.Status, service.Name, hour.ObservedAt)
			}
		}
	}
}

func sortedJSONKeys(value map[string]json.RawMessage) []string {
	keys := make([]string, 0, len(value))
	for key := range value {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	return keys
}
