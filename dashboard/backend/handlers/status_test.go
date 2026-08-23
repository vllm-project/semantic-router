package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"testing"
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

	recorder := httptest.NewRecorder()
	StatusHandler(router.URL).ServeHTTP(
		recorder,
		httptest.NewRequest(http.MethodGet, "/api/status", nil),
	)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body=%s", recorder.Code, http.StatusOK, recorder.Body.String())
	}
	if !reflect.DeepEqual(requestedPaths, []string{"/health"}) {
		t.Fatalf("Router paths = %v, want only /health", requestedPaths)
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(recorder.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode status response: %v", err)
	}
	if got, want := sortedJSONKeys(payload), []string{"overall", "services"}; !reflect.DeepEqual(got, want) {
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

	for _, forbidden := range []string{
		"deployment_type", "endpoint", "router_runtime", "models", "credential", "version",
		"api_key", "environment", "path", "private.invalid", "must-not-be-exposed",
	} {
		if strings.Contains(strings.ToLower(recorder.Body.String()), forbidden) {
			t.Fatalf("public status leaked forbidden value %q: %s", forbidden, recorder.Body.String())
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
