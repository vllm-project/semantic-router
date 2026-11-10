package handlers

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

func TestTaskCatalogUsesObservedRouterResponse(t *testing.T) {
	t.Setenv(instanceSocketEnv, "")
	body := `{"default_deployment":"judge","tasks":[],"models":[],"deployments":[{"deployment":"judge","ready":true}],"bindings":[]}`
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/diagnostics/models/tasks" || r.Header.Get("Authorization") != "Bearer control" || r.Header.Get("Cookie") != "" {
			t.Errorf("unexpected upstream request: %s %v", r.URL, r.Header)
		}
		_, _ = io.WriteString(w, body)
	}))
	defer upstream.Close()
	handler := DecisionTaskCatalogHandler("missing", upstream.URL, decisionModelCredentialProvider{token: "control"})
	request := httptest.NewRequest(http.MethodGet, "/api/decision-model/tasks?url=http://invalid", nil)
	request.Header.Set("Cookie", "browser=secret")
	result := httptest.NewRecorder()
	handler(result, request)
	if result.Code != http.StatusOK || result.Body.String() != body || result.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("observed response changed: %d %s", result.Code, result.Body.String())
	}
}

func TestTaskCatalogFallbackKeepsAuthoringAvailableWithoutInventingReadiness(t *testing.T) {
	t.Setenv(instanceSocketEnv, "")
	upstream := httptest.NewServer(http.NotFoundHandler())
	defer upstream.Close()
	handler := DecisionTaskCatalogHandler(createValidTestConfig(t, t.TempDir()), upstream.URL)
	result := httptest.NewRecorder()
	handler(result, httptest.NewRequest(http.MethodGet, "/api/decision-model/tasks", nil))
	var catalog struct {
		modelservice.TaskCatalogResponse
		RuntimeObserved bool `json:"runtime_observed"`
	}
	if result.Code != http.StatusOK || json.Unmarshal(result.Body.Bytes(), &catalog) != nil {
		t.Fatalf("authoring unavailable: %d %s", result.Code, result.Body.String())
	}
	if len(catalog.Tasks) == 0 || len(catalog.Models) == 0 || catalog.RuntimeObserved {
		t.Fatalf("invalid fallback catalog: %#v", catalog)
	}
	for _, deployment := range catalog.Deployments {
		if deployment.Ready {
			t.Fatalf("unobserved deployment reported ready: %s", deployment.Deployment)
		}
	}
}
