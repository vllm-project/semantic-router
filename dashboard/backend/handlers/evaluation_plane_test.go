package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/evaluationplane"
)

func newEvaluationHandlerService(t *testing.T, dataDir string) *evaluationplane.Service {
	t.Helper()
	if dataDir == "" {
		dataDir = t.TempDir()
	}
	if err := os.Chmod(dataDir, 0o700); err != nil {
		t.Fatalf("protect evaluation data dir: %v", err)
	}
	configPath := filepath.Join(dataDir, "config.yaml")
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		if err := os.WriteFile(configPath, []byte("version: v0.3\nrouting:\n  modelCards: []\n"), 0o600); err != nil {
			t.Fatalf("write config: %v", err)
		}
	}
	service, err := evaluationplane.NewService(evaluationplane.Options{
		DataDir: dataDir, PythonPath: "python3", ConfigPath: configPath,
		RouterAPIURL: "http://router.internal", EnvoyURL: "http://envoy.internal",
		CodeRevision: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	})
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}
	return service
}

func validCreateRunJSON(autoStart bool) string {
	return fmt.Sprintf(`{
        "name":"fixture",
        "description":"handler test",
        "suite_ids":["evaluation-smoke"],
        "track_ids":["routing"],
        "mode":"replay",
        "target_id":"fixture",
		"change_profile":"schema_adapter",
        "sample_limit":4,
        "concurrency":1,
        "seed":17,
        "auto_start":%t
    }`, autoStart)
}

func TestEvaluationPlaneCreateIsStrictAndServerTargetAllowlisted(t *testing.T) {
	service := newEvaluationHandlerService(t, "")
	handler := NewEvaluationPlaneHandler(service, false)
	tests := []struct {
		name string
		body string
	}{
		{name: "unknown endpoint", body: strings.Replace(validCreateRunJSON(false), `"auto_start":false`, `"auto_start":false,"endpoint":"https://attacker.invalid"`, 1)},
		{name: "unknown field", body: strings.Replace(validCreateRunJSON(false), `"auto_start":false`, `"auto_start":false,"command":["sh"]`, 1)},
		{name: "unknown target", body: strings.Replace(validCreateRunJSON(false), `"target_id":"fixture"`, `"target_id":"attacker"`, 1)},
		{name: "unknown suite", body: strings.Replace(validCreateRunJSON(false), `"evaluation-smoke"`, `"unknown-suite"`, 1)},
		{name: "unknown change profile", body: strings.Replace(validCreateRunJSON(false), `"change_profile":"schema_adapter"`, `"change_profile":"unknown"`, 1)},
		{name: "auto start permission bypass", body: validCreateRunJSON(true)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodPost, evaluationAPIBase+"/runs", strings.NewReader(test.body))
			handler.Runs(response, request)
			if response.Code != http.StatusBadRequest {
				t.Fatalf("status=%d want=%d body=%s", response.Code, http.StatusBadRequest, response.Body.String())
			}
		})
	}
	runs, err := service.ListRuns()
	if err != nil || len(runs) != 0 {
		t.Fatalf("rejected creates changed store: runs=%d err=%v", len(runs), err)
	}

	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, evaluationAPIBase+"/runs", strings.NewReader(validCreateRunJSON(false)))
	handler.Runs(response, request)
	if response.Code != http.StatusCreated {
		t.Fatalf("valid create status=%d body=%s", response.Code, response.Body.String())
	}
	var run evaluationplane.Run
	if err := json.NewDecoder(response.Body).Decode(&run); err != nil {
		t.Fatalf("decode run: %v", err)
	}
	if run.Status != evaluationplane.StatusPending || run.TargetID != "fixture" {
		t.Fatalf("unexpected created run: %+v", run)
	}
}

func TestEvaluationPlaneReadonlyDeniesEveryMutation(t *testing.T) {
	service := newEvaluationHandlerService(t, "")
	run, err := service.CreateRun(context.Background(), evaluationplane.CreateRunRequest{
		Name: "fixture", SuiteIDs: []string{"evaluation-smoke"}, TrackIDs: []evaluationplane.TrackID{"routing"},
		Mode: evaluationplane.ModeReplay, TargetID: "fixture", ChangeProfile: "schema_adapter",
		SampleLimit: 4, Concurrency: 1, Seed: 17,
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	handler := NewEvaluationPlaneHandler(service, true)
	tests := []struct {
		method string
		path   string
		body   string
		direct func(http.ResponseWriter, *http.Request)
	}{
		{method: http.MethodPost, path: evaluationAPIBase + "/runs", body: validCreateRunJSON(false), direct: handler.Runs},
		{method: http.MethodPost, path: evaluationAPIBase + "/runs/" + run.ID + "/start", direct: handler.RunRoute},
		{method: http.MethodPost, path: evaluationAPIBase + "/runs/" + run.ID + "/cancel", direct: handler.RunRoute},
		{method: http.MethodDelete, path: evaluationAPIBase + "/runs/" + run.ID, direct: handler.RunRoute},
	}
	for _, test := range tests {
		response := httptest.NewRecorder()
		request := httptest.NewRequest(test.method, test.path, strings.NewReader(test.body))
		test.direct(response, request)
		if response.Code != http.StatusForbidden {
			t.Fatalf("%s %s status=%d want=%d body=%s", test.method, test.path, response.Code, http.StatusForbidden, response.Body.String())
		}
	}
	response := httptest.NewRecorder()
	handler.RunRoute(response, httptest.NewRequest(http.MethodGet, evaluationAPIBase+"/runs/"+run.ID, nil))
	if response.Code != http.StatusOK {
		t.Fatalf("readonly GET status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestEvaluationPlaneSSEReplaysPersistedEventsAfterRestart(t *testing.T) {
	root := t.TempDir()
	service := newEvaluationHandlerService(t, root)
	run, err := service.CreateRun(context.Background(), evaluationplane.CreateRunRequest{
		Name: "fixture", SuiteIDs: []string{"evaluation-smoke"}, TrackIDs: []evaluationplane.TrackID{"routing"},
		Mode: evaluationplane.ModeReplay, TargetID: "fixture", ChangeProfile: "schema_adapter",
		SampleLimit: 4, Concurrency: 1, Seed: 17,
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if _, err := service.CancelRun(run.ID); err != nil {
		t.Fatalf("CancelRun: %v", err)
	}
	restarted := newEvaluationHandlerService(t, root)
	handler := NewEvaluationPlaneHandler(restarted, false)
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, evaluationAPIBase+"/runs/"+run.ID+"/events", nil)
	request.Header.Set("Last-Event-ID", "1")
	handler.RunRoute(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("SSE status=%d body=%s", response.Code, response.Body.String())
	}
	body := response.Body.String()
	if !strings.Contains(body, "id: 2\n") || !strings.Contains(body, "event: cancelled\n") || strings.Contains(body, "id: 1\n") {
		t.Fatalf("unexpected SSE replay body:\n%s", body)
	}
	if contentType := response.Header().Get("Content-Type"); contentType != "text/event-stream" {
		t.Fatalf("SSE Content-Type=%q", contentType)
	}

	badResponse := httptest.NewRecorder()
	badRequest := httptest.NewRequest(http.MethodGet, evaluationAPIBase+"/runs/"+run.ID+"/events", bytes.NewReader(nil))
	badRequest.Header.Set("Last-Event-ID", "not-numeric")
	handler.RunRoute(badResponse, badRequest)
	if badResponse.Code != http.StatusBadRequest {
		t.Fatalf("invalid Last-Event-ID status=%d body=%s", badResponse.Code, badResponse.Body.String())
	}

	for _, cursor := range []string{"2", "999999"} {
		response := httptest.NewRecorder()
		request := httptest.NewRequest(http.MethodGet, evaluationAPIBase+"/runs/"+run.ID+"/events", nil)
		request.Header.Set("Last-Event-ID", cursor)
		done := make(chan struct{})
		go func() {
			handler.RunRoute(response, request)
			close(done)
		}()
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatalf("terminal SSE stream did not close for Last-Event-ID=%s", cursor)
		}
		if response.Code != http.StatusOK || strings.Contains(response.Body.String(), "event:") {
			t.Fatalf("terminal SSE cursor=%s status=%d body=%s", cursor, response.Code, response.Body.String())
		}
	}
}

func TestEvaluationPlaneCORSRejectsHostileOriginsBeforeEvidenceAccess(t *testing.T) {
	service := newEvaluationHandlerService(t, "")
	handler := NewEvaluationPlaneHandler(service, false)
	paths := []string{
		evaluationAPIBase + "/runs/run-id/report",
		evaluationAPIBase + "/runs/run-id/artifacts/metrics-json",
		evaluationAPIBase + "/runs/run-id/events",
	}
	for _, path := range paths {
		response := httptest.NewRecorder()
		request := httptest.NewRequest(http.MethodGet, "https://dashboard.example.test"+path, nil)
		request.Host = "dashboard.example.test"
		request.Header.Set("Origin", "https://sibling.example.test")
		handler.RunRoute(response, request)
		if response.Code != http.StatusForbidden {
			t.Fatalf("hostile Origin %s status=%d body=%s", path, response.Code, response.Body.String())
		}
		if response.Header().Get("Access-Control-Allow-Origin") != "" || response.Header().Get("Access-Control-Allow-Credentials") != "" {
			t.Fatalf("hostile Origin received credentialed CORS headers: %v", response.Header())
		}
	}

	allowed := httptest.NewRecorder()
	allowedRequest := httptest.NewRequest(http.MethodGet, "https://dashboard.example.test"+evaluationAPIBase+"/catalog", nil)
	allowedRequest.Host = "dashboard.example.test"
	allowedRequest.Header.Set("Origin", "https://dashboard.example.test")
	handler.Catalog(allowed, allowedRequest)
	if allowed.Code != http.StatusOK || allowed.Header().Get("Access-Control-Allow-Origin") != "https://dashboard.example.test" {
		t.Fatalf("same-origin catalog status=%d headers=%v", allowed.Code, allowed.Header())
	}

	downgrade := httptest.NewRecorder()
	downgradeRequest := httptest.NewRequest(http.MethodGet, "https://dashboard.example.test"+evaluationAPIBase+"/catalog", nil)
	downgradeRequest.Host = "dashboard.example.test"
	downgradeRequest.Header.Set("Origin", "http://dashboard.example.test")
	handler.Catalog(downgrade, downgradeRequest)
	if downgrade.Code != http.StatusForbidden || downgrade.Header().Get("Access-Control-Allow-Origin") != "" {
		t.Fatalf("mixed-scheme Origin status=%d headers=%v", downgrade.Code, downgrade.Header())
	}

	proxied := httptest.NewRecorder()
	proxiedRequest := httptest.NewRequest(http.MethodGet, "http://dashboard.example.test"+evaluationAPIBase+"/catalog", nil)
	proxiedRequest.Host = "dashboard.example.test"
	proxiedRequest.Header.Set("Origin", "https://dashboard.example.test")
	proxiedRequest.Header.Set("X-Forwarded-Proto", "https")
	handler.Catalog(proxied, proxiedRequest)
	if proxied.Code != http.StatusOK || proxied.Header().Get("Access-Control-Allow-Origin") != "https://dashboard.example.test" {
		t.Fatalf("same-origin proxied request status=%d headers=%v", proxied.Code, proxied.Header())
	}

	preflight := httptest.NewRecorder()
	preflightRequest := httptest.NewRequest(http.MethodOptions, "https://dashboard.example.test"+evaluationAPIBase+"/runs", nil)
	preflightRequest.Host = "dashboard.example.test"
	preflightRequest.Header.Set("Origin", "https://dashboard.example.test")
	preflightRequest.Header.Set("Access-Control-Request-Private-Network", "true")
	handler.Runs(preflight, preflightRequest)
	if preflight.Code != http.StatusNoContent || preflight.Header().Get("Access-Control-Allow-Private-Network") != "true" {
		t.Fatalf("same-origin preflight status=%d headers=%v", preflight.Code, preflight.Header())
	}
}
