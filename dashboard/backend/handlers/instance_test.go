package handlers

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func instanceTestSocket(t *testing.T, handler http.Handler) {
	t.Helper()
	socket := filepath.Join(t.TempDir(), "control.sock")
	listener, err := net.Listen("unix", socket)
	if err != nil {
		t.Fatal(err)
	}
	server := &http.Server{Handler: handler, ReadHeaderTimeout: time.Second}
	go func() { _ = server.Serve(listener) }()
	t.Cleanup(func() { _ = server.Close() })
	t.Setenv(instanceSocketEnv, socket)
}

func TestInstanceUnixStatusAndModels(t *testing.T) {
	var seen []string
	instanceTestSocket(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = append(seen, r.Method+" "+r.URL.Path)
		if r.Header.Get("Authorization") != "" {
			t.Error("browser token leaked to controller")
		}
		switch r.URL.Path {
		case "/status":
			_, _ = io.WriteString(w, `{"ownership":"managed","observed_mode":"engine","active_deployment":"primary"}`)
		case "/models":
			_, _ = io.WriteString(w, `{"object":"list","data":[{"id":"primary","ready":true,"surfaces":["decisions"]}]}`)
		}
	}))
	handler := InstanceHandler()
	for _, route := range []struct {
		method, path, body string
		status             int
	}{{"GET", "/api/instance", "", 200}, {"GET", "/api/instance/models", "", 200}, {"POST", "/api/instance/deploy", `{"mode":"engine","deployment":"primary","request_id":"retry"}`, 405}, {"GET", "/api/instance/deploy", "", 404}} {
		response := httptest.NewRecorder()
		request := httptest.NewRequest(route.method, route.path, strings.NewReader(route.body))
		request.Header.Set("Authorization", "Bearer browser-secret")
		handler(response, request)
		if response.Code != route.status {
			t.Fatalf("%s => %d", route.path, response.Code)
		}
	}
	if len(seen) != 2 {
		t.Fatal(seen)
	}
	if !instanceEngineActive(context.Background()) {
		t.Fatal("engine identity lost")
	}
	response := httptest.NewRecorder()
	InstanceHandler()(response, httptest.NewRequest("PATCH", "/api/instance", strings.NewReader(`{}`)))
	if response.Code != 405 || response.Header().Get("Allow") != "GET" {
		t.Fatal("instance mutation allowed")
	}
}

func TestInstanceMissingControllerNeverClaimsManagedOrEngine(t *testing.T) {
	t.Setenv(instanceSocketEnv, "")
	t.Setenv("KUBERNETES_SERVICE_HOST", "cluster")
	response := httptest.NewRecorder()
	InstanceHandler()(response, httptest.NewRequest("GET", "/api/instance", nil))
	var state map[string]any
	_ = json.Unmarshal(response.Body.Bytes(), &state)
	if state["ownership"] != "kubernetes" || state["observed_mode"] != "unknown" {
		t.Fatal(state)
	}
	response = httptest.NewRecorder()
	InstanceHandler()(response, httptest.NewRequest("POST", "/api/instance/deploy", strings.NewReader(`{}`)))
	if response.Code != 405 {
		t.Fatal("unmanaged operation accepted")
	}
	response = httptest.NewRecorder()
	InstanceHandler()(response, httptest.NewRequest("GET", "/api/instance/models", nil))
	if response.Code != 503 {
		t.Fatal("missing inventory reported success")
	}
}

func TestInstanceEngineUsesPersistentFrontendForInference(t *testing.T) {
	instanceTestSocket(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/status" {
			t.Errorf("inference reached controller: %s", r.URL.Path)
		}
		_, _ = io.WriteString(w, `{"observed_mode":"engine"}`)
	}))
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/diagnostics/models/systemone" {
			t.Errorf("unexpected frontend route %s", r.URL.Path)
		}
		if r.Method == http.MethodGet {
			_, _ = io.WriteString(w, `{"deployments":[{"id":"primary","ready":true,"model":"example/vela","surfaces":["decisions"],"question_types":["noul"]}]}`)
			return
		}
		var request map[string]json.RawMessage
		_ = json.NewDecoder(r.Body).Decode(&request)
		if string(request["deployment"]) != `"primary"` {
			t.Error("deployment lost")
		}
		_, _ = io.WriteString(w, `{"answers":{"q":{"answer":"yes"}},"spans":{},"sets":{}}`)
	}))
	defer upstream.Close()
	handler := DecisionModelHandler(upstream.URL)
	response := httptest.NewRecorder()
	handler(response, httptest.NewRequest("GET", "/api/decision-model/capabilities", nil))
	if response.Code != 200 || !strings.Contains(response.Body.String(), `"primary"`) {
		t.Fatal(response.Body.String())
	}
	response = httptest.NewRecorder()
	handler(response, httptest.NewRequest("POST", "/api/decision-model/test", strings.NewReader(`{"deployment":"primary","request":{"model":"ignored","state":"text","questions":{"q":{"type":"noul"}}}}`)))
	if response.Code != 200 || !strings.Contains(response.Body.String(), `"answers"`) {
		t.Fatalf("%d %s", response.Code, response.Body.String())
	}
}
