//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

type replayRuntimeStub struct {
	method        string
	requestTarget string
	response      routerruntime.ReplayResponse
	handled       bool
	calls         int
}

func (runtime *replayRuntimeStub) HandleReplayRequest(
	method string,
	requestTarget string,
) (routerruntime.ReplayResponse, bool) {
	runtime.calls++
	runtime.method = method
	runtime.requestTarget = requestTarget
	return runtime.response, runtime.handled
}

func TestRouterReplayManagementRouteRequiresAuthAndRedactsViewerPayload(t *testing.T) {
	const (
		viewerToken   = "viewer-replay-token"
		operatorToken = "operator-replay-token"
		adminToken    = "admin-replay-token"
	)
	t.Setenv("VSR_VIEWER_TOKEN", viewerToken)
	t.Setenv("VSR_OPERATOR_TOKEN", operatorToken)
	t.Setenv("VSR_ADMIN_TOKEN", adminToken)
	management := config.ManagementAPIConfig{Auth: config.ManagementAPIAuthConfig{
		Mode: config.ManagementAuthModeBearer,
		Tokens: []config.ManagementAPITokenRef{
			{Env: "VSR_VIEWER_TOKEN", Role: "viewer"},
			{Env: "VSR_OPERATOR_TOKEN", Role: "operator"},
			{Env: "VSR_ADMIN_TOKEN", Role: "admin"},
		},
		Roles: config.DefaultManagementAPIRoles(),
	}}
	cfg := &config.RouterConfig{ManagementAPI: management}
	runtime := &replayRuntimeStub{
		response: routerruntime.ReplayResponse{
			StatusCode: http.StatusOK,
			Body:       []byte(`{"id":"replay-1","request_body":"private prompt","response_body":"private response"}`),
		},
		handled: true,
	}
	registry := routerruntime.NewRegistry(cfg)
	registry.SetReplayRuntime(runtime)
	server := &ClassificationAPIServer{config: cfg, runtimeRegistry: registry}
	mux := server.setupRoutes()

	anonymous := serveReplayManagementRequest(mux, "/v1/router_replay/replay-1", "")
	assertReplayStatus(t, anonymous, http.StatusUnauthorized, "anonymous")
	if runtime.calls != 0 {
		t.Fatalf("runtime calls after anonymous request = %d, want 0", runtime.calls)
	}

	viewerPath := "/v1/router_replay/replay-1?source=dashboard"
	viewer := serveReplayManagementRequest(mux, viewerPath, viewerToken)
	assertViewerReplayResponse(t, viewer, runtime, viewerPath)

	operator := serveReplayManagementRequest(mux, "/v1/router_replay/replay-1", operatorToken)
	assertDetailedReplayResponse(t, operator, runtime.response.Body, "operator")

	admin := serveReplayManagementRequest(mux, "/v1/router_replay/replay-1", adminToken)
	assertDetailedReplayResponse(t, admin, runtime.response.Body, "admin")
}

func serveReplayManagementRequest(handler http.Handler, path, token string) *httptest.ResponseRecorder {
	request := httptest.NewRequest(http.MethodGet, path, nil)
	if token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}

func assertReplayStatus(t *testing.T, response *httptest.ResponseRecorder, status int, principal string) {
	t.Helper()
	if response.Code != status {
		t.Fatalf("%s status = %d, want %d body=%s", principal, response.Code, status, response.Body.String())
	}
}

func assertViewerReplayResponse(
	t *testing.T,
	response *httptest.ResponseRecorder,
	runtime *replayRuntimeStub,
	wantTarget string,
) {
	t.Helper()
	assertReplayStatus(t, response, http.StatusOK, "viewer")
	if runtime.method != http.MethodGet || runtime.requestTarget != wantTarget {
		t.Fatalf("forwarded replay request = %s %s", runtime.method, runtime.requestTarget)
	}
	if response.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("Cache-Control = %q, want no-store", response.Header().Get("Cache-Control"))
	}
	var payload map[string]interface{}
	if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode viewer payload: %v", err)
	}
	if payload["request_body"] != "" || payload["response_body"] != "" {
		t.Fatalf("viewer payload was not redacted: %#v", payload)
	}
}

func assertDetailedReplayResponse(t *testing.T, response *httptest.ResponseRecorder, body []byte, principal string) {
	t.Helper()
	assertReplayStatus(t, response, http.StatusOK, principal)
	if !json.Valid(response.Body.Bytes()) || response.Body.String() != string(body) {
		t.Fatalf("%s replay payload changed: %s", principal, response.Body.String())
	}
}

func TestRouterReplayViewerRedactionCoversListDetailAndTrajectory(t *testing.T) {
	const (
		viewerAccessFixture = "viewer-replay-shapes-token"
		private             = "private-canary"
	)
	t.Setenv("VSR_VIEWER_SHAPES_TOKEN", viewerAccessFixture)
	management := config.ManagementAPIConfig{Auth: config.ManagementAPIAuthConfig{
		Mode: config.ManagementAuthModeBearer,
		Tokens: []config.ManagementAPITokenRef{
			{Env: "VSR_VIEWER_SHAPES_TOKEN", Role: "viewer"},
		},
		Roles: config.DefaultManagementAPIRoles(),
	}}
	cfg := &config.RouterConfig{ManagementAPI: management}
	runtime := &replayRuntimeStub{handled: true}
	registry := routerruntime.NewRegistry(cfg)
	registry.SetReplayRuntime(runtime)
	server := &ClassificationAPIServer{config: cfg, runtimeRegistry: registry}
	mux := server.setupRoutes()

	tests := []struct {
		name string
		path string
		body string
	}{
		{
			name: "list with details",
			path: "/v1/router_replay?showDetails=true",
			body: `{"object":"router_replay.list","data":[{"id":"replay-1","decision":"safe-decision","request_body":"private-canary","prompt":"private-canary","pii_entities":["private-canary"],"hallucination_spans":["private-canary"],"outcomes":[{"source":"user","target":"model","target_ref":"private-canary","verdict":"failed","reason":"private-canary","metadata":{"note":"private-canary"}}],"session_policy":{"note":"private-canary"},"route_diagnostics":{"selected_model":"model-a","annotations":{"note":"private-canary"}}}]}`,
		},
		{
			name: "detail",
			path: "/v1/router_replay/replay-1",
			body: `{"id":"replay-1","decision":"safe-decision","response_body":"private-canary","tool_definitions":"private-canary","hallucination_span_details":[{"text":"private-canary","explanation":"private-canary","severity":2}],"route_diagnostics":{"selected_model":"model-a","signal_errors":{"signal":"private-canary"}},"learning":{"protection":{"method":"protection","reason":"private-canary"}}}`,
		},
		{
			name: "trajectory",
			path: "/v1/router_replay/trajectory?session_id=session-1",
			body: `{"object":"router_replay.trajectory","session_id":"session-1","messages":[{"role":"user","content":"private-canary"},{"role":"assistant","tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"private-canary"}}]},{"role":"tool","content":"private-canary","tool_call_id":"call-1"}]}`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			runtime.response = routerruntime.ReplayResponse{StatusCode: http.StatusOK, Body: []byte(test.body)}
			request := httptest.NewRequest(http.MethodGet, test.path, nil)
			request.Header.Set("Authorization", "Bearer "+viewerAccessFixture)
			response := httptest.NewRecorder()
			mux.ServeHTTP(response, request)

			if response.Code != http.StatusOK {
				t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
			}
			if bytes.Contains(response.Body.Bytes(), []byte(private)) {
				t.Fatalf("viewer response leaked private canary: %s", response.Body.String())
			}
			if !json.Valid(response.Body.Bytes()) {
				t.Fatalf("viewer response is not JSON: %s", response.Body.String())
			}
			if runtime.requestTarget != test.path {
				t.Fatalf("forwarded target = %q, want %q", runtime.requestTarget, test.path)
			}
		})
	}
}

func TestRouterReplayManagementRouteReturnsNotFoundWithoutRuntime(t *testing.T) {
	cfg := &config.RouterConfig{ManagementAPI: config.ManagementAPIConfig{
		Auth: config.ManagementAPIAuthConfig{Mode: config.ManagementAuthModeDisabled},
	}}
	server := &ClassificationAPIServer{
		config:          cfg,
		runtimeRegistry: routerruntime.NewRegistry(cfg),
	}
	recorder := httptest.NewRecorder()
	server.setupRoutes().ServeHTTP(
		recorder,
		httptest.NewRequest(http.MethodGet, "/v1/router_replay", nil),
	)
	if recorder.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404; body=%s", recorder.Code, recorder.Body.String())
	}
}
