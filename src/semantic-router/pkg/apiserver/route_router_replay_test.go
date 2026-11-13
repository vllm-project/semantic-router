//go:build !windows

package apiserver

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
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

	anonymous := serveReplayManagementRequest(mux, "/api/v1/observability/replays/replay-1", "")
	assertReplayStatus(t, anonymous, http.StatusUnauthorized, "anonymous")
	if runtime.calls != 0 {
		t.Fatalf("runtime calls after anonymous request = %d, want 0", runtime.calls)
	}

	viewerPath := "/api/v1/observability/replays/replay-1?source=dashboard"
	viewer := serveReplayManagementRequest(mux, viewerPath, viewerToken)
	assertViewerReplayResponse(t, viewer, runtime, viewerPath)

	operator := serveReplayManagementRequest(mux, "/api/v1/observability/replays/replay-1", operatorToken)
	assertDetailedReplayResponse(t, operator, runtime.response.Body, "operator")

	admin := serveReplayManagementRequest(mux, "/api/v1/observability/replays/replay-1", adminToken)
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
			path: "/api/v1/observability/replays?showDetails=true",
			body: `{"object":"router_replay.list","data":[{"id":"replay-1","decision":"safe-decision","request_body":"private-canary","prompt":"private-canary","pii_entities":["private-canary"],"hallucination_spans":["private-canary"],"outcomes":[{"source":"user","target":"model","target_ref":"private-canary","verdict":"failed","reason":"private-canary","metadata":{"note":"private-canary"}}],"session_policy":{"note":"private-canary"},"route_diagnostics":{"selected_model":"model-a","annotations":{"note":"private-canary"},"prepared_dispatch":{"version":1,"wire_format":"openai.chat.v1","sha256":"private-canary","byte_length":137}}}]}`,
		},
		{
			name: "detail",
			path: "/api/v1/observability/replays/replay-1",
			body: `{"id":"replay-1","decision":"safe-decision","response_body":"private-canary","tool_definitions":"private-canary","hallucination_span_details":[{"text":"private-canary","explanation":"private-canary","severity":2}],"route_diagnostics":{"selected_model":"model-a","signal_errors":{"signal":"private-canary"},"prepared_dispatch":{"version":1,"wire_format":"openai.chat.v1","sha256":"private-canary","byte_length":137}},"learning":{"protection":{"method":"protection","reason":"private-canary"}}}`,
		},
		{
			name: "trajectory",
			path: "/api/v1/observability/replays/trajectory?session_id=session-1",
			body: `{"object":"router_replay.trajectory","session_id":"session-1","messages":[{"role":"user","content":"private-canary"},{"role":"assistant","tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"private-canary"}}]},{"role":"tool","content":"private-canary","tool_call_id":"call-1"}]}`,
		},
		{
			name: "captured tool results",
			path: "/api/v1/observability/replays/replay-2",
			body: `{"id":"replay-2","decision":"agent-loop","request_body":"{\"messages\":[{\"role\":\"tool\",\"content\":\"private-canary\"}]}","tool_trace":{"steps":[{"type":"client_tool_result","tool_call_id":"call-1","status":"succeeded","output":"private-canary","truncated":true}]}}`,
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
		httptest.NewRequest(http.MethodGet, "/api/v1/observability/replays", nil),
	)
	if recorder.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404; body=%s", recorder.Code, recorder.Body.String())
	}
}

// Every Replay read route serves a principal without replay.detail the same
// redacted view, driven through the Router's own replay handler.
func TestRouterReplayEveryReadRouteRedactsRecordedTextForViewer(t *testing.T) {
	const (
		viewerAccessFixture   = "viewer-replay-routes-token"
		operatorAccessFixture = "operator-replay-routes-token"
		private               = "private-canary"
	)
	t.Setenv("VSR_VIEWER_ROUTES_TOKEN", viewerAccessFixture)
	t.Setenv("VSR_OPERATOR_ROUTES_TOKEN", operatorAccessFixture)
	cfg := &config.RouterConfig{ManagementAPI: config.ManagementAPIConfig{Auth: config.ManagementAPIAuthConfig{
		Mode: config.ManagementAuthModeBearer,
		Tokens: []config.ManagementAPITokenRef{
			{Env: "VSR_VIEWER_ROUTES_TOKEN", Role: "viewer"},
			{Env: "VSR_OPERATOR_ROUTES_TOKEN", Role: "operator"},
		},
		Roles: config.DefaultManagementAPIRoles(),
	}}}
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(true, true, 0)
	if _, err := recorder.AddRecord(routerreplay.RoutingRecord{
		ID: "replay-1", SessionID: "session-1", Decision: "safe-decision", SelectedModel: "model-a",
		Prompt: private, RequestBody: private, ResponseBody: private,
		SessionPolicy: map[string]interface{}{"identity": map[string]interface{}{"agent": map[string]interface{}{"hash": private}}},
		RouteDiagnostics: &routerreplay.RouteDiagnostics{
			SelectedModel: "model-a", SelectionReasoning: private, SessionReason: private, DecisionReason: private,
		},
		ToolTrace: &routerreplay.ToolTrace{Steps: []routerreplay.ToolTraceStep{
			{Type: "user_input", Source: "request", Role: "user", Text: private},
			{Type: "assistant_final_response", Source: "response", Role: "assistant", Text: private},
		}},
	}); err != nil {
		t.Fatal(err)
	}
	router := &extproc.OpenAIRouter{ReplayRecorder: recorder}
	extproc.NewRouterServiceForSnapshot(router, nil)
	registry := routerruntime.NewRegistry(cfg)
	registry.SetReplayRuntime(router)
	mux := (&ClassificationAPIServer{config: cfg, runtimeRegistry: registry}).setupRoutes()

	for _, test := range []struct{ route, path string }{
		{"list", "/api/v1/observability/replays"},
		{"list with details", "/api/v1/observability/replays?showDetails=true"},
		{"record", "/api/v1/observability/replays/replay-1"},
		{"trajectory", "/api/v1/observability/replays/trajectory?session_id=session-1"},
		{"aggregate", "/api/v1/observability/replays/aggregate"},
		{"dataset", "/api/v1/observability/replays/dataset?seed=s&split=train:1"},
		{"judge tasks", "/api/v1/observability/replays/dataset/judge-tasks?seed=s&split=train:1&blinding_key=k"},
	} {
		t.Run(test.route, func(t *testing.T) {
			viewer := serveReplayManagementRequest(mux, test.path, viewerAccessFixture)
			if test.route == "judge tasks" {
				assertReplayStatus(t, viewer, http.StatusForbidden, "viewer")
				return
			}
			assertReplayStatus(t, viewer, http.StatusOK, "viewer")
			if bytes.Contains(viewer.Body.Bytes(), []byte(private)) {
				t.Fatalf("viewer read recorded text on %s: %s", test.path, viewer.Body.String())
			}
		})
	}
	// The detail view of the same session still carries the reasoning, so the
	// viewer checks above read a route that returns it.
	operator := serveReplayManagementRequest(mux, "/api/v1/observability/replays/trajectory?session_id=session-1", operatorAccessFixture)
	assertReplayStatus(t, operator, http.StatusOK, "operator")
	if !bytes.Contains(operator.Body.Bytes(), []byte(`"selection_reasoning":"`+private)) {
		t.Fatalf("operator trajectory lost the routing reasoning: %s", operator.Body.String())
	}
}
