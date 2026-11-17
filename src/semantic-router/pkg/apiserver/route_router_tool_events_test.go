//go:build !windows

package apiserver

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

type toolEventRuntimeStub struct {
	replayRuntimeStub
	kept []store.ToolEvent
}

func (stub *toolEventRuntimeStub) RecordToolEvents(events []store.ToolEvent) error {
	stub.kept = append(stub.kept, events...)
	return nil
}

const toolEventsBody = `{"harness":"claude-code","session_id":" s-1 ","events":[
	{"tool_call_id":"toolu_1","tool":"Bash","status":"succeeded","approval_requested_at":"2026-11-17T01:00:00Z",
	 "started_at":"2026-11-17T01:00:04.5Z","ended_at":"2026-11-17T01:00:05Z"},
	{"tool_call_id":"toolu_2","tool":"Read","status":"denied","ended_at":"2026-11-17T01:00:01Z"}]}`

func toolEventsServer(t *testing.T, runtime routerruntime.ReplayRuntime) http.Handler {
	t.Helper()
	t.Setenv("VSR_VIEWER_TOKEN", "viewer-token")
	t.Setenv("VSR_OPERATOR_TOKEN", "operator-token")
	cfg := &config.RouterConfig{ManagementAPI: config.ManagementAPIConfig{Auth: config.ManagementAPIAuthConfig{
		Mode: config.ManagementAuthModeBearer,
		Tokens: []config.ManagementAPITokenRef{
			{Env: "VSR_VIEWER_TOKEN", Role: "viewer"},
			{Env: "VSR_OPERATOR_TOKEN", Role: "operator"},
		},
		Roles: config.DefaultManagementAPIRoles(),
	}}}
	registry := routerruntime.NewRegistry(cfg)
	if runtime != nil {
		registry.SetReplayRuntime(runtime)
	}
	return (&ClassificationAPIServer{config: cfg, runtimeRegistry: registry}).setupRoutes()
}

func postToolEvents(handler http.Handler, token, body string) *httptest.ResponseRecorder {
	request := httptest.NewRequest(http.MethodPost, apiObservabilityToolEventsPath, strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	if token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}

// The route takes a batch from a token whose role may ingest learning
// feedback, as outcomes do, keeps it in the generation's replay runtime with
// the session trimmed and the time the Router took it, and answers how many
// it kept. A viewer and an anonymous caller are refused before the runtime.
func TestToolEventsRouteKeepsAnAuthorizedBatch(t *testing.T) {
	runtime := &toolEventRuntimeStub{}
	handler := toolEventsServer(t, runtime)

	if got := postToolEvents(handler, "", toolEventsBody).Code; got != http.StatusUnauthorized {
		t.Fatalf("anonymous status = %d", got)
	}
	if got := postToolEvents(handler, "viewer-token", toolEventsBody).Code; got != http.StatusForbidden {
		t.Fatalf("viewer status = %d", got)
	}
	if len(runtime.kept) != 0 {
		t.Fatal("a refused caller reached the runtime")
	}
	before := time.Now()
	response := postToolEvents(handler, "operator-token", toolEventsBody)
	if response.Code != http.StatusOK || strings.TrimSpace(response.Body.String()) != `{"accepted":2}` {
		t.Fatalf("operator response = %d %s", response.Code, response.Body.String())
	}
	if len(runtime.kept) != 2 {
		t.Fatalf("kept %d events", len(runtime.kept))
	}
	first := runtime.kept[0]
	if first.SessionID != "s-1" || first.Harness != "claude-code" || first.ToolCallID != "toolu_1" ||
		first.StartedAt == nil || first.ApprovalRequestedAt == nil || first.ReceivedAt.Before(before) {
		t.Fatalf("first event = %+v", first)
	}
	if denied := runtime.kept[1]; denied.Status != store.ToolEventDenied || denied.StartedAt != nil {
		t.Fatalf("denied event = %+v", denied)
	}
}

// A batch that carries anything but names, ids, times and outcomes, such as
// a tool's arguments, is refused whole, as is one the run view could not
// place: no session, an unknown status, an end before its start, a time from
// the future, or more calls than a batch holds. With replay off there is
// nowhere to keep it.
func TestToolEventsRouteRefusesWhatItCannotKeep(t *testing.T) {
	runtime := &toolEventRuntimeStub{}
	handler := toolEventsServer(t, runtime)
	event := func(fields string) string {
		return `{"session_id":"s","events":[{"tool_call_id":"c","tool":"Bash",` + fields + `}]}`
	}
	future := time.Now().Add(time.Hour).UTC().Format(time.RFC3339)
	many := `{"session_id":"s","events":[` + strings.Repeat(`{"tool_call_id":"c","tool":"t","status":"failed"},`, maxToolEventsPerBatch) +
		`{"tool_call_id":"c","tool":"t","status":"failed"}]}`
	for name, body := range map[string]string{
		"arguments":  event(`"status":"succeeded","arguments":"{\"command\":\"ls\"}"`),
		"no session": `{"events":[{"tool_call_id":"c","tool":"Bash","status":"failed"}]}`,
		"status":     event(`"status":"ok"`),
		"order":      event(`"status":"failed","started_at":"2026-11-17T01:00:05Z","ended_at":"2026-11-17T01:00:04Z"`),
		"future":     event(`"status":"failed","ended_at":"` + future + `"`),
		"too many":   many,
	} {
		if got := postToolEvents(handler, "operator-token", body); got.Code != http.StatusBadRequest {
			t.Fatalf("%s: status = %d %s", name, got.Code, got.Body.String())
		}
	}
	if len(runtime.kept) != 0 {
		t.Fatal("a refused batch was kept in part")
	}
	if got := postToolEvents(toolEventsServer(t, nil), "operator-token", toolEventsBody).Code; got != http.StatusNotFound {
		t.Fatalf("replay off: status = %d", got)
	}
}
