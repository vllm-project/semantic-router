package router

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

func TestRedactReplayResponseBodyRemovesSensitiveFields(t *testing.T) {
	t.Parallel()

	body := []byte(`{
		"id":"replay-1",
		"request_body":"secret request",
		"response_body":"secret response",
		"tool_trace":{
			"flow":"User Query -> Tool Calling -> Tool Execute -> LLM Answer",
			"stage":"assistant_final_response",
			"tool_names":["lookup_price"],
			"steps":[
				{"type":"user_input","text":"sensitive user prompt"},
				{"type":"assistant_tool_call","tool_name":"lookup_price","arguments":"{\"ticker\":\"NVDA\"}"},
				{"type":"client_tool_result","tool_name":"lookup_price","text":"sensitive tool result"},
				{"type":"assistant_final_response","text":"sensitive final answer"}
			]
		}
	}`)

	redactedBody, changed, err := redactReplayResponseBody(body)
	if err != nil {
		t.Fatalf("redactReplayResponseBody() error = %v", err)
	}
	if !changed {
		t.Fatalf("redactReplayResponseBody() changed = false, want true")
	}

	var payload map[string]any
	if err := json.Unmarshal(redactedBody, &payload); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	if got := payload["request_body"]; got != "" {
		t.Fatalf("request_body = %#v, want empty string", got)
	}
	if got := payload["response_body"]; got != "" {
		t.Fatalf("response_body = %#v, want empty string", got)
	}

	toolTrace, ok := payload["tool_trace"].(map[string]any)
	if !ok {
		t.Fatalf("tool_trace missing or invalid: %#v", payload["tool_trace"])
	}
	if got := toolTrace["flow"]; got != "" {
		t.Fatalf("tool_trace.flow = %#v, want empty string", got)
	}

	steps, ok := toolTrace["steps"].([]any)
	if !ok || len(steps) != 4 {
		t.Fatalf("tool_trace.steps = %#v, want 4 steps", toolTrace["steps"])
	}

	for index, rawStep := range steps {
		step, ok := rawStep.(map[string]any)
		if !ok {
			t.Fatalf("step %d invalid: %#v", index, rawStep)
		}
		if got, ok := step["text"]; ok && got != "" {
			t.Fatalf("step %d text = %#v, want empty string", index, got)
		}
		if got, ok := step["arguments"]; ok && got != "" {
			t.Fatalf("step %d arguments = %#v, want empty string", index, got)
		}
	}
}

func TestRedactRouterReplayResponseSkipsWriteCapableUsers(t *testing.T) {
	t.Parallel()

	originalBody := []byte(`{"id":"replay-1","request_body":"secret request","tool_trace":{"flow":"secret flow"}}`)
	req, err := http.NewRequest(http.MethodGet, "http://dashboard.local/v1/router_replay/replay-1", nil)
	if err != nil {
		t.Fatalf("http.NewRequest() error = %v", err)
	}
	req = req.WithContext(auth.WithAuthContext(req.Context(), auth.AuthContext{
		Role:  auth.RoleWrite,
		Perms: map[string]bool{auth.PermConfigWrite: true},
	}))

	resp := &http.Response{
		StatusCode:    http.StatusOK,
		Header:        http.Header{"Content-Type": []string{"application/json"}},
		Body:          io.NopCloser(bytes.NewReader(originalBody)),
		ContentLength: int64(len(originalBody)),
		Request:       req,
	}

	if redactErr := redactRouterReplayResponse(resp); redactErr != nil {
		t.Fatalf("redactRouterReplayResponse() error = %v", redactErr)
	}

	actualBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("io.ReadAll() error = %v", err)
	}
	if string(actualBody) != string(originalBody) {
		t.Fatalf("response body changed for write-capable user: got %s want %s", actualBody, originalBody)
	}
}
