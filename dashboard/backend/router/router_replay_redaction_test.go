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
				{"type":"assistant_tool_call","tool_name":"lookup_price","arguments":"{\"ticker\":\"NVDA\"}","raw_arguments":"{\"ticker\":\"NVDA\"}"},
				{"type":"client_tool_result","tool_name":"lookup_price","text":"sensitive tool result","raw_output":"{\"price\":950.25}"},
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
	if got := toolTrace["flow"]; got != "User Query -> Tool Calling -> Tool Execute -> LLM Answer" {
		t.Fatalf("tool_trace.flow = %#v, want preserved flow", got)
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
		if got, ok := step["raw_arguments"]; ok && got != "" {
			t.Fatalf("step %d raw_arguments = %#v, want empty string", index, got)
		}
		if got, ok := step["raw_output"]; ok && got != "" {
			t.Fatalf("step %d raw_output = %#v, want empty string", index, got)
		}
	}

	if got := steps[0].(map[string]any)["content_redacted"]; got != true {
		t.Fatalf("user_input content_redacted = %#v, want true", got)
	}
	if got := steps[1].(map[string]any)["content_redacted"]; got != true {
		t.Fatalf("assistant_tool_call content_redacted = %#v, want true", got)
	}
	if got := steps[2].(map[string]any)["content_redacted"]; got != true {
		t.Fatalf("client_tool_result content_redacted = %#v, want true", got)
	}
	if got := steps[2].(map[string]any)["status"]; got != redactedToolResultStatusSucceeded {
		t.Fatalf("client_tool_result status = %#v, want %q", got, redactedToolResultStatusSucceeded)
	}
	if got := steps[3].(map[string]any)["content_redacted"]; got != true {
		t.Fatalf("assistant_final_response content_redacted = %#v, want true", got)
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

func TestToolTraceResultStatus(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		text any
		want string
	}{
		{name: "meaningful text", text: "{\"temperature\":\"18C\"}", want: redactedToolResultStatusSucceeded},
		{name: "failure prefix", text: "Tool execution failed: timeout", want: redactedToolResultStatusFailed},
		{name: "null text", text: "null", want: redactedToolResultStatusFailed},
		{name: "empty text", text: "", want: redactedToolResultStatusFailed},
		{name: "missing text", text: nil, want: redactedToolResultStatusFailed},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := toolTraceResultStatus(tt.text); got != tt.want {
				t.Fatalf("toolTraceResultStatus(%#v) = %q, want %q", tt.text, got, tt.want)
			}
		})
	}
}
