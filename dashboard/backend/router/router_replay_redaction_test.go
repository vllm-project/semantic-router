package router

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

func requireReplayObject(t *testing.T, value any, label string) map[string]any {
	t.Helper()
	object, ok := value.(map[string]any)
	if !ok {
		t.Fatalf("%s missing or invalid: %#v", label, value)
	}
	return object
}

func requireReplayArray(t *testing.T, value any, label string, length int) []any {
	t.Helper()
	items, ok := value.([]any)
	if !ok || len(items) != length {
		t.Fatalf("%s = %#v, want %d items", label, value, length)
	}
	return items
}

func assertEmptyReplayFields(t *testing.T, object map[string]any, fields ...string) {
	t.Helper()
	for _, field := range fields {
		if got := object[field]; got != "" {
			t.Fatalf("%s = %#v, want empty string", field, got)
		}
	}
}

func assertRedactedTraceStep(t *testing.T, rawStep any, index int) map[string]any {
	t.Helper()
	step := requireReplayObject(t, rawStep, fmt.Sprintf("step %d", index))
	for _, field := range []string{"text", "arguments", "raw_arguments", "raw_output", "output"} {
		if got, ok := step[field]; ok && got != "" {
			t.Fatalf("step %d %s = %#v, want empty string", index, field, got)
		}
	}
	if got := step["content_redacted"]; got != true {
		t.Fatalf("step %d content_redacted = %#v, want true", index, got)
	}
	return step
}

func assertRedactedToolTrace(t *testing.T, payload map[string]any) {
	t.Helper()
	toolTrace := requireReplayObject(t, payload["tool_trace"], "tool_trace")
	if got := toolTrace["flow"]; got != "User Query -> Tool Calling -> Tool Execute -> LLM Answer" {
		t.Fatalf("tool_trace.flow = %#v, want preserved flow", got)
	}
	steps := requireReplayArray(t, toolTrace["steps"], "tool_trace.steps", 4)
	for index, rawStep := range steps {
		assertRedactedTraceStep(t, rawStep, index)
	}
	toolResult := requireReplayObject(t, steps[2], "step 2")
	if got := toolResult["status"]; got != redactedToolResultStatusSucceeded {
		t.Fatalf("client_tool_result status = %#v, want %q", got, redactedToolResultStatusSucceeded)
	}
}

func TestRedactReplayResponseBodyRemovesSensitiveFields(t *testing.T) {
	t.Parallel()

	body := []byte(`{
		"id":"replay-1",
		"request_body":"secret request",
		"response_body":"secret response",
		"prompt":"secret prompt extracted before body truncation",
		"tool_definitions":"[{\"name\":\"private_tool\"}]",
		"tool_trace":{
			"flow":"User Query -> Tool Calling -> Tool Execute -> LLM Answer",
			"stage":"assistant_final_response",
			"tool_names":["lookup_price"],
			"steps":[
				{"type":"user_input","text":"sensitive user prompt"},
				{"type":"assistant_tool_call","tool_name":"lookup_price","arguments":"{\"ticker\":\"NVDA\"}","raw_arguments":"{\"ticker\":\"NVDA\"}"},
				{"type":"client_tool_result","tool_name":"lookup_price","text":"sensitive tool result","output":"{\"price\":950.25}","raw_output":"{\"price\":950.25}"},
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

	assertEmptyReplayFields(t, payload, "request_body", "response_body", "prompt", "tool_definitions")
	assertRedactedToolTrace(t, payload)
}

func TestRedactReplayResponseBodyRedactsListPayload(t *testing.T) {
	t.Parallel()

	body := []byte(`{
		"object":"router_replay.list",
		"count":1,
		"data":[
			{
				"id":"replay-1",
				"request_body":"secret request",
				"response_body":"secret response",
				"tool_trace":{
					"flow":"User Query -> Tool Calling",
					"tool_names":["lookup_price"],
					"steps":[{"type":"user_input","text":"sensitive prompt"}]
				}
			}
		]
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

	data, ok := payload["data"].([]any)
	if !ok || len(data) != 1 {
		t.Fatalf("data = %#v, want one record", payload["data"])
	}
	record, ok := data[0].(map[string]any)
	if !ok {
		t.Fatalf("record invalid: %#v", data[0])
	}
	if got := record["request_body"]; got != "" {
		t.Fatalf("request_body = %#v, want empty string", got)
	}
	if got := record["response_body"]; got != "" {
		t.Fatalf("response_body = %#v, want empty string", got)
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

func TestRedactRouterReplayResponseFailsClosedForInvalidJSON(t *testing.T) {
	t.Parallel()

	body := []byte(`{"request_body":`)
	req, err := http.NewRequest(http.MethodGet, "http://dashboard.local/v1/router_replay/replay-1", nil)
	if err != nil {
		t.Fatalf("http.NewRequest() error = %v", err)
	}
	req = req.WithContext(auth.WithAuthContext(req.Context(), auth.AuthContext{
		Role:  auth.RoleRead,
		Perms: map[string]bool{auth.PermReplayRead: true},
	}))
	resp := &http.Response{
		StatusCode:    http.StatusOK,
		Header:        http.Header{"Content-Type": []string{"application/json"}},
		Body:          io.NopCloser(bytes.NewReader(body)),
		ContentLength: int64(len(body)),
		Request:       req,
	}

	if redactErr := redactRouterReplayResponse(resp); redactErr == nil {
		t.Fatal("redactRouterReplayResponse() error = nil, want fail-closed error")
	}
}

func TestRedactRouterReplayResponseDoesNotTrustContentType(t *testing.T) {
	t.Parallel()

	for _, contentType := range []string{"", "text/plain"} {
		t.Run(contentType, func(t *testing.T) {
			t.Parallel()
			body := []byte(`{"id":"replay-1","request_body":"private-canary"}`)
			req, err := http.NewRequest(http.MethodGet, "http://dashboard.local/v1/router_replay/replay-1", nil)
			if err != nil {
				t.Fatalf("http.NewRequest() error = %v", err)
			}
			req = req.WithContext(auth.WithAuthContext(req.Context(), auth.AuthContext{
				Role:  auth.RoleRead,
				Perms: map[string]bool{auth.PermReplayRead: true},
			}))
			resp := &http.Response{
				StatusCode:    http.StatusOK,
				Header:        make(http.Header),
				Body:          io.NopCloser(bytes.NewReader(body)),
				ContentLength: int64(len(body)),
				Request:       req,
			}
			if contentType != "" {
				resp.Header.Set("Content-Type", contentType)
			}

			if redactErr := redactRouterReplayResponse(resp); redactErr != nil {
				t.Fatalf("redactRouterReplayResponse() error = %v", redactErr)
			}
			redacted, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatalf("io.ReadAll() error = %v", err)
			}
			if bytes.Contains(redacted, []byte("private-canary")) {
				t.Fatalf("Replay response leaked private content for Content-Type %q: %s", contentType, redacted)
			}
		})
	}
}

func TestShouldRedactRouterReplayResponseRejectsLookalikePrefix(t *testing.T) {
	t.Parallel()

	req, err := http.NewRequest(http.MethodGet, "http://dashboard.local/v1/router_replayevil", nil)
	if err != nil {
		t.Fatalf("http.NewRequest() error = %v", err)
	}
	resp := &http.Response{StatusCode: http.StatusOK, Request: req}
	if shouldRedactRouterReplayResponse(resp) {
		t.Fatal("lookalike Router Replay prefix must not enter the replay redaction contract")
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
