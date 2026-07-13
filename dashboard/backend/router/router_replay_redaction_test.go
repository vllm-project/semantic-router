package router

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/proxy"
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

func TestRouterReplayProxyRedactsResponseWithinLimit(t *testing.T) {
	t.Parallel()

	const requestCanary = "bounded replay request canary"
	const responseCanary = "bounded replay response canary"
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(
			w,
			`{"id":"replay-1","request_body":"`+requestCanary+`","response_body":"`+responseCanary+`"}`,
		)
	}))
	defer upstream.Close()

	handler, err := proxy.NewReverseProxy(upstream.URL, "", false)
	if err != nil {
		t.Fatalf("NewReverseProxy() error = %v", err)
	}
	attachRouterReplayResponseRedaction(handler)
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(
		recorder,
		httptest.NewRequest(http.MethodGet, "http://dashboard.local/v1/router_replay/replay-1", nil),
	)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body=%q", recorder.Code, http.StatusOK, recorder.Body.String())
	}
	if strings.Contains(recorder.Body.String(), requestCanary) ||
		strings.Contains(recorder.Body.String(), responseCanary) {
		t.Fatalf("redacted response leaked a canary: %q", recorder.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(recorder.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode redacted response: %v", err)
	}
	if payload["request_body"] != "" || payload["response_body"] != "" {
		t.Fatalf("redacted response = %#v", payload)
	}
	if got, want := recorder.Header().Get("Content-Length"), strconv.Itoa(recorder.Body.Len()); got != want {
		t.Fatalf("Content-Length = %q, want %q", got, want)
	}
}

func TestRouterReplayProxyRejectsOversizedChunkedResponse(t *testing.T) {
	t.Parallel()

	const canary = "oversized-router-replay-canary"
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		payloadSize := int(routerReplayRedactionResponseByteLimit) + 1
		payload := strings.Repeat("x", payloadSize-len(canary)) + canary
		_, _ = io.CopyN(w, strings.NewReader(payload), int64(payloadSize))
	}))
	defer upstream.Close()

	handler, err := proxy.NewReverseProxy(upstream.URL, "", false)
	if err != nil {
		t.Fatalf("NewReverseProxy() error = %v", err)
	}
	attachRouterReplayResponseRedaction(handler)
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(
		recorder,
		httptest.NewRequest(http.MethodGet, "http://dashboard.local/v1/router_replay", nil),
	)

	if recorder.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want %d; body=%q", recorder.Code, http.StatusBadGateway, recorder.Body.String())
	}
	if got := recorder.Body.String(); got != "Bad Gateway\n" || strings.Contains(got, canary) {
		t.Fatalf("oversized upstream response was not replaced by a generic error: %q", got)
	}
}

func TestRouterReplayProxyFailsClosedOnMalformedJSON(t *testing.T) {
	t.Parallel()

	const canary = "malformed-unredacted-canary"
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"request_body":"`+canary+`"`)
	}))
	defer upstream.Close()

	handler, err := proxy.NewReverseProxy(upstream.URL, "", false)
	if err != nil {
		t.Fatalf("NewReverseProxy() error = %v", err)
	}
	attachRouterReplayResponseRedaction(handler)
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(
		recorder,
		httptest.NewRequest(http.MethodGet, "http://dashboard.local/v1/router_replay/replay-1", nil),
	)

	if recorder.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want %d; body=%q", recorder.Code, http.StatusBadGateway, recorder.Body.String())
	}
	if got := recorder.Body.String(); got != "Bad Gateway\n" || strings.Contains(got, canary) {
		t.Fatalf("malformed upstream response failed open: %q", got)
	}
}

func TestReadBoundedRouterReplayResponseBodyAlwaysCloses(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		body    string
		wantErr error
	}{
		{name: "within limit", body: `{"data":[]}`},
		{
			name:    "over limit",
			body:    strings.Repeat("x", int(routerReplayRedactionResponseByteLimit)+1),
			wantErr: errRouterReplayResponseTooLarge,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			body := &closeTrackingReplayBody{Reader: strings.NewReader(test.body)}
			_, err := readBoundedRouterReplayResponseBody(body)
			if !errors.Is(err, test.wantErr) {
				t.Fatalf("error = %v, want %v", err, test.wantErr)
			}
			if !body.closed {
				t.Fatal("upstream response body was not closed")
			}
		})
	}
}

type closeTrackingReplayBody struct {
	io.Reader
	closed bool
}

func (b *closeTrackingReplayBody) Close() error {
	b.closed = true
	return nil
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
