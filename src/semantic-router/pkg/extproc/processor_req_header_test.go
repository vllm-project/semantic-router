package extproc

import (
	"encoding/json"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type requestHeaderTestCase struct {
	name                  string
	method                string
	path                  string
	expectImmediate       bool
	expectResponseAPICtx  bool
	expectContinueHeaders bool
}

func TestExtractResponseIDFromPath(t *testing.T) {
	tests := []struct {
		name string
		path string
		want string
	}{
		{name: "plain response path", path: "/v1/responses/resp_123", want: "resp_123"},
		{name: "response path with query", path: "/v1/responses/resp_123?foo=bar", want: "resp_123"},
		{name: "response path with trailing slash", path: "/v1/responses/resp_123/", want: "resp_123"},
		{name: "input items path should not match", path: "/v1/responses/resp_123/input_items", want: ""},
		{name: "non response id should not match", path: "/v1/responses/abc123", want: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := extractResponseIDFromPath(tt.path); got != tt.want {
				t.Fatalf("extractResponseIDFromPath(%q) = %q, want %q", tt.path, got, tt.want)
			}
		})
	}
}

func TestExtractResponseIDFromInputItemsPath(t *testing.T) {
	tests := []struct {
		name string
		path string
		want string
	}{
		{name: "plain input items path", path: "/v1/responses/resp_123/input_items", want: "resp_123"},
		{name: "input items path with query", path: "/v1/responses/resp_123/input_items?foo=bar", want: "resp_123"},
		{name: "response path without suffix", path: "/v1/responses/resp_123", want: ""},
		{name: "non response id should not match", path: "/v1/responses/abc123/input_items", want: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := extractResponseIDFromInputItemsPath(tt.path); got != tt.want {
				t.Fatalf("extractResponseIDFromInputItemsPath(%q) = %q, want %q", tt.path, got, tt.want)
			}
		})
	}
}

func TestHandleRequestHeadersResponseAPIEndpoints(t *testing.T) {
	router := &OpenAIRouter{
		ResponseAPIFilter: NewResponseAPIFilter(NewMockResponseStore()),
	}

	tests := []requestHeaderTestCase{
		{
			name:            "get response returns immediate response",
			method:          "GET",
			path:            "/v1/responses/resp_test",
			expectImmediate: true,
		},
		{
			name:            "get input items returns immediate response",
			method:          "GET",
			path:            "/v1/responses/resp_test/input_items",
			expectImmediate: true,
		},
		{
			name:            "delete response returns immediate response",
			method:          "DELETE",
			path:            "/v1/responses/resp_test",
			expectImmediate: true,
		},
		{
			name:                  "post response marks body translation context",
			method:                "POST",
			path:                  "/v1/responses",
			expectResponseAPICtx:  true,
			expectContinueHeaders: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			requestHeaders := newRequestHeaders(tt.method, tt.path)
			ctx := &RequestContext{Headers: make(map[string]string)}
			response, err := router.handleRequestHeaders(requestHeaders, ctx)
			if err != nil {
				t.Fatalf("handleRequestHeaders failed: %v", err)
			}

			assertRequestHeaderResponse(t, tt, response, ctx)
		})
	}
}

func newRequestHeaders(method string, path string) *ext_proc.ProcessingRequest_RequestHeaders {
	return &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":method", Value: method},
					{Key: ":path", Value: path},
				},
			},
		},
	}
}

func assertRequestHeaderResponse(
	t *testing.T,
	tt requestHeaderTestCase,
	response *ext_proc.ProcessingResponse,
	ctx *RequestContext,
) {
	t.Helper()

	if tt.expectImmediate && response.GetImmediateResponse() == nil {
		t.Fatalf("expected immediate response for %s %s", tt.method, tt.path)
	}
	if tt.expectContinueHeaders {
		if response.GetRequestHeaders() == nil {
			t.Fatalf("expected continue headers response for %s %s", tt.method, tt.path)
		}
		if response.GetRequestHeaders().Response.Status != ext_proc.CommonResponse_CONTINUE {
			t.Fatalf("expected CONTINUE status, got %v", response.GetRequestHeaders().Response.Status)
		}
	}
	if tt.expectResponseAPICtx && (ctx.ResponseAPICtx == nil || !ctx.ResponseAPICtx.IsResponseAPIRequest) {
		t.Fatalf("expected response API context for %s %s", tt.method, tt.path)
	}
}

func TestHandleRequestHeadersSetsLooperAndStreamingFlags(t *testing.T) {
	router := &OpenAIRouter{}
	requestHeaders := &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":method", Value: "POST"},
					{Key: ":path", Value: "/v1/chat/completions"},
					{Key: "accept", Value: "text/event-stream"},
					{Key: "x-vsr-looper-request", Value: "true"},
					{Key: "x-request-id", Value: "req-123"},
				},
			},
		},
	}

	ctx := &RequestContext{Headers: make(map[string]string)}
	response, err := router.handleRequestHeaders(requestHeaders, ctx)
	if err != nil {
		t.Fatalf("handleRequestHeaders failed: %v", err)
	}
	if response.GetRequestHeaders() == nil {
		t.Fatal("expected continue request headers response")
	}
	if !ctx.ExpectStreamingResponse {
		t.Fatal("expected streaming expectation to be detected")
	}
	if !ctx.LooperRequest {
		t.Fatal("expected looper request to be detected")
	}
	if ctx.RequestID != "req-123" {
		t.Fatalf("expected request ID to be captured, got %q", ctx.RequestID)
	}
}

func TestHandleRequestHeadersReturnsMethodNotAllowedForChatCompletionsGet(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{Headers: make(map[string]string)}

	response, err := router.handleRequestHeaders(newRequestHeaders("GET", "/v1/chat/completions"), ctx)
	if err != nil {
		t.Fatalf("handleRequestHeaders failed: %v", err)
	}

	assertImmediateErrorResponse(t, response, typev3.StatusCode_MethodNotAllowed, "method not allowed")
}

func TestHandleRequestHeadersReturnsMethodNotAllowedForModelsPost(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{Headers: make(map[string]string)}

	response, err := router.handleRequestHeaders(newRequestHeaders("POST", "/v1/models"), ctx)
	if err != nil {
		t.Fatalf("handleRequestHeaders failed: %v", err)
	}

	assertImmediateErrorResponse(t, response, typev3.StatusCode_MethodNotAllowed, "method not allowed")
}

func TestHandleRequestHeadersReturnsNotFoundForUnknownV1Endpoint(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{Headers: make(map[string]string)}

	response, err := router.handleRequestHeaders(newRequestHeaders("POST", "/v1/does-not-exist"), ctx)
	if err != nil {
		t.Fatalf("handleRequestHeaders failed: %v", err)
	}

	assertImmediateErrorResponse(t, response, typev3.StatusCode_NotFound, "endpoint not found")
}

func assertImmediateErrorResponse(
	t *testing.T,
	response *ext_proc.ProcessingResponse,
	wantStatus typev3.StatusCode,
	wantMessage string,
) {
	t.Helper()

	immediate := response.GetImmediateResponse()
	if immediate == nil {
		t.Fatal("expected immediate response")
	}
	if immediate.Status == nil || immediate.Status.Code != wantStatus {
		t.Fatalf("expected status %v, got %v", wantStatus, immediate.Status)
	}

	var decoded struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(immediate.Body, &decoded); err != nil {
		t.Fatalf("failed to decode error response: %v", err)
	}
	if decoded.Error.Message != wantMessage {
		t.Fatalf("expected error message %q, got %q", wantMessage, decoded.Error.Message)
	}
}

func TestValidateRequestHeaders_V1Messages(t *testing.T) {
	router := &OpenAIRouter{}

	tests := []struct {
		name       string
		method     string
		path       string
		wantStatus typev3.StatusCode // 0 means request should pass validation
	}{
		{name: "POST /v1/messages allowed", method: "POST", path: "/v1/messages"},
		{name: "POST /v1/messages with query allowed", method: "POST", path: "/v1/messages?stream=true"},
		{name: "GET /v1/messages returns 405", method: "GET", path: "/v1/messages", wantStatus: typev3.StatusCode_MethodNotAllowed},
		{name: "PUT /v1/messages returns 405", method: "PUT", path: "/v1/messages", wantStatus: typev3.StatusCode_MethodNotAllowed},
		{name: "DELETE /v1/messages returns 405", method: "DELETE", path: "/v1/messages", wantStatus: typev3.StatusCode_MethodNotAllowed},
		{name: "/v1/messages/anything still 404", method: "POST", path: "/v1/messages/count_tokens", wantStatus: typev3.StatusCode_NotFound},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			response := router.validateRequestHeaders(tt.method, tt.path)
			if tt.wantStatus == 0 {
				if response != nil {
					t.Fatalf("expected nil response for %s %s, got %+v", tt.method, tt.path, response)
				}
				return
			}
			if response == nil {
				t.Fatalf("expected immediate response for %s %s, got nil", tt.method, tt.path)
			}
			immediate := response.GetImmediateResponse()
			if immediate == nil || immediate.Status == nil || immediate.Status.Code != tt.wantStatus {
				t.Fatalf("expected status %v for %s %s, got %v", tt.wantStatus, tt.method, tt.path, immediate.Status)
			}
		})
	}
}

func TestDetectClientProtocol(t *testing.T) {
	tests := []struct {
		name      string
		path      string
		wantProto string
	}{
		{name: "anthropic messages path", path: "/v1/messages", wantProto: config.ClientProtocolAnthropic},
		{name: "anthropic messages with query", path: "/v1/messages?stream=true", wantProto: config.ClientProtocolAnthropic},
		{name: "anthropic count tokens subpath", path: "/v1/messages/count_tokens", wantProto: config.ClientProtocolAnthropic},
		{name: "openai chat completions", path: "/v1/chat/completions", wantProto: ""},
		{name: "openai responses", path: "/v1/responses", wantProto: ""},
		{name: "openai models", path: "/v1/models", wantProto: ""},
		{name: "root path", path: "/", wantProto: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := &RequestContext{Headers: make(map[string]string)}
			detectClientProtocol(tt.path, ctx)
			if ctx.ClientProtocol != tt.wantProto {
				t.Fatalf("detectClientProtocol(%q): got ClientProtocol=%q, want %q", tt.path, ctx.ClientProtocol, tt.wantProto)
			}
		})
	}
}

func TestHandleRequestHeadersSetsClientProtocol(t *testing.T) {
	router := &OpenAIRouter{}

	tests := []struct {
		name      string
		path      string
		wantProto string
	}{
		{name: "anthropic messages sets protocol", path: "/v1/messages", wantProto: config.ClientProtocolAnthropic},
		{name: "openai chat keeps default", path: "/v1/chat/completions", wantProto: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			requestHeaders := newRequestHeaders("POST", tt.path)
			ctx := &RequestContext{Headers: make(map[string]string)}
			_, err := router.handleRequestHeaders(requestHeaders, ctx)
			if err != nil {
				t.Fatalf("handleRequestHeaders failed: %v", err)
			}
			if ctx.ClientProtocol != tt.wantProto {
				t.Fatalf("got ClientProtocol=%q, want %q", ctx.ClientProtocol, tt.wantProto)
			}
		})
	}
}
