package extproc

import (
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
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
