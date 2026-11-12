package extproc

import (
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func newEmptyBodyTestRouter(streamed bool) *OpenAIRouter {
	return &OpenAIRouter{
		Config:            &config.RouterConfig{RouterOptions: config.RouterOptions{StreamedBodyMode: streamed}},
		ResponseAPIFilter: NewResponseAPIFilter(NewMockResponseStore()),
	}
}

func newEmptyBodyTestContext() *RequestContext {
	return &RequestContext{Headers: map[string]string{}}
}

// sendRequestHeaders drives the real header stage, including client encoding.
func sendRequestHeaders(t *testing.T, router *OpenAIRouter, request *ext_proc.ProcessingRequest_RequestHeaders, ctx *RequestContext) *ext_proc.ProcessingResponse {
	t.Helper()
	stream := NewMockStream(nil)
	require.NoError(t, router.processRequestHeaders(stream, request, ctx))
	require.Len(t, stream.Responses, 1)
	return stream.Responses[0]
}

// sendEmptyRequestBody drives the body stage with the empty EOS frame of a chunked request.
func sendEmptyRequestBody(t *testing.T, router *OpenAIRouter, ctx *RequestContext) *ext_proc.ProcessingResponse {
	t.Helper()
	stream := NewMockStream(nil)
	body := &ext_proc.ProcessingRequest_RequestBody{RequestBody: &ext_proc.HttpBody{EndOfStream: true}}
	require.NoError(t, router.processRequestBody(stream, body, ctx))
	require.Len(t, stream.Responses, 1)
	return stream.Responses[0]
}

func endOfStreamHeaders(method, path string, extra ...*core.HeaderValue) *ext_proc.ProcessingRequest_RequestHeaders {
	request := newRequestHeaders(method, path)
	request.RequestHeaders.Headers.Headers = append(request.RequestHeaders.Headers.Headers, extra...)
	request.RequestHeaders.EndOfStream = true
	return request
}

func emptyBodyCaseName(streamed bool, path string) string {
	if streamed {
		return "streamed " + path
	}
	return "buffered " + path
}

var bodyRequiredInferenceRoutes = []string{
	"/v1/chat/completions",
	"/v1/chat/completions/",
	"/v1/chat/completions?trace=1",
	"/v1/messages",
	"/v1/responses",
	"/openai/v1/chat/completions",
	"/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21",
	"/openai/responses",
	"/openai/v1/responses",
}

func TestEndOfStreamHeadersRejectBodylessInferenceRequests(t *testing.T) {
	for _, streamed := range []bool{false, true} {
		for _, path := range bodyRequiredInferenceRoutes {
			t.Run(emptyBodyCaseName(streamed, path), func(t *testing.T) {
				router := newEmptyBodyTestRouter(streamed)
				ctx := newEmptyBodyTestContext()
				response := sendRequestHeaders(t, router, endOfStreamHeaders("POST", path), ctx)

				immediate := response.GetImmediateResponse()
				require.NotNil(t, immediate, "empty POST must not reach the upstream: %+v", response)
				require.Equal(t, typev3.StatusCode_BadRequest, immediate.GetStatus().GetCode())
				require.Equal(t, headers.ResponsePathError, immediateHeaderValue(response, headers.VSRResponsePath))
				require.NotNil(t, ctx.ImmediateProtocolError)
				require.Equal(t, "body_limit", ctx.ImmediateProtocolError.Code)
				require.Contains(t, string(immediate.GetBody()), ctx.ImmediateProtocolError.Message)
			})
		}
	}
}

func TestEndOfStreamHeaderRejectionMatchesBodyStage(t *testing.T) {
	for _, streamed := range []bool{false, true} {
		for _, path := range bodyRequiredInferenceRoutes {
			t.Run(emptyBodyCaseName(streamed, path), func(t *testing.T) {
				router := newEmptyBodyTestRouter(streamed)
				headerResponse := sendRequestHeaders(t, router, endOfStreamHeaders("POST", path), newEmptyBodyTestContext())

				bodyCtx := newEmptyBodyTestContext()
				continued := sendRequestHeaders(t, router, newRequestHeaders("POST", path), bodyCtx)
				require.NotNil(t, continued.GetRequestHeaders(), "headers without EOS must continue to the body stage")
				bodyResponse := sendEmptyRequestBody(t, router, bodyCtx)

				fromHeaders, fromBody := headerResponse.GetImmediateResponse(), bodyResponse.GetImmediateResponse()
				require.NotNil(t, fromBody, "body stage must reject an empty body: %+v", bodyResponse)
				require.Equal(t, fromBody.GetStatus().GetCode(), fromHeaders.GetStatus().GetCode())
				require.JSONEq(t, string(fromBody.GetBody()), string(fromHeaders.GetBody()))
				require.Equal(t, "body_limit", bodyCtx.ImmediateProtocolError.Code)
			})
		}
	}
}

func TestEndOfStreamHeadersLeaveBodylessRoutesUnchanged(t *testing.T) {
	tests := []struct {
		name   string
		method string
		path   string
		status typev3.StatusCode // zero means the request continues upstream
		filter bool
	}{
		{name: "models list", method: "GET", path: "/v1/models", status: typev3.StatusCode_OK, filter: true},
		{name: "health passthrough", method: "GET", path: "/health", filter: true},
		{name: "non-v1 POST passthrough", method: "POST", path: "/internal/hook", filter: true},
		{name: "GET chat", method: "GET", path: "/v1/chat/completions", status: typev3.StatusCode_MethodNotAllowed, filter: true},
		{name: "HEAD models", method: "HEAD", path: "/v1/models", status: typev3.StatusCode_MethodNotAllowed, filter: true},
		{name: "OPTIONS chat", method: "OPTIONS", path: "/v1/chat/completions", status: typev3.StatusCode_MethodNotAllowed, filter: true},
		{name: "OPTIONS messages", method: "OPTIONS", path: "/v1/messages", status: typev3.StatusCode_MethodNotAllowed, filter: true},
		{name: "GET response object", method: "GET", path: "/v1/responses/resp_missing", status: typev3.StatusCode_NotFound, filter: true},
		{name: "DELETE response object", method: "DELETE", path: "/v1/responses/resp_missing", status: typev3.StatusCode_NotFound, filter: true},
		{name: "GET input items", method: "GET", path: "/v1/responses/resp_missing/input_items", status: typev3.StatusCode_NotFound, filter: true},
		{name: "responses disabled", method: "POST", path: "/v1/responses", status: typev3.StatusCode_NotFound},
		{name: "unknown v1 path", method: "POST", path: "/v1/embeddings", status: typev3.StatusCode_NotFound, filter: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			router := &OpenAIRouter{Config: &config.RouterConfig{}}
			if test.filter {
				router.ResponseAPIFilter = NewResponseAPIFilter(NewMockResponseStore())
			}
			response := sendRequestHeaders(t, router, endOfStreamHeaders(test.method, test.path), newEmptyBodyTestContext())
			if test.status == 0 {
				require.NotNil(t, response.GetRequestHeaders(), "got %+v", response)
				require.Equal(t, ext_proc.CommonResponse_CONTINUE, response.GetRequestHeaders().GetResponse().GetStatus())
				return
			}
			immediate := response.GetImmediateResponse()
			require.NotNil(t, immediate, "got %+v", response)
			require.Equal(t, test.status, immediate.GetStatus().GetCode(), "body: %s", immediate.GetBody())
		})
	}
}

func TestEndOfStreamHeadersHonorSkipProcessing(t *testing.T) {
	router := newRouterWithSkipProcessingGate(true)
	request := endOfStreamHeaders("POST", "/v1/chat/completions", &core.HeaderValue{Key: headers.VSRSkipProcessing, Value: "true"})

	response := sendRequestHeaders(t, router, request, newEmptyBodyTestContext())

	require.NotNil(t, response.GetRequestHeaders(), "opted-out requests pass through unchanged: %+v", response)
}
