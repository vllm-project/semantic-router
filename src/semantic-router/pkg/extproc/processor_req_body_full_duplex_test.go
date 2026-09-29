package extproc

import (
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func TestFullDuplex_NonEOSChunkDefersResponse(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{
		Headers:               make(map[string]string),
		FullDuplexRequestBody: true,
	}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: []byte(`{"mod`), EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.Nil(t, resp, "full-duplex chunks may be buffered without an intermediate response")
}

func TestFullDuplex_ProtocolConfigDefersBodyResponse(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	stream := NewMockStream(nil)
	req := &ext_proc.ProcessingRequest{
		ProtocolConfig: &ext_proc.ProtocolConfiguration{
			RequestBodyMode: http_ext.ProcessingMode_FULL_DUPLEX_STREAMED,
		},
		Request: &ext_proc.ProcessingRequest_RequestBody{
			RequestBody: &ext_proc.HttpBody{Body: []byte(`{"mod`), EndOfStream: false},
		},
	}

	require.NoError(t, router.handleProcessRequest(stream, req, ctx))
	assert.True(t, ctx.FullDuplexRequestBody)
	assert.Empty(t, stream.Responses)
}

func TestFullDuplex_DisabledAccumulationPassesChunkThrough(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{FullDuplexRequestBody: true}
	chunk := []byte(`{"model":"gpt-4"}`)
	response, err := router.handleRequestBodyDispatch(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: chunk, EndOfStream: true},
	}, ctx)

	require.NoError(t, err)
	streamed := response.GetRequestBody().GetResponse().GetBodyMutation().GetStreamedResponse()
	require.NotNil(t, streamed)
	assert.Equal(t, chunk, streamed.GetBody())
	assert.True(t, streamed.GetEndOfStream())
}

func TestFullDuplex_FinalResponseUsesStreamedMutation(t *testing.T) {
	original := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"hello"}]}`)
	mutated := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}`)
	h := &StreamedBodyHandler{ctx: &RequestContext{FullDuplexRequestBody: true}}
	h.buf.Write(original)
	response := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{Response: &ext_proc.CommonResponse{
				Status: ext_proc.CommonResponse_CONTINUE,
				BodyMutation: &ext_proc.BodyMutation{Mutation: &ext_proc.BodyMutation_Body{
					Body: mutated,
				}},
			}},
		},
	}

	got := h.finalizeResponse(response)
	mutation := got.GetRequestBody().GetResponse().GetBodyMutation()
	streamed := mutation.GetStreamedResponse()
	require.NotNil(t, streamed)
	assert.Equal(t, mutated, streamed.GetBody())
	assert.True(t, streamed.GetEndOfStream())
	assert.Nil(t, mutation.GetBody())
}

func TestFullDuplex_FinalResponseFallsBackToAccumulatedBody(t *testing.T) {
	original := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"hello"}]}`)
	h := &StreamedBodyHandler{ctx: &RequestContext{FullDuplexRequestBody: true}}
	h.buf.Write(original)
	response := newContinueRequestBodyResponse()

	got := h.finalizeResponse(response)
	streamed := got.GetRequestBody().GetResponse().GetBodyMutation().GetStreamedResponse()
	require.NotNil(t, streamed)
	assert.Equal(t, original, streamed.GetBody())
	assert.True(t, streamed.GetEndOfStream())
}

func TestFullDuplex_FinalResponseOmitsHeaderMutation(t *testing.T) {
	original := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"hello"}]}`)
	h := &StreamedBodyHandler{ctx: &RequestContext{FullDuplexRequestBody: true}}
	h.buf.Write(original)
	response := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{Response: &ext_proc.CommonResponse{
				Status: ext_proc.CommonResponse_CONTINUE,
				HeaderMutation: &ext_proc.HeaderMutation{
					RemoveHeaders: []string{"content-length", "x-user-openai-key"},
				},
				BodyMutation: &ext_proc.BodyMutation{Mutation: &ext_proc.BodyMutation_Body{
					Body: original,
				}},
			}},
		},
	}

	got := h.finalizeResponse(response)
	common := got.GetRequestBody().GetResponse()
	require.NotNil(t, common)
	assert.Nil(t, common.GetHeaderMutation(), "header mutation must be nil in full-duplex streamed mode")
	streamed := common.GetBodyMutation().GetStreamedResponse()
	require.NotNil(t, streamed)
	assert.Equal(t, original, streamed.GetBody())
	assert.True(t, streamed.GetEndOfStream())
}

const fullDuplexTestBody = `{"model":"worker","messages":[{"role":"user","content":"hello"}]}`

// fullDuplexRoutingRouter dispatches "worker" to a provider backend with its own
// path prefix and credential, so the body stage produces routing header
// mutations that differ from the client's request and clears the route cache.
func fullDuplexRoutingRouter() *OpenAIRouter {
	router := routingTestRouter("worker")
	router.Config.StreamedBodyMode = true
	router.Config.ClearRouteCache = true
	router.Config.ProviderProfiles["provider"] = config.ProviderProfile{Type: "openai", BaseURL: "http://127.0.0.1:8000/provider/v1"}
	model := router.Config.ModelConfig["worker"]
	model.AccessKeys = map[string]string{"openai": "provider-credential"}
	router.Config.ModelConfig["worker"] = model
	router.Cache = cache.NewInMemoryCache(cache.InMemoryCacheOptions{Enabled: false})
	return router
}

func headersRequest(mode http_ext.ProcessingMode_BodySendMode, method, path string, endOfStream bool, extra ...*core.HeaderValue) *ext_proc.ProcessingRequest {
	values := []*core.HeaderValue{
		{Key: ":method", RawValue: []byte(method)},
		{Key: ":path", RawValue: []byte(path)},
		{Key: "content-type", RawValue: []byte("application/json")},
		{Key: "authorization", RawValue: []byte("Bearer client-credential")},
	}
	return &ext_proc.ProcessingRequest{
		ProtocolConfig: &ext_proc.ProtocolConfiguration{RequestBodyMode: mode},
		Request: &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{
			Headers:     &core.HeaderMap{Headers: append(values, extra...)},
			EndOfStream: endOfStream,
		}},
	}
}

func fullDuplexHeadersRequest(endOfStream bool, extra ...*core.HeaderValue) *ext_proc.ProcessingRequest {
	return headersRequest(http_ext.ProcessingMode_FULL_DUPLEX_STREAMED, "POST", "/v1/chat/completions", endOfStream, extra...)
}

func bodyRequest(body string, endOfStream bool) *ext_proc.ProcessingRequest {
	return &ext_proc.ProcessingRequest{Request: &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(body), EndOfStream: endOfStream},
	}}
}

func trailersRequest() *ext_proc.ProcessingRequest {
	return &ext_proc.ProcessingRequest{Request: &ext_proc.ProcessingRequest_RequestTrailers{
		RequestTrailers: &ext_proc.HttpTrailers{},
	}}
}

func runProcessRequests(t *testing.T, router *OpenAIRouter, ctx *RequestContext, stream *MockStream, requests ...*ext_proc.ProcessingRequest) {
	t.Helper()
	for _, request := range requests {
		require.NoError(t, router.handleProcessRequest(stream, request, ctx))
	}
}

// assertRoutedHeaderReply checks that the header reply carries the provider
// route chosen from the body, not the client's request.
func assertRoutedHeaderReply(t *testing.T, response *ext_proc.ProcessingResponse) {
	t.Helper()
	common := response.GetRequestHeaders().GetResponse()
	require.NotNil(t, common, "the header reply must come first")
	mutation := common.GetHeaderMutation()
	model, _ := accountingHeaderValue(mutation, headers.SelectedModel)
	assert.Equal(t, "worker", model)
	path, _ := accountingHeaderValue(mutation, ":path")
	assert.Equal(t, "/provider/v1/chat/completions", path)
	credential, _ := accountingHeaderValue(mutation, "Authorization")
	assert.Equal(t, "Bearer provider-credential", credential)
	_, ok := accountingHeaderValue(mutation, "accept-encoding")
	assert.True(t, ok, "the header stage's own mutation is kept")
	assert.True(t, common.GetClearRouteCache())
}

func TestFullDuplex_HeaderReplyWaitsForEndOfBody(t *testing.T) {
	router := fullDuplexRoutingRouter()
	ctx := &RequestContext{Headers: make(map[string]string)}
	stream := NewMockStream(nil)

	runProcessRequests(t, router, ctx, stream, fullDuplexHeadersRequest(false), bodyRequest(fullDuplexTestBody[:30], false))
	assert.Empty(t, stream.Responses, "the header reply must wait until the body is routed")

	runProcessRequests(t, router, ctx, stream, bodyRequest(fullDuplexTestBody[30:], true))
	require.Len(t, stream.Responses, 2)
	assertRoutedHeaderReply(t, stream.Responses[0])
	body := stream.Responses[1].GetRequestBody().GetResponse()
	require.NotNil(t, body)
	assert.Nil(t, body.GetHeaderMutation())
	assert.False(t, body.GetClearRouteCache())
	streamed := body.GetBodyMutation().GetStreamedResponse()
	require.NotNil(t, streamed)
	assert.True(t, streamed.GetEndOfStream())
}

func TestFullDuplex_TrailersEndTheBody(t *testing.T) {
	ctx := &RequestContext{Headers: make(map[string]string)}
	stream := NewMockStream(nil)

	runProcessRequests(t, fullDuplexRoutingRouter(), ctx, stream,
		fullDuplexHeadersRequest(false), bodyRequest(fullDuplexTestBody, false), trailersRequest())

	require.Len(t, stream.Responses, 3, "headers, body, trailers")
	assertRoutedHeaderReply(t, stream.Responses[0])
	streamed := stream.Responses[1].GetRequestBody().GetResponse().GetBodyMutation().GetStreamedResponse()
	require.NotNil(t, streamed)
	assert.Contains(t, string(streamed.GetBody()), `"worker"`)
	assert.False(t, streamed.GetEndOfStream(), "the trailers, not the body, end the stream")
	assert.NotNil(t, stream.Responses[2].GetRequestTrailers())
	assert.Nil(t, ctx.StreamedBody, "the finished body's handler is released")
}

func TestFullDuplex_RequestWithoutBodyDataGoesThroughTheBodyStage(t *testing.T) {
	// HTTP/2 headers followed by trailers: Envoy sends no body chunk. The empty
	// body is rejected instead of being forwarded on the client's own route.
	ctx := &RequestContext{Headers: make(map[string]string)}
	stream := NewMockStream(nil)

	runProcessRequests(t, fullDuplexRoutingRouter(), ctx, stream, fullDuplexHeadersRequest(false), trailersRequest())

	require.Len(t, stream.Responses, 1)
	assert.Equal(t, typev3.StatusCode_BadRequest, stream.Responses[0].GetImmediateResponse().GetStatus().GetCode())
}

func TestFullDuplex_ImmediateResponseReplacesHeldHeaderReply(t *testing.T) {
	tests := []struct {
		name       string
		maxBytes   int64
		expire     bool
		wantStatus typev3.StatusCode
		requests   []*ext_proc.ProcessingRequest
	}{
		{name: "invalid body at end of stream", wantStatus: typev3.StatusCode_BadRequest, requests: []*ext_proc.ProcessingRequest{
			fullDuplexHeadersRequest(false), bodyRequest(`{"model":`, true),
		}},
		{name: "invalid body ended by trailers", wantStatus: typev3.StatusCode_BadRequest, requests: []*ext_proc.ProcessingRequest{
			fullDuplexHeadersRequest(false), bodyRequest(`{"model":`, false), trailersRequest(),
		}},
		{name: "body over the size limit mid-stream", maxBytes: 10, wantStatus: typev3.StatusCode_PayloadTooLarge, requests: []*ext_proc.ProcessingRequest{
			fullDuplexHeadersRequest(false), bodyRequest(fullDuplexTestBody, false),
		}},
		{name: "body deadline passed when trailers arrive", expire: true, wantStatus: typev3.StatusCode_RequestTimeout, requests: []*ext_proc.ProcessingRequest{
			fullDuplexHeadersRequest(false), bodyRequest(fullDuplexTestBody, false), trailersRequest(),
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			router := fullDuplexRoutingRouter()
			router.Config.MaxStreamedBodyBytes = test.maxBytes
			ctx := &RequestContext{Headers: make(map[string]string)}
			stream := NewMockStream(nil)
			last := len(test.requests) - 1
			runProcessRequests(t, router, ctx, stream, test.requests[:last]...)
			if test.expire && ctx.StreamedBody != nil {
				ctx.StreamedBody.deadline = time.Now().Add(-time.Second)
			}
			runProcessRequests(t, router, ctx, stream, test.requests[last])

			require.Len(t, stream.Responses, 1)
			assert.Equal(t, test.wantStatus, stream.Responses[0].GetImmediateResponse().GetStatus().GetCode())
		})
	}
}

func TestFullDuplex_RequestsWithoutBodyRoutingReplyAtOnce(t *testing.T) {
	passthrough := fullDuplexRoutingRouter()
	passthrough.Config.StreamedBodyMode = false
	skipping := newRouterWithSkipProcessingGate(true)
	skipping.Config.StreamedBodyMode = true
	tests := []struct {
		name      string
		router    *OpenAIRouter
		request   *ext_proc.ProcessingRequest
		immediate bool
	}{
		{name: "headers end the stream", router: fullDuplexRoutingRouter(), request: fullDuplexHeadersRequest(true)},
		{
			name: "header-stage immediate response", router: fullDuplexRoutingRouter(), immediate: true,
			request: headersRequest(http_ext.ProcessingMode_FULL_DUPLEX_STREAMED, "GET", "/v1/models", false),
		},
		{name: "skip processing", router: skipping, request: fullDuplexHeadersRequest(false,
			&core.HeaderValue{Key: headers.VSRSkipProcessing, RawValue: []byte("true")})},
		{name: "full duplex without body accumulation", router: passthrough, request: fullDuplexHeadersRequest(false)},
		{
			name: "STREAMED body mode", router: fullDuplexRoutingRouter(),
			request: headersRequest(http_ext.ProcessingMode_STREAMED, "POST", "/v1/chat/completions", false),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx := &RequestContext{Headers: make(map[string]string)}
			stream := NewMockStream(nil)
			runProcessRequests(t, test.router, ctx, stream, test.request)

			require.Len(t, stream.Responses, 1)
			if test.immediate {
				assert.NotNil(t, stream.Responses[0].GetImmediateResponse())
			} else {
				assert.NotNil(t, stream.Responses[0].GetRequestHeaders())
			}
		})
	}
}

func TestFullDuplex_PassthroughTrailersFollowTheBody(t *testing.T) {
	passthrough := fullDuplexRoutingRouter()
	passthrough.Config.StreamedBodyMode = false
	skipping := newRouterWithSkipProcessingGate(true)
	skipping.Config.StreamedBodyMode = true
	tests := []struct {
		name    string
		router  *OpenAIRouter
		headers *ext_proc.ProcessingRequest
	}{
		{name: "without body accumulation", router: passthrough, headers: fullDuplexHeadersRequest(false)},
		{name: "skip processing", router: skipping, headers: fullDuplexHeadersRequest(false,
			&core.HeaderValue{Key: headers.VSRSkipProcessing, RawValue: []byte("true")})},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx := &RequestContext{Headers: make(map[string]string)}
			stream := NewMockStream(nil)

			// The Router never parses a pass-through chunk, so any bytes pass.
			chunk := "opaque chunk \x00\x01 not parsed"
			runProcessRequests(t, test.router, ctx, stream, test.headers, bodyRequest(chunk, false), trailersRequest())

			require.Len(t, stream.Responses, 3, "headers, body, trailers")
			assert.NotNil(t, stream.Responses[0].GetRequestHeaders())
			streamed := stream.Responses[1].GetRequestBody().GetResponse().GetBodyMutation().GetStreamedResponse()
			require.NotNil(t, streamed)
			assert.Equal(t, []byte(chunk), streamed.GetBody(), "pass-through forwards the exact bytes")
			assert.False(t, streamed.GetEndOfStream())
			assert.NotNil(t, stream.Responses[2].GetRequestTrailers())
		})
	}
}

// failAfterRequestsStream calls fail instead of returning EOF once its
// requests run out, and records how many responses were sent by then.
type failAfterRequestsStream struct {
	MockStream
	fail       func() error
	sentBefore int
}

func (s *failAfterRequestsStream) Recv() (*ext_proc.ProcessingRequest, error) {
	if s.RecvIndex >= len(s.Requests) {
		s.sentBefore = len(s.Responses)
		return nil, s.fail()
	}
	return s.MockStream.Recv()
}

func TestFullDuplex_HeldHeaderReplyOnStreamError(t *testing.T) {
	// With failure_mode_allow, Envoy forwards the request after a stream error.
	// The header stage's removal of internal headers applies only if the held
	// reply reached Envoy before the error.
	tests := []struct {
		name string
		fail func() error
		code codes.Code
		// wantHeaderReply: the Router ends the stream itself, so the header
		// reply goes out before the error status. Otherwise grpc-go has already
		// sent the status, and nothing more is sent.
		wantHeaderReply bool
	}{
		{
			// Recv panics, as in panicOnRecvStream, standing in for a panic in
			// the body stage.
			name:            "panic in the Router",
			fail:            func() error { panic("simulated CGO OOM panic") },
			code:            codes.Internal,
			wantHeaderReply: true,
		},
		{
			// A real stream is already closed here; this one stays open, so a
			// send after the error would show up.
			name:            "receive error",
			fail:            func() error { return status.Error(codes.ResourceExhausted, "grpc: received message larger than max") },
			code:            codes.ResourceExhausted,
			wantHeaderReply: false,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			stream := &failAfterRequestsStream{
				MockStream: *NewMockStream([]*ext_proc.ProcessingRequest{fullDuplexHeadersRequest(false)}),
				fail:       test.fail,
			}

			assert.Equal(t, test.code, status.Code(fullDuplexRoutingRouter().Process(stream)))

			if !test.wantHeaderReply {
				assert.Len(t, stream.Responses, stream.sentBefore, "nothing is sent after a receive error")
				return
			}
			require.Len(t, stream.Responses, 1)
			mutation := stream.Responses[0].GetRequestHeaders().GetResponse().GetHeaderMutation()
			assert.Contains(t, mutation.GetRemoveHeaders(), headers.VSRInternalAuth)
		})
	}
}
