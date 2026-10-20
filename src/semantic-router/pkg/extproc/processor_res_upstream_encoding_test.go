package extproc

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"strconv"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// A gzip reply is decoded before any response phase reads it, so the client,
// usage accounting and every later phase see what an identity reply gives.
func TestGzipReplyMatchesIdentityReply(t *testing.T) {
	for _, internal := range []bool{false, true} {
		for _, client := range extProcMatrixFormats {
			if internal && client != llmprotocol.OpenAIChatV1 {
				continue
			}
			for _, backend := range extProcMatrixFormats {
				t.Run(fmt.Sprintf("internal_%t/%s/%s", internal, client, backend), func(t *testing.T) {
					reply := extProcResponseFixture(backend)
					newContext := func() *RequestContext {
						ctx := looperTransportContext(backend, false)
						ctx.LooperRequest = internal
						ctx.SourceFormat = client
						return ctx
					}
					identityCtx, gzipCtx := newContext(), newContext()
					identity := runBufferedReply(t, &OpenAIRouter{}, identityCtx, 200, reply, "")
					gzipped := runBufferedReply(t, &OpenAIRouter{}, gzipCtx, 200, gzipBytes(t, reply), "gzip")

					require.Nil(t, gzipped.body.GetImmediateResponse(), "gzip reply was rejected: %v", gzipped.body)
					require.Equal(t, string(identity.clientBody), string(gzipped.clientBody))
					require.NotNil(t, gzipped.body.GetResponseBody().GetResponse().GetBodyMutation(),
						"the decoded reply must replace the provider's compressed bytes")
					assertDecodedReplyHeaders(t, gzipped.body)
					require.Equal(t, identityCtx.SemanticResponse, gzipCtx.SemanticResponse)
					require.Equal(t, identityCtx.ResponseEnvelope, gzipCtx.ResponseEnvelope)
				})
			}
		}
	}
}

// Router Replay and usage accounting record the decoded reply, as they record
// an identity reply.
func TestGzipReplyKeepsReplayAndUsage(t *testing.T) {
	reply := extProcResponseFixture(llmprotocol.AnthropicMessagesV1)
	run := func(encoding string, body []byte) (*RequestContext, routerreplay.RoutingRecord) {
		recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
		recorder.SetCapturePolicy(true, true, 1<<20)
		id, err := recorder.AddRecord(routerreplay.RoutingRecord{ID: "replay-" + encoding})
		require.NoError(t, err)
		router := &OpenAIRouter{ReplayRecorder: recorder}
		ctx := looperTransportContext(llmprotocol.AnthropicMessagesV1, false)
		ctx.LooperRequest = false
		ctx.RouterReplayID = id
		ctx.RouterReplayPluginConfig = &config.RouterReplayPluginConfig{Enabled: true, CaptureResponseBody: true}
		result := runBufferedReply(t, router, ctx, 200, body, encoding)
		require.Nil(t, result.body.GetImmediateResponse())
		record, ok := recorder.GetRecord(id)
		require.True(t, ok)
		return ctx, record
	}
	identityCtx, identityRecord := run("", reply)
	gzipCtx, gzipRecord := run("gzip", gzipBytes(t, reply))

	require.NotEmpty(t, identityRecord.ResponseBody)
	require.Equal(t, identityRecord.ResponseBody, gzipRecord.ResponseBody)
	require.Equal(t, identityRecord.ResponseStatus, gzipRecord.ResponseStatus)
	require.Equal(t, identityRecord.PromptTokens, gzipRecord.PromptTokens)
	require.Equal(t, identityRecord.CompletionTokens, gzipRecord.CompletionTokens)
	require.NotNil(t, gzipCtx.SemanticResponse)
	require.Equal(t, identityCtx.SemanticResponse.Usage, gzipCtx.SemanticResponse.Usage)
}

// A provider error arrives compressed too. Its decoded message reaches the
// client through the transport-error contract, under the provider's status.
func TestGzipErrorReplyKeepsStatusAndMessage(t *testing.T) {
	for _, internal := range []bool{false, true} {
		t.Run(fmt.Sprintf("internal_%t", internal), func(t *testing.T) {
			ctx := looperTransportContext(llmprotocol.OpenAIChatV1, false)
			ctx.LooperRequest = internal
			errorBody := []byte(`{"error":{"message":"backend overloaded","type":"server_error"}}`)
			result := runBufferedReply(t, &OpenAIRouter{}, ctx, 503, gzipBytes(t, errorBody), "gzip")

			require.Nil(t, result.body.GetImmediateResponse(), "the provider's status must be kept: %v", result.body)
			require.Contains(t, string(result.clientBody), "backend overloaded")
			assertDecodedReplyHeaders(t, result.body)
		})
	}
}

// A reply that claims gzip but does not decode is unreadable: a success
// becomes 502 as any unreadable reply does, and an error keeps its status
// with the generic error envelope instead of compressed bytes.
func TestUndecodableGzipReply(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		ctx := looperTransportContext(llmprotocol.OpenAIChatV1, false)
		ctx.LooperRequest = false
		result := runBufferedReply(t, &OpenAIRouter{}, ctx, 200, []byte("not gzip"), "gzip")
		immediate := result.body.GetImmediateResponse()
		require.NotNil(t, immediate)
		require.EqualValues(t, 502, immediate.GetStatus().GetCode())
		require.Contains(t, string(immediate.GetBody()), "The selected model returned an invalid response")
	})
	t.Run("error", func(t *testing.T) {
		ctx := looperTransportContext(llmprotocol.OpenAIChatV1, false)
		ctx.LooperRequest = false
		truncated := gzipBytes(t, []byte(`{"error":{"message":"rate limited"}}`))
		result := runBufferedReply(t, &OpenAIRouter{}, ctx, 429, truncated[:len(truncated)/2], "gzip")
		require.Nil(t, result.body.GetImmediateResponse(), "the provider's status must be kept: %v", result.body)
		require.Contains(t, string(result.clientBody), `"error"`)
		require.NotContains(t, string(result.clientBody), "\x1f\x8b", "compressed bytes reached the client")
		assertDecodedReplyHeaders(t, result.body)
	})
}

// A reply that inflates past the provider body limit fails like an oversized
// identity reply, without decoding more than the limit.
func TestGzipReplyOverTheBodyLimit(t *testing.T) {
	reply := extProcResponseFixture(llmprotocol.OpenAIChatV1)
	previous := maxDecodedUpstreamBodyBytes
	maxDecodedUpstreamBodyBytes = int64(len(reply) - 1)
	t.Cleanup(func() { maxDecodedUpstreamBodyBytes = previous })

	ctx := looperTransportContext(llmprotocol.OpenAIChatV1, false)
	ctx.LooperRequest = false
	result := runBufferedReply(t, &OpenAIRouter{}, ctx, 200, gzipBytes(t, reply), "gzip")
	immediate := result.body.GetImmediateResponse()
	require.NotNil(t, immediate)
	require.EqualValues(t, 502, immediate.GetStatus().GetCode())

	maxDecodedUpstreamBodyBytes = int64(len(reply))
	ctx = looperTransportContext(llmprotocol.OpenAIChatV1, false)
	ctx.LooperRequest = false
	result = runBufferedReply(t, &OpenAIRouter{}, ctx, 200, gzipBytes(t, reply), "gzip")
	require.Nil(t, result.body.GetImmediateResponse(), "a reply at the limit must decode")
}

func TestIsGzipContentEncoding(t *testing.T) {
	tests := []struct {
		values []string
		want   bool
	}{
		{values: []string{"gzip"}, want: true},
		{values: []string{"GZip"}, want: true},
		{values: []string{" x-gzip "}, want: true},
		{values: []string{"identity, gzip"}, want: true},
		{values: []string{"identity", "gzip"}, want: true},
		{values: nil, want: false},
		{values: []string{"identity"}, want: false},
		{values: []string{"br"}, want: false},
		{values: []string{"gzip, br"}, want: false},
		{values: []string{"gzip", "gzip"}, want: false},
	}
	for _, tc := range tests {
		headerMap := &core.HeaderMap{Headers: []*core.HeaderValue{{Key: ":status", RawValue: []byte("200")}}}
		for _, value := range tc.values {
			headerMap.Headers = append(headerMap.Headers, &core.HeaderValue{Key: "Content-Encoding", RawValue: []byte(value)})
		}
		require.Equal(t, tc.want, isGzipContentEncoding(headerMap), "Content-Encoding %q", tc.values)
	}
}

// Each reply is judged by its own headers: an identity reply that follows a
// gzip one on the same request context is not decoded.
func TestUpstreamEncodingIsJudgedPerReply(t *testing.T) {
	ctx := looperTransportContext(llmprotocol.OpenAIChatV1, false)
	ctx.LooperRequest = false
	router := &OpenAIRouter{}
	_, err := router.handleResponseHeaders(upstreamReplyHeaders(503, "application/json", "gzip", 10), ctx)
	require.NoError(t, err)

	reply := extProcResponseFixture(llmprotocol.OpenAIChatV1)
	result := runBufferedReply(t, router, ctx, 200, reply, "")
	require.Nil(t, result.body.GetImmediateResponse())
	require.NotContains(t, result.body.GetResponseBody().GetResponse().GetHeaderMutation().GetRemoveHeaders(), "content-encoding")
}

// Skip-processing and streamed replies are outside the router's buffered
// decode: their bytes and headers pass through untouched.
func TestGzipReplyOutsideBufferedProcessingPassesThrough(t *testing.T) {
	t.Run("skip_processing", func(t *testing.T) {
		ctx := looperTransportContext(llmprotocol.OpenAIChatV1, false)
		ctx.LooperRequest = false
		ctx.SkipProcessing = true
		compressed := gzipBytes(t, extProcResponseFixture(llmprotocol.OpenAIChatV1))
		result := runBufferedReply(t, &OpenAIRouter{}, ctx, 200, compressed, "gzip")
		common := result.body.GetResponseBody().GetResponse()
		require.Nil(t, common.GetBodyMutation())
		require.Empty(t, common.GetHeaderMutation().GetRemoveHeaders())
	})
	t.Run("streamed", func(t *testing.T) {
		ctx := looperTransportContext(llmprotocol.OpenAIChatV1, true)
		ctx.LooperRequest = false
		router := &OpenAIRouter{}
		_, err := router.handleResponseHeaders(upstreamReplyHeaders(200, "text/event-stream", "gzip", 0), ctx)
		require.NoError(t, err)
		require.True(t, ctx.IsStreamingResponse)
		response, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{Body: []byte("data: "), EndOfStream: false},
		}, ctx)
		require.NoError(t, err)
		require.NotContains(t, response.GetResponseBody().GetResponse().GetHeaderMutation().GetRemoveHeaders(), "content-encoding")
	})
}

type bufferedReplyResult struct {
	headers    *ext_proc.ProcessingResponse
	body       *ext_proc.ProcessingResponse
	clientBody []byte
}

// runBufferedReply drives the response-header and response-body phases for
// one buffered upstream reply, as Envoy and the routing engine do, and returns
// both replies with the body the client receives.
func runBufferedReply(
	t *testing.T,
	router *OpenAIRouter,
	ctx *RequestContext,
	status int,
	body []byte,
	encoding string,
) bufferedReplyResult {
	t.Helper()
	headers, err := router.handleResponseHeaders(upstreamReplyHeaders(status, "application/json", encoding, len(body)), ctx)
	require.NoError(t, err)
	require.Nil(t, headers.GetImmediateResponse())
	response, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: body, EndOfStream: true},
	}, ctx)
	require.NoError(t, err)
	clientBody := body
	if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
		clientBody = mutation.GetBody()
	}
	return bufferedReplyResult{headers: headers, body: response, clientBody: clientBody}
}

func upstreamReplyHeaders(status int, contentType, encoding string, length int) *ext_proc.ProcessingRequest_ResponseHeaders {
	headers := looperTransportHeaders(status, contentType)
	fields := &headers.ResponseHeaders.Headers.Headers
	if encoding != "" {
		*fields = append(*fields, &core.HeaderValue{Key: "content-encoding", RawValue: []byte(encoding)})
	}
	if length > 0 {
		*fields = append(*fields, &core.HeaderValue{Key: "content-length", RawValue: []byte(strconv.Itoa(length))})
	}
	return headers
}

// assertDecodedReplyHeaders checks that a decoded reply no longer claims the
// provider's encoding or compressed length.
func assertDecodedReplyHeaders(t *testing.T, response *ext_proc.ProcessingResponse) {
	t.Helper()
	removed := response.GetResponseBody().GetResponse().GetHeaderMutation().GetRemoveHeaders()
	require.Contains(t, removed, "content-encoding")
	require.Contains(t, removed, "content-length")
}

func gzipBytes(t *testing.T, body []byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	writer := gzip.NewWriter(&buf)
	_, err := writer.Write(body)
	require.NoError(t, err)
	require.NoError(t, writer.Close())
	return buf.Bytes()
}
