package extproc

import (
	"bytes"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func makeTestRouter(_ string) *OpenAIRouter {
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: "fallback-model",
			ModelConfig: map[string]config.ModelParams{
				"fallback-model": {},
			},
		},
		RouterOptions: config.RouterOptions{StreamedBodyMode: true},
	}
	return &OpenAIRouter{
		Config: cfg,
		Cache: cache.NewInMemoryCache(cache.InMemoryCacheOptions{
			Enabled: false,
		}),
	}
}

func splitTestChunks(data []byte, chunkSize int) [][]byte {
	var chunks [][]byte
	for len(data) > 0 {
		end := min(chunkSize, len(data))
		chunks = append(chunks, data[:end])
		data = data[end:]
	}
	return chunks
}

func assertChunkEaten(t *testing.T, response *ext_proc.ProcessingResponse) {
	t.Helper()
	require.NotNil(t, response)
	common := response.GetRequestBody().GetResponse()
	require.NotNil(t, common)
	assert.Equal(t, ext_proc.CommonResponse_CONTINUE, common.GetStatus())
	require.NotNil(t, common.GetBodyMutation())
	assert.Empty(t, common.GetBodyMutation().GetBody())
}

func makeTestRouterWithLimits(maxBytes int64, timeoutSec int) *OpenAIRouter {
	router := makeTestRouter("auto")
	router.Config.MaxStreamedBodyBytes = maxBytes
	router.Config.StreamedBodyTimeoutSec = timeoutSec
	return router
}

func TestStreamedBodyAccumulatesProtocolNeutralBytesUntilEOS(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	handler := newStreamedBodyHandler(router, ctx)
	defer handler.Release()

	body := []byte(`{"messages":[{"role":"user","content":"hello"}],"model":"auto","stream":true}`)
	for _, chunk := range splitTestChunks(body, 7) {
		response, err := handler.HandleChunk(&ext_proc.HttpBody{Body: chunk}, ctx)
		require.NoError(t, err)
		assertChunkEaten(t, response)
		assert.Nil(t, ctx.SemanticRequest)
		assert.Empty(t, ctx.RequestModel)
		assert.False(t, ctx.ExpectStreamingResponse)
	}
	assert.Equal(t, body, handler.buf.Bytes())
}

func TestStreamedBodyEOSDelegatesCompleteBodyToCodec(t *testing.T) {
	tests := []struct {
		name   string
		format llmprotocol.WireFormat
		body   []byte
	}{
		{
			name: "openai chat", format: llmprotocol.OpenAIChatV1,
			body: []byte(`{"model":"fallback-model","messages":[{"role":"user","content":"hello"}],"stream":true}`),
		},
		{
			name: "openai responses", format: llmprotocol.OpenAIResponsesV1,
			body: []byte(`{"model":"fallback-model","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}],"stream":true}`),
		},
		{
			name: "anthropic messages", format: llmprotocol.AnthropicMessagesV1,
			body: []byte(`{"model":"fallback-model","max_tokens":8,"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}],"stream":true}`),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx := &RequestContext{Headers: make(map[string]string), SourceFormat: test.format}
			handler := newStreamedBodyHandler(makeTestRouter("auto"), ctx)
			defer handler.Release()
			chunks := splitTestChunks(test.body, 11)
			for _, chunk := range chunks[:len(chunks)-1] {
				response, err := handler.HandleChunk(&ext_proc.HttpBody{Body: chunk}, ctx)
				require.NoError(t, err)
				assertChunkEaten(t, response)
				require.Nil(t, ctx.SemanticRequest)
			}

			response, err := handler.HandleChunk(&ext_proc.HttpBody{
				Body: chunks[len(chunks)-1], EndOfStream: true,
			}, ctx)
			require.NoError(t, err)
			require.NotNil(t, response)
			require.NotNil(t, ctx.SemanticRequest)
			assert.Equal(t, "fallback-model", ctx.SemanticRequest.Model)
			assert.Equal(t, "fallback-model", ctx.RequestModel)
			assert.True(t, ctx.SemanticRequest.Stream)
			assert.True(t, ctx.ExpectStreamingResponse)
			assert.Equal(t, len(test.body), ctx.IngressBodyBytes)
		})
	}
}

func TestStreamedBodyDefersCodecValidationUntilEOS(t *testing.T) {
	ctx := &RequestContext{Headers: make(map[string]string), SourceFormat: llmprotocol.OpenAIChatV1}
	handler := newStreamedBodyHandler(makeTestRouter("auto"), ctx)
	defer handler.Release()

	response, err := handler.HandleChunk(&ext_proc.HttpBody{Body: []byte(`{"model":"auto",`)}, ctx)
	require.NoError(t, err)
	assertChunkEaten(t, response)
	assert.Nil(t, ctx.SemanticRequest)

	response, err = handler.HandleChunk(&ext_proc.HttpBody{
		Body: []byte(`"model":"fallback-model","messages":[]}`), EndOfStream: true,
	}, ctx)
	require.NoError(t, err)
	require.NotNil(t, response.GetImmediateResponse())
	assert.Nil(t, ctx.SemanticRequest)
}

func TestStreamedBodyNonEOSChunkUsesSharedResponse(t *testing.T) {
	ctx := &RequestContext{}
	handler := newStreamedBodyHandler(makeTestRouter("auto"), ctx)
	defer handler.Release()

	response, err := handler.HandleChunk(&ext_proc.HttpBody{Body: []byte("opaque bytes")}, ctx)
	require.NoError(t, err)
	assert.Same(t, sharedContinueEmptyBody, response)
	assertChunkEaten(t, response)
}

func TestStreamedBodyGuardRejectsOversizedAccumulation(t *testing.T) {
	ctx := &RequestContext{}
	handler := newStreamedBodyHandler(makeTestRouterWithLimits(100, 0), ctx)
	defer handler.Release()

	response, err := handler.HandleChunk(&ext_proc.HttpBody{Body: bytes.Repeat([]byte("a"), 100)}, ctx)
	require.NoError(t, err)
	assertChunkEaten(t, response)
	_, err = handler.HandleChunk(&ext_proc.HttpBody{Body: []byte("b")}, ctx)
	require.ErrorContains(t, err, "too large")
}

func TestStreamedBodyGuardRejectsExpiredAccumulation(t *testing.T) {
	ctx := &RequestContext{}
	handler := newStreamedBodyHandler(makeTestRouterWithLimits(0, 1), ctx)
	defer handler.Release()
	handler.deadline = time.Now().Add(-time.Second)

	_, err := handler.HandleChunk(&ext_proc.HttpBody{Body: []byte("opaque")}, ctx)
	require.ErrorContains(t, err, "timed out")
}

func TestStreamedBodyPoolReuseClearsRequestStateAndGuards(t *testing.T) {
	first := newStreamedBodyHandler(makeTestRouterWithLimits(500, 60), &RequestContext{})
	first.buf.WriteString("leftover")
	first.Release()

	second := newStreamedBodyHandler(makeTestRouterWithLimits(0, 0), &RequestContext{})
	defer second.Release()
	assert.Empty(t, second.buf.Bytes())
	assert.Zero(t, second.maxBytes)
	assert.True(t, second.deadline.IsZero())
}
