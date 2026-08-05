package extproc

import (
	"testing"

	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
