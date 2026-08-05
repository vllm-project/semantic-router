package extproc

import (
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ir"
)

// translateResponseBodyForClient is the protocol-keyed dispatcher
// that PR4 wired the Anthropic outbound emitter into. The tests below
// cover its three branches: Anthropic ingress (rewrite to Anthropic
// shape), Response API ingress (existing behavior), and plain OpenAI
// ingress (byte-identical pass-through).

func openaiResponseBytes(t *testing.T, id, content, finishReason string) []byte {
	t.Helper()
	cc := &openai.ChatCompletion{
		ID:     id,
		Object: "chat.completion",
		Model:  "claude-sonnet-4-5",
		Choices: []openai.ChatCompletionChoice{{
			Index: 0,
			Message: openai.ChatCompletionMessage{
				Role:    "assistant",
				Content: content,
			},
			FinishReason: finishReason,
		}},
		Usage: openai.CompletionUsage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}
	body, err := json.Marshal(cc)
	require.NoError(t, err)
	return body
}

func TestTranslateResponseBodyForClient_AnthropicRewrite(t *testing.T) {
	r := &OpenAIRouter{}
	ctx := &RequestContext{
		ClientProtocol: config.ClientProtocolAnthropic,
		RequestModel:   "claude-opus-4-7",
		IRExtensions: &ir.IRExtensions{
			SourceProtocol: "anthropic",
		},
	}
	input := openaiResponseBytes(t, "msg_a1", "hello from anthropic client", "stop")

	out, transformed := r.translateResponseBodyForClient(ctx, input)
	require.True(t, transformed)

	var msg anthropic.Message
	require.NoError(t, json.Unmarshal(out, &msg))
	assert.Equal(t, "msg_a1", msg.ID)
	assert.Equal(t, "message", string(msg.Type))
	assert.Equal(t, anthropic.StopReasonEndTurn, msg.StopReason)
	require.Len(t, msg.Content, 1)
	assert.Equal(t, "hello from anthropic client", msg.Content[0].Text)
}

func TestTranslateResponseBodyForClient_AnthropicEmitFailureReturnsErrorEnvelope(t *testing.T) {
	r := &OpenAIRouter{}
	ctx := &RequestContext{
		ClientProtocol: config.ClientProtocolAnthropic,
		RequestModel:   "claude-opus-4-7",
		IRExtensions:   &ir.IRExtensions{SourceProtocol: "anthropic"},
	}
	out, transformed := r.translateResponseBodyForClient(ctx, []byte("not json"))
	require.True(t, transformed)

	var env struct {
		Type  string `json:"type"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	require.NoError(t, json.Unmarshal(out, &env))
	assert.Equal(t, "error", env.Type)
	assert.Equal(t, "api_error", env.Error.Type)
	assert.NotEmpty(t, env.Error.Message)
}

func TestTranslateResponseBodyForClient_OpenAIIngressUnchanged(t *testing.T) {
	r := &OpenAIRouter{}
	ctx := &RequestContext{
		// ClientProtocol empty == plain OpenAI ingress.
	}
	input := openaiResponseBytes(t, "msg_oa", "unchanged", "stop")

	out, transformed := r.translateResponseBodyForClient(ctx, input)
	assert.False(t, transformed, "OpenAI ingress must not flag a rewrite")
	assert.Equal(t, string(input), string(out), "OpenAI ingress must be byte-identical")
}

func TestTranslateResponseBodyForClient_NilContextOpenAIPath(t *testing.T) {
	r := &OpenAIRouter{}
	input := openaiResponseBytes(t, "msg_nilctx", "x", "stop")
	out, transformed := r.translateResponseBodyForClient(nil, input)
	assert.False(t, transformed)
	assert.Equal(t, string(input), string(out))
}

func TestBuildInitialResponseMutations_NoTransformShortCircuits(t *testing.T) {
	bodyMut, headerMut := buildInitialResponseMutations([]byte("anything"), false)
	assert.Nil(t, bodyMut)
	assert.Nil(t, headerMut)
}

func TestBuildInitialResponseMutations_TransformProducesMutations(t *testing.T) {
	bodyMut, headerMut := buildInitialResponseMutations([]byte("body"), true)
	require.NotNil(t, bodyMut)
	require.NotNil(t, headerMut)
	assert.Equal(t, []byte("body"), bodyMut.GetBody())
	assert.Contains(t, headerMut.RemoveHeaders, "content-length")
}

// responseWarningsHeaderValue returns the x-vsr-response-warnings value emitted
// on a response, and whether the header was present.
func responseWarningsHeaderValue(resp *ext_proc.ProcessingResponse) (string, bool) {
	body, ok := resp.GetResponse().(*ext_proc.ProcessingResponse_ResponseBody)
	if !ok || body.ResponseBody.GetResponse() == nil || body.ResponseBody.GetResponse().GetHeaderMutation() == nil {
		return "", false
	}
	for _, h := range body.ResponseBody.GetResponse().GetHeaderMutation().GetSetHeaders() {
		if h.GetHeader().GetKey() == headers.VSRResponseWarnings {
			return string(h.GetHeader().GetRawValue()), true
		}
	}
	return "", false
}

func TestApplyResponseWarnings_MergesCodesInFixedOrder(t *testing.T) {
	r := &OpenAIRouter{}
	// Default action (nil decision) surfaces each warning as a header code.
	ctx := &RequestContext{
		HallucinationDetected:     true,
		UnverifiedFactualResponse: true,
		ResponseJailbreakDetected: true,
	}

	resp := r.applyResponseWarnings(ctx, []byte(`{"choices":[]}`), nil, nil)

	val, ok := responseWarningsHeaderValue(resp)
	require.True(t, ok, "x-vsr-response-warnings should be present")
	assert.Equal(t, "hallucination,unverified_factual,response_jailbreak", val)
}

func TestApplyResponseWarnings_SingleCode(t *testing.T) {
	r := &OpenAIRouter{}
	ctx := &RequestContext{ResponseJailbreakDetected: true}

	resp := r.applyResponseWarnings(ctx, []byte(`{"choices":[]}`), nil, nil)

	val, ok := responseWarningsHeaderValue(resp)
	require.True(t, ok)
	assert.Equal(t, "response_jailbreak", val)
}

func TestApplyResponseWarnings_NoWarningsOmitsHeader(t *testing.T) {
	r := &OpenAIRouter{}
	ctx := &RequestContext{}

	resp := r.applyResponseWarnings(ctx, []byte(`{"choices":[]}`), nil, nil)

	_, ok := responseWarningsHeaderValue(resp)
	assert.False(t, ok, "no warnings should emit no x-vsr-response-warnings header")
}
