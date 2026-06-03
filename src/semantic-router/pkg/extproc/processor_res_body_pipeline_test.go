package extproc

import (
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
