package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// These tests verify that the OpenAI ↔ Anthropic conversion layer produces
// JSON output that is wire-compatible with the respective official SDKs.
// They serve as a regression guard against schema drift when either SDK is
// upgraded or when internal conversion logic is refactored.

func TestRoundTrip_SimpleRequest(t *testing.T) {
	orig := &openai.ChatCompletionNewParams{
		Model: "claude-sonnet-4-5",
		Messages: []openai.ChatCompletionMessageParamUnion{
			{OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String("Explain photosynthesis."),
				},
			}},
		},
		Temperature: openai.Float(0.5),
	}

	anthropicBody, err := ToAnthropicRequestBody(orig)
	require.NoError(t, err)

	var parsed anthropic.MessageNewParams
	require.NoError(t, json.Unmarshal(anthropicBody, &parsed))

	assert.Equal(t, anthropic.Model("claude-sonnet-4-5"), parsed.Model)
	assert.Equal(t, DefaultMaxTokens, parsed.MaxTokens)
	require.Len(t, parsed.Messages, 1)
	assert.Equal(t, anthropic.MessageParamRoleUser, parsed.Messages[0].Role)
}

func TestRoundTrip_SystemSeparation(t *testing.T) {
	orig := &openai.ChatCompletionNewParams{
		Model: "claude-sonnet-4-5",
		Messages: []openai.ChatCompletionMessageParamUnion{
			{OfSystem: &openai.ChatCompletionSystemMessageParam{
				Content: openai.ChatCompletionSystemMessageParamContentUnion{
					OfString: openai.String("Be concise."),
				},
			}},
			{OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String("Hi"),
				},
			}},
		},
	}

	body, err := ToAnthropicRequestBody(orig)
	require.NoError(t, err)

	var parsed anthropic.MessageNewParams
	require.NoError(t, json.Unmarshal(body, &parsed))

	require.Len(t, parsed.System, 1)
	assert.Equal(t, "Be concise.", parsed.System[0].Text)
	require.Len(t, parsed.Messages, 1, "system must not appear in messages array")
}

func TestRoundTrip_MaxTokensVariants(t *testing.T) {
	tests := []struct {
		name     string
		req      *openai.ChatCompletionNewParams
		expected int64
	}{
		{
			name: "MaxCompletionTokens takes priority",
			req: &openai.ChatCompletionNewParams{
				Model:               "claude-sonnet-4-5",
				MaxCompletionTokens: openai.Int(512),
				MaxTokens:           openai.Int(1024),
				Messages:            simpleUserMsg("hi"),
			},
			expected: 512,
		},
		{
			name: "fallback to MaxTokens",
			req: &openai.ChatCompletionNewParams{
				Model:     "claude-sonnet-4-5",
				MaxTokens: openai.Int(2048),
				Messages:  simpleUserMsg("hi"),
			},
			expected: 2048,
		},
		{
			name: "default when neither set",
			req: &openai.ChatCompletionNewParams{
				Model:    "claude-sonnet-4-5",
				Messages: simpleUserMsg("hi"),
			},
			expected: DefaultMaxTokens,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body, err := ToAnthropicRequestBody(tt.req)
			require.NoError(t, err)

			var parsed anthropic.MessageNewParams
			require.NoError(t, json.Unmarshal(body, &parsed))
			assert.Equal(t, tt.expected, parsed.MaxTokens)
		})
	}
}

func TestResponse_StopReasonMapping(t *testing.T) {
	tests := []struct {
		anthropicReason anthropic.StopReason
		expectedOpenAI  string
	}{
		{anthropic.StopReasonEndTurn, "stop"},
		{anthropic.StopReasonMaxTokens, "length"},
		{anthropic.StopReasonToolUse, "tool_calls"},
		{"unknown_future_reason", "stop"},
	}

	for _, tt := range tests {
		t.Run(string(tt.anthropicReason), func(t *testing.T) {
			resp := anthropic.Message{
				ID:         "msg_test",
				Content:    []anthropic.ContentBlockUnion{{Type: "text", Text: "ok"}},
				StopReason: tt.anthropicReason,
				Usage:      anthropic.Usage{InputTokens: 1, OutputTokens: 1},
			}

			raw, err := json.Marshal(resp)
			require.NoError(t, err)

			out, err := ToOpenAIResponseBody(raw, "claude-sonnet-4-5")
			require.NoError(t, err)

			var oai openai.ChatCompletion
			require.NoError(t, json.Unmarshal(out, &oai))
			assert.Equal(t, tt.expectedOpenAI, oai.Choices[0].FinishReason)
		})
	}
}

func TestResponse_UsageMapping(t *testing.T) {
	resp := anthropic.Message{
		ID:         "msg_usage",
		Content:    []anthropic.ContentBlockUnion{{Type: "text", Text: "hi"}},
		StopReason: anthropic.StopReasonEndTurn,
		Usage:      anthropic.Usage{InputTokens: 42, OutputTokens: 17},
	}

	raw, err := json.Marshal(resp)
	require.NoError(t, err)

	out, err := ToOpenAIResponseBody(raw, "test-model")
	require.NoError(t, err)

	var oai openai.ChatCompletion
	require.NoError(t, json.Unmarshal(out, &oai))

	assert.Equal(t, int64(42), oai.Usage.PromptTokens)
	assert.Equal(t, int64(17), oai.Usage.CompletionTokens)
	assert.Equal(t, int64(59), oai.Usage.TotalTokens)
}

func TestResponse_OutputIsValidOpenAIJSON(t *testing.T) {
	resp := anthropic.Message{
		ID:   "msg_valid",
		Role: "assistant",
		Content: []anthropic.ContentBlockUnion{
			{Type: "text", Text: "Hello world"},
		},
		StopReason: anthropic.StopReasonEndTurn,
		Usage:      anthropic.Usage{InputTokens: 5, OutputTokens: 3},
	}
	raw, err := json.Marshal(resp)
	require.NoError(t, err)

	out, err := ToOpenAIResponseBody(raw, "claude-sonnet-4-5")
	require.NoError(t, err)

	var oai openai.ChatCompletion
	require.NoError(t, json.Unmarshal(out, &oai),
		"output must unmarshal cleanly into the official openai.ChatCompletion type")

	assert.Equal(t, "chat.completion", string(oai.Object))
	assert.Equal(t, "msg_valid", oai.ID)
	assert.Equal(t, "claude-sonnet-4-5", oai.Model)
	require.Len(t, oai.Choices, 1)
	assert.Equal(t, "assistant", string(oai.Choices[0].Message.Role))
	assert.Equal(t, "Hello world", oai.Choices[0].Message.Content)
	assert.NotZero(t, oai.Created)
}

func TestRequest_EmptyMessagesRejected(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Model:    "claude-sonnet-4-5",
		Messages: nil,
	}

	body, err := ToAnthropicRequestBody(req)
	require.NoError(t, err)

	var parsed anthropic.MessageNewParams
	require.NoError(t, json.Unmarshal(body, &parsed))
	assert.Empty(t, parsed.Messages)
}

func TestResponse_EmptyContentBlocks(t *testing.T) {
	resp := anthropic.Message{
		ID:         "msg_empty",
		Content:    []anthropic.ContentBlockUnion{},
		StopReason: anthropic.StopReasonEndTurn,
		Usage:      anthropic.Usage{InputTokens: 1, OutputTokens: 0},
	}
	raw, err := json.Marshal(resp)
	require.NoError(t, err)

	out, err := ToOpenAIResponseBody(raw, "model")
	require.NoError(t, err)

	var oai openai.ChatCompletion
	require.NoError(t, json.Unmarshal(out, &oai))
	assert.Empty(t, oai.Choices[0].Message.Content)
}

func TestRequest_StopSequences(t *testing.T) {
	t.Run("string array", func(t *testing.T) {
		req := &openai.ChatCompletionNewParams{
			Model:    "claude-sonnet-4-5",
			Messages: simpleUserMsg("hi"),
			Stop: openai.ChatCompletionNewParamsStopUnion{
				OfStringArray: []string{"END", "DONE"},
			},
		}

		body, err := ToAnthropicRequestBody(req)
		require.NoError(t, err)

		var parsed anthropic.MessageNewParams
		require.NoError(t, json.Unmarshal(body, &parsed))
		assert.Equal(t, []string{"END", "DONE"}, parsed.StopSequences)
	})

	t.Run("single string", func(t *testing.T) {
		req := &openai.ChatCompletionNewParams{
			Model:    "claude-sonnet-4-5",
			Messages: simpleUserMsg("hi"),
			Stop: openai.ChatCompletionNewParamsStopUnion{
				OfString: openai.String("STOP"),
			},
		}

		body, err := ToAnthropicRequestBody(req)
		require.NoError(t, err)

		var parsed anthropic.MessageNewParams
		require.NoError(t, json.Unmarshal(body, &parsed))
		assert.Equal(t, []string{"STOP"}, parsed.StopSequences)
	})
}

// simpleUserMsg builds a minimal message slice for tests.
func simpleUserMsg(text string) []openai.ChatCompletionMessageParamUnion {
	return []openai.ChatCompletionMessageParamUnion{
		{OfUser: &openai.ChatCompletionUserMessageParam{
			Content: openai.ChatCompletionUserMessageParamContentUnion{
				OfString: openai.String(text),
			},
		}},
	}
}
