package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToAnthropicRequestBody_BasicConversion(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Model: "claude-sonnet-4-5",
		Messages: []openai.ChatCompletionMessageParamUnion{
			{OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String("Hello, world!"),
				},
			}},
		},
	}

	body, err := ToAnthropicRequestBody(req)
	require.NoError(t, err)

	var result anthropic.MessageNewParams
	err = json.Unmarshal(body, &result)
	require.NoError(t, err)

	assert.Equal(t, anthropic.Model("claude-sonnet-4-5"), result.Model)
	assert.Equal(t, DefaultMaxTokens, result.MaxTokens)
	assert.Len(t, result.Messages, 1)
}

func TestToAnthropicRequestBody_WithSystemPrompt(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Model: "claude-sonnet-4-5",
		Messages: []openai.ChatCompletionMessageParamUnion{
			{OfSystem: &openai.ChatCompletionSystemMessageParam{
				Content: openai.ChatCompletionSystemMessageParamContentUnion{
					OfString: openai.String("You are a helpful assistant."),
				},
			}},
			{OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String("Hi"),
				},
			}},
		},
	}

	body, err := ToAnthropicRequestBody(req)
	require.NoError(t, err)

	var result anthropic.MessageNewParams
	err = json.Unmarshal(body, &result)
	require.NoError(t, err)

	assert.Len(t, result.System, 1)
	assert.Equal(t, "You are a helpful assistant.", result.System[0].Text)
	assert.Len(t, result.Messages, 1) // System message not in messages array
}

func TestToAnthropicRequestBody_WithMaxTokens(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Model:     "claude-sonnet-4-5",
		MaxTokens: openai.Int(1024),
		Messages: []openai.ChatCompletionMessageParamUnion{
			{OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String("Hello"),
				},
			}},
		},
	}

	body, err := ToAnthropicRequestBody(req)
	require.NoError(t, err)

	var result anthropic.MessageNewParams
	err = json.Unmarshal(body, &result)
	require.NoError(t, err)

	assert.Equal(t, int64(1024), result.MaxTokens)
}

func TestToAnthropicRequestBody_WithOptionalParams(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Model:       "claude-sonnet-4-5",
		Temperature: openai.Float(0.7),
		TopP:        openai.Float(0.9),
		Stop: openai.ChatCompletionNewParamsStopUnion{
			OfStringArray: []string{"END", "STOP"},
		},
		Messages: []openai.ChatCompletionMessageParamUnion{
			{OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String("Hello"),
				},
			}},
		},
	}

	body, err := ToAnthropicRequestBody(req)
	require.NoError(t, err)

	var result anthropic.MessageNewParams
	err = json.Unmarshal(body, &result)
	require.NoError(t, err)

	assert.NotNil(t, result.Temperature)
	assert.NotNil(t, result.TopP)
	assert.Equal(t, []string{"END", "STOP"}, result.StopSequences)
}

func TestToAnthropicRequestBody_MultiTurnConversation(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Model: "claude-sonnet-4-5",
		Messages: []openai.ChatCompletionMessageParamUnion{
			{OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String("What is 2+2?"),
				},
			}},
			{OfAssistant: &openai.ChatCompletionAssistantMessageParam{
				Content: openai.ChatCompletionAssistantMessageParamContentUnion{
					OfString: openai.String("2+2 equals 4."),
				},
			}},
			{OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String("And 3+3?"),
				},
			}},
		},
	}

	body, err := ToAnthropicRequestBody(req)
	require.NoError(t, err)

	var result anthropic.MessageNewParams
	err = json.Unmarshal(body, &result)
	require.NoError(t, err)

	assert.Len(t, result.Messages, 3)
}

func TestToOpenAIResponseBody_BasicConversion(t *testing.T) {
	anthropicResp := &anthropic.Message{
		ID:   "msg_123",
		Type: "message",
		Role: "assistant",
		Content: []anthropic.ContentBlockUnion{
			{Type: "text", Text: "Hello! How can I help you?"},
		},
		StopReason: anthropic.StopReasonEndTurn,
		Usage: anthropic.Usage{
			InputTokens:  10,
			OutputTokens: 8,
		},
	}

	anthropicJSON, err := json.Marshal(anthropicResp)
	require.NoError(t, err)

	resultJSON, err := ToOpenAIResponseBody(anthropicJSON, "claude-sonnet-4-5")
	require.NoError(t, err)

	var result openai.ChatCompletion
	err = json.Unmarshal(resultJSON, &result)
	require.NoError(t, err)

	assert.Equal(t, "msg_123", result.ID)
	assert.Equal(t, "chat.completion", string(result.Object))
	assert.Equal(t, "claude-sonnet-4-5", result.Model)
	assert.Len(t, result.Choices, 1)
	assert.Equal(t, "assistant", string(result.Choices[0].Message.Role))
	assert.Equal(t, "Hello! How can I help you?", result.Choices[0].Message.Content)
	assert.Equal(t, "stop", result.Choices[0].FinishReason)
	assert.Equal(t, int64(10), result.Usage.PromptTokens)
	assert.Equal(t, int64(8), result.Usage.CompletionTokens)
	assert.Equal(t, int64(18), result.Usage.TotalTokens)
}

func TestToOpenAIResponseBody_MaxTokensStopReason(t *testing.T) {
	anthropicResp := &anthropic.Message{
		ID:         "msg_123",
		Content:    []anthropic.ContentBlockUnion{{Type: "text", Text: "Partial response..."}},
		StopReason: anthropic.StopReasonMaxTokens,
		Usage:      anthropic.Usage{InputTokens: 10, OutputTokens: 100},
	}

	anthropicJSON, err := json.Marshal(anthropicResp)
	require.NoError(t, err)

	resultJSON, err := ToOpenAIResponseBody(anthropicJSON, "claude-sonnet-4-5")
	require.NoError(t, err)

	var result openai.ChatCompletion
	err = json.Unmarshal(resultJSON, &result)
	require.NoError(t, err)

	assert.Equal(t, "length", result.Choices[0].FinishReason)
}

func TestToOpenAIResponseBody_ToolUseStopReason(t *testing.T) {
	anthropicResp := &anthropic.Message{
		ID:         "msg_123",
		Content:    []anthropic.ContentBlockUnion{{Type: "text", Text: "Using tool..."}},
		StopReason: anthropic.StopReasonToolUse,
		Usage:      anthropic.Usage{InputTokens: 10, OutputTokens: 20},
	}

	anthropicJSON, err := json.Marshal(anthropicResp)
	require.NoError(t, err)

	resultJSON, err := ToOpenAIResponseBody(anthropicJSON, "claude-sonnet-4-5")
	require.NoError(t, err)

	var result openai.ChatCompletion
	err = json.Unmarshal(resultJSON, &result)
	require.NoError(t, err)

	assert.Equal(t, "tool_calls", result.Choices[0].FinishReason)
}

func TestToOpenAIResponseBody_MultipleContentBlocks(t *testing.T) {
	anthropicResp := &anthropic.Message{
		ID: "msg_123",
		Content: []anthropic.ContentBlockUnion{
			{Type: "text", Text: "First part. "},
			{Type: "text", Text: "Second part."},
		},
		StopReason: anthropic.StopReasonEndTurn,
		Usage:      anthropic.Usage{InputTokens: 5, OutputTokens: 10},
	}

	anthropicJSON, err := json.Marshal(anthropicResp)
	require.NoError(t, err)

	resultJSON, err := ToOpenAIResponseBody(anthropicJSON, "claude-sonnet-4-5")
	require.NoError(t, err)

	var result openai.ChatCompletion
	err = json.Unmarshal(resultJSON, &result)
	require.NoError(t, err)

	assert.Equal(t, "First part. Second part.", result.Choices[0].Message.Content)
}

func TestBuildRequestHeaders(t *testing.T) {
	headers := BuildRequestHeaders("test-api-key", 1024)

	// Check we have all expected headers
	headerMap := make(map[string]string)
	for _, h := range headers {
		headerMap[h.Key] = h.Value
	}

	assert.Equal(t, "test-api-key", headerMap["x-api-key"])
	assert.Equal(t, AnthropicAPIVersion, headerMap["anthropic-version"])
	assert.Equal(t, "application/json", headerMap["content-type"])
	assert.Equal(t, AnthropicMessagesPath, headerMap[":path"])
	assert.Equal(t, "1024", headerMap["content-length"])
}

func TestHeadersToRemove(t *testing.T) {
	headers := HeadersToRemove()

	assert.Contains(t, headers, "authorization")
	assert.Contains(t, headers, "content-length")
}

func TestExtractSystemContent_StringContent(t *testing.T) {
	msg := &openai.ChatCompletionSystemMessageParam{
		Content: openai.ChatCompletionSystemMessageParamContentUnion{
			OfString: openai.String("You are helpful."),
		},
	}

	result := extractSystemContent(msg)

	assert.Equal(t, "You are helpful.", result)
}

func TestExtractUserContent_StringContent(t *testing.T) {
	msg := &openai.ChatCompletionUserMessageParam{
		Content: openai.ChatCompletionUserMessageParamContentUnion{
			OfString: openai.String("Hello there!"),
		},
	}

	result := extractUserContent(msg)

	assert.Equal(t, "Hello there!", result)
}

func TestExtractAssistantContent_StringContent(t *testing.T) {
	msg := &openai.ChatCompletionAssistantMessageParam{
		Content: openai.ChatCompletionAssistantMessageParamContentUnion{
			OfString: openai.String("Hi! How can I help?"),
		},
	}

	result := extractAssistantContent(msg)

	assert.Equal(t, "Hi! How can I help?", result)
}
