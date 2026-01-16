package anthropic

import (
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
)

func TestToAnthropicRequest_BasicConversion(t *testing.T) {
	client := &Client{}

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

	result := client.toAnthropicRequest(req)

	assert.Equal(t, anthropic.Model("claude-sonnet-4-5"), result.Model)
	assert.Equal(t, DefaultMaxTokens, result.MaxTokens)
	assert.Len(t, result.Messages, 1)
}

func TestToAnthropicRequest_WithSystemPrompt(t *testing.T) {
	client := &Client{}

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

	result := client.toAnthropicRequest(req)

	assert.Len(t, result.System, 1)
	assert.Equal(t, "You are a helpful assistant.", result.System[0].Text)
	assert.Len(t, result.Messages, 1) // System message not in messages array
}

func TestToAnthropicRequest_WithMaxTokens(t *testing.T) {
	client := &Client{}

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

	result := client.toAnthropicRequest(req)

	assert.Equal(t, int64(1024), result.MaxTokens)
}

func TestToAnthropicRequest_WithOptionalParams(t *testing.T) {
	client := &Client{}

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

	result := client.toAnthropicRequest(req)

	assert.NotNil(t, result.Temperature)
	assert.NotNil(t, result.TopP)
	assert.Equal(t, []string{"END", "STOP"}, result.StopSequences)
}

func TestToAnthropicRequest_WithZeroTemperature(t *testing.T) {
	client := &Client{}

	req := &openai.ChatCompletionNewParams{
		Model:       "claude-sonnet-4-5",
		Temperature: openai.Float(0.0), // Explicitly set to 0 for deterministic output
		Messages: []openai.ChatCompletionMessageParamUnion{
			{OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String("Hello"),
				},
			}},
		},
	}

	result := client.toAnthropicRequest(req)

	// Temperature should be set even when 0.0
	assert.True(t, result.Temperature.Valid())
	assert.Equal(t, float64(0.0), result.Temperature.Value)
}

func TestToAnthropicRequest_MultiTurnConversation(t *testing.T) {
	client := &Client{}

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

	result := client.toAnthropicRequest(req)

	assert.Len(t, result.Messages, 3)
}

func TestToOpenAIResponse_BasicConversion(t *testing.T) {
	client := &Client{}

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

	result := client.toOpenAIResponse(anthropicResp, "claude-sonnet-4-5")

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

func TestToOpenAIResponse_MaxTokensStopReason(t *testing.T) {
	client := &Client{}

	anthropicResp := &anthropic.Message{
		ID:         "msg_123",
		Content:    []anthropic.ContentBlockUnion{{Type: "text", Text: "Partial response..."}},
		StopReason: anthropic.StopReasonMaxTokens,
		Usage:      anthropic.Usage{InputTokens: 10, OutputTokens: 100},
	}

	result := client.toOpenAIResponse(anthropicResp, "claude-sonnet-4-5")

	assert.Equal(t, "length", result.Choices[0].FinishReason)
}

func TestToOpenAIResponse_ToolUseStopReason(t *testing.T) {
	client := &Client{}

	anthropicResp := &anthropic.Message{
		ID:         "msg_123",
		Content:    []anthropic.ContentBlockUnion{{Type: "text", Text: "Using tool..."}},
		StopReason: anthropic.StopReasonToolUse,
		Usage:      anthropic.Usage{InputTokens: 10, OutputTokens: 20},
	}

	result := client.toOpenAIResponse(anthropicResp, "claude-sonnet-4-5")

	assert.Equal(t, "tool_calls", result.Choices[0].FinishReason)
}

func TestToOpenAIResponse_MultipleContentBlocks(t *testing.T) {
	client := &Client{}

	anthropicResp := &anthropic.Message{
		ID: "msg_123",
		Content: []anthropic.ContentBlockUnion{
			{Type: "text", Text: "First part. "},
			{Type: "text", Text: "Second part."},
		},
		StopReason: anthropic.StopReasonEndTurn,
		Usage:      anthropic.Usage{InputTokens: 5, OutputTokens: 10},
	}

	result := client.toOpenAIResponse(anthropicResp, "claude-sonnet-4-5")

	assert.Equal(t, "First part. Second part.", result.Choices[0].Message.Content)
}

func TestExtractSystemContent_StringContent(t *testing.T) {
	client := &Client{}

	msg := &openai.ChatCompletionSystemMessageParam{
		Content: openai.ChatCompletionSystemMessageParamContentUnion{
			OfString: openai.String("You are helpful."),
		},
	}

	result := client.extractSystemContent(msg)

	assert.Equal(t, "You are helpful.", result)
}

func TestExtractUserContent_StringContent(t *testing.T) {
	client := &Client{}

	msg := &openai.ChatCompletionUserMessageParam{
		Content: openai.ChatCompletionUserMessageParamContentUnion{
			OfString: openai.String("Hello there!"),
		},
	}

	result := client.extractUserContent(msg)

	assert.Equal(t, "Hello there!", result)
}

func TestExtractAssistantContent_StringContent(t *testing.T) {
	client := &Client{}

	msg := &openai.ChatCompletionAssistantMessageParam{
		Content: openai.ChatCompletionAssistantMessageParamContentUnion{
			OfString: openai.String("Hi! How can I help?"),
		},
	}

	result := client.extractAssistantContent(msg)

	assert.Equal(t, "Hi! How can I help?", result)
}

func TestNewClient(t *testing.T) {
	client := NewClient("test-api-key")

	assert.NotNil(t, client)
}
