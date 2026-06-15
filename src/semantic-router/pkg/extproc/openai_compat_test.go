package extproc

import (
	"encoding/json"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// These tests verify that the pipeline's JSON handling round-trips cleanly
// through the official openai-go SDK types. They guard against schema drift
// when the SDK version is bumped or when internal wire-format assumptions change.

func TestOpenAIRequestRoundTrip_Simple(t *testing.T) {
	raw := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "user", "content": "Hello"}
		]
	}`)

	parsed, err := parseOpenAIRequest(raw)
	require.NoError(t, err)

	assert.Equal(t, "gpt-4o", parsed.Model)
	require.Len(t, parsed.Messages, 1)
	assert.NotNil(t, parsed.Messages[0].OfUser)
	assert.Equal(t, "Hello", parsed.Messages[0].OfUser.Content.OfString.Value)

	reserialized, err := json.Marshal(parsed)
	require.NoError(t, err)

	var reparsed openai.ChatCompletionNewParams
	require.NoError(t, json.Unmarshal(reserialized, &reparsed))
	assert.Equal(t, parsed.Model, reparsed.Model)
}

func TestOpenAIRequestRoundTrip_MultiRole(t *testing.T) {
	raw := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": "What is 2+2?"},
			{"role": "assistant", "content": "4."},
			{"role": "user", "content": "And 3+3?"}
		]
	}`)

	parsed, err := parseOpenAIRequest(raw)
	require.NoError(t, err)

	require.Len(t, parsed.Messages, 4)
	assert.NotNil(t, parsed.Messages[0].OfSystem)
	assert.NotNil(t, parsed.Messages[1].OfUser)
	assert.NotNil(t, parsed.Messages[2].OfAssistant)
	assert.NotNil(t, parsed.Messages[3].OfUser)
}

func TestOpenAIRequestRoundTrip_WithToolCalls(t *testing.T) {
	raw := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "user", "content": "What's the weather in Paris?"}
		],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "get_weather",
					"description": "Get weather for a city",
					"parameters": {
						"type": "object",
						"properties": {
							"city": {"type": "string"}
						},
						"required": ["city"]
					}
				}
			}
		]
	}`)

	parsed, err := parseOpenAIRequest(raw)
	require.NoError(t, err)

	require.Len(t, parsed.Tools, 1)
	assert.Equal(t, "get_weather", parsed.Tools[0].Function.Name)
}

func TestOpenAIRequestRoundTrip_WithOptionalParams(t *testing.T) {
	raw := []byte(`{
		"model": "gpt-4o",
		"messages": [{"role": "user", "content": "Hi"}],
		"temperature": 0.7,
		"top_p": 0.9,
		"max_tokens": 1024,
		"stream": true
	}`)

	parsed, err := parseOpenAIRequest(raw)
	require.NoError(t, err)

	assert.Equal(t, 0.7, parsed.Temperature.Value)
	assert.Equal(t, 0.9, parsed.TopP.Value)
	assert.Equal(t, int64(1024), parsed.MaxTokens.Value)
}

func TestOpenAIRequestRoundTrip_ContentParts(t *testing.T) {
	raw := []byte(`{
		"model": "gpt-4o",
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "Describe this image"},
				{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}}
			]
		}]
	}`)

	parsed, err := parseOpenAIRequest(raw)
	require.NoError(t, err)

	require.Len(t, parsed.Messages, 1)
	msg := parsed.Messages[0].OfUser
	require.NotNil(t, msg)
	require.NotEmpty(t, msg.Content.OfArrayOfContentParts)
}

func TestOpenAIResponseParsing(t *testing.T) {
	raw := []byte(`{
		"id": "chatcmpl-abc123",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Hello! How can I help you today?"
			},
			"finish_reason": "stop"
		}],
		"usage": {
			"prompt_tokens": 10,
			"completion_tokens": 12,
			"total_tokens": 22
		}
	}`)

	var resp openai.ChatCompletion
	require.NoError(t, json.Unmarshal(raw, &resp))

	assert.Equal(t, "chatcmpl-abc123", resp.ID)
	assert.Equal(t, "chat.completion", string(resp.Object))
	assert.Equal(t, "gpt-4o", resp.Model)
	require.Len(t, resp.Choices, 1)
	assert.Equal(t, "assistant", string(resp.Choices[0].Message.Role))
	assert.Equal(t, "Hello! How can I help you today?", resp.Choices[0].Message.Content)
	assert.Equal(t, "stop", resp.Choices[0].FinishReason)
	assert.Equal(t, int64(10), resp.Usage.PromptTokens)
	assert.Equal(t, int64(12), resp.Usage.CompletionTokens)
	assert.Equal(t, int64(22), resp.Usage.TotalTokens)
}

func TestOpenAIResponseWithToolCalls(t *testing.T) {
	raw := []byte(`{
		"id": "chatcmpl-tools",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": null,
				"tool_calls": [{
					"id": "call_abc",
					"type": "function",
					"function": {
						"name": "get_weather",
						"arguments": "{\"city\":\"Paris\"}"
					}
				}]
			},
			"finish_reason": "tool_calls"
		}],
		"usage": {
			"prompt_tokens": 15,
			"completion_tokens": 20,
			"total_tokens": 35
		}
	}`)

	var resp openai.ChatCompletion
	require.NoError(t, json.Unmarshal(raw, &resp))

	require.Len(t, resp.Choices, 1)
	assert.Equal(t, "tool_calls", resp.Choices[0].FinishReason)
	require.Len(t, resp.Choices[0].Message.ToolCalls, 1)
	tc := resp.Choices[0].Message.ToolCalls[0]
	assert.Equal(t, "call_abc", tc.ID)
	assert.Equal(t, "get_weather", tc.Function.Name)
	assert.JSONEq(t, `{"city":"Paris"}`, tc.Function.Arguments)
}

func TestOpenAIStreamChunkParsing(t *testing.T) {
	raw := []byte(`{
		"id": "chatcmpl-stream",
		"object": "chat.completion.chunk",
		"created": 1700000000,
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"delta": {
				"content": "Hello"
			},
			"finish_reason": null
		}]
	}`)

	var chunk openai.ChatCompletionChunk
	require.NoError(t, json.Unmarshal(raw, &chunk))

	assert.Equal(t, "chatcmpl-stream", chunk.ID)
	assert.Equal(t, "chat.completion.chunk", string(chunk.Object))
	require.Len(t, chunk.Choices, 1)
	assert.Equal(t, "Hello", chunk.Choices[0].Delta.Content)
}

func TestOpenAIStreamChunk_FinishReason(t *testing.T) {
	raw := []byte(`{
		"id": "chatcmpl-stream",
		"object": "chat.completion.chunk",
		"created": 1700000000,
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"delta": {},
			"finish_reason": "stop"
		}],
		"usage": {
			"prompt_tokens": 10,
			"completion_tokens": 5,
			"total_tokens": 15
		}
	}`)

	var chunk openai.ChatCompletionChunk
	require.NoError(t, json.Unmarshal(raw, &chunk))

	require.Len(t, chunk.Choices, 1)
	assert.Equal(t, "stop", chunk.Choices[0].FinishReason)
}

func TestExtractTextContent_MatchesSDKParse(t *testing.T) {
	raw := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": "Explain quantum physics briefly."}
		]
	}`)

	fast, err := extractContentFast(raw)
	require.NoError(t, err)

	sdkParsed, err := parseOpenAIRequest(raw)
	require.NoError(t, err)

	sdkUserContent := extractUserContent(sdkParsed)
	assert.Equal(t, fast.UserContent, sdkUserContent,
		"fast-path and SDK-path must extract the same user content")
}

// extractUserContent pulls user content from parsed SDK types — mirrors the
// fast-path extraction logic for cross-validation.
func extractUserContent(req *openai.ChatCompletionNewParams) string {
	for i := len(req.Messages) - 1; i >= 0; i-- {
		msg := req.Messages[i]
		if msg.OfUser != nil {
			if msg.OfUser.Content.OfString.Value != "" {
				return msg.OfUser.Content.OfString.Value
			}
		}
	}
	return ""
}
