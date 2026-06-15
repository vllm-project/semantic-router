package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ir"
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
		ID: "msg_123",
		Content: []anthropic.ContentBlockUnion{{
			Type:  "tool_use",
			ID:    "call_abc",
			Name:  "get_weather",
			Input: json.RawMessage(`{"city":"Paris"}`),
		}},
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
	require.Len(t, result.Choices[0].Message.ToolCalls, 1)
	assert.Equal(t, "call_abc", result.Choices[0].Message.ToolCalls[0].ID)
	assert.Equal(t, "get_weather", result.Choices[0].Message.ToolCalls[0].Function.Name)
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
	headers := BuildRequestHeaders("test-api-key", 1024, "")

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

	custom := BuildRequestHeaders("key", 10, "/Anthropic/v1/messages")
	customMap := make(map[string]string)
	for _, h := range custom {
		customMap[h.Key] = h.Value
	}
	assert.Equal(t, "/Anthropic/v1/messages", customMap[":path"])
}

func TestBuildRequestHeaders_OmitsEmptyAPIKey(t *testing.T) {
	headers := BuildRequestHeaders("", 512, "")
	headerMap := make(map[string]string)
	for _, h := range headers {
		headerMap[h.Key] = h.Value
	}
	_, hasKey := headerMap["x-api-key"]
	assert.False(t, hasKey)
}

func TestHeadersToRemove(t *testing.T) {
	headers := HeadersToRemove()

	assert.Contains(t, headers, "authorization")
	assert.Contains(t, headers, "content-length")
}

func TestToAnthropicRequestBody_WithTools(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Model: "claude-sonnet-4-5",
		Messages: []openai.ChatCompletionMessageParamUnion{
			{OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String("What's the weather in Paris?"),
				},
			}},
		},
		Tools: []openai.ChatCompletionToolParam{{
			Type: "function",
			Function: openai.FunctionDefinitionParam{
				Name:        "get_weather",
				Description: openai.String("Get weather for a city"),
				Parameters: openai.FunctionParameters{
					"type": "object",
					"properties": map[string]any{
						"city": map[string]any{"type": "string"},
					},
					"required": []any{"city"},
				},
			},
		}},
		ToolChoice: openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: openai.String("auto"),
		},
	}

	body, err := ToAnthropicRequestBody(req)
	require.NoError(t, err)

	var parsed anthropic.MessageNewParams
	require.NoError(t, json.Unmarshal(body, &parsed))

	require.Len(t, parsed.Tools, 1)
	require.NotNil(t, parsed.Tools[0].OfTool)
	assert.Equal(t, "get_weather", parsed.Tools[0].OfTool.Name)
	assert.Equal(t, anthropic.ToolTypeCustom, parsed.Tools[0].OfTool.Type)

	var wire map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(body, &wire))
	var tools []map[string]any
	require.NoError(t, json.Unmarshal(wire["tools"], &tools))
	require.Len(t, tools, 1)
	assert.Equal(t, "custom", tools[0]["type"])
}

func TestToAnthropicRequestBody_WithToolHistory(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Model: "claude-sonnet-4-5",
		Messages: []openai.ChatCompletionMessageParamUnion{
			{OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String("Weather in Paris?"),
				},
			}},
			{OfAssistant: &openai.ChatCompletionAssistantMessageParam{
				ToolCalls: []openai.ChatCompletionMessageToolCallParam{{
					ID:   "call_abc",
					Type: "function",
					Function: openai.ChatCompletionMessageToolCallFunctionParam{
						Name:      "get_weather",
						Arguments: `{"city":"Paris"}`,
					},
				}},
			}},
			{OfTool: &openai.ChatCompletionToolMessageParam{
				ToolCallID: "call_abc",
				Content: openai.ChatCompletionToolMessageParamContentUnion{
					OfString: openai.String(`{"temp_c": 18}`),
				},
			}},
		},
	}

	body, err := ToAnthropicRequestBody(req)
	require.NoError(t, err)

	var parsed anthropic.MessageNewParams
	require.NoError(t, json.Unmarshal(body, &parsed))
	require.Len(t, parsed.Messages, 3)

	assistant := parsed.Messages[1]
	assert.Equal(t, anthropic.MessageParamRoleAssistant, assistant.Role)
	require.NotNil(t, assistant.Content[0].OfToolUse)
	assert.Equal(t, "call_abc", assistant.Content[0].OfToolUse.ID)
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

func TestOpenAIFinishReasonFromAnthropic(t *testing.T) {
	cases := []struct {
		name          string
		stopReason    anthropic.StopReason
		toolCallCount int
		want          string
	}{
		{name: "end_turn maps to stop", stopReason: anthropic.StopReasonEndTurn, want: "stop"},
		{name: "max_tokens maps to length", stopReason: anthropic.StopReasonMaxTokens, want: "length"},
		{name: "tool_use maps to tool_calls", stopReason: anthropic.StopReasonToolUse, want: "tool_calls"},
		{name: "stop with tool calls upgrades to tool_calls", stopReason: anthropic.StopReasonEndTurn, toolCallCount: 1, want: "tool_calls"},
		{name: "pause_turn collapses to stop", stopReason: anthropic.StopReasonPauseTurn, want: "stop"},
		{name: "refusal collapses to stop", stopReason: anthropic.StopReasonRefusal, want: "stop"},
		{name: "stop_sequence collapses to stop", stopReason: anthropic.StopReasonStopSequence, want: "stop"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := openAIFinishReasonFromAnthropic(tc.stopReason, tc.toolCallCount)
			assert.Equal(t, tc.want, got)
		})
	}
}

func TestCaptureAnthropicStopReasonIntoExt_PauseTurn(t *testing.T) {
	ext := &ir.IRExtensions{}
	captureAnthropicStopReasonIntoExt(anthropic.StopReasonPauseTurn, "", ext)
	assert.Equal(t, "pause_turn", ext.AnthropicStopReason)
	assert.Empty(t, ext.AnthropicStopSequence)
}

func TestCaptureAnthropicStopReasonIntoExt_Refusal(t *testing.T) {
	ext := &ir.IRExtensions{}
	captureAnthropicStopReasonIntoExt(anthropic.StopReasonRefusal, "", ext)
	assert.Equal(t, "refusal", ext.AnthropicStopReason)
}

func TestCaptureAnthropicStopReasonIntoExt_StopSequence(t *testing.T) {
	ext := &ir.IRExtensions{}
	captureAnthropicStopReasonIntoExt(anthropic.StopReasonStopSequence, "###END###", ext)
	assert.Equal(t, "stop_sequence", ext.AnthropicStopReason,
		"stop_sequence must be captured so the outbound emitter can restore it; "+
			"OpenAI finish_reason collapses it to \"stop\"")
	assert.Equal(t, "###END###", ext.AnthropicStopSequence)
}

func TestCaptureAnthropicStopReasonIntoExt_StopSequenceWithoutSequenceString(t *testing.T) {
	// Defensive: even if the upstream omits the matched sequence string,
	// the stop_reason itself must still round-trip so the emitter
	// reports stop_sequence rather than end_turn.
	ext := &ir.IRExtensions{}
	captureAnthropicStopReasonIntoExt(anthropic.StopReasonStopSequence, "", ext)
	assert.Equal(t, "stop_sequence", ext.AnthropicStopReason)
	assert.Empty(t, ext.AnthropicStopSequence)
}

func TestCaptureAnthropicStopReasonIntoExt_EndTurnLeavesExtClean(t *testing.T) {
	ext := &ir.IRExtensions{}
	captureAnthropicStopReasonIntoExt(anthropic.StopReasonEndTurn, "", ext)
	assert.Empty(t, ext.AnthropicStopReason)
	assert.Empty(t, ext.AnthropicStopSequence)
}

func TestCaptureAnthropicStopReasonIntoExt_NilSafe(t *testing.T) {
	captureAnthropicStopReasonIntoExt(anthropic.StopReasonRefusal, "", nil)
}

func TestExtractThinkingBlocks_PopulatesSignatures(t *testing.T) {
	blocks := []anthropic.ContentBlockUnion{
		{Type: "thinking", Thinking: "let me reason...", Signature: "sig-1"},
		{Type: "text", Text: "the answer is 42"},
		{Type: "thinking", Thinking: "and another...", Signature: "sig-2"},
	}
	ext := &ir.IRExtensions{}
	extractThinkingBlocks(blocks, ext)

	assert.Equal(t, "sig-1", ext.ThinkingSignatures["content[0]"])
	assert.Equal(t, "sig-2", ext.ThinkingSignatures["content[1]"])
}

func TestExtractThinkingBlocks_SkipsEmptySignatures(t *testing.T) {
	blocks := []anthropic.ContentBlockUnion{
		{Type: "thinking", Thinking: "no signature here"},
	}
	ext := &ir.IRExtensions{}
	extractThinkingBlocks(blocks, ext)
	assert.Empty(t, ext.ThinkingSignatures)
}

func TestExtractThinkingBlocks_NilSafe(t *testing.T) {
	blocks := []anthropic.ContentBlockUnion{{Type: "thinking", Signature: "sig"}}
	extractThinkingBlocks(blocks, nil)
}

func TestExtractAnthropicUsageIntoExt_PopulatesAllFields(t *testing.T) {
	u := anthropic.Usage{
		InputTokens:              100,
		OutputTokens:             50,
		CacheReadInputTokens:     120,
		CacheCreationInputTokens: 80,
		CacheCreation: anthropic.CacheCreation{
			Ephemeral5mInputTokens: 30,
			Ephemeral1hInputTokens: 50,
		},
		ServerToolUse: anthropic.ServerToolUsage{
			WebSearchRequests: 2,
		},
	}
	ext := &ir.IRExtensions{}
	extractAnthropicUsageIntoExt(u, ext)

	assert.Equal(t, int64(120), ext.CacheReadInputTokens)
	assert.Equal(t, int64(80), ext.CacheCreationInputTokens)
	assert.Equal(t, int64(30), ext.Ephemeral5mInputTokens)
	assert.Equal(t, int64(50), ext.Ephemeral1hInputTokens)
	assert.Equal(t, int64(2), ext.ServerToolUseCounts["web_search"])
}

func TestExtractAnthropicUsageIntoExt_OmitsServerToolWhenZero(t *testing.T) {
	u := anthropic.Usage{
		InputTokens:  10,
		OutputTokens: 5,
	}
	ext := &ir.IRExtensions{}
	extractAnthropicUsageIntoExt(u, ext)
	assert.Empty(t, ext.ServerToolUseCounts, "no server tool keys for zero counts")
}

func TestExtractAnthropicUsageIntoExt_NilSafe(t *testing.T) {
	extractAnthropicUsageIntoExt(anthropic.Usage{InputTokens: 1}, nil)
}

// ToOpenAIResponseBody must remain byte-identical for the legacy
// OpenAI-client / Anthropic-backend cell after the refactor. The new
// helpers accept an ext but the legacy entrypoint passes nil.
func TestToOpenAIResponseBody_NoSideEffectsOnNilExt(t *testing.T) {
	anthropicResp := &anthropic.Message{
		ID: "msg_refactor",
		Content: []anthropic.ContentBlockUnion{
			{Type: "thinking", Thinking: "reasoning...", Signature: "sig-xyz"},
			{Type: "text", Text: "answer"},
		},
		StopReason: anthropic.StopReasonRefusal,
		Usage: anthropic.Usage{
			InputTokens:              10,
			OutputTokens:             5,
			CacheReadInputTokens:     7,
			CacheCreationInputTokens: 3,
		},
	}
	body, err := json.Marshal(anthropicResp)
	require.NoError(t, err)

	out, err := ToOpenAIResponseBody(body, "claude-sonnet-4-5")
	require.NoError(t, err)

	var oa openai.ChatCompletion
	require.NoError(t, json.Unmarshal(out, &oa))
	assert.Equal(t, "msg_refactor", oa.ID)
	// Refusal collapses to "stop" in the OpenAI envelope; no side
	// channel exists for the legacy entrypoint to expose it.
	assert.Equal(t, "stop", oa.Choices[0].FinishReason)
	assert.Equal(t, "answer", oa.Choices[0].Message.Content)
}

func TestToOpenAIResponseBodyWithExt_PopulatesIRExtensions(t *testing.T) {
	anthropicResp := &anthropic.Message{
		ID: "msg_ext",
		Content: []anthropic.ContentBlockUnion{
			{Type: "thinking", Thinking: "reasoning", Signature: "sig-1"},
			{Type: "text", Text: "answer"},
		},
		StopReason: anthropic.StopReasonPauseTurn,
		Usage: anthropic.Usage{
			InputTokens:              100,
			OutputTokens:             50,
			CacheReadInputTokens:     200,
			CacheCreationInputTokens: 75,
			CacheCreation: anthropic.CacheCreation{
				Ephemeral5mInputTokens: 25,
				Ephemeral1hInputTokens: 50,
			},
			ServerToolUse: anthropic.ServerToolUsage{WebSearchRequests: 4},
		},
	}
	body, err := json.Marshal(anthropicResp)
	require.NoError(t, err)

	ext := &ir.IRExtensions{SourceProtocol: SourceProtocolAnthropic}
	out, err := ToOpenAIResponseBodyWithExt(body, "claude-opus-4-7", ext)
	require.NoError(t, err)

	// Returned envelope is OpenAI-shape (parser does not change with ext).
	var oa openai.ChatCompletion
	require.NoError(t, json.Unmarshal(out, &oa))
	assert.Equal(t, "msg_ext", oa.ID)

	// Ext is populated with the values the OpenAI envelope cannot carry.
	assert.Equal(t, "pause_turn", ext.AnthropicStopReason)
	assert.Equal(t, int64(200), ext.CacheReadInputTokens)
	assert.Equal(t, int64(75), ext.CacheCreationInputTokens)
	assert.Equal(t, int64(25), ext.Ephemeral5mInputTokens)
	assert.Equal(t, int64(50), ext.Ephemeral1hInputTokens)
	assert.Equal(t, int64(4), ext.ServerToolUseCounts["web_search"])
	assert.Equal(t, "sig-1", ext.ThinkingSignatures["content[0]"])
}

func TestToOpenAIResponseBodyWithExt_NilExtMatchesLegacyEntrypoint(t *testing.T) {
	anthropicResp := &anthropic.Message{
		ID: "msg_nil_ext",
		Content: []anthropic.ContentBlockUnion{
			{Type: "text", Text: "x"},
		},
		StopReason: anthropic.StopReasonEndTurn,
		Usage:      anthropic.Usage{InputTokens: 1, OutputTokens: 1},
	}
	body, err := json.Marshal(anthropicResp)
	require.NoError(t, err)

	legacyOut, err := ToOpenAIResponseBody(body, "m")
	require.NoError(t, err)

	withExtOut, err := ToOpenAIResponseBodyWithExt(body, "m", nil)
	require.NoError(t, err)

	// Identical apart from the "created" Unix timestamp.
	var legacy, withExt openai.ChatCompletion
	require.NoError(t, json.Unmarshal(legacyOut, &legacy))
	require.NoError(t, json.Unmarshal(withExtOut, &withExt))
	legacy.Created = 0
	withExt.Created = 0
	assert.Equal(t, legacy, withExt)
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
