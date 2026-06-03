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

// buildOpenAIBody is a test helper that returns a marshalled OpenAI
// ChatCompletion body fitting the given fields. Keeping the helper
// inline avoids a testdata fixture explosion for shapes that are tiny.
func buildOpenAIBody(t *testing.T, choice openai.ChatCompletionChoice, usage openai.CompletionUsage, id string) []byte {
	t.Helper()
	cc := &openai.ChatCompletion{
		ID:      id,
		Object:  "chat.completion",
		Model:   "claude-sonnet-4-5",
		Choices: []openai.ChatCompletionChoice{choice},
		Usage:   usage,
	}
	body, err := json.Marshal(cc)
	require.NoError(t, err)
	return body
}

func parseAnthropicResponse(t *testing.T, body []byte) anthropic.Message {
	t.Helper()
	var msg anthropic.Message
	require.NoError(t, json.Unmarshal(body, &msg), "anthropic-sdk-go must unmarshal emitted body")
	return msg
}

func TestEmitAnthropicResponse_PlainText(t *testing.T) {
	body := buildOpenAIBody(t,
		openai.ChatCompletionChoice{
			FinishReason: "stop",
			Message: openai.ChatCompletionMessage{
				Role:    "assistant",
				Content: "Hello, world!",
			},
		},
		openai.CompletionUsage{PromptTokens: 5, CompletionTokens: 3},
		"msg_text_only",
	)

	out, err := EmitAnthropicResponse(body, nil, "claude-opus-4-7")
	require.NoError(t, err)
	msg := parseAnthropicResponse(t, out)

	assert.Equal(t, "msg_text_only", msg.ID)
	assert.Equal(t, "message", string(msg.Type))
	assert.Equal(t, "assistant", string(msg.Role))
	assert.Equal(t, "claude-opus-4-7", string(msg.Model))
	assert.Equal(t, anthropic.StopReasonEndTurn, msg.StopReason)
	require.Len(t, msg.Content, 1)
	assert.Equal(t, "text", msg.Content[0].Type)
	assert.Equal(t, "Hello, world!", msg.Content[0].Text)
	assert.Equal(t, int64(5), msg.Usage.InputTokens)
	assert.Equal(t, int64(3), msg.Usage.OutputTokens)
}

func TestEmitAnthropicResponse_SingleToolCall(t *testing.T) {
	body := buildOpenAIBody(t,
		openai.ChatCompletionChoice{
			FinishReason: "tool_calls",
			Message: openai.ChatCompletionMessage{
				Role: "assistant",
				ToolCalls: []openai.ChatCompletionMessageToolCall{{
					ID:   "call_abc",
					Type: "function",
					Function: openai.ChatCompletionMessageToolCallFunction{
						Name:      "get_weather",
						Arguments: `{"city":"Paris"}`,
					},
				}},
			},
		},
		openai.CompletionUsage{PromptTokens: 10, CompletionTokens: 20},
		"msg_tool_use",
	)

	out, err := EmitAnthropicResponse(body, nil, "claude-opus-4-7")
	require.NoError(t, err)
	msg := parseAnthropicResponse(t, out)

	assert.Equal(t, anthropic.StopReasonToolUse, msg.StopReason)
	require.Len(t, msg.Content, 1)
	assert.Equal(t, "tool_use", msg.Content[0].Type)
	assert.Equal(t, "call_abc", msg.Content[0].ID)
	assert.Equal(t, "get_weather", msg.Content[0].Name)
	assert.JSONEq(t, `{"city":"Paris"}`, string(msg.Content[0].Input))
}

func TestEmitAnthropicResponse_TextPlusToolCall(t *testing.T) {
	body := buildOpenAIBody(t,
		openai.ChatCompletionChoice{
			FinishReason: "tool_calls",
			Message: openai.ChatCompletionMessage{
				Role:    "assistant",
				Content: "Looking up the weather.",
				ToolCalls: []openai.ChatCompletionMessageToolCall{{
					ID: "call_1",
					Function: openai.ChatCompletionMessageToolCallFunction{
						Name:      "get_weather",
						Arguments: `{"city":"NYC"}`,
					},
				}},
			},
		},
		openai.CompletionUsage{PromptTokens: 15, CompletionTokens: 25},
		"msg_text_and_tool",
	)

	out, err := EmitAnthropicResponse(body, nil, "claude-opus-4-7")
	require.NoError(t, err)
	msg := parseAnthropicResponse(t, out)

	require.Len(t, msg.Content, 2)
	assert.Equal(t, "text", msg.Content[0].Type, "text precedes tool_use in document order")
	assert.Equal(t, "tool_use", msg.Content[1].Type)
}

func TestEmitAnthropicResponse_EmptyToolCallArguments(t *testing.T) {
	body := buildOpenAIBody(t,
		openai.ChatCompletionChoice{
			FinishReason: "tool_calls",
			Message: openai.ChatCompletionMessage{
				ToolCalls: []openai.ChatCompletionMessageToolCall{{
					ID: "call_x",
					Function: openai.ChatCompletionMessageToolCallFunction{
						Name:      "noop",
						Arguments: "",
					},
				}},
			},
		},
		openai.CompletionUsage{},
		"msg_empty_args",
	)

	out, err := EmitAnthropicResponse(body, nil, "model")
	require.NoError(t, err)
	msg := parseAnthropicResponse(t, out)
	assert.JSONEq(t, `{}`, string(msg.Content[0].Input))
}

func TestEmitAnthropicResponse_ThinkingSignaturePrecedesText(t *testing.T) {
	body := buildOpenAIBody(t,
		openai.ChatCompletionChoice{
			FinishReason: "stop",
			Message: openai.ChatCompletionMessage{
				Role:    "assistant",
				Content: "Forty-two.",
			},
		},
		openai.CompletionUsage{PromptTokens: 8, CompletionTokens: 3},
		"msg_thinking",
	)

	ext := &ir.IRExtensions{
		SourceProtocol:     SourceProtocolAnthropic,
		ThinkingSignatures: map[string]string{"content[0]": "sig-deadbeef"},
	}
	out, err := EmitAnthropicResponse(body, ext, "claude-opus-4-7")
	require.NoError(t, err)
	msg := parseAnthropicResponse(t, out)

	require.Len(t, msg.Content, 2)
	assert.Equal(t, "thinking", msg.Content[0].Type)
	assert.Equal(t, "sig-deadbeef", msg.Content[0].Signature)
	assert.NotEmpty(t, msg.Content[0].Thinking, "placeholder fills until PR #1718 lands")
	assert.Equal(t, "text", msg.Content[1].Type)
	assert.Equal(t, "Forty-two.", msg.Content[1].Text)
}

func TestEmitAnthropicResponse_StopReasonLength(t *testing.T) {
	body := buildOpenAIBody(t,
		openai.ChatCompletionChoice{
			FinishReason: "length",
			Message:      openai.ChatCompletionMessage{Content: "Truncated..."},
		},
		openai.CompletionUsage{PromptTokens: 10, CompletionTokens: 100},
		"msg_length",
	)
	out, err := EmitAnthropicResponse(body, nil, "m")
	require.NoError(t, err)
	msg := parseAnthropicResponse(t, out)
	assert.Equal(t, anthropic.StopReasonMaxTokens, msg.StopReason)
}

func TestEmitAnthropicResponse_StopReasonContentFilter(t *testing.T) {
	body := buildOpenAIBody(t,
		openai.ChatCompletionChoice{
			FinishReason: "content_filter",
			Message:      openai.ChatCompletionMessage{Refusal: "I cannot help with that."},
		},
		openai.CompletionUsage{},
		"msg_refusal",
	)
	out, err := EmitAnthropicResponse(body, nil, "m")
	require.NoError(t, err)
	msg := parseAnthropicResponse(t, out)
	assert.Equal(t, anthropic.StopReasonRefusal, msg.StopReason)
	require.Len(t, msg.Content, 1)
	assert.Equal(t, "text", msg.Content[0].Type)
	assert.Equal(t, "I cannot help with that.", msg.Content[0].Text)
}

func TestEmitAnthropicResponse_AnthropicStopReasonOverride_PauseTurn(t *testing.T) {
	body := buildOpenAIBody(t,
		openai.ChatCompletionChoice{
			FinishReason: "stop",
			Message:      openai.ChatCompletionMessage{Content: "Partial..."},
		},
		openai.CompletionUsage{},
		"msg_pause",
	)
	ext := &ir.IRExtensions{
		SourceProtocol:      SourceProtocolAnthropic,
		AnthropicStopReason: string(anthropic.StopReasonPauseTurn),
	}
	out, err := EmitAnthropicResponse(body, ext, "m")
	require.NoError(t, err)
	msg := parseAnthropicResponse(t, out)
	assert.Equal(t, anthropic.StopReasonPauseTurn, msg.StopReason)
	assert.Empty(t, msg.StopSequence)
}

func TestEmitAnthropicResponse_StopSequenceOverride(t *testing.T) {
	body := buildOpenAIBody(t,
		openai.ChatCompletionChoice{
			FinishReason: "stop",
			Message:      openai.ChatCompletionMessage{Content: "Truncated by ###END"},
		},
		openai.CompletionUsage{},
		"msg_seq",
	)
	ext := &ir.IRExtensions{
		SourceProtocol:        SourceProtocolAnthropic,
		AnthropicStopReason:   string(anthropic.StopReasonStopSequence),
		AnthropicStopSequence: "###END",
	}
	out, err := EmitAnthropicResponse(body, ext, "m")
	require.NoError(t, err)
	msg := parseAnthropicResponse(t, out)
	assert.Equal(t, anthropic.StopReasonStopSequence, msg.StopReason)
	assert.Equal(t, "###END", msg.StopSequence)
}

func TestEmitAnthropicResponse_CacheUsageRoundTrip(t *testing.T) {
	body := buildOpenAIBody(t,
		openai.ChatCompletionChoice{
			FinishReason: "stop",
			Message:      openai.ChatCompletionMessage{Content: "cached"},
		},
		openai.CompletionUsage{PromptTokens: 200, CompletionTokens: 10},
		"msg_cache",
	)
	ext := &ir.IRExtensions{
		SourceProtocol:           SourceProtocolAnthropic,
		CacheReadInputTokens:     120,
		CacheCreationInputTokens: 80,
		Ephemeral5mInputTokens:   30,
		Ephemeral1hInputTokens:   50,
	}
	out, err := EmitAnthropicResponse(body, ext, "m")
	require.NoError(t, err)
	msg := parseAnthropicResponse(t, out)
	assert.Equal(t, int64(120), msg.Usage.CacheReadInputTokens)
	assert.Equal(t, int64(80), msg.Usage.CacheCreationInputTokens)
	assert.Equal(t, int64(30), msg.Usage.CacheCreation.Ephemeral5mInputTokens)
	assert.Equal(t, int64(50), msg.Usage.CacheCreation.Ephemeral1hInputTokens)
}

func TestEmitAnthropicResponse_ServerToolUseCounts(t *testing.T) {
	body := buildOpenAIBody(t,
		openai.ChatCompletionChoice{
			FinishReason: "stop",
			Message:      openai.ChatCompletionMessage{Content: "searched"},
		},
		openai.CompletionUsage{},
		"msg_server_tool",
	)
	ext := &ir.IRExtensions{
		SourceProtocol:      SourceProtocolAnthropic,
		ServerToolUseCounts: map[string]int64{"web_search": 3},
		// non-zero cache so the OpenAI-backend warning does not fire
		CacheReadInputTokens: 1,
	}
	out, err := EmitAnthropicResponse(body, ext, "m")
	require.NoError(t, err)
	msg := parseAnthropicResponse(t, out)
	assert.Equal(t, int64(3), msg.Usage.ServerToolUse.WebSearchRequests)
}

func TestEmitAnthropicResponse_OpenAIBackendOmitsCacheObjectAndWarns(t *testing.T) {
	body := buildOpenAIBody(t,
		openai.ChatCompletionChoice{
			FinishReason: "stop",
			Message:      openai.ChatCompletionMessage{Content: "no cache"},
		},
		openai.CompletionUsage{PromptTokens: 20, CompletionTokens: 10},
		"msg_openai_backend",
	)
	ext := &ir.IRExtensions{
		SourceProtocol: SourceProtocolAnthropic,
	}
	out, err := EmitAnthropicResponse(body, ext, "m")
	require.NoError(t, err)

	// cache_creation must be absent in the raw JSON output.
	assert.NotContains(t, string(out), `"cache_creation":`)

	// Info warning recorded so observability captures the structural diff.
	require.NotEmpty(t, ext.Warnings)
	var sawCacheWarning bool
	for _, w := range ext.Warnings {
		if w.Reason == ir.ReasonCacheFieldsAbsent {
			sawCacheWarning = true
			assert.Equal(t, ir.WarningSeverityInfo, w.Severity)
		}
	}
	assert.True(t, sawCacheWarning, "expected cache_fields_absent warning")
}

func TestEmitAnthropicResponse_EmptyContentReturnsEmptyArray(t *testing.T) {
	body := buildOpenAIBody(t,
		openai.ChatCompletionChoice{
			FinishReason: "stop",
			Message:      openai.ChatCompletionMessage{Role: "assistant"},
		},
		openai.CompletionUsage{},
		"msg_empty",
	)
	out, err := EmitAnthropicResponse(body, nil, "m")
	require.NoError(t, err)

	// Distinguish "null" from "[]" at the wire level so Anthropic
	// SDK clients that test typeof content === "object" do not trip.
	assert.Contains(t, string(out), `"content":[]`)
	assert.NotContains(t, string(out), `"content":null`)
}

func TestEmitAnthropicResponse_UnknownFinishReasonFallsBackWithWarning(t *testing.T) {
	body := buildOpenAIBody(t,
		openai.ChatCompletionChoice{
			FinishReason: "something_unexpected",
			Message:      openai.ChatCompletionMessage{Content: "ok"},
		},
		openai.CompletionUsage{},
		"msg_unknown_finish",
	)
	ext := &ir.IRExtensions{SourceProtocol: SourceProtocolAnthropic}
	out, err := EmitAnthropicResponse(body, ext, "m")
	require.NoError(t, err)
	msg := parseAnthropicResponse(t, out)
	assert.Equal(t, anthropic.StopReasonEndTurn, msg.StopReason)

	var sawCoerced bool
	for _, w := range ext.Warnings {
		if w.Reason == ir.ReasonAnthropicStopReasonCoerced {
			sawCoerced = true
		}
	}
	assert.True(t, sawCoerced, "expected anthropic_stop_reason_coerced warning")
}

func TestEmitAnthropicResponse_EmptyFinishReasonCoerces(t *testing.T) {
	body := buildOpenAIBody(t,
		openai.ChatCompletionChoice{
			Message: openai.ChatCompletionMessage{Content: "synth"},
		},
		openai.CompletionUsage{},
		"msg_empty_finish",
	)
	ext := &ir.IRExtensions{SourceProtocol: SourceProtocolAnthropic}
	out, err := EmitAnthropicResponse(body, ext, "m")
	require.NoError(t, err)
	msg := parseAnthropicResponse(t, out)
	assert.Equal(t, anthropic.StopReasonEndTurn, msg.StopReason)
}

func TestEmitAnthropicResponse_InvalidJSONReturnsError(t *testing.T) {
	_, err := EmitAnthropicResponse([]byte("not json"), nil, "m")
	assert.Error(t, err)
}

func TestEmitAnthropicResponse_NoChoicesEmitsEmptyContent(t *testing.T) {
	cc := &openai.ChatCompletion{
		ID:      "msg_no_choices",
		Object:  "chat.completion",
		Model:   "m",
		Choices: nil,
		Usage:   openai.CompletionUsage{PromptTokens: 1},
	}
	body, err := json.Marshal(cc)
	require.NoError(t, err)
	out, err := EmitAnthropicResponse(body, nil, "m")
	require.NoError(t, err)
	msg := parseAnthropicResponse(t, out)
	assert.Equal(t, "msg_no_choices", msg.ID)
	assert.Empty(t, msg.Content)
	assert.Equal(t, anthropic.StopReasonEndTurn, msg.StopReason)
}

func TestEmitAnthropicError_KnownTypes(t *testing.T) {
	cases := []struct {
		errorType string
		message   string
	}{
		{"invalid_request_error", "bad request"},
		{"authentication_error", "no key"},
		{"permission_error", "forbidden"},
		{"not_found_error", "missing"},
		{"request_too_large", "too big"},
		{"rate_limit_error", "slow down"},
		{"api_error", "server hiccup"},
		{"timeout_error", "timed out"},
		{"overloaded_error", "try again"},
	}
	for _, tc := range cases {
		t.Run(tc.errorType, func(t *testing.T) {
			body := EmitAnthropicError(tc.errorType, tc.message)
			var env struct {
				Type  string `json:"type"`
				Error struct {
					Type    string `json:"type"`
					Message string `json:"message"`
				} `json:"error"`
			}
			require.NoError(t, json.Unmarshal(body, &env))
			assert.Equal(t, "error", env.Type)
			assert.Equal(t, tc.errorType, env.Error.Type)
			assert.Equal(t, tc.message, env.Error.Message)
		})
	}
}

func TestEmitAnthropicError_EmptyErrorTypeCoercesToAPIError(t *testing.T) {
	body := EmitAnthropicError("", "boom")
	var env struct {
		Error struct {
			Type string `json:"type"`
		} `json:"error"`
	}
	require.NoError(t, json.Unmarshal(body, &env))
	assert.Equal(t, "api_error", env.Error.Type)
}

// Round-trip: feed an Anthropic Message through the inverse helper
// (toOpenAIResponseBody with a populated ext), then through the
// outbound emitter, and assert the structural identity holds for the
// fields PR4 promises to round-trip.
func TestRoundTrip_AnthropicResponse_StopReasonAndUsage(t *testing.T) {
	original := &anthropic.Message{
		ID: "msg_round_trip",
		Content: []anthropic.ContentBlockUnion{
			{Type: "text", Text: "Hello"},
		},
		StopReason: anthropic.StopReasonPauseTurn,
		Usage: anthropic.Usage{
			InputTokens:              50,
			OutputTokens:             20,
			CacheReadInputTokens:     30,
			CacheCreationInputTokens: 10,
			CacheCreation: anthropic.CacheCreation{
				Ephemeral5mInputTokens: 6,
				Ephemeral1hInputTokens: 4,
			},
		},
	}
	originalBody, err := json.Marshal(original)
	require.NoError(t, err)

	ext := &ir.IRExtensions{SourceProtocol: SourceProtocolAnthropic}
	openaiBody, err := toOpenAIResponseBody(originalBody, "claude-opus-4-7", ext)
	require.NoError(t, err)

	out, err := EmitAnthropicResponse(openaiBody, ext, "claude-opus-4-7")
	require.NoError(t, err)
	restored := parseAnthropicResponse(t, out)

	assert.Equal(t, original.ID, restored.ID)
	assert.Equal(t, original.StopReason, restored.StopReason)
	assert.Equal(t, original.Usage.InputTokens, restored.Usage.InputTokens)
	assert.Equal(t, original.Usage.OutputTokens, restored.Usage.OutputTokens)
	assert.Equal(t, original.Usage.CacheReadInputTokens, restored.Usage.CacheReadInputTokens)
	assert.Equal(t, original.Usage.CacheCreationInputTokens, restored.Usage.CacheCreationInputTokens)
	assert.Equal(t, original.Usage.CacheCreation.Ephemeral5mInputTokens, restored.Usage.CacheCreation.Ephemeral5mInputTokens)
	assert.Equal(t, original.Usage.CacheCreation.Ephemeral1hInputTokens, restored.Usage.CacheCreation.Ephemeral1hInputTokens)
}

func TestOrderedThinkingBlockIDs_LexicographicOrder(t *testing.T) {
	// Verify orderedThinkingBlockIDs returns IDs in deterministic
	// lexicographic order. Multi-digit indices sort by byte value, not
	// by parsed integer, so "content[10]" precedes "content[1]" because
	// '0' (0x30) < ']' (0x5D) at the first differing byte. The SSE
	// emitter consumes this for replay ordering and requires only
	// determinism, not numeric ordering.
	ext := &ir.IRExtensions{
		ThinkingSignatures: map[string]string{
			"content[2]":  "sig2",
			"content[10]": "sig10",
			"content[1]":  "sig1",
			"content[0]":  "sig0",
		},
	}
	got := orderedThinkingBlockIDs(ext)
	assert.Equal(t, []string{"content[0]", "content[10]", "content[1]", "content[2]"}, got)
}

func TestEmitAnthropicResponse_PreservesIDAndModel(t *testing.T) {
	body := buildOpenAIBody(t,
		openai.ChatCompletionChoice{
			FinishReason: "stop",
			Message:      openai.ChatCompletionMessage{Content: "x"},
		},
		openai.CompletionUsage{},
		"msg_preserve_id",
	)
	out, err := EmitAnthropicResponse(body, nil, "claude-haiku-4-5")
	require.NoError(t, err)

	// Spot-check the raw wire output: id and model must be present
	// as top-level strings (Anthropic SDK clients key on them).
	s := string(out)
	assert.Contains(t, s, `"id":"msg_preserve_id"`)
	assert.Contains(t, s, `"model":"claude-haiku-4-5"`)
	assert.Contains(t, s, `"type":"message"`)
	assert.Contains(t, s, `"role":"assistant"`)
}
