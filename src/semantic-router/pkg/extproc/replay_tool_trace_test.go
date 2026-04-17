package extproc

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestBuildReplayRoutingRecordIncludesRequestToolTrace(t *testing.T) {
	ctx := &RequestContext{
		RequestID: "req-tool-1",
		OriginalRequestBody: []byte(`{
			"model": "auto",
			"messages": [
				{"role": "system", "content": "You are helpful."},
				{"role": "user", "content": "Find the weather in San Francisco."},
				{
					"role": "assistant",
					"content": null,
					"tool_calls": [
						{
							"id": "call_weather",
							"type": "function",
							"function": {
								"name": "get_weather",
								"arguments": "{\"location\":\"San Francisco\"}"
							}
						}
					]
				},
				{
					"role": "tool",
					"tool_call_id": "call_weather",
					"content": "{\"temperature\":\"18C\",\"condition\":\"sunny\"}"
				}
			]
		}`),
	}

	record := buildReplayRoutingRecord(ctx, "MoM", "model-a", "default_route")
	if assert.NotNil(t, record.ToolTrace) {
		assert.Equal(t, "User Query -> LLM Tool Call -> Client Tool Result", record.ToolTrace.Flow)
		assert.Equal(t, "Client Tool Result", record.ToolTrace.Stage)
		assert.Equal(t, []string{"get_weather"}, record.ToolTrace.ToolNames)
		assert.Len(t, record.ToolTrace.Steps, 3)
		assert.Equal(t, replayToolStepUserInput, record.ToolTrace.Steps[0].Type)
		assert.Equal(t, replayToolStepAssistantToolCall, record.ToolTrace.Steps[1].Type)
		assert.Equal(t, replayToolStepClientToolResult, record.ToolTrace.Steps[2].Type)
	}
}

func TestAttachRouterReplayResponseMergesToolTrace(t *testing.T) {
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(false, true, 4096)
	requestTrace := newReplayToolTrace([]routerreplay.ToolTraceStep{
		{Type: replayToolStepUserInput, Source: replayToolSourceRequest, Role: "user", Text: "Find the weather."},
		{Type: replayToolStepAssistantToolCall, Source: replayToolSourceRequest, Role: "assistant", ToolName: "get_weather", ToolCallID: "call_weather", Arguments: "{\"location\":\"San Francisco\"}"},
		{Type: replayToolStepClientToolResult, Source: replayToolSourceRequest, Role: "tool", ToolName: "get_weather", ToolCallID: "call_weather", Text: "{\"temperature\":\"18C\"}"},
	})
	replayID, err := recorder.AddRecord(routerreplay.RoutingRecord{
		ID:        "replay-tool-response",
		RequestID: "req-tool-2",
		Decision:  "default_route",
		ToolTrace: requestTrace,
	})
	if err != nil {
		t.Fatalf("failed to add replay record: %v", err)
	}

	router := &OpenAIRouter{ReplayRecorder: recorder}
	ctx := &RequestContext{
		RequestID:            "req-tool-2",
		RouterReplayID:       replayID,
		RouterReplayRecorder: recorder,
	}

	router.attachRouterReplayResponse(ctx, []byte(`{
		"id": "chatcmpl-1",
		"object": "chat.completion",
		"created": 1234567890,
		"model": "model-a",
		"choices": [
			{
				"index": 0,
				"message": {
					"role": "assistant",
					"content": "It is 18C and sunny in San Francisco."
				},
				"finish_reason": "stop"
			}
		]
	}`), true)

	record, found := recorder.GetRecord(replayID)
	if !found {
		t.Fatal("expected replay record to be present")
	}
	if assert.NotNil(t, record.ToolTrace) {
		assert.Equal(t, "User Query -> LLM Tool Call -> Client Tool Result -> LLM Final Response", record.ToolTrace.Flow)
		assert.Equal(t, "LLM Final Response", record.ToolTrace.Stage)
		assert.Len(t, record.ToolTrace.Steps, 4)
		assert.Equal(t, replayToolStepAssistantFinalResponse, record.ToolTrace.Steps[3].Type)
	}
	assert.Contains(t, record.ResponseBody, "sunny in San Francisco")
}

func TestParseResponseAPIRequestToolTrace(t *testing.T) {
	trace := parseResponseAPIRequestToolTrace([]byte(`[
		{
			"type": "message",
			"role": "user",
			"content": [{"type": "input_text", "text": "Find the weather in San Francisco."}]
		},
		{
			"type": "function_call",
			"call_id": "call_weather",
			"name": "get_weather",
			"arguments": "{\"location\":\"San Francisco\"}"
		},
		{
			"type": "function_call_output",
			"call_id": "call_weather",
			"output": {"temperature": "18C", "condition": "sunny"}
		}
	]`))

	if assert.NotNil(t, trace) {
		assert.Equal(t, "User Query -> LLM Tool Call -> Client Tool Result", trace.Flow)
		assert.Equal(t, []string{"get_weather"}, trace.ToolNames)
		assert.Len(t, trace.Steps, 3)
		assert.Equal(t, replayToolStepClientToolResult, trace.Steps[2].Type)
		assert.Contains(t, trace.Steps[2].Text, "temperature")
	}
}

func TestParseChatCompletionRequestToolTracePreservesNullToolResult(t *testing.T) {
	trace := parseChatCompletionRequestToolTrace([]byte(`{
		"model": "auto",
		"messages": [
			{"role": "user", "content": "Call the tool."},
			{
				"role": "assistant",
				"content": null,
				"tool_calls": [
					{
						"id": "call_weather",
						"type": "function",
						"function": {
							"name": "get_weather",
							"arguments": "{\"location\":\"San Francisco\"}"
						}
					}
				]
			},
			{
				"role": "tool",
				"tool_call_id": "call_weather",
				"content": null
			}
		]
	}`))

	if assert.NotNil(t, trace) {
		assert.Len(t, trace.Steps, 3)

		// Chat Completions tool call step should preserve RawArguments.
		assert.Equal(t, replayToolStepAssistantToolCall, trace.Steps[1].Type)
		assert.JSONEq(t, "{\"location\":\"San Francisco\"}", trace.Steps[1].RawArguments)

		// Null tool result: Text, RawOutput should both be empty.
		assert.Equal(t, replayToolStepClientToolResult, trace.Steps[2].Type)
		assert.Equal(t, "get_weather", trace.Steps[2].ToolName)
		assert.Equal(t, "call_weather", trace.Steps[2].ToolCallID)
		assert.Empty(t, trace.Steps[2].Text)
		assert.Empty(t, trace.Steps[2].RawOutput)
	}
}

func TestBuildReplayStreamingToolTrace(t *testing.T) {
	ctx := &RequestContext{
		StreamingContent: "It is 18C and sunny in San Francisco.",
		StreamingToolCalls: map[int]*StreamingToolCallState{
			0: {
				ID:        "call_weather",
				Name:      "get_weather",
				Arguments: "{\"location\":\"San Francisco\"}",
			},
		},
	}

	trace := buildReplayStreamingToolTrace(ctx)
	if assert.NotNil(t, trace) {
		assert.Equal(t, "LLM Tool Call -> LLM Final Response", trace.Flow)
		assert.Equal(t, "LLM Final Response", trace.Stage)
		assert.Equal(t, []string{"get_weather"}, trace.ToolNames)
		assert.Len(t, trace.Steps, 2)

		// Streaming tool call step should preserve RawArguments.
		assert.Equal(t, replayToolStepAssistantToolCall, trace.Steps[0].Type)
		assert.JSONEq(t, "{\"location\":\"San Francisco\"}", trace.Steps[0].RawArguments)
	}
}

func TestParseResponseAPIResponseToolTraceWithComplexContent(t *testing.T) {
	trace := parseResponseAPIResponseToolTrace([]byte(`{
		"output": [
			{
				"type": "function_call",
				"call_id": "call_weather",
				"name": "get_weather",
				"arguments": "{\"location\":\"San Francisco\"}"
			},
			{
				"type": "message",
				"role": "assistant",
				"content": [
					{"type": "output_text", "text": "The weather is sunny."}
				]
			}
		]
	}`))

	if assert.NotNil(t, trace) {
		assert.Equal(t, "LLM Tool Call -> LLM Final Response", trace.Flow)
		assert.Len(t, trace.Steps, 2)
		assert.Equal(t, replayToolStepAssistantToolCall, trace.Steps[0].Type)
		assert.Equal(t, "call_weather", trace.Steps[0].ToolCallID)
		assert.Equal(t, "get_weather", trace.Steps[0].ToolName)
		assert.JSONEq(t, "{\"location\":\"San Francisco\"}", trace.Steps[0].RawArguments)
		assert.Equal(t, replayToolStepAssistantFinalResponse, trace.Steps[1].Type)
		assert.Equal(t, "The weather is sunny.", trace.Steps[1].Text)
	}
}

func TestParseResponseAPIRequestToolTraceMultiToolStructuredOutput(t *testing.T) {
	trace := parseResponseAPIRequestToolTrace([]byte(`[
		{
			"type": "message",
			"role": "user",
			"content": "Find weather and news."
		},
		{
			"type": "function_call",
			"call_id": "call_weather",
			"name": "get_weather",
			"arguments": "{\"location\":\"San Francisco\"}"
		},
		{
			"type": "function_call_output",
			"call_id": "call_weather",
			"output": {"temperature": "18C", "condition": "sunny"}
		},
		{
			"type": "function_call",
			"call_id": "call_news",
			"name": "get_news",
			"arguments": "{\"topic\":\"tech\"}"
		},
		{
			"type": "function_call_output",
			"call_id": "call_news",
			"output": [{"headline": "AI advances", "source": "TechDaily"}]
		}
	]`))

	if assert.NotNil(t, trace) {
		assert.Equal(t, "User Query -> LLM Tool Call -> Client Tool Result -> LLM Tool Call -> Client Tool Result", trace.Flow)
		assert.Len(t, trace.Steps, 5)

		// First tool call
		assert.Equal(t, replayToolStepAssistantToolCall, trace.Steps[1].Type)
		assert.Equal(t, "call_weather", trace.Steps[1].ToolCallID)
		assert.Equal(t, "get_weather", trace.Steps[1].ToolName)
		assert.JSONEq(t, "{\"location\":\"San Francisco\"}", trace.Steps[1].RawArguments)

		// First structured output
		assert.Equal(t, replayToolStepClientToolResult, trace.Steps[2].Type)
		assert.Equal(t, "call_weather", trace.Steps[2].ToolCallID)
		assert.Equal(t, "get_weather", trace.Steps[2].ToolName)
		assert.JSONEq(t, `{"temperature":"18C","condition":"sunny"}`, trace.Steps[2].RawOutput)
		assert.Contains(t, trace.Steps[2].Text, "temperature")

		// Second tool call
		assert.Equal(t, replayToolStepAssistantToolCall, trace.Steps[3].Type)
		assert.Equal(t, "call_news", trace.Steps[3].ToolCallID)
		assert.Equal(t, "get_news", trace.Steps[3].ToolName)
		assert.JSONEq(t, "{\"topic\":\"tech\"}", trace.Steps[3].RawArguments)

		// Second structured output (array)
		assert.Equal(t, replayToolStepClientToolResult, trace.Steps[4].Type)
		assert.Equal(t, "call_news", trace.Steps[4].ToolCallID)
		assert.Equal(t, "get_news", trace.Steps[4].ToolName)
		assert.JSONEq(t, `[{"headline":"AI advances","source":"TechDaily"}]`, trace.Steps[4].RawOutput)
	}
}

func TestResponseAPIToolTraceRoundTrip(t *testing.T) {
	requestTrace := parseResponseAPIRequestToolTrace([]byte(`[
		{
			"type": "message",
			"role": "user",
			"content": "What's the weather?"
		},
		{
			"type": "function_call",
			"call_id": "call_weather_123",
			"name": "get_weather",
			"arguments": "{\"city\":\"Berlin\"}"
		},
		{
			"type": "function_call_output",
			"call_id": "call_weather_123",
			"output": {"temp": 22, "unit": "C"}
		}
	]`))

	responseTrace := parseResponseAPIResponseToolTrace([]byte(`{
		"output": [
			{
				"type": "message",
				"role": "assistant",
				"content": "It is 22C in Berlin."
			}
		]
	}`))

	merged := mergeReplayToolTraces(requestTrace, responseTrace)
	if !assert.NotNil(t, merged) {
		return
	}
	assert.Len(t, merged.Steps, 4)

	// Round-trip through JSON marshal/unmarshal to verify persistence fidelity.
	serialized, err := json.Marshal(merged)
	if !assert.NoError(t, err) {
		return
	}

	var restored routerreplay.ToolTrace
	if !assert.NoError(t, json.Unmarshal(serialized, &restored)) {
		return
	}

	if !assert.Len(t, restored.Steps, 4) {
		return
	}

	// Verify call_id -> ToolCallID survives round-trip.
	assert.Equal(t, "call_weather_123", restored.Steps[1].ToolCallID)
	assert.Equal(t, "call_weather_123", restored.Steps[2].ToolCallID)

	// Verify raw arguments preserved as exact string through round-trip.
	assert.JSONEq(t, "{\"city\":\"Berlin\"}", restored.Steps[1].RawArguments)

	// Verify raw structured output preserved as exact string through round-trip.
	// This confirms that RawOutput (a string field) is not altered by JSON
	// marshal/unmarshal — e.g. numeric values like 22 stay as "22" not "22.0".
	assert.Equal(t, restored.Steps[2].RawOutput, merged.Steps[2].RawOutput)
	assert.JSONEq(t, `{"temp":22,"unit":"C"}`, restored.Steps[2].RawOutput)

	// Verify text extracted from structured output is present.
	assert.Contains(t, restored.Steps[2].Text, "temp")

	// Verify flow and stage survive round-trip.
	assert.Equal(t, merged.Flow, restored.Flow)
	assert.Equal(t, merged.Stage, restored.Stage)
	assert.Equal(t, merged.ToolNames, restored.ToolNames)
}

func TestBuildReplayStreamingToolTracePreservesResponseAPICallID(t *testing.T) {
	ctx := &RequestContext{
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
		},
		StreamingContent: "It is 18C and sunny in San Francisco.",
		StreamingToolCalls: map[int]*StreamingToolCallState{
			0: {
				ID:        "call_weather_123",
				Name:      "get_weather",
				Arguments: "{\"location\":\"San Francisco\"}",
			},
			1: {
				ID:        "call_news_456",
				Name:      "get_news",
				Arguments: "{\"topic\":\"tech\"}",
			},
		},
	}

	trace := buildReplayStreamingToolTrace(ctx)
	if assert.NotNil(t, trace) {
		// Flow deduplicates consecutive identical step labels, so two assistant_tool_calls
		// collapse into a single "LLM Tool Call" label.
		assert.Equal(t, "LLM Tool Call -> LLM Final Response", trace.Flow)
		assert.Equal(t, []string{"get_weather", "get_news"}, trace.ToolNames)
		assert.Len(t, trace.Steps, 3)

		// First streamed tool call should preserve call_id as ToolCallID.
		assert.Equal(t, replayToolStepAssistantToolCall, trace.Steps[0].Type)
		assert.Equal(t, "call_weather_123", trace.Steps[0].ToolCallID)
		assert.Equal(t, "get_weather", trace.Steps[0].ToolName)
		assert.JSONEq(t, "{\"location\":\"San Francisco\"}", trace.Steps[0].RawArguments)

		// Second streamed tool call should preserve call_id as ToolCallID.
		assert.Equal(t, replayToolStepAssistantToolCall, trace.Steps[1].Type)
		assert.Equal(t, "call_news_456", trace.Steps[1].ToolCallID)
		assert.Equal(t, "get_news", trace.Steps[1].ToolName)
		assert.JSONEq(t, "{\"topic\":\"tech\"}", trace.Steps[1].RawArguments)

		// Final assistant response.
		assert.Equal(t, replayToolStepAssistantFinalResponse, trace.Steps[2].Type)
		assert.Equal(t, "It is 18C and sunny in San Francisco.", trace.Steps[2].Text)
	}
}

// ---------------------------------------------------------------------------
// Tests for issue #1780 – structured tool-call fields
// ---------------------------------------------------------------------------

func TestExtractChatCompletionPromptAndTools(t *testing.T) {
	body := []byte(`{
		"model": "auto",
		"messages": [
			{"role": "system", "content": "You are helpful."},
			{"role": "user", "content": "What is the weather in Paris?"}
		],
		"tools": [
			{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {}}}
		]
	}`)

	prompt, toolDefs := extractChatCompletionPromptAndTools(body)

	assert.Equal(t, "What is the weather in Paris?", prompt)
	assert.NotEmpty(t, toolDefs)
	assert.Contains(t, toolDefs, "get_weather")
}

func TestExtractChatCompletionPromptAndTools_NoTools(t *testing.T) {
	body := []byte(`{
		"model": "auto",
		"messages": [
			{"role": "user", "content": "Hello"}
		]
	}`)

	prompt, toolDefs := extractChatCompletionPromptAndTools(body)

	assert.Equal(t, "Hello", prompt)
	assert.Empty(t, toolDefs)
}

func TestExtractChatCompletionPromptAndTools_Empty(t *testing.T) {
	prompt, toolDefs := extractChatCompletionPromptAndTools(nil)
	assert.Empty(t, prompt)
	assert.Empty(t, toolDefs)
}

func TestBuildReplayRoutingRecord_PopulatesPromptAndToolDefs(t *testing.T) {
	ctx := &RequestContext{
		RequestID: "req-structured-1",
		OriginalRequestBody: []byte(`{
			"model": "auto",
			"messages": [
				{"role": "system", "content": "Be helpful."},
				{"role": "user", "content": "Find the weather in Tokyo."}
			],
			"tools": [
				{"type": "function", "function": {"name": "get_weather", "description": "Returns weather", "parameters": {}}}
			]
		}`),
	}

	record := buildReplayRoutingRecord(ctx, "MoM", "model-a", "default")

	assert.Equal(t, "Find the weather in Tokyo.", record.Prompt)
	assert.False(t, record.PromptTruncated)
	assert.NotEmpty(t, record.ToolDefinitions)
	assert.Contains(t, record.ToolDefinitions, "get_weather")
}

func TestBuildReplayRoutingRecord_NoTools_EmptyToolDefs(t *testing.T) {
	ctx := &RequestContext{
		RequestID: "req-structured-2",
		OriginalRequestBody: []byte(`{
			"model": "auto",
			"messages": [{"role": "user", "content": "Hello world"}]
		}`),
	}

	record := buildReplayRoutingRecord(ctx, "MoM", "model-a", "default")

	assert.Equal(t, "Hello world", record.Prompt)
	assert.Empty(t, record.ToolDefinitions)
}

func TestChatCompletionToolSteps_HaveAPITypeAndOutput(t *testing.T) {
	ctx := &RequestContext{
		RequestID: "req-apitype-chat",
		OriginalRequestBody: []byte(`{
			"model": "auto",
			"messages": [
				{"role": "user", "content": "Get weather."},
				{
					"role": "assistant",
					"content": null,
					"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}}]
				},
				{"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\":\"22C\"}"}
			]
		}`),
	}

	record := buildReplayRoutingRecord(ctx, "MoM", "model-a", "default")

	if assert.NotNil(t, record.ToolTrace) && assert.Len(t, record.ToolTrace.Steps, 3) {
		toolCallStep := record.ToolTrace.Steps[1]
		assert.Equal(t, replayAPITypeChatCompletions, toolCallStep.APIType)

		toolResultStep := record.ToolTrace.Steps[2]
		assert.Equal(t, replayAPITypeChatCompletions, toolResultStep.APIType)
		assert.JSONEq(t, `{"temp":"22C"}`, toolResultStep.Output)
		assert.Equal(t, toolResultStep.RawOutput, toolResultStep.Output)
	}
}

func TestMaxToolTraceBytesPrompTruncation(t *testing.T) {
	storage := store.NewMemoryStore(10, 0)
	recorder := routerreplay.NewRecorder(storage)
	recorder.SetCapturePolicy(false, false, 4096)
	recorder.SetMaxToolTraceBytes(10) // very small limit

	record := routerreplay.RoutingRecord{
		RequestID:       "req-trunc-1",
		Prompt:          "This is a prompt longer than ten bytes",
		ToolDefinitions: `[{"type":"function","function":{"name":"long_tool"}}]`,
	}

	id, err := recorder.AddRecord(record)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	stored, found := recorder.GetRecord(id)
	if !found {
		t.Fatal("record not found")
	}
	assert.Len(t, stored.Prompt, 10)
	assert.True(t, stored.PromptTruncated)
	assert.Len(t, stored.ToolDefinitions, 10)
}

func TestMaxToolTraceBytesStepTruncation(t *testing.T) {
	storage := store.NewMemoryStore(10, 0)
	recorder := routerreplay.NewRecorder(storage)
	recorder.SetCapturePolicy(false, false, 4096)
	recorder.SetMaxToolTraceBytes(5)

	trace := &routerreplay.ToolTrace{
		Steps: []routerreplay.ToolTraceStep{
			{
				Type:      replayToolStepAssistantToolCall,
				Arguments: `{"city":"New York"}`,
				Output:    "",
			},
			{
				Type:   replayToolStepClientToolResult,
				Output: `{"temperature":"22C","condition":"sunny"}`,
			},
		},
	}
	record := routerreplay.RoutingRecord{
		RequestID: "req-trunc-2",
		ToolTrace: trace,
	}

	id, err := recorder.AddRecord(record)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	stored, found := recorder.GetRecord(id)
	if !found {
		t.Fatal("record not found")
	}

	if assert.Len(t, stored.ToolTrace.Steps, 2) {
		assert.Len(t, stored.ToolTrace.Steps[0].Arguments, 5)
		assert.True(t, stored.ToolTrace.Steps[0].Truncated)

		assert.Len(t, stored.ToolTrace.Steps[1].Output, 5)
		assert.True(t, stored.ToolTrace.Steps[1].Truncated)
	}
}
