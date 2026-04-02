package extproc

import (
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
		assert.Equal(t, replayToolStepClientToolResult, trace.Steps[2].Type)
		assert.Equal(t, "get_weather", trace.Steps[2].ToolName)
		assert.Equal(t, "call_weather", trace.Steps[2].ToolCallID)
		assert.Empty(t, trace.Steps[2].Text)
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
	}
}
