package extproc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestBuildReplayRoutingRecordUsesNeutralRequestToolTrace(t *testing.T) {
	ctx := &RequestContext{
		RequestID:    "req-tool-1",
		SourceFormat: llmprotocol.OpenAIResponsesV1,
		SemanticRequest: &llmprotocol.Request{
			Model: "auto",
			Messages: []llmprotocol.Message{
				{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Find the weather in San Francisco."}}},
				{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{
					ID: "call_weather", Name: "get_weather", Arguments: `{"location":"San Francisco"}`,
				}}}},
				{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{
					CallID:  "call_weather",
					Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: `{"temperature":"18C","condition":"sunny"}`}},
				}}}},
			},
			Tools: []llmprotocol.Tool{{Name: "get_weather", InputSchema: []byte(`{"type":"object"}`)}},
		},
	}

	record := buildReplayRoutingRecord(ctx, "MoM", "model-a", "default_route")
	require.NotNil(t, record.ToolTrace)
	require.Equal(t, "User Query -> LLM Tool Call -> Client Tool Result", record.ToolTrace.Flow)
	require.Equal(t, "Client Tool Result", record.ToolTrace.Stage)
	require.Equal(t, []string{"get_weather"}, record.ToolTrace.ToolNames)
	require.Len(t, record.ToolTrace.Steps, 3)
	require.Equal(t, string(llmprotocol.OpenAIResponsesV1), record.ToolTrace.Steps[0].APIType)
	require.Equal(t, replayToolStepAssistantToolCall, record.ToolTrace.Steps[1].Type)
	require.JSONEq(t, `{"location":"San Francisco"}`, record.ToolTrace.Steps[1].RawArguments)
	require.Equal(t, replayToolStepClientToolResult, record.ToolTrace.Steps[2].Type)
	require.Contains(t, record.ToolTrace.Steps[2].RawOutput, "temperature")
	require.Equal(t, "Find the weather in San Francisco.", record.Prompt)
	require.Contains(t, record.ToolDefinitions, "get_weather")
}

func TestBuildReplayResponseToolTraceUsesNeutralResponse(t *testing.T) {
	ctx := &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1,
		SemanticResponse: &llmprotocol.Response{
			Output: []llmprotocol.OutputItem{{
				Role: llmprotocol.RoleAssistant,
				Content: []llmprotocol.Content{
					{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "call_weather", Name: "get_weather", Arguments: `{"city":"SF"}`}},
					{Kind: llmprotocol.ContentText, Text: "It is sunny."},
				},
			}},
		},
	}

	trace := buildReplayResponseToolTrace(ctx, []byte("ignored transport body"))
	require.NotNil(t, trace)
	require.Equal(t, "LLM Tool Call -> LLM Final Response", trace.Flow)
	require.Equal(t, []string{"get_weather"}, trace.ToolNames)
	require.Len(t, trace.Steps, 2)
	require.Equal(t, replayToolSourceResponse, trace.Steps[0].Source)
	require.Equal(t, string(llmprotocol.AnthropicMessagesV1), trace.Steps[0].APIType)
	require.Equal(t, "It is sunny.", trace.Steps[1].Text)
}

func TestBuildReplayStreamingToolTraceUsesSemanticAccumulator(t *testing.T) {
	ctx := &RequestContext{
		SourceFormat:            llmprotocol.OpenAIChatV1,
		ExpectStreamingResponse: true,
		SemanticResponse: &llmprotocol.Response{
			Output: []llmprotocol.OutputItem{{
				Role: llmprotocol.RoleAssistant,
				Content: []llmprotocol.Content{
					{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "call_weather", Name: "get_weather", Arguments: `{"city":"SF"}`}},
					{Kind: llmprotocol.ContentText, Text: "It is sunny."},
				},
			}},
		},
	}

	trace := buildReplayStreamingToolTrace(ctx)
	require.NotNil(t, trace)
	require.Equal(t, "LLM Tool Call -> LLM Final Response", trace.Flow)
	require.Len(t, trace.Steps, 2)
	for _, step := range trace.Steps {
		require.Equal(t, replayToolSourceStream, step.Source)
	}
}

func TestBuildReplayTraceDoesNotPersistReasoningText(t *testing.T) {
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIResponsesV1,
		SemanticResponse: &llmprotocol.Response{Output: []llmprotocol.OutputItem{{
			Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentReasoning, Text: "private reasoning", Signature: "signed",
			}},
		}}},
	}

	trace := buildReplayStreamingToolTrace(ctx)
	require.NotNil(t, trace)
	require.Equal(t, "LLM Reasoning Complete", trace.Flow)
	require.Len(t, trace.Steps, 1)
	require.Empty(t, trace.Steps[0].Text)
	require.NotContains(t, trace.Steps[0].RawOutput, "private reasoning")
}

func TestAttachRouterReplayResponseMergesNeutralTrace(t *testing.T) {
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(false, true, 4096)
	replayID, err := recorder.AddRecord(routerreplay.RoutingRecord{
		ID:        "replay-tool-response",
		RequestID: "req-tool-2",
		Decision:  "default_route",
		ToolTrace: newReplayToolTrace([]routerreplay.ToolTraceStep{
			{Type: replayToolStepUserInput, Source: replayToolSourceRequest, Role: "user", Text: "Find the weather."},
			{Type: replayToolStepAssistantToolCall, Source: replayToolSourceRequest, Role: "assistant", ToolName: "get_weather", ToolCallID: "call_weather", Arguments: `{"city":"SF"}`},
		}),
	})
	require.NoError(t, err)

	ctx := &RequestContext{
		RequestID:            "req-tool-2",
		RouterReplayID:       replayID,
		RouterReplayRecorder: recorder,
		SourceFormat:         llmprotocol.OpenAIChatV1,
		SemanticResponse: &llmprotocol.Response{Output: []llmprotocol.OutputItem{{
			Role:    llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "It is sunny."}},
		}}},
	}
	(&OpenAIRouter{ReplayRecorder: recorder}).attachRouterReplayResponse(ctx, []byte(`{"safe":"client body"}`), true)

	record, found := recorder.GetRecord(replayID)
	require.True(t, found)
	require.NotNil(t, record.ToolTrace)
	require.Equal(t, "User Query -> LLM Tool Call -> LLM Final Response", record.ToolTrace.Flow)
	require.Len(t, record.ToolTrace.Steps, 3)
	require.Contains(t, record.ResponseBody, "client body")
}

func TestMergeReplayToolTracesDeduplicatesBoundaryStep(t *testing.T) {
	step := routerreplay.ToolTraceStep{Type: replayToolStepAssistantToolCall, ToolName: "lookup", ToolCallID: "call-1"}
	merged := mergeReplayToolTraces(newReplayToolTrace([]routerreplay.ToolTraceStep{step}), newReplayToolTrace([]routerreplay.ToolTraceStep{step}))
	require.NotNil(t, merged)
	require.Len(t, merged.Steps, 1)
}
