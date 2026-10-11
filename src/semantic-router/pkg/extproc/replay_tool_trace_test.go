package extproc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestBuildReplayRoutingRecordUsesNeutralRequestToolTrace(t *testing.T) {
	ctx := &RequestContext{
		RequestID:    "req-tool-1",
		SourceFormat: llmprotocol.OpenAIResponsesV1,
		SemanticRequest: &llmprotocol.Request{
			Model: "vllm-sr/auto",
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

	record := buildReplayRoutingRecord(ctx, "vllm-sr/auto", "model-a", "default_route")
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

// Review #3806 (P1): suppressing RequestBody and Prompt was not enough. The
// tool trace carries user text and tool arguments and results, and every
// record is written before masking runs, so a masking route persisted
// alice@example.com in the trace while the body and prompt were empty.
func TestBuildReplayRoutingRecordSuppressesRequestTraceWhenMasking(t *testing.T) {
	const raw = "alice@example.com"
	newCtx := func(looper bool) *RequestContext {
		return &RequestContext{
			RequestID:     "req-mask-trace",
			SourceFormat:  llmprotocol.OpenAIChatV1,
			LooperRequest: looper,
			SemanticRequest: &llmprotocol.Request{
				Model: "auto",
				Messages: []llmprotocol.Message{
					{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{
						{Kind: llmprotocol.ContentText, Text: "email " + raw},
					}},
					{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
						{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{
							ID: "call_1", Name: "send", Arguments: `{"to":"` + raw + `"}`,
						}},
					}},
					{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{
						{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{
							CallID:  "call_1",
							Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "sent to " + raw}},
						}},
					}},
				},
			},
		}
	}

	// Baseline: without the plugin the trace is recorded as before.
	plain := buildReplayRoutingRecord(newCtx(false), "auto", "model-a", "route")
	require.NotNil(t, plain.ToolTrace, "a route without masking must still record its trace")

	maskingDecision := &config.Decision{Plugins: []config.DecisionPlugin{{
		Type:          config.DecisionPluginMasking,
		Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true}),
	}}}

	// Both paths write through this one builder: the ordinary request path and
	// the internal Looper path, which records before the final mask.
	for _, looper := range []bool{false, true} {
		ctx := newCtx(looper)
		ctx.VSRSelectedDecision = maskingDecision

		record := buildReplayRoutingRecord(ctx, "auto", "model-a", "route")

		require.Nil(t, record.ToolTrace, "looper=%v: request tool trace must be suppressed", looper)
		require.Empty(t, record.RequestBody, "looper=%v: request body must be suppressed", looper)
		require.Empty(t, record.Prompt, "looper=%v: prompt must be suppressed", looper)
		// Routing metadata still records; only request content is dropped.
		require.Equal(t, "route", record.Decision)
		require.Equal(t, "model-a", record.SelectedModel)
	}
}
