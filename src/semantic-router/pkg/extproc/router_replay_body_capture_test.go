package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

const (
	replayEarlierTurnText = "EARLIER-USER-TURN"
	replayToolArgsText    = "TOOL-ARGUMENTS"
	replayToolResultText  = "TOOL-RESULT"
	replayCurrentTurnText = "CURRENT-USER-TURN"
	replayToolSchemaText  = "TOOL-SCHEMA"
	replayAnswerText      = "MODEL-ANSWER"
	replayAnswerArgsText  = "ANSWER-ARGUMENTS"
)

var (
	replayRequestTexts  = []string{replayEarlierTurnText, replayToolArgsText, replayToolResultText, replayCurrentTurnText, replayToolSchemaText}
	replayResponseTexts = []string{replayAnswerText, replayAnswerArgsText}
)

// replayConversationRequest is a tool loop whose every text field carries a
// marker, so a stored record can be searched for any request text.
func replayConversationRequest(streaming bool) *RequestContext {
	ctx := looperTransportContext(llmprotocol.OpenAIChatV1, streaming)
	ctx.LooperRequest, ctx.RequestID, ctx.ExpectStreamingResponse = false, "replay-body-capture", streaming
	text := func(s string) []llmprotocol.Content {
		return []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: s + " " + strings.Repeat("x", 64)}}
	}
	ctx.SemanticRequest.Messages = []llmprotocol.Message{
		{Role: llmprotocol.RoleUser, Content: text(replayEarlierTurnText)},
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{
			ID: "call-1", Name: "lookup", Arguments: `{"q":"` + replayToolArgsText + `"}`,
		}}}},
		{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{
			CallID: "call-1", Content: text(replayToolResultText),
		}}}},
		{Role: llmprotocol.RoleUser, Content: text(replayCurrentTurnText)},
	}
	ctx.SemanticRequest.Tools = []llmprotocol.Tool{{Name: "lookup", Description: replayToolSchemaText, InputSchema: []byte(`{"type":"object"}`)}}
	return ctx
}

func replayConversationReply(t *testing.T, router *OpenAIRouter, ctx *RequestContext, streaming bool) {
	t.Helper()
	answer := replayAnswerText + " " + strings.Repeat("y", 64)
	call := `{"index":0,"id":"call-2","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"` + replayAnswerArgsText + `\"}"}}`
	if !streaming {
		runBufferedReply(t, router, ctx, 200, []byte(`{"id":"r1","object":"chat.completion","model":"public-model","choices":[{"index":0,"message":{"role":"assistant","content":"`+answer+`","tool_calls":[`+call+`]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`), "")
		return
	}
	stream := strings.Join([]string{
		`data: {"id":"r1","model":"public-model","choices":[{"index":0,"delta":{"role":"assistant","content":"` + answer + `"}}]}`,
		`data: {"id":"r1","model":"public-model","choices":[{"index":0,"delta":{"tool_calls":[` + call + `]},"finish_reason":"tool_calls"}]}`,
		`data: {"id":"r1","model":"public-model","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`,
		`data: [DONE]`,
	}, "\n\n") + "\n\n"
	headers, err := router.handleResponseHeaders(upstreamReplyHeaders(200, "text/event-stream", "", 0), ctx)
	require.NoError(t, err)
	require.Nil(t, headers.GetImmediateResponse())
	_, err = router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: []byte(stream), EndOfStream: true},
	}, ctx)
	require.NoError(t, err)
}

func recordReplayConversation(t *testing.T, plugin map[string]interface{}, streaming bool) routerreplay.RoutingRecord {
	t.Helper()
	plugin["enabled"], plugin["capture_tool_results"] = true, true
	cfg := withGlobalPlugins(t, &config.RouterConfig{RouterReplay: config.RouterReplayConfig{StoreBackend: "memory"}}, replayTestPlugin(plugin))
	router := replayRuntimeRouter(t, cfg)
	ctx := replayConversationRequest(streaming)
	ctx.RouterReplayPluginConfig = cfg.EffectiveRouterReplayConfig(nil)
	router.startRouterReplay(ctx, "public-model", "public-model", "")
	replayConversationReply(t, router, ctx, streaming)
	record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleCompleted, record.LifecycleState)
	return record
}

// With a body capture switch off, that side's text reaches no field of the
// record, while the trace keeps its steps, tool names and call IDs.
func TestReplayBodyCaptureOffStoresNoConversationText(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		for _, capture := range []struct{ request, response bool }{{false, false}, {true, false}, {false, true}, {true, true}} {
			name := map[bool]string{false: "buffered", true: "streaming"}[streaming]
			name += map[bool]string{false: "/request_off", true: "/request_on"}[capture.request]
			name += map[bool]string{false: "/response_off", true: "/response_on"}[capture.response]
			t.Run(name, func(t *testing.T) {
				record := recordReplayConversation(t, map[string]interface{}{
					"capture_request_body": capture.request, "capture_response_body": capture.response,
				}, streaming)
				stored, err := json.Marshal(record)
				require.NoError(t, err)
				for _, text := range replayRequestTexts {
					require.Equal(t, capture.request, strings.Contains(string(stored), text), "request text %s with capture_request_body=%t", text, capture.request)
				}
				for _, text := range replayResponseTexts {
					require.Equal(t, capture.response, strings.Contains(string(stored), text), "response text %s with capture_response_body=%t", text, capture.response)
				}
				require.NotNil(t, record.ToolTrace)
				require.Equal(t, []string{"lookup"}, record.ToolTrace.ToolNames)
				types, ids := []string{}, []string{}
				for _, step := range record.ToolTrace.Steps {
					types, ids = append(types, step.Type), append(ids, step.ToolCallID)
					require.False(t, step.ContentRedacted, "step %s: text the capture switch left out is not role redaction", step.Type)
				}
				require.Equal(t, []string{
					replayToolStepUserInput, replayToolStepAssistantToolCall, replayToolStepClientToolResult, replayToolStepUserInput,
					replayToolStepAssistantFinalResponse, replayToolStepAssistantToolCall,
				}, types)
				require.Equal(t, []string{"", "call-1", "call-1", "", "", "call-2"}, ids)
			})
		}
	}
}

// max_tool_trace_bytes bounds every stored trace text, user turns and the
// answer included, and marks each step and trajectory message it cuts.
func TestReplayToolTraceByteLimitBoundsStepText(t *testing.T) {
	const limit = 24
	for _, streaming := range []bool{false, true} {
		record := recordReplayConversation(t, map[string]interface{}{
			"capture_request_body": true, "capture_response_body": true, "max_tool_trace_bytes": limit,
		}, streaming)
		require.NotNil(t, record.ToolTrace)
		texts := 0
		for _, step := range record.ToolTrace.Steps {
			for _, value := range []string{step.Text, step.Arguments, step.Output} {
				require.LessOrEqual(t, len(value), limit, "step %s stored %d bytes", step.Type, len(value))
			}
			if step.Text != "" {
				texts++
				require.True(t, step.Truncated, "step %s was cut without its flag", step.Type)
			}
		}
		require.Equal(t, 3, texts, "streaming=%t: both user turns and the answer are kept, cut", streaming)
		messages := buildTrajectoryMessages(buildTrajectoryTurns([]routerreplay.RoutingRecord{record}))
		for _, message := range messages {
			if message.Role != "tool" && message.Content != "" {
				require.True(t, message.ContentTruncated, "trajectory %s text was cut without its flag", message.Role)
			}
		}
	}
}

// A provider's error reply is response content. With capture_response_body
// off, the fallback execution record keeps each attempt's status and a fixed
// reason, not the reply.
func TestReplayFallbackExecutionKeepsProviderRepliesOnlyWhenCaptured(t *testing.T) {
	const echo = "ECHOED-PROMPT"
	for _, capture := range []bool{false, true} {
		policy := fallback.DefaultEnabledPolicy()
		policy.MaxAttempts = 2
		router, _ := setupFallbackTestRouter(t, policy)
		recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
		recorder.SetCapturePolicy(capture, capture, 4096)
		t.Cleanup(func() { _ = recorder.Close() })
		recordID, err := recorder.AddRecord(routerreplay.RoutingRecord{ID: "replay-fallback-capture"})
		require.NoError(t, err)

		ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
		ctx.RouterReplayID, ctx.RouterReplayRecorder = recordID, recorder
		ctx.UpstreamStatusCode = http.StatusServiceUnavailable
		router.fallbackCaller = func(context.Context, string, []byte, map[string]string) ([]byte, int, error) {
			return []byte(`{"error":"cannot answer ` + echo + `"}`), http.StatusServiceUnavailable, nil
		}
		require.Nil(t, router.handleUpstreamTransportError([]byte("overloaded: "+echo), ctx).GetImmediateResponse())

		record, ok := recorder.GetRecord(recordID)
		require.True(t, ok)
		stored, err := json.Marshal(record)
		require.NoError(t, err)
		require.Equal(t, capture, strings.Contains(string(stored), echo), "capture_response_body=%t", capture)
		var execution fallback.ExecutionRecord
		require.NoError(t, json.Unmarshal([]byte(record.Outcomes[len(record.Outcomes)-1].Metadata["execution_record"]), &execution))
		require.Len(t, execution.Attempts, 2)
		for _, attempt := range execution.Attempts {
			require.Equal(t, http.StatusServiceUnavailable, attempt.StatusCode)
			if !capture {
				require.Equal(t, "upstream returned status 503", attempt.ErrorMessage)
			}
		}
	}
}
