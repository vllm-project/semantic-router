package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func replayUserTurnWire(format llmprotocol.WireFormat, cycles int, nextUser bool) []byte {
	user := `{"role":"user","content":"Read the neutral state."}`
	call := `{"role":"assistant","tool_calls":[{"id":"call-%d","type":"function","function":{"name":"lookup","arguments":"{}"}}]}`
	result := `{"role":"tool","tool_call_id":"call-%d","content":"ready"}`
	field, extra := "messages", ""
	switch format {
	case llmprotocol.OpenAIResponsesV1:
		field, extra = "input", `,"store":false`
		call = `{"type":"function_call","call_id":"call-%d","name":"lookup","arguments":"{}"}`
		result = `{"type":"function_call_output","call_id":"call-%d","output":"ready"}`
	case llmprotocol.AnthropicMessagesV1:
		extra = `,"max_tokens":64`
		call = `{"role":"assistant","content":[{"type":"tool_use","id":"call-%d","name":"lookup","input":{}}]}`
		result = `{"role":"user","content":[{"type":"tool_result","tool_use_id":"call-%d","content":"ready"}]}`
	}
	messages := []string{user}
	for cycle := 1; cycle <= cycles; cycle++ {
		messages = append(messages, fmt.Sprintf(call, cycle), fmt.Sprintf(result, cycle))
	}
	if nextUser {
		messages = append(messages, `{"role":"assistant","content":"Done."}`, `{"role":"user","content":"Now summarize the state."}`)
	}
	return fmt.Appendf(nil, `{"model":"public","%s":[%s]%s}`, field, strings.Join(messages, ","), extra)
}

func recordReplayUserTurn(t *testing.T, router *OpenAIRouter, format llmprotocol.WireFormat, body []byte, conversation string) (*RequestContext, routerreplay.RoutingRecord) {
	t.Helper()
	ctx := &RequestContext{
		SourceFormat: format, TraceContext: t.Context(),
		Headers:                  map[string]string{"x-session-id": "shared-session", "x-conversation-id": conversation},
		RouterReplayPluginConfig: &config.RouterReplayPluginConfig{Enabled: true},
	}
	request, immediate := router.prepareProtocolRequest(body, ctx)
	require.Nil(t, immediate)
	require.NotNil(t, request)
	_, err := router.extractRequestSignalSnapshot(ctx)
	require.NoError(t, err)
	populateSessionTransitionFields(ctx)
	invocationIndex := ctx.TurnIndex
	router.startRouterReplay(ctx, "public", "backend", "route")
	require.Equal(t, invocationIndex, ctx.TurnIndex, "Replay must not change selection/telemetry invocation indexing")
	record, found := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
	require.True(t, found)
	return ctx, record
}

func TestReplayUserTurnsThroughProtocolPipeline(t *testing.T) {
	for _, format := range extProcMatrixFormats {
		t.Run(string(format), func(t *testing.T) {
			router := &OpenAIRouter{Config: &config.RouterConfig{}, ReplayRecorder: routerreplay.NewRecorder(store.NewMemoryStore(10, 0)), ReplayStoreShared: true}
			for cycles := 0; cycles <= 2; cycles++ {
				ctx, record := recordReplayUserTurn(t, router, format, replayUserTurnWire(format, cycles, false), "conversation-a")
				require.Zero(t, record.TurnIndex, "tool results do not start another user turn")
				if format != llmprotocol.OpenAIResponsesV1 {
					require.Equal(t, cycles, ctx.TurnIndex, "assistant invocation semantics must remain unchanged")
				}
			}
			_, separate := recordReplayUserTurn(t, router, format, replayUserTurnWire(format, 0, false), "conversation-b")
			require.Zero(t, separate.TurnIndex)

			response := router.handleRouterReplayTrajectoryAPI("GET", "session_id=shared-session").GetImmediateResponse()
			require.EqualValues(t, 200, response.Status.Code)
			var trajectory struct {
				RecordCount int                 `json:"record_count"`
				TurnCount   int                 `json:"turn_count"`
				Messages    []trajectoryMessage `json:"messages"`
			}
			require.NoError(t, json.Unmarshal(response.Body, &trajectory))
			require.Equal(t, 4, trajectory.RecordCount)
			require.Equal(t, 2, trajectory.TurnCount)
			require.Len(t, trajectory.Messages, 6, "one user, two call/result pairs, then the other conversation's user")
			for i, message := range trajectory.Messages {
				wantCID := "conversation-a"
				if i == 5 {
					wantCID = "conversation-b"
				}
				require.Equal(t, wantCID, message.ConversationID)
				require.Zero(t, message.TurnIndex)
			}
			require.Equal(t, "call-1", trajectory.Messages[1].ToolCalls[0].ID)
			require.Equal(t, "call-1", trajectory.Messages[2].ToolCallID)
			require.Equal(t, "call-2", trajectory.Messages[3].ToolCalls[0].ID)
			require.Equal(t, "call-2", trajectory.Messages[4].ToolCallID)

			_, next := recordReplayUserTurn(t, router, format, replayUserTurnWire(format, 2, true), "conversation-a")
			require.Equal(t, 1, next.TurnIndex, "an ordinary new user message advances the user turn once")
			require.Len(t, next.ToolTrace.Steps, 6, "the individual record retains its complete input trace")
			response = router.handleRouterReplayTrajectoryAPI("GET", "session_id=shared-session").GetImmediateResponse()
			require.NoError(t, json.Unmarshal(response.Body, &trajectory))
			require.Equal(t, 5, trajectory.RecordCount)
			require.Equal(t, 3, trajectory.TurnCount)
			require.Len(t, trajectory.Messages, 7, "the new user turn must not repeat the previous tool history")
			require.Equal(t, "Now summarize the state.", trajectory.Messages[6].Content)
			require.Equal(t, "conversation-a", trajectory.Messages[6].ConversationID)
			require.Equal(t, 1, trajectory.Messages[6].TurnIndex)
		})
	}
}

func TestReplayUserTurnUsesMaterializedResponsesHistory(t *testing.T) {
	responseStore := NewMockResponseStore()
	require.NoError(t, responseStore.StoreResponse(t.Context(), &responseapi.StoredResponse{
		ID:     "resp-parent",
		Input:  []responseapi.InputItem{{Type: responseapi.ItemTypeMessage, Role: responseapi.RoleUser, Content: json.RawMessage(`"Read the neutral state."`)}},
		Output: []responseapi.OutputItem{{Type: responseapi.ItemTypeFunctionCall, CallID: "call-1", Name: "lookup", Arguments: `{}`}},
	}))
	router := &OpenAIRouter{Config: &config.RouterConfig{}, ResponseAPIFilter: NewResponseAPIFilter(responseStore), ReplayRecorder: routerreplay.NewRecorder(store.NewMemoryStore(10, 0))}
	ctx, record := recordReplayUserTurn(t, router, llmprotocol.OpenAIResponsesV1, []byte(`{"model":"public","store":false,"previous_response_id":"resp-parent","input":[{"type":"function_call_output","call_id":"call-1","output":"ready"}]}`), "conversation-a")
	require.Equal(t, 1, ctx.TurnIndex, "stored assistant invocation count is unchanged")
	require.Zero(t, record.TurnIndex)
	require.Len(t, record.ToolTrace.Steps, 3)
}

func TestReplayUserTurnsPreserveRepeatedQuestions(t *testing.T) {
	for _, format := range extProcMatrixFormats {
		t.Run(string(format), func(t *testing.T) {
			router := &OpenAIRouter{Config: &config.RouterConfig{}, ReplayRecorder: routerreplay.NewRecorder(store.NewMemoryStore(10, 0)), ReplayStoreShared: true}
			recordReplayUserTurn(t, router, format, replayUserTurnWire(format, 0, false), "conversation-a")
			body := strings.ReplaceAll(string(replayUserTurnWire(format, 0, true)), "Now summarize the state.", "Read the neutral state.")
			recordReplayUserTurn(t, router, format, []byte(body), "conversation-a")
			response := router.handleRouterReplayTrajectoryAPI("GET", "session_id=shared-session").GetImmediateResponse()
			var trajectory routerReplayTrajectoryResponse
			require.NoError(t, json.Unmarshal(response.Body, &trajectory))
			require.Equal(t, 2, trajectory.TurnCount)
			require.Len(t, trajectory.Messages, 2)
			for index, message := range trajectory.Messages {
				require.Equal(t, index, message.TurnIndex)
				require.Equal(t, "Read the neutral state.", message.Content)
			}
		})
	}
}

func TestReplayUserTurnWithoutVisibleHistoryKeepsFallback(t *testing.T) {
	for _, request := range []*llmprotocol.Request{nil, {}, {Messages: []llmprotocol.Message{{Role: llmprotocol.RoleTool}}}} {
		ctx := &RequestContext{TurnIndex: 7, SemanticRequest: request}
		require.Equal(t, 7, buildReplayRoutingRecord(ctx, "public", "backend", "route").TurnIndex)
		require.Equal(t, 7, ctx.TurnIndex)
	}
}

func TestReplayUserTurnUsesHistoryBeforeContextChanges(t *testing.T) {
	for _, trim := range []bool{false, true} {
		t.Run(fmt.Sprintf("trim=%t", trim), func(t *testing.T) {
			router := &OpenAIRouter{Config: &config.RouterConfig{}, ReplayRecorder: routerreplay.NewRecorder(store.NewMemoryStore(10, 0))}
			ctx := &RequestContext{
				SourceFormat:             llmprotocol.OpenAIChatV1,
				TraceContext:             t.Context(),
				RouterReplayPluginConfig: &config.RouterReplayPluginConfig{Enabled: true},
			}
			request, immediate := router.prepareProtocolRequest(replayUserTurnWire(llmprotocol.OpenAIChatV1, 0, true), ctx)
			require.Nil(t, immediate)
			require.NotNil(t, request)
			_, err := router.extractRequestSignalSnapshot(ctx)
			require.NoError(t, err)
			if trim {
				ctx.ContextHistorySteps = []contextcompression.TransformationStep{{
					Kind: contextcompression.TransformSelectTurns, Enabled: true, FailureMode: contextcompression.FailureClosed,
					Propose: func(context.Context, contextcompression.TransformationView) (contextcompression.TransformationEdits, error) {
						return contextcompression.TransformationEdits{RemoveMessages: []int{0, 1}}, nil
					},
				}}
				require.NoError(t, router.applyContextTransformationPlan(ctx, request))
				require.Len(t, request.Messages, 1)
			} else {
				request.Messages = append([]llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Saved context."}}}}, request.Messages...)
				ctx.MemoryMessageIndexes = map[int]struct{}{0: {}}
				require.NoError(t, router.applyContextTransformationPlan(ctx, request))
				require.Len(t, request.Messages, 4)
			}
			router.startRouterReplay(ctx, "public", "backend", "route")
			record, found := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			require.True(t, found)
			require.Equal(t, 1, record.TurnIndex)
		})
	}
}
