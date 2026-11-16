package extproc

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/codexreview"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// The turn the captured review judges, which its client_metadata names as
// parent_turn_id.
const reviewedTurn = "00000000-0000-4000-8000-000000000010"

// recordCodexCall runs one request through ingress and starts its replay
// record, as the Router does for a call Codex sends.
func recordCodexCall(t *testing.T, router *OpenAIRouter, id, header string, body []byte) string {
	t.Helper()
	ctx := &RequestContext{SourceFormat: llmprotocol.OpenAIResponsesV1, RequestID: id, TraceContext: t.Context()}
	if header != "" {
		ctx.Headers = map[string]string{headers.XCodexTurnMetadata: header}
	}
	_, immediate := router.prepareProtocolRequest(body, ctx)
	require.Nil(t, immediate)
	ctx.RouterReplayPluginConfig = router.Config.EffectiveRouterReplayConfig(nil)
	router.startRouterReplay(ctx, "vllm-sr/auto", "gpt-6-luna", "coding")
	require.NotEmpty(t, ctx.RouterReplayID)
	return ctx.RouterReplayID
}

func replayIDs(t *testing.T, router *OpenAIRouter, query string) []string {
	t.Helper()
	var list routerreplay.ListResponse
	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays?limit=100&"+query, &list))
	ids := make([]string, 0, len(list.Data))
	for _, record := range list.Data {
		ids = append(ids, record.ID)
	}
	return ids
}

// Every call a Codex turn sends names the turn, by header or in the body's
// client_metadata, and its approval reviews and subagents name it as their
// parent turn: one turn_id query finds all of them and nothing of another
// turn, and the session filter still narrows it.
func TestReplayFindsACodexTurnAndWhatItStarted(t *testing.T) {
	captured, err := os.ReadFile(codexReviewFixture)
	require.NoError(t, err)
	cfg := withGlobalPlugins(t, &config.RouterConfig{RouterReplay: config.RouterReplayConfig{StoreBackend: "memory"}},
		replayTestPlugin(map[string]interface{}{"enabled": true}))
	router := replayRuntimeRouter(t, cfg)
	metadata := func(fields map[string]string) string {
		encoded, _ := json.Marshal(fields)
		return string(encoded)
	}
	withMetadata := func(fields map[string]string) []byte {
		body, _ := json.Marshal(map[string]interface{}{"model": "vllm-sr/auto", "input": "fix the build", "client_metadata": fields})
		return body
	}

	byHeader := recordCodexCall(t, router, "by-header", metadata(map[string]string{"session_id": "s1", "thread_id": "s1", "turn_id": reviewedTurn}),
		[]byte(`{"model":"vllm-sr/auto","input":"fix the build"}`))
	byBody := recordCodexCall(t, router, "by-body", "", withMetadata(map[string]string{
		headers.XCodexTurnMetadata: metadata(map[string]string{"session_id": "s1", "thread_id": "s1", "turn_id": reviewedTurn}),
	}))
	review := recordCodexCall(t, router, "review", "", captured)
	subagent := recordCodexCall(t, router, "subagent", "", withMetadata(map[string]string{
		"session_id": "s1", "thread_id": "child-thread", "turn_id": "child-turn", "parent_turn_id": reviewedTurn,
	}))
	other := recordCodexCall(t, router, "next-turn", metadata(map[string]string{"thread_id": "s1", "turn_id": "next-turn"}),
		[]byte(`{"model":"vllm-sr/auto","input":"and the tests"}`))
	recordCodexCall(t, router, "no-turn", "", []byte(`{"model":"vllm-sr/auto","input":"hello"}`))

	record, ok := router.ReplayRecorder.GetRecord(review)
	require.True(t, ok)
	require.Equal(t, store.RequestKindCodexApprovalReview, record.RequestKind)
	require.Equal(t, []string{"00000000-0000-4000-8000-000000000007", "00000000-0000-4000-8000-000000000008", reviewedTurn},
		[]string{record.ThreadID, record.TurnID, record.ParentTurnID})
	record, ok = router.ReplayRecorder.GetRecord(byHeader)
	require.True(t, ok)
	require.Equal(t, []string{"s1", reviewedTurn, ""}, []string{record.ThreadID, record.TurnID, record.ParentTurnID})

	require.ElementsMatch(t, []string{byHeader, byBody, review, subagent}, replayIDs(t, router, "turn_id="+reviewedTurn))
	require.ElementsMatch(t, []string{review}, replayIDs(t, router, "turn_id="+reviewedTurn+"&request_kind=codex_approval_review"))
	require.ElementsMatch(t, []string{other}, replayIDs(t, router, "turn_id=next-turn"))
	require.Empty(t, replayIDs(t, router, "turn_id="+reviewedTurn+"&session_id=another-session"))

	var aggregate routerreplay.AggregateResponse
	replayAPIJSON(t, router, "/api/v1/observability/replays/aggregate?turn_id="+reviewedTurn, &aggregate)
	require.Equal(t, 4, aggregate.RecordCount)
	require.NotNil(t, aggregate.Reviews)
}

// A Chat Completions body has no client_metadata to read, so only the
// header names its turn.
func TestAgentTurnReadsTheBodyOnlyOnResponses(t *testing.T) {
	body := []byte(`{"client_metadata":{"turn_id":"from-body"}}`)
	chat := &RequestContext{SourceFormat: llmprotocol.OpenAIChatV1}
	observeAgentTurn(chat, body)
	require.Equal(t, codexreview.Turn{}, chat.AgentTurn)
	chat.Headers = map[string]string{headers.XCodexTurnMetadata: `{"turn_id":"from-header"}`}
	observeAgentTurn(chat, body)
	require.Equal(t, "from-header", chat.AgentTurn.TurnID)
	responses := &RequestContext{SourceFormat: llmprotocol.OpenAIResponsesV1}
	observeAgentTurn(responses, body)
	require.Equal(t, "from-body", responses.AgentTurn.TurnID)
}
