package extproc

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/redaction"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func replayRunStep(source, kind, toolCallID, status string) routerreplay.ToolTraceStep {
	return routerreplay.ToolTraceStep{Source: source, Type: kind, ToolCallID: toolCallID, Status: status}
}

// newReplayRunRouter holds one agent run: a main agent that calls a tool, a
// subagent it spawns, a subagent that one spawns, a Looper hop the first call
// made, and the main agent's answer after a failed tool on another model.
// Another session's call is not part of it.
func newReplayRunRouter(t *testing.T) (*OpenAIRouter, map[string]string) {
	t.Helper()
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(20, 0))
	recorder.SetCapturePolicy(true, true, 0)
	router := &OpenAIRouter{ReplayStoreShared: true, ReplayRecorder: recorder}
	start := time.Date(2026, 11, 16, 9, 0, 0, 0, time.UTC)
	usd := "USD"
	cost := func(value float64) *float64 { return &value }
	tokens := func(value int) *int { return &value }
	ids := make(map[string]string)
	for _, record := range []routerreplay.RoutingRecord{
		{
			ID: "main-1", RequestID: "req-1", SessionID: "run-1", Timestamp: start, DurationMS: 1000,
			OriginalModel: "vllm-sr/auto", SelectedModel: "small", Decision: "code",
			Prompt: "<system-reminder>\nworkspace notes\n</system-reminder>\nfix the build",
			Agent:  &store.AgentIdentity{Harness: store.HarnessCodex}, LifecycleState: routerreplay.LifecycleCompleted,
			ResponseStatus: 200, PromptTokens: tokens(100), CompletionTokens: tokens(10), ActualCost: cost(0.002), Currency: &usd,
			Signals:   routerreplay.Signal{Keyword: []string{"build"}},
			ToolTrace: &routerreplay.ToolTrace{Steps: []routerreplay.ToolTraceStep{replayRunStep("response", replayToolStepAssistantToolCall, "call-1", "")}},
		},
		{
			ID: "hop-1", RequestID: "hop-req", ParentRequestID: "req-1", SessionID: "run-1", Timestamp: start.Add(-500 * time.Millisecond), DurationMS: 300,
			SelectedModel: "small", LifecycleState: routerreplay.LifecycleCompleted, ResponseStatus: 200,
			CostAccounting: &store.CostAccounting{UnpricedReason: store.UnpricedCountedInParent},
		},
		{
			ID: "sub-1", RequestID: "req-2", SessionID: "run-1", Timestamp: start.Add(2 * time.Second), DurationMS: 500,
			OriginalModel: "vllm-sr/auto", SelectedModel: "small", LifecycleState: routerreplay.LifecycleCompleted, ResponseStatus: 200,
			Agent:      &store.AgentIdentity{Harness: store.HarnessCodex, ID: "agenthash-a", Kind: "collab_spawn"},
			ActualCost: cost(0.001), Currency: &usd,
		},
		{
			ID: "sub-2", RequestID: "req-3", SessionID: "run-1", Timestamp: start.Add(3 * time.Second), DurationMS: 500,
			OriginalModel: "vllm-sr/auto", SelectedModel: "small", LifecycleState: routerreplay.LifecycleFailed, ResponseStatus: 502,
			Agent: &store.AgentIdentity{Harness: store.HarnessCodex, ID: "agenthash-b", ParentID: "agenthash-a", Kind: "collab_spawn"},
		},
		{
			ID: "main-2", RequestID: "req-4", SessionID: "run-1", Timestamp: start.Add(5 * time.Second), DurationMS: 2000,
			OriginalModel: "vllm-sr/auto", SelectedModel: "large", LifecycleState: routerreplay.LifecycleCompleted, ResponseStatus: 200,
			Agent:            &store.AgentIdentity{Harness: store.HarnessCodex},
			RouteDiagnostics: &routerreplay.RouteDiagnostics{SessionAction: "switch", SessionReason: "complexity rose"},
			ActualCost:       cost(0.01), Currency: &usd,
			ToolTrace: &routerreplay.ToolTrace{Steps: []routerreplay.ToolTraceStep{
				replayRunStep("request", replayToolStepAssistantToolCall, "call-1", ""),
				replayRunStep("request", replayToolStepClientToolResult, "call-1", "failed"),
				replayRunStep("response", replayToolStepAssistantFinalResponse, "", ""),
			}},
			Outcomes: []routerreplay.Outcome{
				{Source: primaryResponseOutcomeSource, Target: "model", Verdict: "completed"},
				{Source: "agent", Target: "model", Verdict: "good_fit", Reason: "tests pass"},
			},
		},
		{ID: "other", SessionID: "run-2", Timestamp: start.Add(time.Second), SelectedModel: "small", LifecycleState: routerreplay.LifecycleCompleted},
	} {
		id, err := router.ReplayRecorder.AddRecord(record)
		require.NoError(t, err)
		ids[record.ID] = id
	}
	return router, ids
}

// The run read returns one session as a tree: its agents, each naming the
// agent that spawned it, and its calls in time order with the Looper hop
// under the call that made it, the model switch, the failures, the final
// answer and the outcome the agent reported.
func TestReplayRunReturnsTheAgentTree(t *testing.T) {
	router, ids := newReplayRunRouter(t)
	var run routerreplay.RunResponse
	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/run?session_id=run-1", &run))

	summary := run.Run
	require.Equal(t, "run-1", summary.SessionID)
	require.Equal(t, store.HarnessCodex, summary.Harness)
	require.Equal(t, "fix the build", summary.Prompt)
	require.Equal(t, 4, summary.Calls)
	// The Looper hop started half a second before the record of its call.
	require.Equal(t, int64(7500), summary.DurationMS)
	agents := make([][4]string, 0, len(summary.Agents))
	for _, agent := range summary.Agents {
		agents = append(agents, [4]string{agent.Label, agent.Parent, agent.Kind, agent.SpawnedAfterRecordID})
	}
	require.Equal(t, [][4]string{
		{"main", "", "", ""},
		{"subagent 1", "main", "collab_spawn", ids["main-1"]},
		{"subagent 2", "subagent 1", "collab_spawn", ids["sub-1"]},
	}, agents)

	lanes := make([]string, 0, len(run.Calls))
	for _, call := range run.Calls {
		lanes = append(lanes, call.Agent+"/"+call.Model)
	}
	require.Equal(t, []string{"main/small", "subagent 1/small", "subagent 2/small", "main/large"}, lanes)
	require.Len(t, run.Calls[0].Hops, 1)
	require.Equal(t, ids["hop-1"], run.Calls[0].Hops[0].RecordID)
	require.Equal(t, []string{"keyword:build"}, run.Calls[0].Signals)
	require.Equal(t, []string{routerreplay.RunMarkerFailure}, run.Calls[2].Markers)
	last := run.Calls[3]
	require.Equal(t, "small", last.PreviousModel)
	require.Equal(t, []string{routerreplay.RunMarkerSwitch, routerreplay.RunMarkerToolFailure}, last.Markers)
	require.Equal(t, "complexity rose", last.Reason)
	require.True(t, last.FinalAnswer)
	require.Len(t, last.Outcomes, 1)

	require.Equal(t, routerreplay.RunOutcome{
		Status: routerreplay.RunStatusAnswered, FinalAnswerRecordID: ids["main-2"], ToolResults: 1, ToolFailures: 1,
		Reported: []routerreplay.RunOutcomeAt{{RecordID: ids["main-2"], Outcome: last.Outcomes[0]}},
	}, summary.Outcome)
	require.Equal(t, map[string]int{"failure": 1, "switch": 1, "tool_failure": 1}, summary.Events)
	require.Len(t, summary.Cost.ByCurrency, 1)
	require.InDelta(t, 0.013, summary.Cost.ByCurrency[0].Total, 1e-9)
	require.Equal(t, []routerreplay.AggregateValue{{Name: "small", Value: 4}, {Name: "large", Value: 1}}, summary.Models)

	var missing map[string]interface{}
	require.Equal(t, 404, replayAPIJSON(t, router, "/api/v1/observability/replays/run?session_id=nobody", &missing))
	require.Equal(t, 400, replayAPIJSON(t, router, "/api/v1/observability/replays/run", &missing))
}

// The runs list holds one summary per session, latest activity first, and
// pages them.
func TestReplayRunsListsOneSummaryPerSession(t *testing.T) {
	router, _ := newReplayRunRouter(t)
	var runs routerreplay.RunListResponse
	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/runs", &runs))
	require.Equal(t, 2, runs.Total)
	require.Equal(t, []string{"run-1", "run-2"}, []string{runs.Data[0].SessionID, runs.Data[1].SessionID})
	require.Equal(t, 4, runs.Data[0].Calls)
	require.Equal(t, "fix the build", runs.Data[0].Prompt)
	require.Equal(t, routerreplay.RunStatusAnswered, runs.Data[0].Outcome.Status)

	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/runs?limit=1&offset=1", &runs))
	require.Equal(t, 1, runs.Count)
	require.False(t, runs.HasMore)
	require.Equal(t, "run-2", runs.Data[0].SessionID)
}

// Without replay.detail a run keeps its tree, labels, markers and counts,
// and loses what its records would: the prompt, the agent hashes, the
// protection reason and the reported outcome's text.
func TestReplayRunRedactsWhatItsRecordsWould(t *testing.T) {
	router, ids := newReplayRunRouter(t)
	for _, path := range []string{"/api/v1/observability/replays/run?session_id=run-1", "/api/v1/observability/replays/runs"} {
		response := router.handleRouterReplayAPI(context.Background(), "GET", path).GetImmediateResponse()
		redacted, changed, err := redaction.RedactResponseBody(response.Body)
		require.NoError(t, err)
		require.True(t, changed, path)
		body := string(redacted)
		for _, private := range []string{"fix the build", "agenthash-a", "agenthash-b", "complexity rose", "tests pass"} {
			require.NotContains(t, body, private, path)
		}
		require.Contains(t, body, `"label":"subagent 2"`, path)
		require.Contains(t, body, `"parent":"subagent 1"`, path)
	}
	var record map[string]interface{}
	response := router.handleRouterReplayAPI(context.Background(), "GET", "/api/v1/observability/replays/"+ids["sub-2"]).GetImmediateResponse()
	require.NotNil(t, response)
	redacted, _, err := redaction.RedactResponseBody(response.Body)
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal(redacted, &record))
	if agent, ok := record["agent"].(map[string]interface{}); ok {
		require.Empty(t, agent["id"])
		require.Empty(t, agent["parent_id"])
	}
}

// newNamedRunRouter holds a Claude Code run whose title request came before
// its first turn and that the user renamed later, and the lone thread Codex
// starts for a title.
func newNamedRunRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(20, 0))
	recorder.SetCapturePolicy(true, true, 0)
	router := &OpenAIRouter{ReplayStoreShared: true, ReplayRecorder: recorder}
	start := time.Date(2026, 11, 16, 9, 0, 0, 0, time.UTC)
	claude := &store.AgentIdentity{Harness: store.HarnessClaudeCode}
	for _, record := range []routerreplay.RoutingRecord{
		{
			ID: "title", SessionID: "cc-run", Timestamp: start, RequestKind: store.RequestKindSessionTitle, Agent: claude,
			Prompt: "<session>\nfix average\n</session>", SessionName: "Average helper", SelectedModel: "small",
			LifecycleState: routerreplay.LifecycleCompleted,
		},
		{
			ID: "turn-1", SessionID: "cc-run", Timestamp: start.Add(50 * time.Millisecond), Agent: claude,
			Prompt: "fix average", SelectedModel: "large", LifecycleState: routerreplay.LifecycleCompleted,
		},
		{
			ID: "turn-2", SessionID: "cc-run", Timestamp: start.Add(time.Minute), Agent: claude,
			SessionName: "Renamed by the user", SelectedModel: "large", LifecycleState: routerreplay.LifecycleCompleted,
		},
		{
			ID: "codex-title", SessionID: "title-thread", Timestamp: start.Add(2 * time.Minute), RequestKind: store.RequestKindSessionTitle,
			Agent: &store.AgentIdentity{Harness: store.HarnessCodex}, SelectedModel: "large", LifecycleState: routerreplay.LifecycleCompleted,
		},
	} {
		_, err := router.ReplayRecorder.AddRecord(record)
		require.NoError(t, err)
	}
	return router
}

// A run takes the name its harness last gave the session and keeps the
// first turn's prompt as the task; the title request is not the task, and a
// session that only asked for a title is no agent run.
func TestReplayRunTakesTheSessionName(t *testing.T) {
	router := newNamedRunRouter(t)
	var run routerreplay.RunResponse
	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/run?session_id=cc-run", &run))
	require.Equal(t, "Renamed by the user", run.Run.Name)
	require.Equal(t, "fix average", run.Run.Prompt)
	require.Empty(t, run.Calls[1].Markers, "a side request's model is not the one the agent switched from")

	var runs routerreplay.RunListResponse
	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/runs", &runs))
	require.Equal(t, 1, runs.Total)
	require.Equal(t, "cc-run", runs.Data[0].SessionID)
	require.Equal(t, "Renamed by the user", runs.Data[0].Name)
	require.Equal(t, "fix average", runs.Data[0].Prompt)

	for _, path := range []string{"/api/v1/observability/replays/run?session_id=cc-run", "/api/v1/observability/replays/runs"} {
		response := router.handleRouterReplayAPI(context.Background(), "GET", path).GetImmediateResponse()
		redacted, _, err := redaction.RedactResponseBody(response.Body)
		require.NoError(t, err)
		require.NotContains(t, string(redacted), "Renamed by the user", path)
	}
}
