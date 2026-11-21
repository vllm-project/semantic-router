package extproc

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// Tool events go in a file beside the records where the file store keeps
// them, in memory where another store does, and nowhere while replay records
// nothing; a Router without them refuses a report rather than drop it.
func TestReplayToolEventsLiveBesideTheRecords(t *testing.T) {
	cfg := &config.RouterConfig{}
	require.Nil(t, openReplayToolEvents(cfg, nil, nil))
	require.ErrorIs(t, (&OpenAIRouter{}).RecordToolEvents(nil), errReplayToolEventsOff)

	dir := t.TempDir()
	file, err := store.OpenFileStore(filepath.Join(dir, "router-replay.jsonl"), 10, 0)
	require.NoError(t, err)
	t.Cleanup(func() { _ = file.Close() })
	onFile := openReplayToolEvents(cfg, routerreplay.NewRecorder(file), nil)
	router := &OpenAIRouter{toolEvents: onFile}
	ended := time.Now()
	require.NoError(t, router.RecordToolEvents([]store.ToolEvent{{
		SessionID: "s", ToolCallID: "toolu_1", Tool: "Bash", Status: store.ToolEventSucceeded, EndedAt: &ended, ReceivedAt: ended,
	}}))
	require.NoError(t, onFile.Close())
	kept, err := os.ReadFile(filepath.Join(dir, "router-tool-events.jsonl"))
	require.NoError(t, err)
	require.Contains(t, string(kept), `"tool_call_id":"toolu_1"`)

	inMemory := openReplayToolEvents(cfg, nil, map[string]*routerreplay.Recorder{"d": routerreplay.NewRecorder(store.NewMemoryStore(10, 0))})
	t.Cleanup(func() { _ = inMemory.Close() })
	require.NotNil(t, inMemory)
	require.NotSame(t, onFile, inMemory)
}

// A run's tool calls take what the harness plugins reported of them in the
// run's session, by tool call id: calls of one response that ran side by
// side each get their own times, a start after the approval ask gives the
// wait apart, a start before it (Codex) keeps the wait in, a refused call
// reads as denied, and a call no plugin reported keeps the gateway's window.
// Another session's report of the same id is not this run's.
func TestReplayRunJoinsPluginToolTimes(t *testing.T) {
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	router := &OpenAIRouter{ReplayStoreShared: true, ReplayRecorder: recorder, toolEvents: store.NewToolEventStore(10, time.Hour)}
	start := time.Date(2026, 11, 17, 1, 0, 0, 0, time.UTC)
	asks := func(ids ...string) *routerreplay.ToolTrace {
		trace := &routerreplay.ToolTrace{}
		for _, id := range ids {
			trace.Steps = append(trace.Steps, routerreplay.ToolTraceStep{Source: "response", Type: replayToolStepAssistantToolCall, ToolName: "Bash", ToolCallID: id})
		}
		return trace
	}
	carries := &routerreplay.ToolTrace{}
	for _, id := range []string{"toolu_a", "toolu_b", "toolu_c", "toolu_d", "call_e"} {
		carries.Steps = append(carries.Steps, routerreplay.ToolTraceStep{Source: "request", Type: replayToolStepAssistantToolCall, ToolName: "Bash", ToolCallID: id})
	}
	for _, id := range []string{"toolu_a", "toolu_b", "toolu_c", "toolu_d", "call_e"} {
		carries.Steps = append(carries.Steps, routerreplay.ToolTraceStep{Source: "request", Type: replayToolStepClientToolResult, ToolCallID: id, Status: "failed"})
	}
	for _, record := range []routerreplay.RoutingRecord{
		{
			ID: "ask", SessionID: "run-1", Timestamp: start, DurationMS: 1000, SelectedModel: "m", LifecycleState: routerreplay.LifecycleCompleted, ResponseStatus: 200,
			ToolTrace: asks("toolu_a", "toolu_b", "toolu_c", "toolu_d", "call_e"),
		},
		{
			ID: "next", SessionID: "run-1", Timestamp: start.Add(10 * time.Second), DurationMS: 1000, SelectedModel: "m", LifecycleState: routerreplay.LifecycleCompleted, ResponseStatus: 200,
			ToolTrace: carries,
		},
	} {
		_, err := recorder.AddRecord(record)
		require.NoError(t, err)
	}
	at := func(ms int) *time.Time { value := start.Add(time.Duration(ms) * time.Millisecond); return &value }
	report := func(session, id, status string, started, ended, asked *time.Time) store.ToolEvent {
		return store.ToolEvent{SessionID: session, ToolCallID: id, Tool: "Bash", Status: status, StartedAt: started, EndedAt: ended, ApprovalRequestedAt: asked, ReceivedAt: time.Now()}
	}
	require.NoError(t, router.RecordToolEvents([]store.ToolEvent{
		report("run-1", "toolu_a", store.ToolEventFailed, at(6000), at(6400), at(1000)),
		report("run-1", "toolu_b", store.ToolEventFailed, at(1100), at(1300), nil),
		report("run-1", "toolu_d", store.ToolEventDenied, nil, at(4000), at(1000)),
		report("run-1", "call_e", store.ToolEventSucceeded, at(1050), at(9000), at(2000)),
		report("run-2", "toolu_c", store.ToolEventSucceeded, at(1100), at(1200), nil),
	}))

	var run routerreplay.RunResponse
	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/run?session_id=run-1", &run))
	tools := map[string]routerreplay.RunToolCall{}
	for _, tool := range run.Calls[0].ToolCalls {
		tools[tool.ID] = tool
	}
	waited := tools["toolu_a"]
	require.Equal(t, "plugin", waited.TimingSource)
	require.Equal(t, int64(5000), *waited.ApprovalMS)
	require.Equal(t, int64(400), waited.EndedAt.Sub(*waited.StartedAt).Milliseconds())
	side := tools["toolu_b"]
	require.Equal(t, int64(200), side.EndedAt.Sub(*side.StartedAt).Milliseconds())
	require.Nil(t, side.ApprovalMS)
	require.Equal(t, "denied", tools["toolu_d"].Status)
	require.Equal(t, "failed", side.Status)
	codex := tools["call_e"]
	require.Nil(t, codex.ApprovalMS)
	require.Equal(t, start.Add(2*time.Second), *codex.ApprovalRequestedAt)
	gateway := tools["toolu_c"]
	require.Equal(t, "gateway", gateway.TimingSource)
	require.Nil(t, gateway.StartedAt)
	require.Equal(t, int64(9000), *gateway.ClientMS)
}

// A Codex result carries no error flag, so the gateway reads every one as
// succeeded: the status the plugin reported names how each call ended, on
// its chip, in the run's failure count and marker and in Runs. A call a
// person refused is denied, not a failed tool, whatever its result says.
func TestReplayRunCountsFailuresAsThePluginReported(t *testing.T) {
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	router := &OpenAIRouter{ReplayStoreShared: true, ReplayRecorder: recorder, toolEvents: store.NewToolEventStore(10, time.Hour)}
	start := time.Date(2026, 11, 17, 13, 14, 0, 0, time.UTC)
	step := func(source, kind, id, status string) routerreplay.ToolTraceStep {
		return routerreplay.ToolTraceStep{Source: source, Type: kind, ToolName: "Bash", ToolCallID: id, Status: status}
	}
	ids := []string{"call_tests", "call_curl", "call_echo", "call_rm"}
	asked, carried := &routerreplay.ToolTrace{}, &routerreplay.ToolTrace{}
	for _, id := range ids {
		asked.Steps = append(asked.Steps, step("response", replayToolStepAssistantToolCall, id, ""))
		carried.Steps = append(carried.Steps, step("request", replayToolStepAssistantToolCall, id, ""))
	}
	for _, id := range ids {
		status := "succeeded"
		if id == "call_rm" {
			status = "failed"
		}
		carried.Steps = append(carried.Steps, step("request", replayToolStepClientToolResult, id, status))
	}
	for index, trace := range []*routerreplay.ToolTrace{asked, carried} {
		_, err := recorder.AddRecord(routerreplay.RoutingRecord{
			ID: fmt.Sprintf("call-%d", index), SessionID: "cx-failed", Timestamp: start.Add(time.Duration(index) * 10 * time.Second),
			SelectedModel: "qwen-local", Agent: &store.AgentIdentity{Harness: store.HarnessCodex},
			LifecycleState: routerreplay.LifecycleCompleted, ResponseStatus: 200, ToolTrace: trace,
		})
		require.NoError(t, err)
	}
	ended := start.Add(5 * time.Second)
	var reports []store.ToolEvent
	for id, status := range map[string]string{
		"call_tests": store.ToolEventFailed, "call_curl": store.ToolEventFailed, "call_echo": store.ToolEventSucceeded, "call_rm": store.ToolEventDenied,
	} {
		reports = append(reports, store.ToolEvent{SessionID: "cx-failed", ToolCallID: id, Tool: "Bash", Status: status, EndedAt: &ended, ReceivedAt: time.Now()})
	}
	require.NoError(t, router.RecordToolEvents(reports))

	var run routerreplay.RunResponse
	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/run?session_id=cx-failed", &run))
	chips := map[string]string{}
	for _, tool := range run.Calls[0].ToolCalls {
		chips[tool.ID] = tool.Status
	}
	require.Equal(t, map[string]string{"call_tests": "failed", "call_curl": "failed", "call_echo": "succeeded", "call_rm": "denied"}, chips)
	require.Equal(t, 4, run.Run.Outcome.ToolResults)
	require.Equal(t, 2, run.Run.Outcome.ToolFailures)
	require.Equal(t, 2, run.Calls[1].ToolFailures)
	require.Equal(t, map[string]int{routerreplay.RunMarkerToolLoop: 1, routerreplay.RunMarkerToolFailure: 1}, run.Run.Events)
	var runs routerreplay.RunListResponse
	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/runs", &runs))
	require.Len(t, runs.Data, 1)
	require.Equal(t, 2, runs.Data[0].Outcome.ToolFailures)
	require.Equal(t, map[string]int{routerreplay.RunMarkerToolLoop: 1, routerreplay.RunMarkerToolFailure: 1}, runs.Data[0].Events)
}
