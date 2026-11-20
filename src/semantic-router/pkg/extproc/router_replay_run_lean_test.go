package extproc

import (
	"context"
	"encoding/json"
	"sort"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// leanRunStore answers a run read as the Postgres store does: one session's
// records oldest first, without their bodies, prompt or tool definitions, and
// each tool step without the text, arguments or output it carried.
type leanRunStore struct {
	*store.MemoryStore
	runReads int
}

func (s *leanRunStore) QueryRun(ctx context.Context, filters store.QueryFilters) ([]store.Record, error) {
	s.runReads++
	all, err := s.List(ctx)
	if err != nil {
		return nil, err
	}
	var records []store.Record
	for _, record := range all {
		if record.SessionID != filters.SessionID {
			continue
		}
		record.RequestBody, record.ResponseBody, record.Prompt, record.ToolDefinitions = "", "", "", ""
		record.ProjectionTrace = nil
		if record.ToolTrace != nil {
			trace := *record.ToolTrace
			trace.Steps = nil
			for _, step := range record.ToolTrace.Steps {
				trace.Steps = append(trace.Steps, store.ToolTraceStep{
					Type: step.Type, Source: step.Source, ToolName: step.ToolName, ToolCallID: step.ToolCallID, Status: step.Status,
				})
			}
			record.ToolTrace = &trace
		}
		records = append(records, record)
	}
	sort.SliceStable(records, func(i, j int) bool { return records[i].Timestamp.Before(records[j].Timestamp) })
	return records, nil
}

// A store that reads a run without its bodies answers the same run as one
// that reads every record whole: the same agents, calls, markers, outcome and
// cost, and the task, which the run takes from its first main call's prompt.
func TestReplayRunReadsTheSameRunWithoutBodies(t *testing.T) {
	full, _ := newReplayRunRouter(t)
	var want routerreplay.RunResponse
	require.Equal(t, 200, replayAPIJSON(t, full, "/api/v1/observability/replays/run?session_id=run-1", &want))
	require.Equal(t, "fix the build", want.Run.Prompt)

	records, err := full.ReplayRecorder.ListRecords()
	require.NoError(t, err)
	lean := &leanRunStore{MemoryStore: store.NewMemoryStore(20, 0)}
	recorder := routerreplay.NewRecorder(lean)
	recorder.SetCapturePolicy(true, true, 0)
	router := &OpenAIRouter{ReplayStoreShared: true, ReplayRecorder: recorder}
	for _, record := range records {
		_, err = recorder.AddRecord(record)
		require.NoError(t, err)
	}

	var got routerreplay.RunResponse
	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/run?session_id=run-1", &got))
	require.Equal(t, 1, lean.runReads, "the run is read through the store's lean query")
	wantJSON, err := json.Marshal(want)
	require.NoError(t, err)
	gotJSON, err := json.Marshal(got)
	require.NoError(t, err)
	require.JSONEq(t, string(wantJSON), string(gotJSON))

	var missing routerreplay.RunResponse
	require.Equal(t, 404, replayAPIJSON(t, router, "/api/v1/observability/replays/run?session_id=nobody", &missing))
}
