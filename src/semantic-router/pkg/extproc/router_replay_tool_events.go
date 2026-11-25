package extproc

import (
	"errors"
	"path/filepath"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// The tool events harness plugins report are kept beside the replay records:
// at most replayToolEventLimit of them (about 1 MB), and none older than the
// records' TTL, or replayToolEventMaxAge where the records have none.
const (
	replayToolEventLimit  = 5000
	replayToolEventMaxAge = 7 * 24 * time.Hour
)

var errReplayToolEventsOff = errors.New("router replay keeps no tool events")

// openReplayToolEvents keeps the tool events in a file beside the records
// where the file store keeps them, else in memory, shared by every
// generation of the process so that a reload keeps them. Backends other than
// the file store keep their records elsewhere and the tool events in memory:
// a restart loses those, and the run view falls back to the gateway's timing.
func openReplayToolEvents(cfg *config.RouterConfig, recorder *routerreplay.Recorder, recorders map[string]*routerreplay.Recorder) *store.ToolEventStore {
	if recorder == nil && len(recorders) == 0 {
		return nil
	}
	ttl := replayToolEventMaxAge
	if cfg.RouterReplay.TTLSeconds > 0 {
		ttl = time.Duration(cfg.RouterReplay.TTLSeconds) * time.Second
	}
	path := ""
	if records, ok := replayFilePath(recorder); ok {
		path = filepath.Join(filepath.Dir(records), "router-tool-events.jsonl")
	} else {
		logging.Warnf("Router replay keeps no records in a file, so tool events stay in memory and are lost on restart.")
	}
	events, err := store.OpenToolEventStore(path, replayToolEventLimit, ttl)
	if err == nil {
		return events
	}
	logging.Warnf("Router replay cannot keep tool events in %s (%v); they stay in memory and are lost on restart.", path, err)
	events, _ = store.OpenToolEventStore("", replayToolEventLimit, ttl)
	return events
}

// useReplayToolEvents opens the generation's tool events beside its replay
// records, released with the generation.
func (components *routerComponents) useReplayToolEvents(cfg *config.RouterConfig) {
	components.toolEvents = openReplayToolEvents(cfg, components.replayRecorder, components.replayRecorders)
	if components.toolEvents != nil {
		components.resources.add(components.toolEvents.Close)
	}
}

// RecordToolEvents keeps tool events a harness plugin reported, for the run
// view to join by session and tool call id.
func (r *OpenAIRouter) RecordToolEvents(events []store.ToolEvent) error {
	if r == nil || r.toolEvents == nil {
		return errReplayToolEventsOff
	}
	return r.toolEvents.Add(events)
}

// replayFilePath is the file a recorder keeps its records in, where it keeps
// them in one: the file store is shared, so the global recorder names it. The
// store may be wrapped, as a prepared generation's is, so this asks for the
// path rather than the type.
func replayFilePath(recorder *routerreplay.Recorder) (string, bool) {
	if recorder == nil {
		return "", false
	}
	file, ok := recorder.Reader().(interface{ Path() string })
	if !ok {
		return "", false
	}
	return file.Path(), true
}

// joinReplayToolReports gives each of a run's tool calls what a harness
// plugin reported of it, by tool call id: its times replace the gateway's
// window, which stays for a call no plugin reported, and timing_source says
// so, and its status replaces the gateway's.
func joinReplayToolReports(calls []routerreplay.RunCall, reports map[string]store.ToolEvent) {
	if len(reports) == 0 {
		return
	}
	for index := range calls {
		for tool := range calls[index].ToolCalls {
			timed := &calls[index].ToolCalls[tool]
			if report, ok := reports[timed.ID]; ok && timed.ID != "" {
				applyReplayToolReport(timed, report)
			}
		}
	}
}

// applyReplayToolReport takes a report's times and its status. The harness
// that ran the call knows how it ended, refused included, where the gateway
// reads only an error flag on the result, which a Responses result has none
// of and a refused call's result sets.
func applyReplayToolReport(timed *routerreplay.RunToolCall, report store.ToolEvent) {
	timed.TimingSource = replayToolTimingPlugin
	timed.StartedAt, timed.EndedAt, timed.ApprovalRequestedAt = report.StartedAt, report.EndedAt, report.ApprovalRequestedAt
	if asked, started := report.ApprovalRequestedAt, report.StartedAt; asked != nil && started != nil && !started.Before(*asked) {
		wait := started.Sub(*asked).Milliseconds()
		timed.ApprovalMS = &wait
	}
	if report.Status != "" {
		timed.Status = report.Status
	}
}
