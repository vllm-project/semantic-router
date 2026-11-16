package extproc

import (
	"net/url"
	"sort"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

const (
	routerReplayRunsPath = routerReplayAPIBasePath + "/runs"
	routerReplayRunPath  = routerReplayAPIBasePath + "/run"
)

// handleRouterReplayRunsAPI serves GET /api/v1/observability/replays/runs:
// one summary per agent run, a session, over the filtered records.
func (r *OpenAIRouter) handleRouterReplayRunsAPI(method string, rawQuery string) *ext_proc.ProcessingResponse {
	if method != "GET" {
		return r.createErrorResponse(405, "method not allowed")
	}
	query, err := parseRouterReplayListQuery(rawQuery)
	if err != nil {
		return r.createErrorResponse(400, err.Error())
	}
	payload, err := r.queryRouterReplayRuns(query)
	if err != nil {
		return r.createErrorResponse(500, "router replay storage query failed")
	}
	return r.createJSONResponse(200, payload)
}

// handleRouterReplayRunAPI serves GET /api/v1/observability/replays/run?session_id={id}:
// one agent run as a tree of its agents and their calls, across recipes.
func (r *OpenAIRouter) handleRouterReplayRunAPI(method string, rawQuery string) *ext_proc.ProcessingResponse {
	if method != "GET" {
		return r.createErrorResponse(405, "method not allowed")
	}
	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		return r.createErrorResponse(400, "invalid query parameters")
	}
	sessionID := strings.TrimSpace(values.Get("session_id"))
	if sessionID == "" {
		return r.createErrorResponse(400, "session_id is required")
	}
	records, err := r.queryRouterReplaySession(sessionID, nil)
	if err != nil {
		return r.createErrorResponse(500, "router replay storage query failed")
	}
	if len(records) == 0 {
		return r.createErrorResponse(404, "run not found")
	}
	builder := newReplayRunBuilder(sessionID)
	for _, record := range records {
		builder.add(record)
	}
	summary, calls := builder.build()
	return r.createJSONResponse(200, routerreplay.RunResponse{Object: "router_replay.run", Run: summary, Calls: calls})
}

func (r *OpenAIRouter) queryRouterReplayRuns(query routerReplayListQuery) (routerreplay.RunListResponse, error) {
	search := strings.ToLower(query.filters.search)
	builders := make(map[string]*replayRunBuilder)
	for _, reader := range r.routerReplayReaders() {
		if err := reader.ScanMetadata(func(record routerreplay.RoutingRecord) error {
			if record.SessionID == "" || !doesRouterReplayRecordMatchFilters(record, query.filters, search) {
				return nil
			}
			builder := builders[record.SessionID]
			if builder == nil {
				builder = newReplayRunBuilder(record.SessionID)
				builders[record.SessionID] = builder
			}
			builder.add(record)
			return nil
		}); err != nil {
			return routerreplay.RunListResponse{}, err
		}
	}
	runs := make([]routerreplay.RunSummary, 0, len(builders))
	prompts := make(map[string]string, len(builders))
	for sessionID, builder := range builders {
		if builder.sidesOnly() {
			continue
		}
		summary, _ := builder.build()
		runs = append(runs, summary)
		prompts[sessionID] = builder.firstMain.recordID
	}
	sort.Slice(runs, func(i, j int) bool {
		if runs[i].EndedAt.Equal(runs[j].EndedAt) {
			return runs[i].SessionID < runs[j].SessionID
		}
		return runs[i].EndedAt.After(runs[j].EndedAt)
	})
	payload := pageReplayRuns(runs, query)
	for index, run := range payload.Data {
		// A metadata scan may leave the prompt out; read the first main call.
		if run.Prompt == "" && prompts[run.SessionID] != "" {
			if record, ok := r.findRouterReplayRecord(prompts[run.SessionID]); ok {
				payload.Data[index].Prompt = looper.WithoutSystemReminders(record.Prompt)
			}
		}
	}
	return payload, nil
}

func pageReplayRuns(runs []routerreplay.RunSummary, query routerReplayListQuery) routerreplay.RunListResponse {
	total := len(runs)
	offset := min(query.offset, total)
	end := min(offset+query.limit, total)
	payload := routerreplay.RunListResponse{
		Object: "router_replay.runs", Count: end - offset, Total: total,
		Limit: query.limit, Offset: offset, HasMore: end < total, Data: runs[offset:end],
	}
	if payload.HasMore {
		next := end
		payload.NextOffset = &next
	}
	return payload
}
