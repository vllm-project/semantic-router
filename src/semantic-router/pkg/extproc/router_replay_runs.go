package extproc

import (
	"context"
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

// replayRunList is every run the filters select, newest first, and for each
// run the record whose prompt is its task.
type replayRunList struct {
	runs    []routerreplay.RunSummary
	prompts map[string]string
}

// handleRouterReplayRunsAPI serves GET /api/v1/observability/replays/runs:
// one summary per agent run, a session, over the filtered records. Every page
// of one filter comes from one cached scan.
func (r *OpenAIRouter) handleRouterReplayRunsAPI(ctx context.Context, method string, rawQuery string) *ext_proc.ProcessingResponse {
	if method != "GET" {
		return r.createErrorResponse(405, "method not allowed")
	}
	query, err := parseRouterReplayListQuery(rawQuery)
	if err != nil {
		return r.createErrorResponse(400, err.Error())
	}
	list, err := r.replayRuns.get(ctx, query.filters.cacheKey(), func(ctx context.Context) (replayRunList, error) {
		return r.queryRouterReplayRuns(ctx, query.filters)
	})
	if err != nil {
		return r.createErrorResponse(500, "router replay storage query failed")
	}
	return r.createJSONResponse(200, r.pageRouterReplayRuns(list, query))
}

// handleRouterReplayRunAPI serves GET /api/v1/observability/replays/run?session_id={id}:
// one agent run as a tree of its agents and their calls, across recipes.
func (r *OpenAIRouter) handleRouterReplayRunAPI(ctx context.Context, method string, rawQuery string) *ext_proc.ProcessingResponse {
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
	records, err := r.queryRouterReplaySession(ctx, sessionID, nil)
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

func (r *OpenAIRouter) queryRouterReplayRuns(ctx context.Context, filters routerReplayFilters) (replayRunList, error) {
	search := strings.ToLower(filters.search)
	builders := make(map[string]*replayRunBuilder)
	for _, reader := range r.routerReplayReaders() {
		if err := reader.ScanMetadata(ctx, filters.storageFilters(), func(record routerreplay.RoutingRecord) error {
			if record.SessionID == "" || !doesRouterReplayRecordMatchFilters(record, filters, search) {
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
			return replayRunList{}, err
		}
	}
	list := replayRunList{
		runs:    make([]routerreplay.RunSummary, 0, len(builders)),
		prompts: make(map[string]string, len(builders)),
	}
	for sessionID, builder := range builders {
		if builder.titlesOnly() {
			continue
		}
		summary, _ := builder.build()
		list.runs = append(list.runs, summary)
		list.prompts[sessionID] = builder.firstMain.recordID
	}
	sort.Slice(list.runs, func(i, j int) bool {
		if list.runs[i].EndedAt.Equal(list.runs[j].EndedAt) {
			return list.runs[i].SessionID < list.runs[j].SessionID
		}
		return list.runs[i].EndedAt.After(list.runs[j].EndedAt)
	})
	return list, nil
}

// pageRouterReplayRuns copies one page out of the shared list, then reads the
// first main call of each run whose prompt the metadata scan left out.
func (r *OpenAIRouter) pageRouterReplayRuns(list replayRunList, query routerReplayListQuery) routerreplay.RunListResponse {
	payload := pageReplayRuns(list.runs, query)
	payload.Data = append(make([]routerreplay.RunSummary, 0, len(payload.Data)), payload.Data...)
	for index, run := range payload.Data {
		if run.Prompt == "" && list.prompts[run.SessionID] != "" {
			if record, ok := r.findRouterReplayRecord(list.prompts[run.SessionID]); ok {
				payload.Data[index].Prompt = looper.WithoutSystemReminders(record.Prompt)
			}
		}
	}
	return payload
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
