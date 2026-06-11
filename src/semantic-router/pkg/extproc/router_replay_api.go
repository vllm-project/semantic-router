package extproc

import (
	"encoding/json"
	"fmt"
	"net/url"
	"sort"
	"strconv"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/protobuf/proto"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

const (
	routerReplayAggregatePath          = routerReplayAPIBasePath + "/aggregate"
	routerReplayDefaultListLimit       = 20
	routerReplayMaxListLimit           = 100
	routerReplayMaxImmediateResponseMB = 4 * 1024 * 1024
)

type routerReplayFilters struct {
	search      string
	decision    string
	model       string
	cacheStatus string
	sessionID   string
}

type routerReplayListQuery struct {
	filters routerReplayFilters
	limit   int
	offset  int
}

type routerReplayListResponse struct {
	Object     string                       `json:"object"`
	Count      int                          `json:"count"`
	Total      int                          `json:"total"`
	Limit      int                          `json:"limit"`
	Offset     int                          `json:"offset"`
	HasMore    bool                         `json:"has_more"`
	NextOffset *int                         `json:"next_offset,omitempty"`
	Data       []routerreplay.RoutingRecord `json:"data"`
}

// handleRouterReplayAPI serves read-only endpoints for router replay records.
func (r *OpenAIRouter) handleRouterReplayAPI(method string, path string) *ext_proc.ProcessingResponse {
	if !r.hasRouterReplayRecorders() {
		return nil
	}

	normalizedPath, rawQuery := splitRouterReplayRequestPath(path)
	if !isRouterReplayPath(normalizedPath) {
		return nil
	}

	switch {
	case isRouterReplayListPath(normalizedPath):
		return r.handleRouterReplayListAPI(method, rawQuery)
	case normalizedPath == routerReplayAggregatePath:
		return r.handleRouterReplayAggregateAPI(method, rawQuery)
	case normalizedPath == routerReplayTrajectoryPath:
		return r.handleRouterReplayTrajectoryAPI(method, rawQuery)
	case strings.HasPrefix(normalizedPath, routerReplayAPIBasePath+"/"):
		replayID := strings.TrimPrefix(normalizedPath, routerReplayAPIBasePath+"/")
		return r.handleRouterReplayRecordAPI(method, replayID)
	default:
		return nil
	}
}

func isRouterReplayPath(path string) bool {
	return isRouterReplayListPath(path) ||
		path == routerReplayAggregatePath ||
		strings.HasPrefix(path, routerReplayAPIBasePath+"/")
}

func (r *OpenAIRouter) hasRouterReplayRecorders() bool {
	return len(r.ReplayRecorders) > 0 || r.ReplayRecorder != nil
}

func splitRouterReplayRequestPath(path string) (string, string) {
	normalizedPath, rawQuery, hasQuery := strings.Cut(path, "?")
	if !hasQuery {
		return path, ""
	}
	return normalizedPath, rawQuery
}

func isRouterReplayListPath(path string) bool {
	return path == routerReplayAPIBasePath || path == routerReplayAPIBasePath+"/"
}

func (r *OpenAIRouter) handleRouterReplayListAPI(method string, rawQuery string) *ext_proc.ProcessingResponse {
	if method != "GET" {
		return r.createErrorResponse(405, "method not allowed")
	}

	query, err := parseRouterReplayListQuery(rawQuery)
	if err != nil {
		return r.createErrorResponse(400, err.Error())
	}

	records := filterRouterReplayRecords(r.collectRouterReplayRecords(), query.filters)
	payload := buildRouterReplayListPayload(records, query)
	boundRouterReplayListPayloadSize(&payload)
	return r.createRouterReplayJSONResponse(200, payload)
}

// boundRouterReplayListPayloadSize guarantees the list page serializes under
// the ext-proc message-size cap regardless of per-record capture config.
//
// The list response embeds whole records, and the heavy text fields
// (tool-trace step Arguments/Output, request/response bodies, prompt,
// tool_definitions) are individually capped only by the replay capture policy
// — which an operator may raise or disable (max_tool_trace_bytes: 0). A single
// large request, or a long agent tool-trace under the default config, can
// therefore push a normal page past the cap. Hard-failing the whole history
// list with a 413 in that case is the wrong behavior for a read endpoint, so
// this trims the heaviest fields on the page records (which are clones from the
// store, never the stored originals) until the page fits, setting the existing
// *_truncated flags so callers can see fidelity was reduced for transport. Full
// fidelity remains available via the single-record GET /v1/router_replay/{id}.
//
// Trimming order is heaviest-first so the cheapest loss of fidelity is taken
// first: per-step tool-trace text, then response/request bodies, then the
// structured prompt and tool definitions.
func boundRouterReplayListPayloadSize(payload *routerReplayListResponse) {
	if listPayloadSizeWithinLimit(*payload) {
		return
	}
	// Budget per heavy field per record, shrunk on each pass until the page
	// fits or we reach a hard floor. A floor keeps each record minimally useful
	// (id, decision, metadata, a snippet of each text field) rather than
	// blanking them entirely.
	for _, budget := range []int{4096, 1024, 256, 0} {
		for i := range payload.Data {
			trimRoutingRecordForList(&payload.Data[i], budget)
		}
		if listPayloadSizeWithinLimit(*payload) {
			return
		}
	}
	// Even fully trimmed the page is too large (pathological: a huge page of
	// records whose non-text metadata alone exceeds the cap). Fall back to
	// halving the page until it fits, recording the reduced count.
	for len(payload.Data) > 1 && !listPayloadSizeWithinLimit(*payload) {
		keep := len(payload.Data) / 2
		payload.Data = payload.Data[:keep]
		payload.Count = keep
		payload.HasMore = true
		nextOffset := payload.Offset + keep
		payload.NextOffset = &nextOffset
	}
}

// listPayloadSizeWithinLimit reports whether the payload serializes (as the
// ext-proc immediate response) within the size cap. It measures the same
// proto envelope createRouterReplayJSONResponse will emit.
func listPayloadSizeWithinLimit(payload routerReplayListResponse) bool {
	body, err := json.Marshal(payload)
	if err != nil {
		return false
	}
	// The immediate-response envelope adds fixed framing around the JSON body;
	// measuring the JSON length against the cap is a safe lower bound, and the
	// final proto.Size check in createRouterReplayJSONResponse remains the
	// authoritative gate.
	return len(body) <= routerReplayMaxImmediateResponseMB
}

// trimRoutingRecordForList shrinks the heavy text fields of a single list
// record to at most budget bytes each, setting the matching *_truncated flag
// when it cuts. budget == 0 trims the fields to empty (flag still set). The
// record is a per-request clone, so this never mutates the stored original.
func trimRoutingRecordForList(record *routerreplay.RoutingRecord, budget int) {
	if record.ToolTrace != nil {
		for i := range record.ToolTrace.Steps {
			trimToolTraceStepForList(&record.ToolTrace.Steps[i], budget)
		}
	}
	truncateListField(&record.ResponseBody, &record.ResponseBodyTruncated, budget)
	truncateListField(&record.RequestBody, &record.RequestBodyTruncated, budget)
	truncateListField(&record.Prompt, &record.PromptTruncated, budget)
	truncateListField(&record.ToolDefinitions, &record.ToolDefinitionsTruncated, budget)
}

// trimToolTraceStepForList trims every free-text field a tool-trace step can
// carry (each can hold a full message or tool payload) to budget bytes,
// flagging Truncated if any field is cut.
func trimToolTraceStepForList(step *routerreplay.ToolTraceStep, budget int) {
	truncateListField(&step.Arguments, &step.Truncated, budget)
	truncateListField(&step.RawArguments, &step.Truncated, budget)
	truncateListField(&step.Output, &step.Truncated, budget)
	truncateListField(&step.RawOutput, &step.Truncated, budget)
	truncateListField(&step.Text, &step.Truncated, budget)
}

// truncateListField cuts *field to at most budget bytes, setting *flag true
// when it removes anything. A no-op when the field already fits.
func truncateListField(field *string, flag *bool, budget int) {
	if len(*field) > budget {
		*field = (*field)[:budget]
		*flag = true
	}
}

func (r *OpenAIRouter) collectRouterReplayRecords() []routerreplay.RoutingRecord {
	if r.ReplayStoreShared && r.ReplayRecorder != nil {
		return sortRouterReplayRecords(r.ReplayRecorder.ListAllRecords())
	}

	var records []routerreplay.RoutingRecord
	for _, recorder := range r.ReplayRecorders {
		records = append(records, recorder.ListAllRecords()...)
	}
	if len(records) == 0 && r.ReplayRecorder != nil {
		records = r.ReplayRecorder.ListAllRecords()
	}

	return sortRouterReplayRecords(records)
}

func sortRouterReplayRecords(records []routerreplay.RoutingRecord) []routerreplay.RoutingRecord {
	sort.SliceStable(records, func(i, j int) bool {
		if records[i].Timestamp.Equal(records[j].Timestamp) {
			return records[i].ID > records[j].ID
		}
		return records[i].Timestamp.After(records[j].Timestamp)
	})
	return records
}

func (r *OpenAIRouter) handleRouterReplayRecordAPI(method string, replayID string) *ext_proc.ProcessingResponse {
	if method != "GET" {
		return r.createErrorResponse(405, "method not allowed")
	}
	if replayID == "" {
		return r.createErrorResponse(400, "replay id is required")
	}

	record, ok := r.findRouterReplayRecord(replayID)
	if !ok {
		return r.createErrorResponse(404, "replay record not found")
	}
	return r.createRouterReplayJSONResponse(200, record)
}

func (r *OpenAIRouter) findRouterReplayRecord(replayID string) (routerreplay.RoutingRecord, bool) {
	if r.ReplayStoreShared && r.ReplayRecorder != nil {
		return r.ReplayRecorder.GetRecord(replayID)
	}

	for _, recorder := range r.ReplayRecorders {
		if record, ok := recorder.GetRecord(replayID); ok {
			return record, true
		}
	}
	if r.ReplayRecorder != nil {
		return r.ReplayRecorder.GetRecord(replayID)
	}
	return routerreplay.RoutingRecord{}, false
}

func parseRouterReplayListQuery(rawQuery string) (routerReplayListQuery, error) {
	query := routerReplayListQuery{
		limit: routerReplayDefaultListLimit,
	}
	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		return query, fmt.Errorf("invalid router replay query")
	}
	filters, err := parseRouterReplayFilters(values)
	if err != nil {
		return query, err
	}
	query.filters = filters

	if values.Has("limit") {
		limit, err := parseRouterReplayQueryInt(values, "limit", true)
		if err != nil {
			return query, err
		}
		if limit > routerReplayMaxListLimit {
			limit = routerReplayMaxListLimit
		}
		query.limit = limit
	}

	if values.Has("offset") {
		offset, err := parseRouterReplayQueryInt(values, "offset", false)
		if err != nil {
			return query, err
		}
		query.offset = offset
	}

	return query, nil
}

func parseRouterReplayFilters(values url.Values) (routerReplayFilters, error) {
	filters := routerReplayFilters{
		search:    strings.TrimSpace(values.Get("search")),
		decision:  strings.TrimSpace(values.Get("decision")),
		model:     strings.TrimSpace(values.Get("model")),
		sessionID: strings.TrimSpace(values.Get("session_id")),
	}

	cacheStatus := strings.TrimSpace(values.Get("cache_status"))
	switch cacheStatus {
	case "", "all":
		filters.cacheStatus = ""
	case "cached", "streamed":
		filters.cacheStatus = cacheStatus
	default:
		return filters, fmt.Errorf("cache_status must be one of all, cached, or streamed")
	}

	return filters, nil
}

func parseRouterReplayQueryInt(values url.Values, key string, positiveOnly bool) (int, error) {
	value, err := strconv.Atoi(values.Get(key))
	if err != nil {
		return 0, fmt.Errorf("%s must be an integer", key)
	}
	if positiveOnly && value <= 0 {
		return 0, fmt.Errorf("%s must be greater than zero", key)
	}
	if !positiveOnly && value < 0 {
		return 0, fmt.Errorf("%s must be zero or greater", key)
	}
	return value, nil
}

func filterRouterReplayRecords(
	records []routerreplay.RoutingRecord,
	filters routerReplayFilters,
) []routerreplay.RoutingRecord {
	if filters == (routerReplayFilters{}) {
		return records
	}

	search := strings.ToLower(filters.search)
	filtered := make([]routerreplay.RoutingRecord, 0, len(records))
	for _, record := range records {
		if !doesRouterReplayRecordMatchFilters(record, filters, search) {
			continue
		}
		filtered = append(filtered, record)
	}
	return filtered
}

func doesRouterReplayRecordMatchFilters(
	record routerreplay.RoutingRecord,
	filters routerReplayFilters,
	search string,
) bool {
	if filters.cacheStatus != "" && !hasMatchingCacheStatus(record, filters.cacheStatus) {
		return false
	}
	if filters.decision != "" && record.Decision != filters.decision {
		return false
	}
	if filters.model != "" && !doesModelMatch(record, filters.model) {
		return false
	}
	if filters.sessionID != "" && record.SessionID != filters.sessionID {
		return false
	}
	if search != "" && !strings.Contains(strings.ToLower(record.RequestID), search) {
		return false
	}
	return true
}

func hasMatchingCacheStatus(record routerreplay.RoutingRecord, cacheStatus string) bool {
	switch cacheStatus {
	case "cached":
		return record.FromCache
	case "streamed":
		return record.Streaming
	default:
		return true
	}
}

func doesModelMatch(record routerreplay.RoutingRecord, model string) bool {
	return record.SelectedModel == model || record.OriginalModel == model
}

func buildRouterReplayListPayload(
	records []routerreplay.RoutingRecord,
	query routerReplayListQuery,
) routerReplayListResponse {
	total := len(records)
	offset := query.offset
	if offset > total {
		offset = total
	}

	end := offset + query.limit
	if end > total {
		end = total
	}

	page := records[offset:end]
	payload := routerReplayListResponse{
		Object:  "router_replay.list",
		Count:   len(page),
		Total:   total,
		Limit:   query.limit,
		Offset:  offset,
		HasMore: end < total,
		Data:    page,
	}
	if payload.HasMore {
		nextOffset := end
		payload.NextOffset = &nextOffset
	}
	return payload
}

func (r *OpenAIRouter) createRouterReplayJSONResponse(
	statusCode int,
	data interface{},
) *ext_proc.ProcessingResponse {
	response := r.createJSONResponse(statusCode, data)
	if response == nil {
		return nil
	}
	if proto.Size(response) <= routerReplayMaxImmediateResponseMB {
		return response
	}
	return r.createErrorResponse(
		413,
		"router replay response exceeds the ext-proc message size limit; retry with a smaller page or reduce captured replay body bytes",
	)
}
