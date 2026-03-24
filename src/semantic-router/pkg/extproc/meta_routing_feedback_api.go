package extproc

import (
	"encoding/json"
	"fmt"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/protobuf/proto"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

const (
	metaRoutingFeedbackAPIBasePath    = "/v1/meta_routing_feedback"
	metaRoutingFeedbackAggregatePath  = metaRoutingFeedbackAPIBasePath + "/aggregate"
	metaRoutingFeedbackDefaultLimit   = 20
	metaRoutingFeedbackMaxLimit       = 100
	metaRoutingFeedbackQueryPreview   = 160
	routerAPIMaxImmediateResponseSize = 4 * 1024 * 1024
)

type metaRoutingFeedbackStoredRecord struct {
	Metadata routerreplay.RoutingRecord
	Record   FeedbackRecord
}

type metaRoutingFeedbackFilters struct {
	search         string
	mode           string
	trigger        string
	rootCause      string
	actionType     string
	signalFamily   string
	overturned     string
	decision       string
	model          string
	responseStatus *int
}

type metaRoutingFeedbackListQuery struct {
	filters metaRoutingFeedbackFilters
	limit   int
	offset  int
}

type metaRoutingFeedbackSummary struct {
	ID                        string    `json:"id"`
	Timestamp                 time.Time `json:"timestamp"`
	Mode                      string    `json:"mode,omitempty"`
	RequestID                 string    `json:"request_id,omitempty"`
	RequestModel              string    `json:"request_model,omitempty"`
	RequestQueryPreview       string    `json:"request_query_preview,omitempty"`
	PassCount                 int       `json:"pass_count,omitempty"`
	Planned                   bool      `json:"planned,omitempty"`
	Executed                  bool      `json:"executed,omitempty"`
	ExecutedPassCount         int       `json:"executed_pass_count,omitempty"`
	TriggerNames              []string  `json:"trigger_names,omitempty"`
	RootCauses                []string  `json:"root_causes,omitempty"`
	ActionTypes               []string  `json:"action_types,omitempty"`
	RefinedSignalFamilies     []string  `json:"refined_signal_families,omitempty"`
	OverturnedDecision        bool      `json:"overturned_decision,omitempty"`
	LatencyDeltaMs            int64     `json:"latency_delta_ms,omitempty"`
	DecisionMarginDelta       float64   `json:"decision_margin_delta,omitempty"`
	ProjectionBoundaryDelta   *float64  `json:"projection_boundary_delta,omitempty"`
	FinalDecisionName         string    `json:"final_decision_name,omitempty"`
	FinalDecisionConfidence   float64   `json:"final_decision_confidence,omitempty"`
	FinalModel                string    `json:"final_model,omitempty"`
	ResponseStatus            int       `json:"response_status,omitempty"`
	Streaming                 bool      `json:"streaming,omitempty"`
	CacheHit                  bool      `json:"cache_hit,omitempty"`
	PIIBlocked                bool      `json:"pii_blocked,omitempty"`
	HallucinationDetected     bool      `json:"hallucination_detected,omitempty"`
	ResponseJailbreakDetected bool      `json:"response_jailbreak_detected,omitempty"`
	RouterReplayID            string    `json:"router_replay_id,omitempty"`
	UserFeedbackSignals       []string  `json:"user_feedback_signals,omitempty"`
}

type metaRoutingFeedbackListResponse struct {
	Object     string                       `json:"object"`
	Count      int                          `json:"count"`
	Total      int                          `json:"total"`
	Limit      int                          `json:"limit"`
	Offset     int                          `json:"offset"`
	HasMore    bool                         `json:"has_more"`
	NextOffset *int                         `json:"next_offset,omitempty"`
	Data       []metaRoutingFeedbackSummary `json:"data"`
}

type metaRoutingFeedbackDetailResponse struct {
	Object    string         `json:"object"`
	ID        string         `json:"id"`
	Timestamp time.Time      `json:"timestamp"`
	Record    FeedbackRecord `json:"record"`
}

// handleMetaRoutingFeedbackAPI serves read-only endpoints for persisted
// meta-routing feedback records.
func (r *OpenAIRouter) handleMetaRoutingFeedbackAPI(method string, path string) *ext_proc.ProcessingResponse {
	normalizedPath, rawQuery := splitRouterReplayRequestPath(path)
	switch {
	case isMetaRoutingFeedbackListPath(normalizedPath):
		return r.handleMetaRoutingFeedbackListAPI(method, rawQuery)
	case normalizedPath == metaRoutingFeedbackAggregatePath:
		return r.handleMetaRoutingFeedbackAggregateAPI(method, rawQuery)
	case strings.HasPrefix(normalizedPath, metaRoutingFeedbackAPIBasePath+"/"):
		recordID := strings.TrimPrefix(normalizedPath, metaRoutingFeedbackAPIBasePath+"/")
		return r.handleMetaRoutingFeedbackRecordAPI(method, recordID)
	default:
		return nil
	}
}

func isMetaRoutingFeedbackListPath(path string) bool {
	return path == metaRoutingFeedbackAPIBasePath || path == metaRoutingFeedbackAPIBasePath+"/"
}

func (r *OpenAIRouter) handleMetaRoutingFeedbackListAPI(method string, rawQuery string) *ext_proc.ProcessingResponse {
	if method != "GET" {
		return r.createErrorResponse(405, "method not allowed")
	}

	query, err := parseMetaRoutingFeedbackListQuery(rawQuery)
	if err != nil {
		return r.createErrorResponse(400, err.Error())
	}

	records := filterMetaRoutingFeedbackRecords(r.collectMetaRoutingFeedbackStoredRecords(), query.filters)
	return r.createMetaRoutingFeedbackJSONResponse(200, buildMetaRoutingFeedbackListPayload(records, query))
}

func (r *OpenAIRouter) handleMetaRoutingFeedbackRecordAPI(method string, recordID string) *ext_proc.ProcessingResponse {
	if method != "GET" {
		return r.createErrorResponse(405, "method not allowed")
	}
	if recordID == "" {
		return r.createErrorResponse(400, "feedback record id is required")
	}

	record, ok := r.findMetaRoutingFeedbackRecord(recordID)
	if !ok {
		return r.createErrorResponse(404, "meta-routing feedback record not found")
	}

	return r.createMetaRoutingFeedbackJSONResponse(200, metaRoutingFeedbackDetailResponse{
		Object:    "meta_routing_feedback.record",
		ID:        record.Metadata.ID,
		Timestamp: record.Metadata.Timestamp,
		Record:    record.Record,
	})
}

func parseMetaRoutingFeedbackListQuery(rawQuery string) (metaRoutingFeedbackListQuery, error) {
	query := metaRoutingFeedbackListQuery{
		limit: metaRoutingFeedbackDefaultLimit,
	}

	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		return query, fmt.Errorf("invalid meta-routing feedback query")
	}

	filters, err := parseMetaRoutingFeedbackFilters(values)
	if err != nil {
		return query, err
	}
	query.filters = filters

	if values.Has("limit") {
		limit, err := parseRouterReplayQueryInt(values, "limit", true)
		if err != nil {
			return query, err
		}
		if limit > metaRoutingFeedbackMaxLimit {
			limit = metaRoutingFeedbackMaxLimit
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

func parseMetaRoutingFeedbackFilters(values url.Values) (metaRoutingFeedbackFilters, error) {
	filters := metaRoutingFeedbackFilters{
		search:       strings.TrimSpace(values.Get("search")),
		mode:         normalizeMetaRoutingFilter(values.Get("mode")),
		trigger:      normalizeMetaRoutingFilter(values.Get("trigger")),
		rootCause:    normalizeMetaRoutingFilter(values.Get("root_cause")),
		actionType:   normalizeMetaRoutingFilter(values.Get("action_type")),
		signalFamily: normalizeMetaRoutingFilter(values.Get("signal_family")),
		decision:     strings.TrimSpace(values.Get("decision")),
		model:        strings.TrimSpace(values.Get("model")),
	}

	switch overturned := normalizeMetaRoutingFilter(values.Get("overturned")); overturned {
	case "", "all":
		filters.overturned = ""
	case "true", "false":
		filters.overturned = overturned
	default:
		return filters, fmt.Errorf("overturned must be one of all, true, or false")
	}

	if rawStatus := normalizeMetaRoutingFilter(values.Get("response_status")); rawStatus != "" && rawStatus != "all" {
		status, err := strconv.Atoi(rawStatus)
		if err != nil {
			return filters, fmt.Errorf("response_status must be an integer")
		}
		filters.responseStatus = &status
	}

	return filters, nil
}

func normalizeMetaRoutingFilter(value string) string {
	return strings.TrimSpace(strings.ToLower(value))
}

func (r *OpenAIRouter) collectMetaRoutingFeedbackStoredRecords() []metaRoutingFeedbackStoredRecord {
	if r == nil || r.FeedbackRecorder == nil {
		return nil
	}

	rawRecords := r.FeedbackRecorder.ListAllRecords()
	records := make([]metaRoutingFeedbackStoredRecord, 0, len(rawRecords))
	for _, raw := range rawRecords {
		if raw.RequestBody == "" {
			continue
		}
		var parsed FeedbackRecord
		if err := json.Unmarshal([]byte(raw.RequestBody), &parsed); err != nil {
			continue
		}
		records = append(records, metaRoutingFeedbackStoredRecord{
			Metadata: raw,
			Record:   parsed,
		})
	}

	sort.SliceStable(records, func(i, j int) bool {
		if records[i].Metadata.Timestamp.Equal(records[j].Metadata.Timestamp) {
			return records[i].Metadata.ID > records[j].Metadata.ID
		}
		return records[i].Metadata.Timestamp.After(records[j].Metadata.Timestamp)
	})
	return records
}

func (r *OpenAIRouter) findMetaRoutingFeedbackRecord(recordID string) (metaRoutingFeedbackStoredRecord, bool) {
	if r == nil || r.FeedbackRecorder == nil {
		return metaRoutingFeedbackStoredRecord{}, false
	}

	raw, ok := r.FeedbackRecorder.GetRecord(recordID)
	if !ok || raw.RequestBody == "" {
		return metaRoutingFeedbackStoredRecord{}, false
	}

	var parsed FeedbackRecord
	if err := json.Unmarshal([]byte(raw.RequestBody), &parsed); err != nil {
		return metaRoutingFeedbackStoredRecord{}, false
	}

	return metaRoutingFeedbackStoredRecord{
		Metadata: raw,
		Record:   parsed,
	}, true
}

func filterMetaRoutingFeedbackRecords(
	records []metaRoutingFeedbackStoredRecord,
	filters metaRoutingFeedbackFilters,
) []metaRoutingFeedbackStoredRecord {
	if filters == (metaRoutingFeedbackFilters{}) {
		return records
	}

	search := strings.ToLower(filters.search)
	filtered := make([]metaRoutingFeedbackStoredRecord, 0, len(records))
	for _, record := range records {
		if !doesMetaRoutingFeedbackMatchFilters(record, filters, search) {
			continue
		}
		filtered = append(filtered, record)
	}
	return filtered
}

func doesMetaRoutingFeedbackMatchFilters(
	record metaRoutingFeedbackStoredRecord,
	filters metaRoutingFeedbackFilters,
	search string,
) bool {
	trace := record.Record.Observation.Trace
	return matchMetaRoutingMode(record.Record, filters) &&
		matchMetaRoutingTrigger(trace, filters) &&
		matchMetaRoutingRootCause(trace, filters) &&
		matchMetaRoutingActionType(record.Record, filters) &&
		matchMetaRoutingSignalFamily(record.Record, filters) &&
		matchMetaRoutingOverturned(trace, filters) &&
		matchMetaRoutingDecision(record.Record, filters) &&
		matchMetaRoutingModel(record.Record, filters) &&
		matchMetaRoutingResponseStatus(record.Record, filters) &&
		matchMetaRoutingSearch(record, search)
}

func matchMetaRoutingMode(record FeedbackRecord, filters metaRoutingFeedbackFilters) bool {
	return filters.mode == "" || strings.ToLower(record.Mode) == filters.mode
}

func matchMetaRoutingTrigger(trace *RoutingTrace, filters metaRoutingFeedbackFilters) bool {
	return filters.trigger == "" || containsFold(traceTriggerNames(trace), filters.trigger)
}

func matchMetaRoutingRootCause(trace *RoutingTrace, filters metaRoutingFeedbackFilters) bool {
	return filters.rootCause == "" || containsFold(traceRootCauses(trace), filters.rootCause)
}

func matchMetaRoutingActionType(record FeedbackRecord, filters metaRoutingFeedbackFilters) bool {
	return filters.actionType == "" || containsFold(actionTypesForFeedbackRecord(record), filters.actionType)
}

func matchMetaRoutingSignalFamily(record FeedbackRecord, filters metaRoutingFeedbackFilters) bool {
	return filters.signalFamily == "" || containsFold(signalFamiliesForFeedbackRecord(record), filters.signalFamily)
}

func matchMetaRoutingOverturned(trace *RoutingTrace, filters metaRoutingFeedbackFilters) bool {
	if filters.overturned == "" {
		return true
	}
	expected := filters.overturned == "true"
	return trace != nil && trace.OverturnedDecision == expected
}

func matchMetaRoutingDecision(record FeedbackRecord, filters metaRoutingFeedbackFilters) bool {
	return filters.decision == "" || record.Outcome.FinalDecisionName == filters.decision
}

func matchMetaRoutingModel(record FeedbackRecord, filters metaRoutingFeedbackFilters) bool {
	return filters.model == "" || doesMetaRoutingModelMatch(record, filters.model)
}

func matchMetaRoutingResponseStatus(record FeedbackRecord, filters metaRoutingFeedbackFilters) bool {
	return filters.responseStatus == nil || record.Outcome.ResponseStatus == *filters.responseStatus
}

func matchMetaRoutingSearch(record metaRoutingFeedbackStoredRecord, search string) bool {
	return search == "" || doesMetaRoutingSearchMatch(record, search)
}

func traceTriggerNames(trace *RoutingTrace) []string {
	if trace == nil {
		return nil
	}
	return trace.TriggerNames
}

func traceRootCauses(trace *RoutingTrace) []string {
	if trace == nil || trace.FinalAssessment == nil {
		return nil
	}
	return trace.FinalAssessment.RootCauses
}

func containsFold(values []string, target string) bool {
	for _, value := range values {
		if strings.EqualFold(value, target) {
			return true
		}
	}
	return false
}

func doesMetaRoutingModelMatch(record FeedbackRecord, target string) bool {
	return strings.EqualFold(record.Outcome.FinalModel, target) ||
		strings.EqualFold(record.Observation.RequestModel, target)
}

func doesMetaRoutingSearchMatch(record metaRoutingFeedbackStoredRecord, search string) bool {
	candidates := []string{
		record.Metadata.ID,
		record.Record.Observation.RequestID,
		record.Record.Observation.RequestModel,
		record.Record.Observation.RequestQuery,
		record.Record.Outcome.FinalDecisionName,
		record.Record.Outcome.FinalModel,
		record.Record.Outcome.RouterReplayID,
	}
	for _, candidate := range candidates {
		if strings.Contains(strings.ToLower(candidate), search) {
			return true
		}
	}
	return false
}

func buildMetaRoutingFeedbackListPayload(
	records []metaRoutingFeedbackStoredRecord,
	query metaRoutingFeedbackListQuery,
) metaRoutingFeedbackListResponse {
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
	data := make([]metaRoutingFeedbackSummary, 0, len(page))
	for _, record := range page {
		data = append(data, summarizeMetaRoutingFeedbackRecord(record))
	}

	payload := metaRoutingFeedbackListResponse{
		Object:  "meta_routing_feedback.list",
		Count:   len(data),
		Total:   total,
		Limit:   query.limit,
		Offset:  offset,
		HasMore: end < total,
		Data:    data,
	}
	if payload.HasMore {
		nextOffset := end
		payload.NextOffset = &nextOffset
	}
	return payload
}

func summarizeMetaRoutingFeedbackRecord(record metaRoutingFeedbackStoredRecord) metaRoutingFeedbackSummary {
	trace := record.Record.Observation.Trace
	summary := metaRoutingFeedbackSummary{
		ID:                        record.Metadata.ID,
		Timestamp:                 record.Metadata.Timestamp,
		Mode:                      record.Record.Mode,
		RequestID:                 record.Record.Observation.RequestID,
		RequestModel:              record.Record.Observation.RequestModel,
		RequestQueryPreview:       truncateMetaRoutingQuery(record.Record.Observation.RequestQuery),
		Planned:                   record.Record.Action.Planned,
		Executed:                  record.Record.Action.Executed,
		ExecutedPassCount:         record.Record.Action.ExecutedPassCount,
		ActionTypes:               actionTypesForFeedbackRecord(record.Record),
		RefinedSignalFamilies:     signalFamiliesForFeedbackRecord(record.Record),
		FinalDecisionName:         record.Record.Outcome.FinalDecisionName,
		FinalDecisionConfidence:   record.Record.Outcome.FinalDecisionConfidence,
		FinalModel:                record.Record.Outcome.FinalModel,
		ResponseStatus:            record.Record.Outcome.ResponseStatus,
		Streaming:                 record.Record.Outcome.Streaming,
		CacheHit:                  record.Record.Outcome.CacheHit,
		PIIBlocked:                record.Record.Outcome.PIIBlocked,
		HallucinationDetected:     record.Record.Outcome.HallucinationDetected,
		ResponseJailbreakDetected: record.Record.Outcome.ResponseJailbreakDetected,
		RouterReplayID:            record.Record.Outcome.RouterReplayID,
		UserFeedbackSignals:       append([]string(nil), record.Record.Outcome.UserFeedbackSignals...),
	}

	if trace != nil {
		summary.PassCount = trace.PassCount
		summary.TriggerNames = append([]string(nil), trace.TriggerNames...)
		summary.OverturnedDecision = trace.OverturnedDecision
		summary.LatencyDeltaMs = trace.LatencyDeltaMs
		summary.DecisionMarginDelta = trace.DecisionMarginDelta
		summary.ProjectionBoundaryDelta = cloneMetaRoutingFloat64Ptr(trace.ProjectionBoundaryDelta)
		if trace.FinalAssessment != nil {
			summary.RootCauses = append([]string(nil), trace.FinalAssessment.RootCauses...)
		}
	}

	return summary
}

func truncateMetaRoutingQuery(value string) string {
	value = strings.TrimSpace(value)
	if len(value) <= metaRoutingFeedbackQueryPreview {
		return value
	}
	return strings.TrimSpace(value[:metaRoutingFeedbackQueryPreview]) + "..."
}

func actionTypesForFeedbackRecord(record FeedbackRecord) []string {
	if len(record.Action.ExecutedActionTypes) > 0 {
		return append([]string(nil), record.Action.ExecutedActionTypes...)
	}
	if record.Action.Plan == nil || len(record.Action.Plan.Actions) == 0 {
		return nil
	}
	types := make([]string, 0, len(record.Action.Plan.Actions))
	for _, action := range record.Action.Plan.Actions {
		if action.Type != "" {
			types = append(types, action.Type)
		}
	}
	return uniqueSortedStrings(types)
}

func signalFamiliesForFeedbackRecord(record FeedbackRecord) []string {
	if len(record.Action.ExecutedSignalFamilies) > 0 {
		return append([]string(nil), record.Action.ExecutedSignalFamilies...)
	}
	if record.Action.Plan == nil || len(record.Action.Plan.Actions) == 0 {
		return nil
	}
	families := make([]string, 0)
	for _, action := range record.Action.Plan.Actions {
		families = append(families, action.SignalFamilies...)
	}
	return uniqueSortedStrings(families)
}

func cloneMetaRoutingFloat64Ptr(value *float64) *float64 {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func (r *OpenAIRouter) createMetaRoutingFeedbackJSONResponse(
	statusCode int,
	data interface{},
) *ext_proc.ProcessingResponse {
	response := r.createJSONResponse(statusCode, data)
	if response == nil {
		return nil
	}
	if proto.Size(response) <= routerAPIMaxImmediateResponseSize {
		return response
	}
	return r.createErrorResponse(
		413,
		"meta-routing feedback response exceeds the ext-proc message size limit; retry with a smaller page or reduce captured feedback body bytes",
	)
}
