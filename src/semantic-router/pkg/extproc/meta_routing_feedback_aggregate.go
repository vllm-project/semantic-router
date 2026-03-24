package extproc

import (
	"net/url"
	"sort"
	"strconv"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

type metaRoutingFeedbackAggregateValue struct {
	Name  string `json:"name"`
	Value int    `json:"value"`
}

type metaRoutingFeedbackAggregateSummary struct {
	PlannedRefinementRate  float64 `json:"planned_refinement_rate"`
	ExecutedRefinementRate float64 `json:"executed_refinement_rate"`
	OverturnRate           float64 `json:"overturn_rate"`
	AverageLatencyDeltaMs  float64 `json:"average_latency_delta_ms"`
	P95LatencyDeltaMs      int64   `json:"p95_latency_delta_ms"`
	TopTrigger             string  `json:"top_trigger,omitempty"`
	TopRootCause           string  `json:"top_root_cause,omitempty"`
}

type metaRoutingFeedbackAggregateResponse struct {
	Object                     string                              `json:"object"`
	RecordCount                int                                 `json:"record_count"`
	Summary                    metaRoutingFeedbackAggregateSummary `json:"summary"`
	ModeDistribution           []metaRoutingFeedbackAggregateValue `json:"mode_distribution"`
	TriggerDistribution        []metaRoutingFeedbackAggregateValue `json:"trigger_distribution"`
	RootCauseDistribution      []metaRoutingFeedbackAggregateValue `json:"root_cause_distribution"`
	ActionTypeDistribution     []metaRoutingFeedbackAggregateValue `json:"action_type_distribution"`
	SignalFamilyDistribution   []metaRoutingFeedbackAggregateValue `json:"signal_family_distribution"`
	DecisionChangeDistribution []metaRoutingFeedbackAggregateValue `json:"decision_change_distribution"`
	DecisionDistribution       []metaRoutingFeedbackAggregateValue `json:"decision_distribution"`
	ModelDistribution          []metaRoutingFeedbackAggregateValue `json:"model_distribution"`
	ResponseStatusDistribution []metaRoutingFeedbackAggregateValue `json:"response_status_distribution"`
	AvailableModes             []string                            `json:"available_modes"`
	AvailableTriggers          []string                            `json:"available_triggers"`
	AvailableRootCauses        []string                            `json:"available_root_causes"`
	AvailableActionTypes       []string                            `json:"available_action_types"`
	AvailableSignalFamilies    []string                            `json:"available_signal_families"`
	AvailableDecisions         []string                            `json:"available_decisions"`
	AvailableModels            []string                            `json:"available_models"`
	AvailableResponseStatuses  []int                               `json:"available_response_statuses"`
}

func (r *OpenAIRouter) handleMetaRoutingFeedbackAggregateAPI(
	method string,
	rawQuery string,
) *ext_proc.ProcessingResponse {
	if method != "GET" {
		return r.createErrorResponse(405, "method not allowed")
	}

	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		return r.createErrorResponse(400, "invalid meta-routing feedback query")
	}
	filters, err := parseMetaRoutingFeedbackFilters(values)
	if err != nil {
		return r.createErrorResponse(400, err.Error())
	}

	allRecords := r.collectMetaRoutingFeedbackStoredRecords()
	filteredRecords := filterMetaRoutingFeedbackRecords(allRecords, filters)
	return r.createMetaRoutingFeedbackJSONResponse(200, buildMetaRoutingFeedbackAggregatePayload(allRecords, filteredRecords))
}

func buildMetaRoutingFeedbackAggregatePayload(
	allRecords []metaRoutingFeedbackStoredRecord,
	filteredRecords []metaRoutingFeedbackStoredRecord,
) metaRoutingFeedbackAggregateResponse {
	triggerDistribution := buildMetaRoutingFeedbackStringDistribution(filteredRecords, func(record metaRoutingFeedbackStoredRecord) []string {
		return traceTriggerNames(record.Record.Observation.Trace)
	}, 0)
	rootCauseDistribution := buildMetaRoutingFeedbackStringDistribution(filteredRecords, func(record metaRoutingFeedbackStoredRecord) []string {
		return traceRootCauses(record.Record.Observation.Trace)
	}, 0)

	return metaRoutingFeedbackAggregateResponse{
		Object:                "meta_routing_feedback.aggregate",
		RecordCount:           len(filteredRecords),
		Summary:               buildMetaRoutingFeedbackAggregateSummary(filteredRecords, triggerDistribution, rootCauseDistribution),
		ModeDistribution:      buildMetaRoutingFeedbackScalarDistribution(filteredRecords, func(record metaRoutingFeedbackStoredRecord) string { return record.Record.Mode }, 0),
		TriggerDistribution:   triggerDistribution,
		RootCauseDistribution: rootCauseDistribution,
		ActionTypeDistribution: buildMetaRoutingFeedbackStringDistribution(filteredRecords, func(record metaRoutingFeedbackStoredRecord) []string {
			return actionTypesForFeedbackRecord(record.Record)
		}, 0),
		SignalFamilyDistribution: buildMetaRoutingFeedbackStringDistribution(filteredRecords, func(record metaRoutingFeedbackStoredRecord) []string {
			return signalFamiliesForFeedbackRecord(record.Record)
		}, 0),
		DecisionChangeDistribution: buildMetaRoutingFeedbackDecisionChangeDistribution(filteredRecords),
		DecisionDistribution:       buildMetaRoutingFeedbackScalarDistribution(filteredRecords, func(record metaRoutingFeedbackStoredRecord) string { return record.Record.Outcome.FinalDecisionName }, 0),
		ModelDistribution:          buildMetaRoutingFeedbackScalarDistribution(filteredRecords, func(record metaRoutingFeedbackStoredRecord) string { return record.Record.Outcome.FinalModel }, 10),
		ResponseStatusDistribution: buildMetaRoutingFeedbackResponseStatusDistribution(filteredRecords),
		AvailableModes:             collectMetaRoutingFeedbackStringOptions(allRecords, func(record metaRoutingFeedbackStoredRecord) []string { return []string{record.Record.Mode} }),
		AvailableTriggers: collectMetaRoutingFeedbackStringOptions(allRecords, func(record metaRoutingFeedbackStoredRecord) []string {
			return traceTriggerNames(record.Record.Observation.Trace)
		}),
		AvailableRootCauses: collectMetaRoutingFeedbackStringOptions(allRecords, func(record metaRoutingFeedbackStoredRecord) []string {
			return traceRootCauses(record.Record.Observation.Trace)
		}),
		AvailableActionTypes: collectMetaRoutingFeedbackStringOptions(allRecords, func(record metaRoutingFeedbackStoredRecord) []string {
			return actionTypesForFeedbackRecord(record.Record)
		}),
		AvailableSignalFamilies: collectMetaRoutingFeedbackStringOptions(allRecords, func(record metaRoutingFeedbackStoredRecord) []string {
			return signalFamiliesForFeedbackRecord(record.Record)
		}),
		AvailableDecisions: collectMetaRoutingFeedbackStringOptions(allRecords, func(record metaRoutingFeedbackStoredRecord) []string {
			return []string{record.Record.Outcome.FinalDecisionName}
		}),
		AvailableModels: collectMetaRoutingFeedbackStringOptions(allRecords, func(record metaRoutingFeedbackStoredRecord) []string {
			return []string{record.Record.Outcome.FinalModel, record.Record.Observation.RequestModel}
		}),
		AvailableResponseStatuses: collectMetaRoutingFeedbackStatusOptions(allRecords),
	}
}

func buildMetaRoutingFeedbackAggregateSummary(
	records []metaRoutingFeedbackStoredRecord,
	triggerDistribution []metaRoutingFeedbackAggregateValue,
	rootCauseDistribution []metaRoutingFeedbackAggregateValue,
) metaRoutingFeedbackAggregateSummary {
	var summary metaRoutingFeedbackAggregateSummary
	if len(records) == 0 {
		return summary
	}

	latencyDeltas := make([]int64, 0, len(records))
	plannedCount := 0
	executedCount := 0
	overturnedCount := 0
	for _, record := range records {
		if record.Record.Action.Planned {
			plannedCount++
		}
		if record.Record.Action.Executed {
			executedCount++
		}
		if record.Record.Observation.Trace != nil {
			latencyDeltas = append(latencyDeltas, record.Record.Observation.Trace.LatencyDeltaMs)
			if record.Record.Observation.Trace.OverturnedDecision {
				overturnedCount++
			}
		}
	}

	summary.PlannedRefinementRate = float64(plannedCount) / float64(len(records))
	summary.ExecutedRefinementRate = float64(executedCount) / float64(len(records))
	summary.OverturnRate = float64(overturnedCount) / float64(len(records))
	summary.AverageLatencyDeltaMs = averageInt64(latencyDeltas)
	summary.P95LatencyDeltaMs = percentileInt64(latencyDeltas, 0.95)
	if len(triggerDistribution) > 0 {
		summary.TopTrigger = triggerDistribution[0].Name
	}
	if len(rootCauseDistribution) > 0 {
		summary.TopRootCause = rootCauseDistribution[0].Name
	}
	return summary
}

func buildMetaRoutingFeedbackScalarDistribution(
	records []metaRoutingFeedbackStoredRecord,
	getter func(metaRoutingFeedbackStoredRecord) string,
	limit int,
) []metaRoutingFeedbackAggregateValue {
	counts := make(map[string]int)
	for _, record := range records {
		name := getter(record)
		if name == "" {
			name = "Unknown"
		}
		counts[name]++
	}
	return sortMetaRoutingFeedbackAggregateValues(counts, limit)
}

func buildMetaRoutingFeedbackStringDistribution(
	records []metaRoutingFeedbackStoredRecord,
	getter func(metaRoutingFeedbackStoredRecord) []string,
	limit int,
) []metaRoutingFeedbackAggregateValue {
	counts := make(map[string]int)
	for _, record := range records {
		for _, value := range getter(record) {
			if value == "" {
				continue
			}
			counts[value]++
		}
	}
	return sortMetaRoutingFeedbackAggregateValues(counts, limit)
}

func buildMetaRoutingFeedbackDecisionChangeDistribution(
	records []metaRoutingFeedbackStoredRecord,
) []metaRoutingFeedbackAggregateValue {
	counts := map[string]int{
		"stable":     0,
		"overturned": 0,
	}
	for _, record := range records {
		if record.Record.Observation.Trace != nil && record.Record.Observation.Trace.OverturnedDecision {
			counts["overturned"]++
			continue
		}
		counts["stable"]++
	}
	return sortMetaRoutingFeedbackAggregateValues(counts, 0)
}

func buildMetaRoutingFeedbackResponseStatusDistribution(
	records []metaRoutingFeedbackStoredRecord,
) []metaRoutingFeedbackAggregateValue {
	counts := make(map[string]int)
	for _, record := range records {
		name := "Unknown"
		if record.Record.Outcome.ResponseStatus != 0 {
			name = strconv.Itoa(record.Record.Outcome.ResponseStatus)
		}
		counts[name]++
	}
	return sortMetaRoutingFeedbackAggregateValues(counts, 0)
}

func sortMetaRoutingFeedbackAggregateValues(
	counts map[string]int,
	limit int,
) []metaRoutingFeedbackAggregateValue {
	values := make([]metaRoutingFeedbackAggregateValue, 0, len(counts))
	for name, value := range counts {
		values = append(values, metaRoutingFeedbackAggregateValue{Name: name, Value: value})
	}

	sort.SliceStable(values, func(i, j int) bool {
		if values[i].Value == values[j].Value {
			return values[i].Name < values[j].Name
		}
		return values[i].Value > values[j].Value
	})

	if limit > 0 && len(values) > limit {
		return values[:limit]
	}
	return values
}

func collectMetaRoutingFeedbackStringOptions(
	records []metaRoutingFeedbackStoredRecord,
	getter func(metaRoutingFeedbackStoredRecord) []string,
) []string {
	values := make(map[string]struct{})
	for _, record := range records {
		for _, value := range getter(record) {
			if value == "" {
				continue
			}
			values[value] = struct{}{}
		}
	}

	options := make([]string, 0, len(values))
	for value := range values {
		options = append(options, value)
	}
	sort.Strings(options)
	return options
}

func collectMetaRoutingFeedbackStatusOptions(records []metaRoutingFeedbackStoredRecord) []int {
	values := make(map[int]struct{})
	for _, record := range records {
		if record.Record.Outcome.ResponseStatus != 0 {
			values[record.Record.Outcome.ResponseStatus] = struct{}{}
		}
	}

	options := make([]int, 0, len(values))
	for value := range values {
		options = append(options, value)
	}
	sort.Ints(options)
	return options
}

func averageInt64(values []int64) float64 {
	if len(values) == 0 {
		return 0
	}
	var total int64
	for _, value := range values {
		total += value
	}
	return float64(total) / float64(len(values))
}

func percentileInt64(values []int64, percentile float64) int64 {
	if len(values) == 0 {
		return 0
	}
	cloned := append([]int64(nil), values...)
	sort.Slice(cloned, func(i, j int) bool {
		return cloned[i] < cloned[j]
	})
	index := int(float64(len(cloned)-1) * percentile)
	if index < 0 {
		index = 0
	}
	if index >= len(cloned) {
		index = len(cloned) - 1
	}
	return cloned[index]
}
