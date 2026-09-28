package extproc

import (
	"math"
	"net/url"
	"sort"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

type routerReplayAggregateResponse = routerreplay.AggregateResponse

type routerReplayLifecycleSummary = routerreplay.LifecycleSummary

type routerReplayAggregateCostSummary = routerreplay.AggregateCostSummary

type routerReplayAggregateValue = routerreplay.AggregateValue

type routerReplayAggregateTokenVolume = routerreplay.AggregateTokenVolume

type routerReplayAggregateTokenBuckets = routerreplay.AggregateTokenBuckets

type routerReplayAggregateTokenEntry = routerreplay.AggregateTokenEntry

func (r *OpenAIRouter) handleRouterReplayAggregateAPI(
	method string,
	rawQuery string,
) *ext_proc.ProcessingResponse {
	if method != "GET" {
		return r.createErrorResponse(405, "method not allowed")
	}

	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		return r.createErrorResponse(400, err.Error())
	}
	filters, err := parseRouterReplayFilters(values)
	if err != nil {
		return r.createErrorResponse(400, err.Error())
	}

	payload, err := r.queryRouterReplayAggregate(filters)
	if err != nil {
		return r.createErrorResponse(500, "router replay storage query failed")
	}
	return r.createRouterReplayJSONResponse(200, payload)
}

func buildRouterReplayAggregatePayload(
	allRecords []routerreplay.RoutingRecord,
	filteredRecords []routerreplay.RoutingRecord,
) routerReplayAggregateResponse {
	return routerReplayAggregateResponse{
		Object:               "router_replay.aggregate",
		RecordCount:          len(filteredRecords),
		Lifecycle:            buildRouterReplayLifecycleSummary(filteredRecords),
		Summary:              buildRouterReplayAggregateCostSummary(filteredRecords),
		ModelSelection:       buildRouterReplayModelSelection(filteredRecords),
		DecisionDistribution: buildRouterReplayDecisionDistribution(filteredRecords),
		SignalDistribution:   buildRouterReplaySignalDistribution(filteredRecords),
		TokenVolume:          buildRouterReplayTokenVolume(filteredRecords),
		TokenBreakdown:       buildRouterReplayTokenBreakdown(filteredRecords),
		AvailableRecipes:     collectRouterReplayRecipeOptions(allRecords),
		AvailableDecisions:   collectRouterReplayDecisionOptions(allRecords),
		AvailableModels:      collectRouterReplayModelOptions(allRecords),
	}
}

func buildRouterReplayLifecycleSummary(
	records []routerreplay.RoutingRecord,
) routerReplayLifecycleSummary {
	summary := routerReplayLifecycleSummary{}
	for _, record := range records {
		switch record.LifecycleState {
		case routerreplay.LifecycleCompleted:
			summary.Completed++
		case routerreplay.LifecycleFailed:
			summary.Failed++
		case routerreplay.LifecycleAborted:
			summary.Aborted++
		case routerreplay.LifecycleInProgress:
			summary.InProgress++
		default:
			summary.Unknown++
		}
	}
	return summary
}

func buildRouterReplayAggregateCostSummary(
	records []routerreplay.RoutingRecord,
) routerReplayAggregateCostSummary {
	summary := routerReplayAggregateCostSummary{}
	byCurrency := make(map[string]*routerreplay.CurrencyCostSummary)
	for _, record := range records {
		if record.LifecycleState != routerreplay.LifecycleCompleted ||
			record.TotalTokens == nil || record.BaselineModel == nil || *record.BaselineModel == "" ||
			record.Currency == nil || normalizeReplayCurrency(*record.Currency) == "" {
			continue
		}
		if !finiteReplayCost(record.ActualCost) || !finiteReplayCost(record.BaselineCost) || !finiteReplayCost(record.CostSavings) {
			continue
		}
		currency := normalizeReplayCurrency(*record.Currency)
		group := byCurrency[currency]
		if group == nil {
			group = &routerreplay.CurrencyCostSummary{Currency: currency}
			byCurrency[currency] = group
		}
		group.TotalSaved += *record.CostSavings
		group.BaselineSpend += *record.BaselineCost
		group.ActualSpend += *record.ActualCost
		group.CostRecordCount++
		summary.CostRecordCount++
	}
	for _, group := range byCurrency {
		summary.ByCurrency = append(summary.ByCurrency, *group)
	}
	sort.Slice(summary.ByCurrency, func(i, j int) bool {
		return summary.ByCurrency[i].Currency < summary.ByCurrency[j].Currency
	})
	if len(summary.ByCurrency) == 1 {
		group := summary.ByCurrency[0]
		summary.Currency = group.Currency
		summary.TotalSaved = group.TotalSaved
		summary.BaselineSpend = group.BaselineSpend
		summary.ActualSpend = group.ActualSpend
	}
	summary.ExcludedRecordCount = len(records) - summary.CostRecordCount
	return summary
}

func finiteReplayCost(value *float64) bool {
	return value != nil && !math.IsNaN(*value) && !math.IsInf(*value, 0)
}

func buildRouterReplayModelSelection(
	records []routerreplay.RoutingRecord,
) []routerReplayAggregateValue {
	counts := make(map[string]int)
	for _, record := range records {
		name := record.SelectedModel
		if name == "" {
			name = "Unknown"
		}
		counts[name]++
	}
	return sortRouterReplayAggregateValues(counts, 10)
}

func buildRouterReplayDecisionDistribution(
	records []routerreplay.RoutingRecord,
) []routerReplayAggregateValue {
	counts := make(map[string]int)
	for _, record := range records {
		name := config.RoutingDecisionKey(config.RecipeName(record.Recipe), record.Decision)
		if name == "" {
			name = "Unknown"
		}
		counts[name]++
	}
	return sortRouterReplayAggregateValues(counts, 0)
}

func buildRouterReplaySignalDistribution(
	records []routerreplay.RoutingRecord,
) []routerReplayAggregateValue {
	counts := make(map[string]int)
	for _, record := range records {
		if n := len(record.Signals.Keyword); n > 0 {
			counts["keyword"] += n
		}
		if n := len(record.Signals.Embedding); n > 0 {
			counts["embedding"] += n
		}
		if n := len(record.Signals.Domain); n > 0 {
			counts["domain"] += n
		}
		if n := len(record.Signals.FactCheck); n > 0 {
			counts["fact_check"] += n
		}
		if n := len(record.Signals.UserFeedback); n > 0 {
			counts["user_feedback"] += n
		}
		if n := len(record.Signals.Reask); n > 0 {
			counts["reask"] += n
		}
		if n := len(record.Signals.Preference); n > 0 {
			counts["preference"] += n
		}
		if n := len(record.Signals.Language); n > 0 {
			counts["language"] += n
		}
		if n := len(record.Signals.Context); n > 0 {
			counts["context"] += n
		}
		if n := len(record.Signals.Complexity); n > 0 {
			counts["complexity"] += n
		}
	}
	return sortRouterReplayAggregateValues(counts, 0)
}

func buildRouterReplayTokenVolume(
	records []routerreplay.RoutingRecord,
) routerReplayAggregateTokenVolume {
	var volume routerReplayAggregateTokenVolume
	usageRecordCount := 0

	for _, record := range records {
		promptTokens, completionTokens, totalTokens, hasUsage := routerReplayUsageTriplet(record)
		if !hasUsage {
			continue
		}

		volume.InputTokens += promptTokens
		volume.OutputTokens += completionTokens
		volume.TotalTokens += totalTokens
		usageRecordCount++
	}

	volume.ExcludedRecordCount = len(records) - usageRecordCount
	return volume
}

func buildRouterReplayTokenBreakdown(
	records []routerreplay.RoutingRecord,
) routerReplayAggregateTokenBuckets {
	decisionBuckets := make(map[string]*routerReplayAggregateTokenEntry)
	modelBuckets := make(map[string]*routerReplayAggregateTokenEntry)

	for _, record := range records {
		promptTokens, completionTokens, totalTokens, hasUsage := routerReplayUsageTriplet(record)
		if !hasUsage {
			continue
		}

		accumulateRouterReplayTokenEntry(
			decisionBuckets,
			routerReplayFallbackName(config.RoutingDecisionKey(config.RecipeName(record.Recipe), record.Decision)),
			promptTokens,
			completionTokens,
			totalTokens,
		)
		accumulateRouterReplayTokenEntry(
			modelBuckets,
			routerReplayFallbackName(record.SelectedModel),
			promptTokens,
			completionTokens,
			totalTokens,
		)
	}

	return routerReplayAggregateTokenBuckets{
		ByDecision:      sortRouterReplayTokenEntries(decisionBuckets, 8),
		BySelectedModel: sortRouterReplayTokenEntries(modelBuckets, 8),
	}
}

func routerReplayUsageTriplet(record routerreplay.RoutingRecord) (int, int, int, bool) {
	promptTokens := 0
	completionTokens := 0
	hasPrompt := record.PromptTokens != nil
	hasCompletion := record.CompletionTokens != nil
	if hasPrompt {
		promptTokens = *record.PromptTokens
	}
	if hasCompletion {
		completionTokens = *record.CompletionTokens
	}

	if record.TotalTokens != nil {
		return promptTokens, completionTokens, *record.TotalTokens, true
	}
	if hasPrompt || hasCompletion {
		return promptTokens, completionTokens, promptTokens + completionTokens, true
	}
	return 0, 0, 0, false
}

func accumulateRouterReplayTokenEntry(
	buckets map[string]*routerReplayAggregateTokenEntry,
	name string,
	inputTokens int,
	outputTokens int,
	totalTokens int,
) {
	entry, ok := buckets[name]
	if !ok {
		entry = &routerReplayAggregateTokenEntry{Name: name}
		buckets[name] = entry
	}
	entry.InputTokens += inputTokens
	entry.OutputTokens += outputTokens
	entry.TotalTokens += totalTokens
}

func sortRouterReplayAggregateValues(
	counts map[string]int,
	limit int,
) []routerReplayAggregateValue {
	values := make([]routerReplayAggregateValue, 0, len(counts))
	for name, value := range counts {
		values = append(values, routerReplayAggregateValue{Name: name, Value: value})
	}
	sort.Slice(values, func(i, j int) bool {
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

func sortRouterReplayTokenEntries(
	buckets map[string]*routerReplayAggregateTokenEntry,
	limit int,
) []routerReplayAggregateTokenEntry {
	values := make([]routerReplayAggregateTokenEntry, 0, len(buckets))
	for _, entry := range buckets {
		values = append(values, *entry)
	}
	sort.Slice(values, func(i, j int) bool {
		if values[i].TotalTokens == values[j].TotalTokens {
			return values[i].Name < values[j].Name
		}
		return values[i].TotalTokens > values[j].TotalTokens
	})
	if limit > 0 && len(values) > limit {
		return values[:limit]
	}
	return values
}

func collectRouterReplayDecisionOptions(records []routerreplay.RoutingRecord) []string {
	values := make(map[string]struct{})
	for _, record := range records {
		if record.Decision != "" {
			values[record.Decision] = struct{}{}
		}
	}
	return sortRouterReplayOptionSet(values)
}

func collectRouterReplayRecipeOptions(records []routerreplay.RoutingRecord) []string {
	values := make(map[string]struct{})
	for _, record := range records {
		if record.Recipe != "" {
			values[record.Recipe] = struct{}{}
		}
	}
	return sortRouterReplayOptionSet(values)
}

func collectRouterReplayModelOptions(records []routerreplay.RoutingRecord) []string {
	values := make(map[string]struct{})
	for _, record := range records {
		if record.SelectedModel != "" {
			values[record.SelectedModel] = struct{}{}
		}
		if record.OriginalModel != "" {
			values[record.OriginalModel] = struct{}{}
		}
	}
	return sortRouterReplayOptionSet(values)
}

func sortRouterReplayOptionSet(values map[string]struct{}) []string {
	options := make([]string, 0, len(values))
	for value := range values {
		options = append(options, value)
	}
	sort.Strings(options)
	return options
}

func routerReplayFallbackName(value string) string {
	if value == "" {
		return "Unknown"
	}
	return value
}
