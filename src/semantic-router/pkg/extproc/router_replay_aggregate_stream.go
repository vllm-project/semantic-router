package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// The accumulator retains buckets, never a row list or captured bodies. The
// one-record builders keep cost, lifecycle, and usage eligibility in one place.
type replayAggregateAccumulator struct {
	payload                                               routerReplayAggregateResponse
	models, decisions, signals                            map[string]int
	costs                                                 *replayCostAccumulator
	providerCache                                         *replayProviderCacheTotals
	decisionTokens, modelTokens                           map[string]*routerReplayAggregateTokenEntry
	availableRecipes, availableDecisions, availableModels map[string]struct{}
	availableReviewModels                                 map[string]struct{}
	reviews                                               *replayReviewAccumulator
}

func newReplayAggregateAccumulator() *replayAggregateAccumulator {
	return &replayAggregateAccumulator{
		payload: routerReplayAggregateResponse{Object: "router_replay.aggregate"},
		models:  make(map[string]int), decisions: make(map[string]int), signals: make(map[string]int),
		costs:          newReplayCostAccumulator(),
		providerCache:  newReplayProviderCacheTotals(),
		decisionTokens: make(map[string]*routerReplayAggregateTokenEntry), modelTokens: make(map[string]*routerReplayAggregateTokenEntry),
		availableRecipes: make(map[string]struct{}), availableDecisions: make(map[string]struct{}), availableModels: make(map[string]struct{}),
		availableReviewModels: make(map[string]struct{}), reviews: newReplayReviewAccumulator(),
	}
}

func (r *OpenAIRouter) queryRouterReplayAggregate(filters routerReplayFilters) (routerReplayAggregateResponse, error) {
	accumulator := newReplayAggregateAccumulator()
	search := strings.ToLower(filters.search)
	for _, reader := range r.routerReplayReaders() {
		if err := reader.ScanMetadata(func(record routerreplay.RoutingRecord) error {
			accumulator.addOptions(record)
			if doesRouterReplayRecordMatchFilters(record, filters, search) {
				accumulator.add(record)
			}
			return nil
		}); err != nil {
			return routerReplayAggregateResponse{}, err
		}
	}
	return accumulator.result(), nil
}

func (a *replayAggregateAccumulator) addOptions(record routerreplay.RoutingRecord) {
	if record.RequestKind != "" && record.SelectedModel != "" {
		a.availableReviewModels[record.SelectedModel] = struct{}{}
	}
	for _, option := range []struct {
		set   map[string]struct{}
		value string
	}{
		{a.availableRecipes, record.Recipe},
		{a.availableDecisions, record.Decision},
		{a.availableModels, record.SelectedModel},
		{a.availableModels, record.OriginalModel},
	} {
		if option.value != "" {
			option.set[option.value] = struct{}{}
		}
	}
}

func (a *replayAggregateAccumulator) add(record routerreplay.RoutingRecord) {
	part := buildRouterReplayAggregatePayload(nil, []routerreplay.RoutingRecord{record})
	a.payload.RecordCount++
	a.payload.Lifecycle.Completed += part.Lifecycle.Completed
	a.payload.Lifecycle.Failed += part.Lifecycle.Failed
	a.payload.Lifecycle.Aborted += part.Lifecycle.Aborted
	a.payload.Lifecycle.InProgress += part.Lifecycle.InProgress
	a.payload.Lifecycle.Unknown += part.Lifecycle.Unknown
	a.costs.add(record)
	a.providerCache.add(record)
	a.reviews.add(record)
	for _, value := range part.ModelSelection {
		a.models[value.Name] += value.Value
	}
	for _, value := range part.DecisionDistribution {
		a.decisions[value.Name] += value.Value
	}
	for _, value := range part.SignalDistribution {
		a.signals[value.Name] += value.Value
	}
	a.payload.TokenVolume.InputTokens += part.TokenVolume.InputTokens
	a.payload.TokenVolume.OutputTokens += part.TokenVolume.OutputTokens
	a.payload.TokenVolume.TotalTokens += part.TokenVolume.TotalTokens
	a.payload.TokenVolume.ExcludedRecordCount += part.TokenVolume.ExcludedRecordCount
	for _, value := range part.TokenBreakdown.ByDecision {
		accumulateRouterReplayTokenEntry(a.decisionTokens, value.Name, value.InputTokens, value.OutputTokens, value.TotalTokens)
	}
	for _, value := range part.TokenBreakdown.BySelectedModel {
		accumulateRouterReplayTokenEntry(a.modelTokens, value.Name, value.InputTokens, value.OutputTokens, value.TotalTokens)
	}
}

func (a *replayAggregateAccumulator) result() routerReplayAggregateResponse {
	a.payload.Summary = a.costs.result()
	a.payload.ProviderCache = a.providerCache.summary()
	a.payload.ModelSelection = sortRouterReplayAggregateValues(a.models, 10)
	a.payload.DecisionDistribution = sortRouterReplayAggregateValues(a.decisions, 0)
	a.payload.SignalDistribution = sortRouterReplayAggregateValues(a.signals, 0)
	a.payload.TokenBreakdown.ByDecision = sortRouterReplayTokenEntries(a.decisionTokens, 8)
	a.payload.TokenBreakdown.BySelectedModel = sortRouterReplayTokenEntries(a.modelTokens, 8)
	a.payload.AvailableRecipes = sortRouterReplayOptionSet(a.availableRecipes)
	a.payload.AvailableDecisions = sortRouterReplayOptionSet(a.availableDecisions)
	a.payload.AvailableModels = sortRouterReplayOptionSet(a.availableModels)
	a.payload.Reviews = a.reviews.result()
	if len(a.availableReviewModels) > 0 {
		a.payload.AvailableReviewModels = sortRouterReplayOptionSet(a.availableReviewModels)
	}
	return a.payload
}
