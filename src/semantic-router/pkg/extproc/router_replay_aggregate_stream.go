package extproc

import (
	"context"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// The accumulator retains buckets, never a row list or captured bodies. It
// counts each record through the same one-record helpers as the batch
// builders, so cost, lifecycle, and usage eligibility stay in one place, and
// allocates almost nothing per record: a scan visits the whole history.
type replayAggregateAccumulator struct {
	payload                                               routerReplayAggregateResponse
	models, decisions, signals                            map[string]int
	costs                                                 *replayCostAccumulator
	providerCache                                         *replayProviderCacheTotals
	decisionTokens, modelTokens                           map[string]*routerReplayAggregateTokenEntry
	availableRecipes, availableDecisions, availableModels map[string]struct{}
	availableReviewModels                                 map[string]struct{}
	reviews                                               *replayReviewAccumulator
	decisionNames                                         map[[2]string]string
	traffic                                               *replayTrafficAccumulator
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
		decisionNames: make(map[[2]string]string),
	}
}

// queryRouterReplayAggregate scans the records the filters select. The filter
// options cover every record: a database reader that narrows the scan reads
// them in a separate grouped query, and otherwise the scan collects them.
func (r *OpenAIRouter) queryRouterReplayAggregate(ctx context.Context, filters routerReplayFilters) (routerReplayAggregateResponse, error) {
	return r.queryRouterReplayAggregateTraffic(ctx, filters, nil)
}

// queryRouterReplayAggregateTraffic is queryRouterReplayAggregate with the
// traffic series a non-nil query asks for, counted in the same scan.
func (r *OpenAIRouter) queryRouterReplayAggregateTraffic(
	ctx context.Context,
	filters routerReplayFilters,
	traffic *routerReplayTrafficQuery,
) (routerReplayAggregateResponse, error) {
	accumulator := newReplayAggregateAccumulator()
	accumulator.traffic = newReplayTrafficAccumulator(traffic)
	search := strings.ToLower(filters.search)
	storageFilters := filters.storageFilters()
	for _, reader := range r.routerReplayReaders() {
		separate, err := reader.ScanFilterOptions(ctx, storageFilters, func(record routerreplay.RoutingRecord) error {
			accumulator.addOptions(record)
			return nil
		})
		if err != nil {
			return routerReplayAggregateResponse{}, err
		}
		if err := reader.ScanMetadata(ctx, storageFilters, func(record routerreplay.RoutingRecord) error {
			if !separate {
				accumulator.addOptions(record)
			}
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
	if record.RequestKind == store.RequestKindCodexApprovalReview && record.SelectedModel != "" {
		a.availableReviewModels[record.SelectedModel] = struct{}{}
	}
	for _, option := range [...]struct {
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
	a.payload.RecordCount++
	countRouterReplayLifecycle(&a.payload.Lifecycle, record.LifecycleState)
	a.costs.add(record)
	a.providerCache.add(record)
	a.reviews.add(record)
	a.traffic.add(record)
	model := routerReplayFallbackName(record.SelectedModel)
	decision := a.decisionName(record)
	a.models[model]++
	a.decisions[decision]++
	countRouterReplaySignals(a.signals, record.Signals)
	promptTokens, completionTokens, totalTokens, hasUsage := routerReplayUsageTriplet(record)
	if !hasUsage {
		a.payload.TokenVolume.ExcludedRecordCount++
		return
	}
	a.payload.TokenVolume.InputTokens += promptTokens
	a.payload.TokenVolume.OutputTokens += completionTokens
	a.payload.TokenVolume.TotalTokens += totalTokens
	accumulateRouterReplayTokenEntry(a.decisionTokens, decision, promptTokens, completionTokens, totalTokens)
	accumulateRouterReplayTokenEntry(a.modelTokens, model, promptTokens, completionTokens, totalTokens)
}

// decisionName is routerReplayDecisionName, built once per recipe and decision.
func (a *replayAggregateAccumulator) decisionName(record routerreplay.RoutingRecord) string {
	key := [2]string{record.Recipe, record.Decision}
	name, ok := a.decisionNames[key]
	if !ok {
		name = routerReplayDecisionName(record)
		a.decisionNames[key] = name
	}
	return name
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
	a.payload.Traffic = a.traffic.result()
	return a.payload
}
