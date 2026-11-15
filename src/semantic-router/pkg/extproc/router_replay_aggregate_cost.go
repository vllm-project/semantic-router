package extproc

import (
	"math"
	"slices"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// baselineSourceUnrecorded labels a record compared with a baseline before
// records said how the baseline was chosen.
const baselineSourceUnrecorded = "unrecorded"

// replayCostAccumulator builds the aggregate cost summary one record at a
// time, so the list builder and the streaming scan share one definition of
// what is priced, what is compared with a baseline, and what is left out.
type replayCostAccumulator struct {
	summary    routerreplay.AggregateCostSummary
	currencies map[string]*routerreplay.CurrencyCostSummary
}

func newReplayCostAccumulator() *replayCostAccumulator {
	return &replayCostAccumulator{currencies: make(map[string]*routerreplay.CurrencyCostSummary)}
}

func buildRouterReplayAggregateCostSummary(records []routerreplay.RoutingRecord) routerReplayAggregateCostSummary {
	accumulator := newReplayCostAccumulator()
	for _, record := range records {
		accumulator.add(record)
	}
	return accumulator.result()
}

func (a *replayCostAccumulator) add(record routerreplay.RoutingRecord) {
	a.summary.ExcludedRecordCount++
	if record.CostAccounting != nil && record.CostAccounting.UnpricedReason == store.UnpricedCountedInParent {
		a.summary.Coverage.LooperHopRecords++
		return
	}
	if record.CostAccounting != nil && record.CostAccounting.CallerLogin {
		a.summary.Coverage.CallerLoginRecords++
	}
	// Spend counts whatever the lifecycle: a request that failed after paid
	// calls, such as an escalation whose last attempt failed, paid for them.
	modelPriced := a.addModelCost(record)
	routerPriced, routerGap := a.addRouterCalls(record)
	if record.LifecycleState != routerreplay.LifecycleCompleted {
		// Its cost may still miss calls, and only a completed request is
		// compared with a baseline.
		a.countRecord(modelPriced || routerPriced, false, store.UnpricedNotCompleted)
		return
	}
	a.addSavings(record, modelPriced)
	gap := ""
	if !modelPriced && !isLooperReplayRecord(record) {
		gap = replayModelUnpricedReason(record)
	}
	if gap == "" {
		gap = routerGap
	}
	a.countRecord(modelPriced || routerPriced, gap == "", gap)
}

// addModelCost totals the selected model's priced cost and reports whether
// the record has one.
func (a *replayCostAccumulator) addModelCost(record routerreplay.RoutingRecord) bool {
	currency := replayRecordCurrency(record)
	if currency == "" || !finiteReplayCost(record.ActualCost) {
		return false
	}
	a.currency(currency).TotalSpend += *record.ActualCost
	if record.CostAccounting != nil && record.CostAccounting.PricingSource == store.PricingSourceCatalogList {
		a.summary.Coverage.ListPriceRecords++
	}
	return true
}

// addRouterCalls totals the priced calls the Router made itself and returns
// whether any was priced and the reason of the first one that was not.
func (a *replayCostAccumulator) addRouterCalls(record routerreplay.RoutingRecord) (bool, string) {
	if record.CostAccounting == nil {
		return false, ""
	}
	priced, gap := false, ""
	for _, call := range record.CostAccounting.RouterCalls {
		a.summary.Coverage.RouterCalls += call.Calls
		a.summary.Coverage.UnpricedRouterCalls += call.UnpricedCalls
		currency := normalizeReplayCurrency(call.Currency)
		if currency != "" && finiteReplayCost(call.Cost) {
			group := a.currency(currency)
			group.RouterCallSpend += *call.Cost
			group.TotalSpend += *call.Cost
			priced = true
		}
		if gap == "" && call.UnpricedCalls > 0 {
			gap = firstReplayReason(call.UnpricedReasons)
		}
	}
	return priced, gap
}

// addSavings compares a record with its baseline when both are complete.
func (a *replayCostAccumulator) addSavings(record routerreplay.RoutingRecord, modelPriced bool) {
	coverage := &a.summary.Coverage
	if !modelPriced {
		return
	}
	if !replaySavingsComplete(record) {
		reason := store.UnpricedNotRecorded
		if record.CostAccounting != nil && record.CostAccounting.BaselineUnavailable != "" {
			reason = record.CostAccounting.BaselineUnavailable
		}
		coverage.BaselineUnavailable = incrementReplayCount(coverage.BaselineUnavailable, reason)
		return
	}
	group := a.currency(replayRecordCurrency(record))
	group.TotalSaved += *record.CostSavings
	group.BaselineSpend += *record.BaselineCost
	group.ActualSpend += *record.ActualCost
	group.CostRecordCount++
	a.summary.CostRecordCount++
	a.summary.ExcludedRecordCount--
	source := baselineSourceUnrecorded
	if record.CostAccounting != nil && record.CostAccounting.BaselineSource != "" {
		source = record.CostAccounting.BaselineSource
	}
	coverage.BaselineSources = incrementReplayCount(coverage.BaselineSources, source)
}

func replaySavingsComplete(record routerreplay.RoutingRecord) bool {
	return record.TotalTokens != nil && record.BaselineModel != nil && *record.BaselineModel != "" &&
		finiteReplayCost(record.BaselineCost) && finiteReplayCost(record.CostSavings)
}

func (a *replayCostAccumulator) countRecord(anyPriced, complete bool, gap string) {
	coverage := &a.summary.Coverage
	switch {
	case anyPriced && complete:
		coverage.PricedRecords++
		return
	case anyPriced:
		coverage.PartlyPricedRecords++
	default:
		coverage.UnpricedRecords++
	}
	if gap == "" {
		gap = store.UnpricedNotRecorded
	}
	coverage.UnpricedReasons = incrementReplayCount(coverage.UnpricedReasons, gap)
}

func (a *replayCostAccumulator) currency(code string) *routerreplay.CurrencyCostSummary {
	group := a.currencies[code]
	if group == nil {
		group = &routerreplay.CurrencyCostSummary{Currency: code}
		a.currencies[code] = group
	}
	return group
}

func (a *replayCostAccumulator) result() routerReplayAggregateCostSummary {
	summary := a.summary
	summary.ByCurrency = nil
	for _, group := range a.currencies {
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
		summary.RouterCallSpend = group.RouterCallSpend
		summary.TotalSpend = group.TotalSpend
	}
	return summary
}

// isLooperReplayRecord reports a Looper request, whose cost is its attempts'
// (a router call) rather than one selected model call.
func isLooperReplayRecord(record routerreplay.RoutingRecord) bool {
	return record.CostAccounting != nil && slices.ContainsFunc(record.CostAccounting.RouterCalls,
		func(call store.RouterCallCost) bool { return call.Kind == store.RouterCallKindLooper })
}

// replayModelUnpricedReason says why a completed record has no model cost.
func replayModelUnpricedReason(record routerreplay.RoutingRecord) string {
	switch {
	case record.CostAccounting != nil && record.CostAccounting.UnpricedReason != "":
		return record.CostAccounting.UnpricedReason
	case record.TotalTokens == nil:
		return store.UnpricedNoUsage
	default:
		return store.UnpricedNotRecorded
	}
}

func replayRecordCurrency(record routerreplay.RoutingRecord) string {
	if record.Currency == nil {
		return ""
	}
	return normalizeReplayCurrency(*record.Currency)
}

func firstReplayReason(reasons map[string]int) string {
	keys := make([]string, 0, len(reasons))
	for reason, count := range reasons {
		if count > 0 {
			keys = append(keys, reason)
		}
	}
	if len(keys) == 0 {
		return store.UnpricedNotRecorded
	}
	sort.Strings(keys)
	return keys[0]
}

func incrementReplayCount(counts map[string]int, key string) map[string]int {
	if counts == nil {
		counts = make(map[string]int)
	}
	counts[key]++
	return counts
}

func finiteReplayCost(value *float64) bool {
	return value != nil && !math.IsNaN(*value) && !math.IsInf(*value, 0)
}
