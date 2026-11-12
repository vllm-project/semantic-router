package extproc

import (
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

type providerCacheSpendKey struct{ name, currency string }

// replayProviderCacheTotals sums settled records in visit order, so the
// streaming and slice aggregates produce identical floating-point totals.
type replayProviderCacheTotals struct {
	currencies        map[string]*routerreplay.ProviderCacheCurrencyTotal
	decisions, models map[providerCacheSpendKey]*routerreplay.ProviderCacheSpendEntry
	priced, excluded  int
}

func newReplayProviderCacheTotals() *replayProviderCacheTotals {
	return &replayProviderCacheTotals{
		currencies: make(map[string]*routerreplay.ProviderCacheCurrencyTotal),
		decisions:  make(map[providerCacheSpendKey]*routerreplay.ProviderCacheSpendEntry),
		models:     make(map[providerCacheSpendKey]*routerreplay.ProviderCacheSpendEntry),
	}
}

func buildRouterReplayProviderCacheSummary(
	records []routerreplay.RoutingRecord,
) routerreplay.AggregateProviderCacheSummary {
	totals := newReplayProviderCacheTotals()
	for _, record := range records {
		totals.add(record)
	}
	return totals.summary()
}

func (t *replayProviderCacheTotals) add(record routerreplay.RoutingRecord) {
	currency, noCache, actual, ok := settledProviderCacheSpend(record)
	if !ok {
		t.excluded++
		return
	}
	t.priced++
	group := t.currencies[currency]
	if group == nil {
		group = &routerreplay.ProviderCacheCurrencyTotal{Currency: currency}
		t.currencies[currency] = group
	}
	group.NoCacheSpend += noCache
	group.ActualSpend += actual
	group.PricedRecords++
	decision := config.RoutingDecisionKey(config.RecipeName(record.Recipe), record.Decision)
	addProviderCacheSpend(t.decisions, providerCacheSpendKey{routerReplayFallbackName(decision), currency}, noCache, actual)
	addProviderCacheSpend(t.models, providerCacheSpendKey{routerReplayFallbackName(record.SelectedModel), currency}, noCache, actual)
}

// settledProviderCacheSpend returns a record's same-model no-cache and actual
// provider cost from the prices captured when it settled. A Router cache hit
// made no provider request, and partial or unpriced usage has no
// counterfactual.
func settledProviderCacheSpend(record routerreplay.RoutingRecord) (string, float64, float64, bool) {
	accounting := record.ProviderCache
	if record.LifecycleState != routerreplay.LifecycleCompleted || record.FromCache ||
		accounting == nil || accounting.State != "complete" ||
		!finiteReplayCost(accounting.NoCacheCost) || !finiteReplayCost(accounting.CacheSavings) ||
		record.Currency == nil || normalizeReplayCurrency(*record.Currency) == "" {
		return "", 0, 0, false
	}
	noCache := *accounting.NoCacheCost
	return normalizeReplayCurrency(*record.Currency), noCache, noCache - *accounting.CacheSavings, true
}

func addProviderCacheSpend(
	entries map[providerCacheSpendKey]*routerreplay.ProviderCacheSpendEntry,
	key providerCacheSpendKey,
	noCache, actual float64,
) {
	entry := entries[key]
	if entry == nil {
		entry = &routerreplay.ProviderCacheSpendEntry{Name: key.name, Currency: key.currency}
		entries[key] = entry
	}
	entry.NoCacheSpend += noCache
	entry.ActualSpend += actual
	entry.PricedRecords++
}

func (t *replayProviderCacheTotals) summary() routerreplay.AggregateProviderCacheSummary {
	summary := routerreplay.AggregateProviderCacheSummary{
		PricedRecords:   t.priced,
		ExcludedRecords: t.excluded,
		ByDecision:      sortProviderCacheSpend(t.decisions),
		BySelectedModel: sortProviderCacheSpend(t.models),
	}
	for _, group := range t.currencies {
		total := *group
		total.NetSavings = total.NoCacheSpend - total.ActualSpend
		summary.ByCurrency = append(summary.ByCurrency, total)
	}
	sort.Slice(summary.ByCurrency, func(i, j int) bool {
		return summary.ByCurrency[i].Currency < summary.ByCurrency[j].Currency
	})
	if len(summary.ByCurrency) == 1 {
		only := summary.ByCurrency[0]
		summary.Currency = only.Currency
		summary.NoCacheSpend = only.NoCacheSpend
		summary.ActualSpend = only.ActualSpend
		summary.NetSavings = only.NetSavings
	}
	return summary
}

// sortProviderCacheSpend keeps every entry so each currency's split adds up to
// its total. Entries are grouped by currency, largest actual spend first.
func sortProviderCacheSpend(
	entries map[providerCacheSpendKey]*routerreplay.ProviderCacheSpendEntry,
) []routerreplay.ProviderCacheSpendEntry {
	values := make([]routerreplay.ProviderCacheSpendEntry, 0, len(entries))
	for _, entry := range entries {
		value := *entry
		value.NetSavings = value.NoCacheSpend - value.ActualSpend
		values = append(values, value)
	}
	sort.Slice(values, func(i, j int) bool {
		if values[i].Currency != values[j].Currency {
			return values[i].Currency < values[j].Currency
		}
		if values[i].ActualSpend != values[j].ActualSpend {
			return values[i].ActualSpend > values[j].ActualSpend
		}
		return values[i].Name < values[j].Name
	})
	return values
}
