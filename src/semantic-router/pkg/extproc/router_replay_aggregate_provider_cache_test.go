package extproc

import (
	"math"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func providerCacheAggregateRecord(decision, model, currency string, noCache, savings float64) routerreplay.RoutingRecord {
	return routerreplay.RoutingRecord{
		Recipe: "agents", Decision: decision, SelectedModel: model,
		LifecycleState: routerreplay.LifecycleCompleted, Currency: replayStringPtr(currency),
		ProviderCache: &store.ProviderCacheAccounting{
			State: "complete", NoCacheCost: replayFloat64Ptr(noCache), CacheSavings: replayFloat64Ptr(savings),
		},
	}
}

func assertProviderCacheAmount(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("%s=%v, want %v", name, got, want)
	}
}

func TestRouterReplayProviderCacheSummaryNetsWithoutClamping(t *testing.T) {
	reads := providerCacheAggregateRecord("reads", "model-a", "USD", .00022, .00009)
	writes := providerCacheAggregateRecord("writes", "model-a", "usd ", .00022, -.00035)
	cacheHit := providerCacheAggregateRecord("reads", "model-a", "USD", .00022, .00022)
	cacheHit.FromCache = true
	aborted := providerCacheAggregateRecord("reads", "model-a", "USD", .00022, .00009)
	aborted.LifecycleState = routerreplay.LifecycleAborted
	partial := providerCacheAggregateRecord("reads", "model-a", "USD", .00022, .00009)
	partial.ProviderCache.State = "partial"
	unpriced := providerCacheAggregateRecord("reads", "model-a", "USD", .00022, .00009)
	unpriced.ProviderCache.NoCacheCost = nil
	noCurrency := providerCacheAggregateRecord("reads", "model-a", "USD", .00022, .00009)
	noCurrency.Currency = nil
	historical := routerreplay.RoutingRecord{LifecycleState: routerreplay.LifecycleCompleted, Currency: replayStringPtr("USD")}

	got := buildRouterReplayProviderCacheSummary([]routerreplay.RoutingRecord{
		reads, writes, cacheHit, aborted, partial, unpriced, noCurrency, historical,
	})
	if got.PricedRecords != 2 || got.ExcludedRecords != 6 || got.Currency != "USD" || len(got.ByCurrency) != 1 {
		t.Fatalf("settled records were miscounted: %+v", got)
	}
	// A per-request max(0, saving) would report +0.00009 here.
	assertProviderCacheAmount(t, "no_cache_spend", got.NoCacheSpend, .00044)
	assertProviderCacheAmount(t, "actual_spend", got.ActualSpend, .0007)
	assertProviderCacheAmount(t, "net_savings", got.NetSavings, -.00026)
	if len(got.ByDecision) != 2 || got.ByDecision[0].Name != "agents::writes" || got.ByDecision[1].Name != "agents::reads" {
		t.Fatalf("decision split = %+v", got.ByDecision)
	}
	assertProviderCacheAmount(t, "writes net", got.ByDecision[0].NetSavings, -.00035)
	assertProviderCacheAmount(t, "reads net", got.ByDecision[1].NetSavings, .00009)
	if len(got.BySelectedModel) != 1 || got.BySelectedModel[0].Name != "model-a" || got.BySelectedModel[0].PricedRecords != 2 {
		t.Fatalf("model split = %+v", got.BySelectedModel)
	}
	assertProviderCacheAmount(t, "model net", got.BySelectedModel[0].NetSavings, -.00026)
}

func TestRouterReplayProviderCacheSummaryKeepsCurrenciesApart(t *testing.T) {
	got := buildRouterReplayProviderCacheSummary([]routerreplay.RoutingRecord{
		providerCacheAggregateRecord("reads", "model-a", "USD", 2, 1),
		providerCacheAggregateRecord("reads", "model-a", "EUR", 4, -1),
	})
	if got.Currency != "" || got.NoCacheSpend != 0 || got.ActualSpend != 0 || got.NetSavings != 0 || got.PricedRecords != 2 {
		t.Fatalf("mixed currencies were summed: %+v", got)
	}
	if len(got.ByCurrency) != 2 || got.ByCurrency[0].Currency != "EUR" || got.ByCurrency[1].Currency != "USD" {
		t.Fatalf("currency groups = %+v", got.ByCurrency)
	}
	assertProviderCacheAmount(t, "EUR actual", got.ByCurrency[0].ActualSpend, 5)
	assertProviderCacheAmount(t, "EUR net", got.ByCurrency[0].NetSavings, -1)
	assertProviderCacheAmount(t, "USD net", got.ByCurrency[1].NetSavings, 1)
	if len(got.ByDecision) != 2 || got.ByDecision[0].Currency != "EUR" || got.ByDecision[1].Currency != "USD" {
		t.Fatalf("decision split mixed currencies: %+v", got.ByDecision)
	}
}

func TestHandleRouterReplayAggregateAPIReturnsProviderCacheSummary(t *testing.T) {
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	for i, record := range []routerreplay.RoutingRecord{
		providerCacheAggregateRecord("reads", "model-a", "USD", .00022, .00009),
		providerCacheAggregateRecord("writes", "model-b", "USD", .00022, -.00035),
		{LifecycleState: routerreplay.LifecycleCompleted},
	} {
		record.ID = string(rune('a' + i))
		record.Timestamp = time.Unix(int64(i+1), 0).UTC()
		if _, err := recorder.AddRecord(record); err != nil {
			t.Fatal(err)
		}
	}
	router := &OpenAIRouter{ReplayRecorders: map[string]*routerreplay.Recorder{"agents": recorder}}
	response := router.handleRouterReplayAPI("GET", routerReplayAggregatePath+"?model=model-b")
	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	summary := body["provider_cache"].(map[string]interface{})
	if summary["currency"] != "USD" || summary["priced_records"] != float64(1) || summary["excluded_records"] != float64(0) {
		t.Fatalf("filtered provider cache summary = %#v", summary)
	}
	assertProviderCacheAmount(t, "net_savings", summary["net_savings"].(float64), -.00035)
	assertProviderCacheAmount(t, "actual_spend", summary["actual_spend"].(float64), .00057)
	byDecision := summary["by_decision"].([]interface{})
	if len(byDecision) != 1 || byDecision[0].(map[string]interface{})["name"] != "agents::writes" {
		t.Fatalf("decision split = %#v", byDecision)
	}
	if _, ok := summary["by_selected_model"].([]interface{}); !ok {
		t.Fatalf("model split missing: %#v", summary)
	}
}
