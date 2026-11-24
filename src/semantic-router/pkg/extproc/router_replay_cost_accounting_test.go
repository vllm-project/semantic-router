package extproc

import (
	"context"
	"encoding/json"
	"math"
	"reflect"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func costAccountingTestRouter() *OpenAIRouter {
	return &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{
		ModelConfig: map[string]config.ModelParams{
			"cheap":     {Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: 1, CompletionPer1M: 2}},
			"expensive": {Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: 4, CompletionPer1M: 8}},
			"euro":      {Pricing: config.ModelPricing{Currency: "EUR", PromptPer1M: 1, CompletionPer1M: 1}},
			// A model whose operator set no pricing, materialized with the
			// catalog's list price.
			"listed":   {Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: 2, CompletionPer1M: 4}, AuthoredModel: &config.CanonicalProviderModel{Name: "listed"}},
			"unpriced": {},
		},
	}}}
}

func withReplayBaseline(ctx *RequestContext, model string) *RequestContext {
	ctx.RouterReplayPluginConfig = &config.RouterReplayPluginConfig{Enabled: true, CostBaselineModel: model}
	return ctx
}

func TestBuildReplayUsageCostComparesWithTheConfiguredBaseline(t *testing.T) {
	router := costAccountingTestRouter()
	usage := responseUsageMetrics{promptTokens: 1000, completionTokens: 500}

	// The recipe pool's most expensive model is the selected one, so the
	// default baseline could never show a loss; the configured one does.
	got := router.buildReplayUsageCost(withReplayBaseline(replayCostContext("expensive", "cheap", "expensive"), " cheap "), usage)
	if got.BaselineModel == nil || *got.BaselineModel != "cheap" {
		t.Fatalf("baseline model = %v, want cheap", got.BaselineModel)
	}
	assertApproxFloat64(t, got.ActualCost, 0.008)
	assertApproxFloat64(t, got.BaselineCost, 0.002)
	assertApproxFloat64(t, got.CostSavings, -0.006)
	if got.CostAccounting == nil || got.CostAccounting.BaselineSource != store.BaselineSourceConfigured ||
		got.CostAccounting.PricingSource != store.PricingSourceConfigured {
		t.Fatalf("cost accounting = %+v", got.CostAccounting)
	}

	// A passthrough request takes the configured baseline as well.
	passthrough := withReplayBaseline(&RequestContext{RequestModel: "cheap"}, "expensive")
	got = router.buildReplayUsageCost(passthrough, usage)
	assertApproxFloat64(t, got.CostSavings, 0.006)
	if got.CostAccounting.BaselineSource != store.BaselineSourceConfigured {
		t.Fatalf("passthrough baseline source = %q", got.CostAccounting.BaselineSource)
	}
}

func TestBuildReplayUsageCostNeverFallsBackFromAnUnusableConfiguredBaseline(t *testing.T) {
	router := costAccountingTestRouter()
	usage := responseUsageMetrics{promptTokens: 1000, completionTokens: 500}
	for baseline, want := range map[string]string{
		"euro":     store.UnpricedCurrencyMismatch,
		"unpriced": store.UnpricedPricingNotConfigured,
	} {
		got := router.buildReplayUsageCost(withReplayBaseline(replayCostContext("cheap", "cheap", "expensive"), baseline), usage)
		if got.BaselineModel != nil || got.BaselineCost != nil || got.CostSavings != nil {
			t.Fatalf("%s: fell back to another baseline: %+v", baseline, got)
		}
		if got.ActualCost == nil || got.CostAccounting.BaselineSource != store.BaselineSourceConfigured ||
			got.CostAccounting.BaselineUnavailable != want {
			t.Fatalf("%s: cost accounting = %+v, actual %v", baseline, got.CostAccounting, got.ActualCost)
		}
	}
}

func TestBuildReplayUsageCostLabelsTheDefaultBaselineAndThePriceSource(t *testing.T) {
	router := costAccountingTestRouter()
	usage := responseUsageMetrics{promptTokens: 1000, completionTokens: 500}

	pooled := router.buildReplayUsageCost(replayCostContext("listed", "listed", "expensive"), usage)
	if pooled.CostAccounting.BaselineSource != store.BaselineSourceRecipePool ||
		pooled.CostAccounting.PricingSource != store.PricingSourceCatalogList || pooled.CostAccounting.CallerLogin {
		t.Fatalf("pooled cost accounting = %+v", pooled.CostAccounting)
	}
	direct := router.buildReplayUsageCost(&RequestContext{RequestModel: "cheap"}, usage)
	if direct.CostAccounting.BaselineSource != store.BaselineSourceSelectedModel || direct.CostSavings == nil || *direct.CostSavings != 0 {
		t.Fatalf("direct cost accounting = %+v, savings %v", direct.CostAccounting, direct.CostSavings)
	}
	foreign := router.buildReplayUsageCost(replayCostContext("cheap", "euro"), usage)
	if foreign.BaselineModel != nil || foreign.CostAccounting.BaselineUnavailable != store.UnpricedCurrencyMismatch {
		t.Fatalf("foreign-currency pool = %+v", foreign.CostAccounting)
	}
}

func TestBuildReplayUsageCostSaysWhyThereIsNoPrice(t *testing.T) {
	router := costAccountingTestRouter()
	usage := responseUsageMetrics{promptTokens: 1000, completionTokens: 500}
	unpriced := router.buildReplayUsageCost(replayCostContext("unpriced", "unpriced"), usage)
	if unpriced.ActualCost != nil || unpriced.CostAccounting.UnpricedReason != store.UnpricedPricingNotConfigured ||
		unpriced.CostAccounting.PricingSource != "" {
		t.Fatalf("unpriced cost accounting = %+v", unpriced.CostAccounting)
	}
	empty := router.buildReplayUsageCost(replayCostContext("cheap", "cheap"), responseUsageMetrics{})
	if empty.CostAccounting.UnpricedReason != store.UnpricedNoUsage {
		t.Fatalf("usage-free cost accounting = %+v", empty.CostAccounting)
	}
}

func TestLooperRouterCallsCountEveryUnpricedAttempt(t *testing.T) {
	usd, eur := 0.01, 0.5
	second := 0.02
	diagnostics := &routerreplay.LooperDiagnostics{
		Attempts: []routerreplay.LooperAttempt{
			{Model: "a", ActualCost: &usd, Currency: "usd", Usage: routerreplay.LooperUsage{PromptTokens: 10, CompletionTokens: 5}},
			{Model: "b", ActualCost: &second, Currency: "USD", Usage: routerreplay.LooperUsage{PromptTokens: 10, CompletionTokens: 5}},
			{Model: "unpriced", Usage: routerreplay.LooperUsage{PromptTokens: 10, CompletionTokens: 5}},
			{Model: "failed", Currency: "USD"},
			{Model: "partial", Currency: "USD", Usage: routerreplay.LooperUsage{PromptTokens: 10}},
			{Model: "euro", ActualCost: &eur, Currency: "EUR", Usage: routerreplay.LooperUsage{PromptTokens: 10, CompletionTokens: 5}},
		},
		DroppedAttemptCount: 2,
	}
	got := looperRouterCalls(diagnostics, looper.TokenUsage{PromptTokens: 60, CompletionTokens: 25, TotalTokens: 85}, 8)
	want := store.RouterCallCost{
		Kind: store.RouterCallKindLooper, Calls: 8, PromptTokens: 60, CompletionTokens: 25,
		Currency: "USD", UnpricedCalls: 6,
		UnpricedReasons: map[string]int{
			store.UnpricedPricingNotConfigured: 1, store.UnpricedNoUsage: 1, store.UnpricedUsageIncomplete: 1,
			store.UnpricedCurrencyMismatch: 1, store.UnpricedAttemptNotTraced: 2,
		},
	}
	if got.Cost == nil || math.Abs(*got.Cost-0.03) > 1e-12 {
		t.Fatalf("priced attempts cost %v, want 0.03", got.Cost)
	}
	got.Cost = nil
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("router calls = %+v, want %+v", got, want)
	}

	for name, trace := range map[string]*routerreplay.LooperDiagnostics{"no trace": nil, "empty trace": {Version: 1}} {
		untraced := looperRouterCalls(trace, looper.TokenUsage{PromptTokens: 3}, 3)
		if untraced.Calls != 3 || untraced.UnpricedCalls != 3 || untraced.Cost != nil ||
			untraced.UnpricedReasons[store.UnpricedAttemptNotTraced] != 3 {
			t.Fatalf("%s: untraced Looper calls counted as priced: %+v", name, untraced)
		}
	}
	if unknown := looperRouterCalls(nil, looper.TokenUsage{PromptTokens: 3}, 0); unknown.UnpricedCalls != 1 {
		t.Fatalf("a Looper that reports no calls made none: %+v", unknown)
	}
}

// costCoverageRecords covers every way a record is priced, partly priced or
// left out.
func costCoverageRecords() []routerreplay.RoutingRecord {
	completed := func(id string) routerreplay.RoutingRecord {
		return routerreplay.RoutingRecord{ID: id, Timestamp: time.Unix(int64(len(id)), 0).UTC(), LifecycleState: routerreplay.LifecycleCompleted, TotalTokens: replayIntPtr(100)}
	}
	priced := func(id string, accounting store.CostAccounting) routerreplay.RoutingRecord {
		record := completed(id)
		record.ActualCost, record.BaselineCost, record.CostSavings = replayFloat64Ptr(1), replayFloat64Ptr(3), replayFloat64Ptr(2)
		record.Currency, record.BaselineModel = replayStringPtr("USD"), replayStringPtr("base")
		record.CostAccounting = &accounting
		return record
	}
	pooled := priced("pooled", store.CostAccounting{PricingSource: store.PricingSourceConfigured, BaselineSource: store.BaselineSourceRecipePool})
	listed := priced("listed", store.CostAccounting{PricingSource: store.PricingSourceCatalogList, CallerLogin: true, BaselineSource: store.BaselineSourceConfigured})
	noBaseline := priced("no-baseline", store.CostAccounting{PricingSource: store.PricingSourceConfigured, BaselineSource: store.BaselineSourceConfigured, BaselineUnavailable: store.UnpricedCurrencyMismatch})
	noBaseline.BaselineModel, noBaseline.BaselineCost, noBaseline.CostSavings = nil, nil, nil
	looperRecord := completed("looper")
	looperRecord.CostAccounting = &store.CostAccounting{RouterCalls: []store.RouterCallCost{{
		Kind: store.RouterCallKindLooper, Calls: 3, Cost: replayFloat64Ptr(0.5), Currency: "USD",
		UnpricedCalls: 1, UnpricedReasons: map[string]int{store.UnpricedPricingNotConfigured: 1},
	}}}
	unpriced := completed("unpriced")
	// A subscription login without a price is unpriced, and still counted as
	// on the caller's login.
	unpriced.CostAccounting = &store.CostAccounting{UnpricedReason: store.UnpricedPricingNotConfigured, CallerLogin: true}
	legacy := completed("legacy")
	// A request still running and an escalation that failed after paid
	// attempts keep their known spend, but neither is compared with a
	// baseline.
	running := pooled
	running.ID, running.LifecycleState = "running", routerreplay.LifecycleInProgress
	failedLooper := looperRecord
	failedLooper.ID, failedLooper.LifecycleState = "failed-looper", routerreplay.LifecycleFailed
	failedLooper.CostAccounting = &store.CostAccounting{RouterCalls: []store.RouterCallCost{{
		Kind: store.RouterCallKindLooper, Calls: 2, Cost: replayFloat64Ptr(0.25), Currency: "USD",
	}}}
	aborted := completed("aborted")
	aborted.LifecycleState = routerreplay.LifecycleAborted
	hop := completed("hop")
	hop.TotalTokens, hop.CostAccounting = nil, &store.CostAccounting{UnpricedReason: store.UnpricedCountedInParent}
	return []routerreplay.RoutingRecord{pooled, listed, noBaseline, looperRecord, unpriced, legacy, running, failedLooper, aborted, hop}
}

func TestRouterReplayAggregateReportsCostCoverage(t *testing.T) {
	summary := buildRouterReplayAggregatePayload(nil, costCoverageRecords()).Summary
	want := routerreplay.AggregateCostCoverage{
		PricedRecords: 3, PartlyPricedRecords: 3, UnpricedRecords: 3, LooperHopRecords: 1,
		UnpricedReasons: map[string]int{
			store.UnpricedPricingNotConfigured: 2, store.UnpricedNotRecorded: 1, store.UnpricedNotCompleted: 3,
		},
		ListPriceRecords: 1, CallerLoginRecords: 2, RouterCalls: 5, UnpricedRouterCalls: 1,
		BaselineSources:     map[string]int{store.BaselineSourceRecipePool: 1, store.BaselineSourceConfigured: 1},
		BaselineUnavailable: map[string]int{store.UnpricedCurrencyMismatch: 1},
	}
	if !reflect.DeepEqual(summary.Coverage, want) {
		t.Fatalf("coverage = %+v, want %+v", summary.Coverage, want)
	}
	if summary.CostRecordCount != 2 || summary.ExcludedRecordCount != 8 || summary.Currency != "USD" ||
		summary.ActualSpend != 2 || summary.BaselineSpend != 6 || summary.TotalSaved != 4 ||
		summary.RouterCallSpend != 0.75 || summary.TotalSpend != 4.75 {
		t.Fatalf("cost summary = %+v", summary)
	}
}

func TestRouterReplayAggregateAPIStreamsTheSameCostSummary(t *testing.T) {
	records := costCoverageRecords()
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(len(records), 0))
	for _, record := range records {
		if _, err := recorder.AddRecord(record); err != nil {
			t.Fatal(err)
		}
	}
	router := &OpenAIRouter{ReplayRecorders: map[string]*routerreplay.Recorder{"route": recorder}}
	response := router.handleRouterReplayAPI(context.Background(), "GET", "/api/v1/observability/replays/aggregate")
	var got routerreplay.AggregateResponse
	if err := json.Unmarshal(response.GetImmediateResponse().Body, &got); err != nil {
		t.Fatal(err)
	}
	want := buildRouterReplayAggregatePayload(nil, records).Summary
	if !reflect.DeepEqual(got.Summary, want) {
		t.Fatalf("API summary = %+v\nwant %+v", got.Summary, want)
	}
}

// netSavingsRecords are requests compared with a baseline whose Router calls
// are priced, free, unpriced or in another currency, and a failed request
// whose Router call is spend but not part of any saving.
func netSavingsRecords() []routerreplay.RoutingRecord {
	compared := func(id string, calls ...store.RouterCallCost) routerreplay.RoutingRecord {
		record := routerreplay.RoutingRecord{ID: id, Timestamp: time.Unix(int64(len(id)), 0).UTC(), LifecycleState: routerreplay.LifecycleCompleted, TotalTokens: replayIntPtr(100)}
		record.ActualCost, record.BaselineCost, record.CostSavings = replayFloat64Ptr(1), replayFloat64Ptr(3), replayFloat64Ptr(2)
		record.Currency, record.BaselineModel = replayStringPtr("USD"), replayStringPtr("base")
		record.CostAccounting = &store.CostAccounting{PricingSource: store.PricingSourceConfigured, BaselineSource: store.BaselineSourceConfigured, RouterCalls: calls}
		return record
	}
	priced := func(kind string, calls int, cost float64, currency string) store.RouterCallCost {
		return store.RouterCallCost{Kind: kind, Calls: calls, Cost: replayFloat64Ptr(cost), Currency: currency}
	}
	failed := compared("failed", priced("decision_model", 1, 0.125, "USD"))
	failed.LifecycleState = routerreplay.LifecycleFailed
	return []routerreplay.RoutingRecord{
		compared("routed", priced("systemone", 1, 0.25, "USD"), priced("decision_model", 2, 0.5, "USD")),
		compared("free", priced("classify", 1, 0, "USD")),
		compared("unpriced", store.RouterCallCost{Kind: "systemone", Calls: 1, UnpricedCalls: 1, UnpricedReasons: map[string]int{store.UnpricedNoUsage: 1}}),
		failed,
		compared("euro-call", priced("systemone", 1, 0.25, "EUR")),
	}
}

func replayAggregateSummaryJSON(t *testing.T, records []routerreplay.RoutingRecord) map[string]interface{} {
	t.Helper()
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(len(records), 0))
	for _, record := range records {
		if _, err := recorder.AddRecord(record); err != nil {
			t.Fatal(err)
		}
	}
	router := &OpenAIRouter{ReplayRecorders: map[string]*routerreplay.Recorder{"route": recorder}}
	response := router.handleRouterReplayAPI(context.Background(), "GET", "/api/v1/observability/replays/aggregate")
	return decodeJSONBody(t, response.GetImmediateResponse().Body)["summary"].(map[string]interface{})
}

// Savings are net of the Router's own calls on the compared requests. A
// request whose Router call has no price in its currency leaves savings,
// gross and net, and is counted by why instead of being netted at zero.
func TestRouterReplayAggregateNetsRouterCallsOutOfSavings(t *testing.T) {
	records := netSavingsRecords()
	summary := replayAggregateSummaryJSON(t, records[:4])
	want := map[string]interface{}{
		"total_saved": 3.25, "gross_saved": 4.0, "netted_router_call_spend": 0.75,
		"baseline_spend": 6.0, "actual_spend": 2.0, "router_call_spend": 0.875, "total_spend": 4.875,
		"cost_record_count": 2.0, "excluded_record_count": 2.0,
	}
	for key, value := range want {
		if summary[key] != value {
			t.Errorf("summary[%q] = %v, want %v", key, summary[key], value)
		}
	}
	coverage := summary["coverage"].(map[string]interface{})
	if got, want := coverage["netted_router_calls"], map[string]interface{}{"systemone": 1.0, "decision_model": 2.0, "classify": 1.0}; !reflect.DeepEqual(got, want) {
		t.Errorf("netted_router_calls = %v, want %v", got, want)
	}
	if got, want := coverage["net_unavailable"], map[string]interface{}{store.UnpricedNoUsage: 1.0}; !reflect.DeepEqual(got, want) {
		t.Errorf("net_unavailable = %v, want %v", got, want)
	}

	// A Router call priced in another currency cannot be netted either.
	summary = replayAggregateSummaryJSON(t, records)
	coverage = summary["coverage"].(map[string]interface{})
	if got, _ := coverage["net_unavailable"].(map[string]interface{}); got[store.UnpricedCurrencyMismatch] != 1.0 {
		t.Errorf("net_unavailable = %v, want currency_mismatch 1", got)
	}
	for _, group := range summary["by_currency"].([]interface{}) {
		group := group.(map[string]interface{})
		if group["currency"] == "USD" && (group["total_saved"] != 3.25 || group["gross_saved"] != 4.0 || group["netted_router_call_spend"] != 0.75 || group["cost_record_count"] != 2.0) {
			t.Errorf("USD group = %v", group)
		}
	}
}

// A Router call that cost nothing nets at zero whatever its currency, so a
// free call in USD, the currency an empty pricing block defaults to, leaves a
// record priced in another currency compared.
func TestRouterReplayAggregateNetsAFreeRouterCallInAnyCurrency(t *testing.T) {
	record := netSavingsRecords()[1]
	record.ID, record.Currency = "euro-record", replayStringPtr("EUR")
	summary := replayAggregateSummaryJSON(t, []routerreplay.RoutingRecord{record})
	compared := false
	for _, group := range summary["by_currency"].([]interface{}) {
		group := group.(map[string]interface{})
		compared = compared || group["currency"] == "EUR" && group["cost_record_count"] == 1.0 && group["total_saved"] == 2.0
	}
	if !compared {
		t.Fatalf("by_currency = %v, want the EUR record compared with its free USD call netted at zero", summary["by_currency"])
	}
	if unavailable := summary["coverage"].(map[string]interface{})["net_unavailable"]; unavailable != nil {
		t.Fatalf("net_unavailable = %v, want none", unavailable)
	}
}
