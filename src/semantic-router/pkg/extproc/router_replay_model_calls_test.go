package extproc

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

func TestReplayModelCallsPricingAndCoverage(t *testing.T) {
	r := costAccountingTestRouter()
	r.Config.ModelDeployments = map[string]config.ModelDeployment{
		"priced":   {Pricing: &config.ModelPricing{PromptPer1M: 2, CompletionPer1M: 4}},
		"unpriced": {},
	}
	measured := []modelservice.ModelCall{
		{Deployment: "priced", Kind: "decision_model", Model: "decision", HasUsage: true, InputTokens: 1000, OutputTokens: 500},
		{Deployment: "unpriced", Kind: "systemone", Model: "remote", HasUsage: true, InputTokens: 100},
		{Deployment: "priced", Kind: "embeddings", Model: "embed"},
		{Deployment: "priced", Kind: "systemone", Model: "partial", OutputTokens: 100},
	}
	ctx := &RequestContext{}
	ctx.TraceContext, ctx.ModelCallMeter = modelservice.WithRequestMeter(context.Background())
	for _, call := range measured {
		modelservice.RecordModelCall(ctx.TraceContext, call)
	}
	accounting := r.replaceReplayModelCalls(ctx, nil)
	calls := accounting.RouterCalls
	if len(calls) != 4 || calls[0].Model != "decision" {
		t.Fatalf("calls = %+v", calls)
	}
	assertApproxFloat64(t, calls[0].Cost, 0.004)
	if calls[1].UnpricedReasons[store.UnpricedPricingNotConfigured] != 1 || calls[2].UnpricedReasons[store.UnpricedNoUsage] != 1 ||
		calls[3].UnpricedReasons[store.UnpricedUsageIncomplete] != 1 {
		t.Fatalf("calls = %+v", calls)
	}
	// Refreshing at response/failure finalization replaces the meter snapshot.
	accounting = r.replaceReplayModelCalls(ctx, accounting)
	if len(accounting.RouterCalls) != 4 {
		t.Fatal("snapshot duplicated calls")
	}
	for _, state := range []string{routerreplay.LifecycleCompleted, routerreplay.LifecycleFailed} {
		summary := buildRouterReplayAggregateCostSummary([]routerreplay.RoutingRecord{{LifecycleState: state, CostAccounting: accounting}})
		if summary.RouterCallSpend != 0.004 || summary.Coverage.RouterCalls != 4 || summary.Coverage.UnpricedRouterCalls != 3 {
			t.Fatalf("%s summary = %+v", state, summary)
		}
	}
}

func TestReplayFailedDecisionRetainsSpentUsage(t *testing.T) {
	r := costAccountingTestRouter()
	r.Config.ModelDeployments = map[string]config.ModelDeployment{"primary": {Pricing: &config.ModelPricing{PromptPer1M: 1}}}
	r.ReplayRecorder = routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	ctx := &RequestContext{RequestID: "failed-decision", RouterReplayPluginConfig: &config.RouterReplayPluginConfig{Enabled: true}}
	startRequestHeaderSpan(newRequestHeaders("POST", "/v1/chat/completions"), ctx).End()
	// A model may consume tokens and return an invalid answer; no generation follows.
	modelservice.RecordModelCall(ctx.TraceContext, modelservice.ModelCall{Deployment: "primary", Kind: "decision_model", Model: "primary", HasUsage: true, InputTokens: 1000})
	r.startRouterReplay(ctx, "public", "", "failed-decision")
	r.finalizeRouterReplay(ctx, routerreplay.LifecycleFailed, "decision_failed")
	record, ok := r.ReplayRecorder.GetRecord(ctx.RouterReplayID)
	if !ok || record.CostAccounting == nil || len(record.CostAccounting.RouterCalls) != 1 {
		t.Fatalf("record = %+v", record)
	}
	assertApproxFloat64(t, record.CostAccounting.RouterCalls[0].Cost, 0.001)
	if record.ActualCost != nil || record.TotalTokens != nil {
		t.Fatal("router tokens leaked into selected-model usage")
	}
}

func TestReplayWithoutModelCallsUnchanged(t *testing.T) {
	r := costAccountingTestRouter()
	if got := r.replaceReplayModelCalls(&RequestContext{}, nil); got != nil {
		t.Fatalf("request with no calls changed: %+v", got)
	}
}

func TestReplayLooperHopModelCallsCountedInParent(t *testing.T) {
	r := costAccountingTestRouter()
	r.Config.ModelDeployments = map[string]config.ModelDeployment{"primary": {Pricing: &config.ModelPricing{PromptPer1M: 1}}}
	r.ReplayRecorder = routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	parent := &RequestContext{RequestID: "parent", RouterReplayPluginConfig: &config.RouterReplayPluginConfig{Enabled: true}}
	startRequestHeaderSpan(newRequestHeaders("POST", "/v1/chat/completions"), parent).End()
	r.startRouterReplay(parent, "public", "", "loop")
	r.writeRouterReplayUsageCost(parent, routerreplay.UsageCost{CostAccounting: &store.CostAccounting{
		RouterCalls: []store.RouterCallCost{{Kind: store.RouterCallKindLooper, Calls: 1, Cost: replayFloat64Ptr(0.002), Currency: "USD"}},
	}})
	hop := r.newRoutingSession(routing.WithHop(parent.TraceContext, routing.Hop{Decision: "loop"}), nil).ctx
	hop.RequestID, hop.RouterReplayPluginConfig = "hop", parent.RouterReplayPluginConfig
	startRequestHeaderSpan(newRequestHeaders("POST", "/v1/chat/completions"), hop).End()
	modelservice.RecordModelCall(hop.TraceContext, modelservice.ModelCall{Deployment: "primary", Kind: "decision_model", Model: "primary", HasUsage: true, InputTokens: 1000})
	modelservice.RecordModelCall(hop.TraceContext, modelservice.ModelCall{Deployment: "primary", Kind: "decision_model", Model: "primary"})
	r.startLooperInternalReplay(hop, "cheap", "loop")
	r.writeRouterReplayUsageCost(hop, routerreplay.UsageCost{
		ActualCost: replayFloat64Ptr(0.002), Currency: replayStringPtr("USD"),
		CostAccounting: &store.CostAccounting{UnpricedReason: store.UnpricedCountedInParent},
	})
	r.finalizeRouterReplay(hop, routerreplay.LifecycleCompleted, "")
	r.finalizeRouterReplay(parent, routerreplay.LifecycleCompleted, "")
	record, ok := r.ReplayRecorder.GetRecord(parent.RouterReplayID)
	if !ok || record.CostAccounting == nil || len(record.CostAccounting.RouterCalls) != 3 {
		t.Fatalf("parent accounting = %+v", record.CostAccounting)
	}
	assertApproxFloat64(t, record.CostAccounting.RouterCalls[1].Cost, 0.001)
	summary := buildRouterReplayAggregateCostSummary(r.ReplayRecorder.ListAllRecords())
	if summary.RouterCallSpend != 0.003 || summary.TotalSpend != 0.003 || summary.Coverage.RouterCalls != 3 ||
		summary.Coverage.UnpricedRouterCalls != 1 || summary.Coverage.LooperHopRecords != 1 || summary.Coverage.PartlyPricedRecords != 1 {
		t.Fatalf("summary = %+v", summary)
	}
}
