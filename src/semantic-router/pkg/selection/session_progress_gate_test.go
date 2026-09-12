package selection

import (
	"testing"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestSwitchGateDowngradeRequiresComparableEvidence(t *testing.T) {
	sel := NewSessionAwareSelector(nil)
	a := modelParamsWithTestQuality(0.9)
	b := modelParamsWithTestQuality(0.6)
	sel.InitializeFromConfig(map[string]config.ModelParams{"a": a, "b": b})
	ctx := &SelectionContext{CandidateModels: []config.ModelRef{{Model: "a"}, {Model: "b"}}}
	if !sel.IsDowngrade(ctx, "a", "b") || sel.IsDowngrade(ctx, "b", "a") {
		t.Fatal("same-index quality direction was not used")
	}
	ctx.CandidateModels[1].ReasoningEffort = "high"
	if sel.IsDowngrade(ctx, "a", "b") {
		t.Fatal("missing exact-effort evidence must not borrow the preferred score")
	}
	low := 40.0
	b.IndexResultsByEffort = map[string]map[string]modelcatalog.IndexResult{
		"high": {testIntelligenceIndex: {Index: testIntelligenceIndex, Status: "available", Score: &low}},
	}
	sel.InitializeFromConfig(map[string]config.ModelParams{"a": a, "b": b})
	if !sel.IsDowngrade(ctx, "a", "b") {
		t.Fatal("exact effort evidence was ignored")
	}
	b.QualityIndex = "different/index@1.0.0"
	sel.InitializeFromConfig(map[string]config.ModelParams{"a": a, "b": b})
	if sel.IsDowngrade(ctx, "a", "b") {
		t.Fatal("unrelated indexes are not comparable")
	}
}

func TestMultiFactorCandidateEligibilityDoesNotUseFallback(t *testing.T) {
	cfg := DefaultMultiFactorConfig()
	cfg.SLO.MaxCostPer1M = 5
	cfg.OnNoCandidates = "cheapest"
	selector := NewMultiFactorSelector(cfg)
	selector.InitializeFromConfig(map[string]config.ModelParams{
		"expensive":  {Pricing: config.ModelPricing{PromptPer1M: 20, CompletionPer1M: 20}},
		"affordable": {Pricing: config.ModelPricing{PromptPer1M: 1, CompletionPer1M: 1}},
	})
	ctx := &SelectionContext{CandidateModels: candidates("expensive", "affordable")}
	if selector.CandidateEligible(ctx, "expensive") || !selector.CandidateEligible(ctx, "affordable") {
		t.Fatal("override eligibility ignored the configured SLO")
	}
	ctx.CandidateModels = candidates("expensive")
	if selector.CandidateEligible(ctx, "expensive") {
		t.Fatal("on_no_candidates fallback cannot authorize a disqualified hold")
	}
}

func TestProgressEvidenceMetricAvailability(t *testing.T) {
	window := []TurnOutcomeFact{
		{Category: OutcomeProgress, ModelAttributable: true, Confidence: 0.4, ConfidenceKnown: true, Cost: 0.2, CostKnown: true, LatencyMs: 100, LatencyKnown: true},
		{Category: OutcomeProgress, ModelAttributable: true, OutputTokens: 999},
		{Category: OutcomeProgress, ModelAttributable: true, Confidence: 0.8, ConfidenceKnown: true, Cost: 0.1, CostKnown: true, LatencyMs: 50, LatencyKnown: true},
	}
	got := EvaluateProgressEvidence(window)
	if !got.ConfidenceTrendKnown || got.ConfidenceTrend != 1 || !got.CostTrendKnown || got.CostTrend != -0.5 || !got.LatencyTrendKnown || got.LatencyTrend != -0.5 {
		t.Fatalf("known metrics: %+v", got)
	}
	missing := EvaluateProgressEvidence(window[1:2])
	if missing.CostTrendKnown || missing.ConfidenceTrendKnown || missing.LatencyTrendKnown {
		t.Fatalf("missing metrics invented a trend: %+v", missing)
	}
	cfg := enforceConfig()
	cfg.CalibrationID = "fixture/gate@1.0.0"
	if got := EvaluateSwitchGate(cfg, SwitchGateInput{}); got.CalibrationID != cfg.CalibrationID {
		t.Fatalf("calibration identity lost: %+v", got)
	}
}
