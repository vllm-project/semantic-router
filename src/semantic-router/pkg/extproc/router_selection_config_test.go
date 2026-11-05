package extproc

import (
	"context"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestBuildModelSelectionConfigPreservesSelectorDefaultsWhenUnset(t *testing.T) {
	defaults := config.DefaultGlobalConfig()
	got := buildModelSelectionConfig(&defaults)

	defaultRouterDC := selection.DefaultRouterDCConfig()
	if !reflect.DeepEqual(got.RouterDC, defaultRouterDC) {
		t.Fatalf("RouterDC default drift:\n got: %+v\nwant: %+v", got.RouterDC, defaultRouterDC)
	}

	defaultHybrid := selection.DefaultHybridConfig()
	if !reflect.DeepEqual(got.Hybrid, defaultHybrid) {
		t.Fatalf("Hybrid default drift:\n got: %+v\nwant: %+v", got.Hybrid, defaultHybrid)
	}
}

func TestBuildModelSelectionConfigPreservesExplicitFalseOverrides(t *testing.T) {
	cfg := config.DefaultGlobalConfig()
	cfg.IntelligentRouting.ModelSelection.RouterDC.UseCapabilities = false
	cfg.IntelligentRouting.ModelSelection.Hybrid.NormalizeScores = false

	got := buildModelSelectionConfig(&cfg)

	if got.RouterDC == nil {
		t.Fatal("expected RouterDC config to be built")
	}
	if got.RouterDC.UseCapabilities {
		t.Fatalf("expected explicit router_dc.use_capabilities=false to be preserved, got %+v", got.RouterDC)
	}
	if got.RouterDC.Temperature != selection.DefaultRouterDCConfig().Temperature {
		t.Fatalf("expected router_dc defaults to survive explicit false override, got %+v", got.RouterDC)
	}

	if got.Hybrid == nil {
		t.Fatal("expected Hybrid config to be built")
	}
	if got.Hybrid.NormalizeScores {
		t.Fatalf("expected explicit hybrid.normalize_scores=false to be preserved, got %+v", got.Hybrid)
	}
	if got.Hybrid.RouterDCWeight != selection.DefaultHybridConfig().RouterDCWeight {
		t.Fatalf("expected hybrid defaults to survive explicit false override, got %+v", got.Hybrid)
	}
}

func TestBuildHybridSelectionConfigMergesDecisionOverrides(t *testing.T) {
	cfg := config.DefaultGlobalConfig()

	got := buildHybridSelectionConfig(&cfg, &config.HybridSelectionConfig{
		ExperienceWeight: 0.6,
		RouterDCWeight:   0.4,
	})

	if got == nil {
		t.Fatal("expected Hybrid config to be built")
	}
	if got.ExperienceWeight != 0.6 || got.RouterDCWeight != 0.4 {
		t.Fatalf("expected decision-scoped hybrid weights, got %+v", got)
	}
	if got.AutoMixWeight != selection.DefaultHybridConfig().AutoMixWeight || !got.NormalizeScores {
		t.Fatalf("expected decision-scoped hybrid config to preserve remaining defaults, got %+v", got)
	}
}

// TestHybridCostAdjustmentPreservesResolvedGlobalWeight drives the real
// selection path: buildSelectionContext fills the request-level default
// weights (0.5 when no decision weight is configured), while
// newDecisionHybridSelector resolves the effective hybrid configuration from
// the global block plus any decision override. The cost adjustment must
// consume the resolved configuration — the request default must neither
// replace a configured global weight nor re-enable a disabled one.
func TestHybridCostAdjustmentPreservesResolvedGlobalWeight(t *testing.T) {
	newRouter := func(globalCostWeight float64) *OpenAIRouter {
		return &OpenAIRouter{Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				ModelSelection: config.ModelSelectionConfig{
					Hybrid: config.HybridSelectionConfig{CostWeight: globalCostWeight},
				},
			},
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"model-cheap": {Pricing: config.ModelPricing{PromptPer1M: 1.0}},
					"model-dear":  {Pricing: config.ModelPricing{PromptPer1M: 2.0}},
				},
			},
		}}
	}
	candidates := []config.ModelRef{{Model: "model-cheap"}, {Model: "model-dear"}}

	// With global cost_weight=0.2 the cheap candidate earns the configured
	// bonus on its 0.5 blended base: 0.5 * (1 + 0.2) = 0.6, not the request
	// default's 0.75.
	weighted := newRouter(0.2)
	selCtx := weighted.buildSelectionContext(candidates, "pick", "solve", nil, "", nil, nil)
	result, err := weighted.newDecisionHybridSelector(nil, nil).Select(context.Background(), selCtx)
	if err != nil {
		t.Fatalf("hybrid selection returned error: %v", err)
	}
	if result.SelectedModel != "model-cheap" {
		t.Fatalf("expected the cheap model to win the configured cost bonus, got %q", result.SelectedModel)
	}
	if result.Score != 0.6 {
		t.Fatalf("cheap candidate score = %v, want 0.6", result.Score)
	}

	// With global cost_weight=0 the adjustment stays disabled: the request
	// default of 0.5 must not re-enable cost weighting.
	disabled := newRouter(0)
	selCtx = disabled.buildSelectionContext(candidates, "pick", "solve", nil, "", nil, nil)
	result, err = disabled.newDecisionHybridSelector(nil, nil).Select(context.Background(), selCtx)
	if err != nil {
		t.Fatalf("hybrid selection returned error: %v", err)
	}
	if result.Score != 0.5 {
		t.Fatalf("score with cost weighting disabled = %v, want 0.5", result.Score)
	}

	// An explicit decision-level cost_weight still reaches the adjustment
	// through the resolved configuration: 0.5 * (1 + 0.9) = 0.95.
	decisionOverride := &config.HybridSelectionConfig{CostWeight: 0.9}
	overridden := newRouter(0)
	selCtx = overridden.buildSelectionContext(candidates, "pick", "solve", nil, "", nil, nil)
	result, err = overridden.newDecisionHybridSelector(decisionOverride, nil).Select(context.Background(), selCtx)
	if err != nil {
		t.Fatalf("hybrid selection returned error: %v", err)
	}
	if result.SelectedModel != "model-cheap" {
		t.Fatalf("expected the cheap model to win the decision cost bonus, got %q", result.SelectedModel)
	}
	if result.Score != 0.95 {
		t.Fatalf("cheap candidate score with decision override = %v, want 0.95", result.Score)
	}
}

func TestBuildMultiFactorSelectionConfigCopiesQualityEvidencePolicy(t *testing.T) {
	minimumScore := 40.0
	got := buildMultiFactorSelectionConfig(&config.MultiFactorSelectionConfig{
		LatencyMetric: "ttft",
		Objective: &config.MultiFactorObjectiveConfig{
			Strategy: config.MultiFactorObjectiveLexicographic,
			Priorities: []config.MultiFactorPriorityConfig{
				{Factor: config.MultiFactorFactorQuality, Tolerance: 0.02},
				{Factor: config.MultiFactorFactorCost},
			},
		},
		Quality: &config.QualityEvidenceConfig{
			Index:       "vllm-sr/agentic@1.0.0",
			OnMissing:   "exclude",
			MinCoverage: 0.8,
			MinScore:    &minimumScore,
		},
	})
	assertString(t, got.QualityIndex, "vllm-sr/agentic@1.0.0", "multi_factor.quality.index")
	assertString(t, got.LatencyMetric, "ttft", "multi_factor.latency_metric")
	assertString(t, got.QualityOnMissing, "exclude", "multi_factor.quality.on_missing")
	assertFloat(t, got.QualityMinCoverage, 0.8, "multi_factor.quality.min_coverage")
	if got.QualityMinScore == nil {
		t.Fatal("multi_factor.quality.min_score was not copied")
	}
	assertFloat(t, *got.QualityMinScore, minimumScore, "multi_factor.quality.min_score")
	assertString(t, got.Objective.Strategy, config.MultiFactorObjectiveLexicographic, "multi_factor.objective.strategy")
	if len(got.Objective.Priorities) != 2 {
		t.Fatalf("multi_factor.objective.priorities = %#v, want two entries", got.Objective.Priorities)
	}
	assertString(t, got.Objective.Priorities[0].Factor, config.MultiFactorFactorQuality, "multi_factor.objective.priorities[0].factor")
	assertFloat(t, got.Objective.Priorities[0].Tolerance, 0.02, "multi_factor.objective.priorities[0].tolerance")

	strict := buildMultiFactorSelectionConfig(&config.MultiFactorSelectionConfig{
		Quality: &config.QualityEvidenceConfig{Index: "vllm-sr/intelligence@1.0.0"},
	})
	assertString(t, strict.QualityOnMissing, "exclude", "multi_factor.quality.on_missing default")
}

func TestBuildModelSelectionConfigDoesNotPromoteDecisionMultiFactorConfig(t *testing.T) {
	got := buildModelSelectionConfig(&config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name: "weighted-latency-router",
					Algorithm: &config.AlgorithmConfig{
						Type: string(selection.MethodMultiFactor),
						MultiFactor: &config.MultiFactorSelectionConfig{
							Weights: &config.MultiFactorWeightsConfig{
								Quality: 0.7,
								Latency: 0.2,
								Cost:    0.1,
								Load:    0.0,
							},
							SLO: &config.MultiFactorSLOConfig{
								MaxTPOTMs:    85,
								MaxTTFTMs:    550,
								MaxCostPer1M: 2.4,
								MaxInflight:  32,
							},
							LatencyPercentile: 90,
							OnNoCandidates:    "fail",
						},
					},
				},
			},
		},
	})

	if got.MultiFactor == nil {
		t.Fatal("expected MultiFactor config to be built")
	}
	defaults := selection.DefaultMultiFactorConfig()
	assertFloat(t, got.MultiFactor.Weights.Quality, defaults.Weights.Quality, "multi_factor.weights.quality")
	assertFloat(t, got.MultiFactor.Weights.Latency, defaults.Weights.Latency, "multi_factor.weights.latency")
	assertFloat(t, got.MultiFactor.Weights.Cost, defaults.Weights.Cost, "multi_factor.weights.cost")
	assertFloat(t, got.MultiFactor.Weights.Load, defaults.Weights.Load, "multi_factor.weights.load")
	assertFloat(t, got.MultiFactor.SLO.MaxTPOTMs, 0, "multi_factor.slo.max_tpot_ms")
	assertFloat(t, got.MultiFactor.SLO.MaxTTFTMs, 0, "multi_factor.slo.max_ttft_ms")
	assertFloat(t, got.MultiFactor.SLO.MaxCostPer1M, 0, "multi_factor.slo.max_cost_per_1m")
	assertInt(t, got.MultiFactor.SLO.MaxInflight, 0, "multi_factor.slo.max_inflight")
	assertInt(t, got.MultiFactor.LatencyPercentile, defaults.LatencyPercentile, "multi_factor.latency_percentile")
	assertString(t, got.MultiFactor.OnNoCandidates, defaults.OnNoCandidates, "multi_factor.on_no_candidates")
}

func assertFloat(t *testing.T, got, want float64, field string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s = %v, want %v", field, got, want)
	}
}

func assertString(t *testing.T, got, want, field string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s = %q, want %q", field, got, want)
	}
}

func assertInt(t *testing.T, got, want int, field string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s = %d, want %d", field, got, want)
	}
}
