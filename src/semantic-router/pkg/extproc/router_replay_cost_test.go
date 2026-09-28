package extproc

import (
	"math"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBuildReplayUsageCostComputesBaselineSavings(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"cheap-model": {
						Pricing: config.ModelPricing{
							Currency:        "USD",
							PromptPer1M:     1,
							CompletionPer1M: 2,
						},
					},
					"expensive-model": {
						Pricing: config.ModelPricing{
							Currency:        "USD",
							PromptPer1M:     4,
							CompletionPer1M: 8,
						},
					},
				},
			},
		},
	}

	usage := responseUsageMetrics{promptTokens: 1000, completionTokens: 500}
	snapshot := router.buildReplayUsageCost(replayCostContext("cheap-model", "cheap-model", "expensive-model"), usage)

	if snapshot.PromptTokens == nil || *snapshot.PromptTokens != 1000 {
		t.Fatalf("expected prompt tokens to be recorded, got %#v", snapshot.PromptTokens)
	}
	if snapshot.CompletionTokens == nil || *snapshot.CompletionTokens != 500 {
		t.Fatalf("expected completion tokens to be recorded, got %#v", snapshot.CompletionTokens)
	}
	if snapshot.TotalTokens == nil || *snapshot.TotalTokens != 1500 {
		t.Fatalf("expected total tokens to be recorded, got %#v", snapshot.TotalTokens)
	}
	if snapshot.BaselineModel == nil || *snapshot.BaselineModel != "expensive-model" {
		t.Fatalf("expected baseline model to be expensive-model, got %#v", snapshot.BaselineModel)
	}
	assertApproxFloat64(t, snapshot.ActualCost, 0.002)
	assertApproxFloat64(t, snapshot.BaselineCost, 0.008)
	assertApproxFloat64(t, snapshot.CostSavings, 0.006)
	if snapshot.Currency == nil || *snapshot.Currency != "USD" {
		t.Fatalf("expected USD currency, got %#v", snapshot.Currency)
	}
}

func TestBuildCacheHitReplayUsageRecordsZeroActualCost(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"cheap-model": {
						Pricing: config.ModelPricing{
							Currency:        "USD",
							PromptPer1M:     1,
							CompletionPer1M: 2,
						},
					},
					"expensive-model": {
						Pricing: config.ModelPricing{
							Currency:        "USD",
							PromptPer1M:     10,
							CompletionPer1M: 20,
						},
					},
				},
			},
		},
	}
	usage := responseUsageMetrics{promptTokens: 1000, completionTokens: 500}

	snapshot := router.buildCacheHitReplayUsage(
		replayCostContext("cheap-model", "cheap-model", "expensive-model"),
		usage,
	)

	assertApproxFloat64(t, snapshot.ActualCost, 0)
	assertApproxFloat64(t, snapshot.BaselineCost, 0.02)
	assertApproxFloat64(t, snapshot.CostSavings, 0.02)
}

func TestBuildReplayUsageCostIncludesCacheWrites(t *testing.T) {
	selectedWriteRate := 1.25
	baselineWriteRate := 6.25
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"cheap-model": {
						Pricing: config.ModelPricing{
							Currency:         "USD",
							PromptPer1M:      1,
							CachedInputPer1M: 0.1,
							CacheWritePer1M:  &selectedWriteRate,
							CompletionPer1M:  6,
						},
					},
					"expensive-model": {
						Pricing: config.ModelPricing{
							Currency:         "USD",
							PromptPer1M:      5,
							CachedInputPer1M: 0.5,
							CacheWritePer1M:  &baselineWriteRate,
							CompletionPer1M:  30,
						},
					},
				},
			},
		},
	}

	usage := responseUsageMetrics{
		promptTokens:       1_000,
		cachedPromptTokens: 200,
		cacheWriteTokens:   300,
		completionTokens:   100,
	}
	snapshot := router.buildReplayUsageCost(replayCostContext("cheap-model", "cheap-model", "expensive-model"), usage)

	assertApproxFloat64(t, snapshot.ActualCost, 0.001495)
	assertApproxFloat64(t, snapshot.BaselineCost, 0.007475)
	assertApproxFloat64(t, snapshot.CostSavings, 0.00598)
	if snapshot.CacheWriteTokens == nil || *snapshot.CacheWriteTokens != 300 {
		t.Fatalf("expected cache-write tokens to be recorded, got %#v", snapshot.CacheWriteTokens)
	}
}

func TestBuildReplayUsageCostKeepsTokenCountsWhenPricingIsMissing(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"unpriced-model": {},
					"expensive-model": {
						Pricing: config.ModelPricing{
							Currency:        "USD",
							PromptPer1M:     4,
							CompletionPer1M: 8,
						},
					},
				},
			},
		},
	}

	snapshot := router.buildReplayUsageCost(
		&RequestContext{RequestModel: "unpriced-model"},
		responseUsageMetrics{promptTokens: 100, completionTokens: 50},
	)

	if snapshot.TotalTokens == nil || *snapshot.TotalTokens != 150 {
		t.Fatalf("expected total tokens to still be recorded, got %#v", snapshot.TotalTokens)
	}
	if snapshot.ActualCost != nil || snapshot.BaselineCost != nil || snapshot.CostSavings != nil {
		t.Fatalf("expected cost fields to stay empty when selected model pricing is missing, got %#v", snapshot)
	}
	if snapshot.Currency != nil || snapshot.BaselineModel != nil {
		t.Fatalf("expected currency and baseline model to stay empty without cost data, got %#v", snapshot)
	}
}

func TestBuildReplayUsageCostTreatsZeroPricedModelAsFree(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"free-model": {
						Pricing: config.ModelPricing{
							Currency:        "USD",
							PromptPer1M:     0,
							CompletionPer1M: 0,
						},
					},
					"expensive-model": {
						Pricing: config.ModelPricing{
							Currency:        "USD",
							PromptPer1M:     4,
							CompletionPer1M: 8,
						},
					},
				},
			},
		},
	}

	usage := responseUsageMetrics{promptTokens: 1000, completionTokens: 500}
	snapshot := router.buildReplayUsageCost(replayCostContext("free-model", "free-model", "expensive-model"), usage)

	if snapshot.BaselineModel == nil || *snapshot.BaselineModel != "expensive-model" {
		t.Fatalf("expected baseline model to be expensive-model, got %#v", snapshot.BaselineModel)
	}
	assertApproxFloat64(t, snapshot.ActualCost, 0.0)
	assertApproxFloat64(t, snapshot.BaselineCost, 0.008)
	assertApproxFloat64(t, snapshot.CostSavings, 0.008)
	if snapshot.Currency == nil || *snapshot.Currency != "USD" {
		t.Fatalf("expected USD currency for zero-priced model, got %#v", snapshot.Currency)
	}
}

func TestBuildReplayUsageCostSkipsEmptyUsage(t *testing.T) {
	snapshot := (&OpenAIRouter{}).buildReplayUsageCost(
		replayCostContext("cheap-model", "cheap-model", "expensive-model"),
		responseUsageMetrics{},
	)

	if snapshot.PromptTokens != nil ||
		snapshot.CacheWriteTokens != nil ||
		snapshot.CompletionTokens != nil ||
		snapshot.TotalTokens != nil ||
		snapshot.ActualCost != nil ||
		snapshot.BaselineCost != nil ||
		snapshot.CostSavings != nil ||
		snapshot.Currency != nil ||
		snapshot.BaselineModel != nil {
		t.Fatalf("expected empty usage snapshot, got %#v", snapshot)
	}
}

func assertApproxFloat64(t *testing.T, value *float64, expected float64) {
	t.Helper()
	if value == nil {
		t.Fatalf("expected float value %.6f, got nil", expected)
	}
	if math.Abs(*value-expected) > 1e-9 {
		t.Fatalf("expected %.6f, got %.6f", expected, *value)
	}
}

func replayCostContext(selected string, candidates ...string) *RequestContext {
	ctx := &RequestContext{RequestModel: selected, VSRSelectedDecision: &config.Decision{Name: "route"}}
	for _, model := range candidates {
		ctx.VSRSelectedDecision.ModelRefs = append(ctx.VSRSelectedDecision.ModelRefs, config.ModelRef{Model: model})
	}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{
		Name: "cost-recipe", Profile: config.RoutingProfile{Decisions: []config.Decision{*ctx.VSRSelectedDecision}},
	})
	return ctx
}

func TestBuildReplayUsageCostScopesBaselineAndUsesRecordedUsage(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{
		ModelConfig: map[string]config.ModelParams{
			"selected":       {Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: 1, CompletionPer1M: 1}},
			"input-heavy":    {Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: 4, CompletionPer1M: 1}},
			"output-heavy":   {Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: 1, CompletionPer1M: 10}},
			"foreign-recipe": {Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: 100, CompletionPer1M: 100}},
			"other-currency": {Pricing: config.ModelPricing{Currency: "CNY", PromptPer1M: 200, CompletionPer1M: 200}},
			"unpriced":       {},
		},
	}}}
	ctx := replayCostContext("selected", "selected", "input-heavy", "output-heavy", "other-currency", "unpriced")
	// The output-heavy model has the highest combined rate, but costs less for
	// this recorded input-heavy request. A model outside this recipe and a
	// different currency must not contribute to the baseline.
	got := router.buildReplayUsageCost(ctx, responseUsageMetrics{promptTokens: 1000, completionTokens: 1})
	if got.BaselineModel == nil || *got.BaselineModel != "input-heavy" {
		t.Fatalf("unexpected baseline: %#v", got)
	}
	assertApproxFloat64(t, got.ActualCost, 0.001001)
	assertApproxFloat64(t, got.BaselineCost, 0.004001)
	assertApproxFloat64(t, got.CostSavings, 0.003)
}

func TestBuildReplayUsageCostUsesWholeSelectedRecipePool(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"selected":     {Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: 1, CompletionPer1M: 1}},
			"input-heavy":  {Pricing: config.ModelPricing{Currency: " usd ", PromptPer1M: 4, CompletionPer1M: 1}},
			"output-heavy": {Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: 1, CompletionPer1M: 10}},
			"foreign":      {Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: 100, CompletionPer1M: 100}},
			"other-money":  {Pricing: config.ModelPricing{Currency: "EUR", PromptPer1M: 200, CompletionPer1M: 200}},
			"unpriced":     {},
		}},
		Recipes: []config.RoutingRecipe{
			{Name: "selected-recipe", Profile: config.RoutingProfile{Decisions: []config.Decision{
				{Name: "simple", ModelRefs: []config.ModelRef{{Model: "selected"}}},
				{Name: "complex", ModelRefs: []config.ModelRef{
					{Model: "input-heavy"}, {Model: "output-heavy"}, {Model: "other-money"}, {Model: "unpriced"},
				}},
			}}},
			{Name: "foreign-recipe", Profile: config.RoutingProfile{Decisions: []config.Decision{
				{Name: "expensive", ModelRefs: []config.ModelRef{{Model: "foreign"}}},
			}}},
		},
	}}
	recipe, found := router.Config.RecipeByName("selected-recipe")
	if !found {
		t.Fatal("missing selected recipe")
	}
	ctx := &RequestContext{RequestModel: "selected", VSRSelectedDecision: &recipe.Profile.Decisions[0]}
	ctx.Routing.SelectRecipe(recipe)
	for _, test := range []struct {
		name, baseline string
		usage          responseUsageMetrics
		actual, cost   float64
	}{
		{"input-heavy request", "input-heavy", responseUsageMetrics{promptTokens: 1000, completionTokens: 1}, 0.001001, 0.004001},
		{"output-heavy request", "output-heavy", responseUsageMetrics{promptTokens: 1, completionTokens: 1000}, 0.001001, 0.010001},
	} {
		t.Run(test.name, func(t *testing.T) {
			got := router.buildReplayUsageCost(ctx, test.usage)
			if got.BaselineModel == nil || *got.BaselineModel != test.baseline {
				t.Fatalf("expected %s from a different decision in the same recipe, got %#v", test.baseline, got)
			}
			assertApproxFloat64(t, got.ActualCost, test.actual)
			assertApproxFloat64(t, got.BaselineCost, test.cost)
			assertApproxFloat64(t, got.CostSavings, test.cost-test.actual)
		})
	}
}

func TestBuildReplayUsageCostIncludesRecipeGenerationTargets(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{
		DefaultModel: "default",
		ModelConfig: map[string]config.ModelParams{
			"selected":  {Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: 1}},
			"explicit":  {Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: 10}},
			"default":   {Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: 50}},
			"auxiliary": {Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: 100}},
		},
	}}}
	strict := &config.CandidateRequirements{Context: config.CandidateContextKnownLimits}
	for _, test := range []struct {
		name         string
		decision     config.Decision
		requirements *config.CandidateRequirements
		baseline     string
		cost         float64
	}{
		{
			name: "explicit candidate iteration in another decision",
			decision: config.Decision{
				ModelRefs: []config.ModelRef{{Model: "selected"}},
				CandidateIterations: []config.CandidateIterationConfig{{
					Source: "models", Models: []config.ModelRef{{Model: "explicit"}},
				}},
			},
			baseline: "explicit", cost: 0.01,
		},
		{
			name: "route action overrides empty candidate fallback",
			decision: config.Decision{Action: &config.DecisionAction{
				Type: config.DecisionActionRoute, Destination: "explicit",
			}},
			baseline: "explicit", cost: 0.01,
		},
		{
			name: "strict route action still belongs to recipe",
			decision: config.Decision{Action: &config.DecisionAction{
				Type: config.DecisionActionRoute, Destination: "explicit",
			}},
			requirements: strict, baseline: "explicit", cost: 0.01,
		},
		{
			name:     "empty candidate decision admits router default",
			baseline: "default", cost: 0.05,
		},
		{
			name:         "strict requirements disable empty candidate fallback",
			requirements: strict, baseline: "selected", cost: 0.001,
		},
		{
			name:     "minimum candidates prevents empty candidate fallback",
			decision: config.Decision{Algorithm: &config.AlgorithmConfig{MinimumCandidates: 1}},
			baseline: "selected", cost: 0.001,
		},
		{
			name: "fast response does not admit default",
			decision: config.Decision{Plugins: []config.DecisionPlugin{{
				Type: config.DecisionPluginFastResponse,
				Configuration: config.MustStructuredPayload(config.FastResponsePluginConfig{
					Message: "Synthetic immediate response.",
				}),
			}}},
			baseline: "selected", cost: 0.001,
		},
		{
			name: "auxiliary selector and unassigned default are excluded",
			decision: config.Decision{
				ModelRefs: []config.ModelRef{{Model: "selected"}},
				Algorithm: &config.AlgorithmConfig{Type: "prompt", Prompt: &config.PromptSelectionConfig{Model: "auxiliary"}},
			},
			baseline: "selected", cost: 0.001,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx := replayCostContext("selected", "selected")
			recipe := ctx.Routing.SelectedRecipe()
			recipe.Profile.Decisions = append(recipe.Profile.Decisions, test.decision)
			recipe.Profile.CandidateRequirements = test.requirements
			got := router.buildReplayUsageCost(ctx, responseUsageMetrics{promptTokens: 1000})
			if got.BaselineModel == nil || *got.BaselineModel != test.baseline {
				t.Fatalf("unexpected recipe baseline: %#v", got)
			}
			assertApproxFloat64(t, got.ActualCost, 0.001)
			assertApproxFloat64(t, got.BaselineCost, test.cost)
			assertApproxFloat64(t, got.CostSavings, test.cost-0.001)
		})
	}
}

func TestBuildReplayUsageCostNeverAddsForeignSelectedModelToRecipePool(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{
		ModelConfig: map[string]config.ModelParams{
			"selected": {Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: 10}},
			"pooled":   {Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: 2}},
			"unpriced": {},
		},
	}}}
	usage := responseUsageMetrics{promptTokens: 1000}
	got := router.buildReplayUsageCost(replayCostContext("selected", "pooled"), usage)
	if got.BaselineModel == nil || *got.BaselineModel != "pooled" {
		t.Fatalf("selected model outside the recipe must not replace its baseline: %#v", got)
	}
	assertApproxFloat64(t, got.ActualCost, 0.01)
	assertApproxFloat64(t, got.BaselineCost, 0.002)
	assertApproxFloat64(t, got.CostSavings, -0.008)

	got = router.buildReplayUsageCost(replayCostContext("selected", "unpriced"), usage)
	assertApproxFloat64(t, got.ActualCost, 0.01)
	if got.BaselineModel != nil || got.BaselineCost != nil || got.CostSavings != nil {
		t.Fatalf("unpriced recipe pool must leave the baseline unavailable: %#v", got)
	}
}

func TestBuildReplayUsageCostBreaksEqualCostTiesByModelName(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{
		ModelConfig: map[string]config.ModelParams{
			"selected": {Pricing: config.ModelPricing{PromptPer1M: 1}},
			"a-model":  {Pricing: config.ModelPricing{PromptPer1M: 2}},
			"z-model":  {Pricing: config.ModelPricing{PromptPer1M: 2}},
		},
	}}}
	for _, candidates := range [][]string{{"z-model", "a-model"}, {"a-model", "z-model"}} {
		got := router.buildReplayUsageCost(replayCostContext("selected", candidates...), responseUsageMetrics{promptTokens: 100})
		if got.BaselineModel == nil || *got.BaselineModel != "a-model" {
			t.Fatalf("unstable baseline: %#v", got)
		}
	}
}

func TestBuildReplayUsageCostPassthroughDoesNotInventSavings(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{
		ModelConfig: map[string]config.ModelParams{
			"selected":  {Pricing: config.ModelPricing{PromptPer1M: 1}},
			"unrelated": {Pricing: config.ModelPricing{PromptPer1M: 100}},
		},
	}}}
	ctx := &RequestContext{RequestModel: "selected"}
	ctx.Routing.SelectPassthrough()
	got := router.buildReplayUsageCost(ctx, responseUsageMetrics{promptTokens: 100})
	if got.BaselineModel == nil || *got.BaselineModel != "selected" {
		t.Fatalf("unexpected baseline: %#v", got)
	}
	assertApproxFloat64(t, got.ActualCost, 0.0001)
	assertApproxFloat64(t, got.BaselineCost, 0.0001)
	assertApproxFloat64(t, got.CostSavings, 0)
}

func TestBuildReplayUsageCostRetainsExplicitZeroRecipeBaseline(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{
		ModelConfig: map[string]config.ModelParams{
			"free-a": {Pricing: config.ModelPricing{Currency: "USD"}},
			"free-b": {Pricing: config.ModelPricing{Currency: "USD"}},
		},
	}}}
	got := router.buildReplayUsageCost(replayCostContext("free-b", "free-b", "free-a"), responseUsageMetrics{promptTokens: 100})
	if got.BaselineModel == nil || *got.BaselineModel != "free-a" {
		t.Fatalf("zero-priced pool is available and ties are stable: %#v", got)
	}
	assertApproxFloat64(t, got.ActualCost, 0)
	assertApproxFloat64(t, got.BaselineCost, 0)
	assertApproxFloat64(t, got.CostSavings, 0)
}
