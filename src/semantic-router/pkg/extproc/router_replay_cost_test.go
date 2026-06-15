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
	snapshot := router.buildReplayUsageCost(&RequestContext{RequestModel: "cheap-model"}, usage)

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
	snapshot := router.buildReplayUsageCost(&RequestContext{RequestModel: "free-model"}, usage)

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
		&RequestContext{RequestModel: "cheap-model"},
		responseUsageMetrics{},
	)

	if snapshot.PromptTokens != nil ||
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
