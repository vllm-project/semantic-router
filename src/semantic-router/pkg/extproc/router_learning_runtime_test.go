package extproc

import (
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/outcomefeedback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

type staticOutcomeProjection struct {
	projection outcomefeedback.Projection
	err        error
}

func (projection staticOutcomeProjection) Read(context.Context, string) (outcomefeedback.Projection, error) {
	return projection.projection, projection.err
}

func TestDurableRoutingLearningUsesOnlyRevisionedOutcomeProjection(t *testing.T) {
	namespaceID := "00000000-0000-4000-8000-000000000101"
	projection := outcomefeedback.Projection{
		Schema: outcomefeedback.ProjectionSchema, NamespaceID: namespaceID, Revision: 8,
		Entries: []outcomefeedback.ProjectionEntry{{
			RecipeID: "recipe-balanced", RecipeName: "balanced", RecipeRevision: 4,
			DecisionID: "complex", DecisionName: "complex", DecisionTier: 3,
			ModelID: "model-a", ModelName: "model-a", ModelRevision: 9,
			GoodFitCount: 2, FailedCount: 1,
		}},
	}
	request := &RequestContext{
		TraceContext: context.Background(),
		InferenceAccess: &inferenceRequestAccess{
			tenant: accessruntime.TenantContext{NamespaceID: namespaceID},
		},
	}
	first := newRouterLearningRuntime(nil, nil, nil)
	first.outcomeProjection = staticOutcomeProjection{projection: projection}
	first.recordModelExperience("complex", 3, "model-a", routerLearningOutcomeFailed, 25)
	observed := first.experienceSnapshotForRequest(request, "complex", 3, "model-a")
	if observed.GoodFitCount != 2 || observed.FailedCount != 1 {
		t.Fatalf("durable experience = %+v, want only durable projection counts", observed)
	}

	// A fresh runtime observes the same global state without inheriting a local
	// experience map from the previous process.
	restarted := newRouterLearningRuntime(nil, nil, nil)
	restarted.outcomeProjection = staticOutcomeProjection{projection: projection}
	restartedObserved := restarted.experienceSnapshotForRequest(request, "complex", 3, "model-a")
	if restartedObserved.GoodFitCount != observed.GoodFitCount || restartedObserved.FailedCount != observed.FailedCount {
		t.Fatalf("restart experience = %+v, first = %+v", restartedObserved, observed)
	}
}

func TestRouterLearningRuntimeRecordsTelemetryAcrossFallbackKeys(t *testing.T) {
	runtime := newRouterLearningRuntime(nil, nil, nil)
	runtime.recordModelTelemetry("domain_code", 4, "model-a", routerLearningTelemetryObservation{
		LatencySeconds: 0.8, LatencyObserved: true,
		CacheHitRatio: 0.25, CacheWritePressure: 0.75, CacheObserved: true,
		InputCostMultiplier: 0.9, InputCostObserved: true,
	})

	exact := runtime.experienceSnapshot("domain_code", 4, "model-a")
	if exact.LatencyEWMA != 0.8 || exact.CacheHitEWMA != 0.25 ||
		exact.CacheWriteEWMA != 0.75 || exact.InputCostMultiplierEWMA != 0.9 {
		t.Fatalf("exact telemetry = %#v", exact)
	}
	if tier := runtime.experienceSnapshot("other_code", 4, "model-a"); tier.LatencyEWMA != 0.8 {
		t.Fatalf("tier telemetry = %#v", tier)
	}
	if global := runtime.experienceSnapshot("other_code", 0, "model-a"); global.CacheHitEWMA != 0.25 {
		t.Fatalf("global telemetry = %#v", global)
	}
}

func TestRouterLearningRuntimeTelemetryAllowsZeroCacheObservation(t *testing.T) {
	runtime := newRouterLearningRuntime(nil, nil, nil)
	runtime.recordModelTelemetry("domain_code", 4, "model-a", routerLearningTelemetryObservation{
		CacheHitRatio: 1, CacheObserved: true,
	})
	runtime.recordModelTelemetry("domain_code", 4, "model-a", routerLearningTelemetryObservation{
		CacheWritePressure: 1, CacheObserved: true,
	})

	exact := runtime.experienceSnapshot("domain_code", 4, "model-a")
	if exact.CacheHitEWMA >= 1 || exact.CacheHitEWMA <= 0 || exact.CacheWriteEWMA <= 0 {
		t.Fatalf("zero cache observation = %#v", exact)
	}
}

func TestRouterLearningRuntimeRecordsProviderFailureAsReliabilityEvidence(t *testing.T) {
	runtime := newRouterLearningRuntime(nil, nil, nil)
	runtime.recordModelTelemetry("domain_code", 4, "model-a", routerLearningTelemetryObservation{
		ProviderFailureObserved: true,
	})
	if exact := runtime.experienceSnapshot("domain_code", 4, "model-a"); exact.FailedCount != 1 {
		t.Fatalf("failure evidence = %#v", exact)
	}
}

func TestObserveRouterLearningUsageTelemetryUsesEffectiveInputCost(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		RouterLearning: config.RouterLearningConfig{Enabled: true},
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"model-a": {Pricing: config.ModelPricing{PromptPer1M: 10, CachedInputPer1M: 2}},
		}},
	}}
	router.observeRouterLearningUsageTelemetry(&RequestContext{
		RequestModel: "model-a", VSRSelectedDecisionName: "domain_code",
		VSRSelectedDecision: &config.Decision{Name: "domain_code", Tier: 4},
	}, 800*time.Millisecond, responseUsageMetrics{
		promptTokens: 100, cachedPromptTokens: 50, cachedPromptTokensReported: true,
		completionTokens: 20,
	}, routerreplay.UsageCost{})

	exact := router.routerLearningRuntimeState().experienceSnapshot("domain_code", 4, "model-a")
	if exact.LatencyEWMA != 0.8 || exact.CacheHitEWMA != 0.5 ||
		exact.CacheWriteEWMA != 0.5 || exact.InputCostMultiplierEWMA != 0.6 {
		t.Fatalf("usage telemetry = %#v", exact)
	}
}

func TestObserveRouterLearningUsageTelemetryUsesReportedCacheWrites(t *testing.T) {
	cacheWriteRate := 12.5
	router := &OpenAIRouter{Config: &config.RouterConfig{
		RouterLearning: config.RouterLearningConfig{Enabled: true},
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"model-a": {Pricing: config.ModelPricing{
				PromptPer1M: 10, CachedInputPer1M: 2, CacheWritePer1M: &cacheWriteRate,
			}},
		}},
	}}
	router.observeRouterLearningUsageTelemetry(&RequestContext{
		RequestModel: "model-a", VSRSelectedDecisionName: "domain_code",
		VSRSelectedDecision: &config.Decision{Name: "domain_code", Tier: 4},
	}, 800*time.Millisecond, responseUsageMetrics{
		promptTokens: 100, cachedPromptTokens: 20, cachedPromptTokensReported: true,
		cacheWriteTokens: 30, cacheWriteTokensReported: true, completionTokens: 20,
	}, routerreplay.UsageCost{})

	exact := router.routerLearningRuntimeState().experienceSnapshot("domain_code", 4, "model-a")
	if exact.CacheHitEWMA != 0.2 || exact.CacheWriteEWMA != 0.3 ||
		exact.InputCostMultiplierEWMA != 0.732 {
		t.Fatalf("cache-write telemetry = %#v", exact)
	}
}
