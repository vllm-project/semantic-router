package extproc

import (
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestRouterLearningRuntimeUpdateOutcomeUsesTargetRefAndTier(t *testing.T) {
	rt := newRouterLearningRuntime(nil, nil, nil)

	result := rt.UpdateOutcome(context.Background(), &routerruntime.RouterOutcome{
		ReplayID:  "replay-1",
		Source:    routerruntime.RouterOutcomeSourceAgent,
		Target:    routerruntime.RouterOutcomeTargetModel,
		TargetRef: "model-a",
		Verdict:   routerruntime.RouterOutcomeVerdictGoodFit,
		Score:     1,
		Metadata: map[string]string{
			"decision":      "domain_code",
			"decision_tier": "4",
		},
	})
	if result.Updated != 1 {
		t.Fatalf("expected one outcome update, got %#v", result)
	}

	exact := rt.experienceSnapshot("domain_code", 4, "model-a")
	if exact.GoodFitCount != 1 {
		t.Fatalf("expected exact experience update, got %#v", exact)
	}
	tier := rt.experienceSnapshot("other_code", 4, "model-a")
	if tier.GoodFitCount != 1 {
		t.Fatalf("expected tier experience fallback update, got %#v", tier)
	}
	global := rt.experienceSnapshot("other_code", 0, "model-a")
	if global.GoodFitCount != 1 {
		t.Fatalf("expected global model experience fallback update, got %#v", global)
	}
}

func TestRouterLearningRuntimeIgnoresNonModelOutcomes(t *testing.T) {
	rt := newRouterLearningRuntime(nil, nil, nil)

	result := rt.UpdateOutcome(context.Background(), &routerruntime.RouterOutcome{
		ReplayID:  "replay-1",
		Source:    routerruntime.RouterOutcomeSourceEval,
		Target:    routerruntime.RouterOutcomeTargetRoute,
		TargetRef: "domain_code",
		Verdict:   routerruntime.RouterOutcomeVerdictUnderpowered,
	})
	if result.Updated != 0 {
		t.Fatalf("expected route outcome to skip online model update, got %#v", result)
	}
}

func TestRouterLearningRuntimeRecordsReplayOutcome(t *testing.T) {
	storage := store.NewMemoryStore(10, 0)
	recorder := routerreplay.NewRecorder(storage)
	rt := newRouterLearningRuntime(nil, recorder, nil)

	if _, err := recorder.AddRecord(routerreplay.RoutingRecord{
		ID:            "replay-1",
		Decision:      "domain_code",
		DecisionTier:  4,
		SelectedModel: "model-a",
	}); err != nil {
		t.Fatalf("add replay record: %v", err)
	}

	result := rt.UpdateOutcome(context.Background(), &routerruntime.RouterOutcome{
		ReplayID:  "replay-1",
		Source:    routerruntime.RouterOutcomeSourceEval,
		Target:    routerruntime.RouterOutcomeTargetModel,
		TargetRef: "model-a",
		Verdict:   routerruntime.RouterOutcomeVerdictOverprovisioned,
		Score:     0.25,
		Metadata: map[string]string{
			"decision":      "domain_code",
			"decision_tier": "4",
			"eval_id":       "eval-1",
		},
	})
	if result.Updated != 1 || !result.Recorded {
		t.Fatalf("expected outcome to update experience and replay, got %#v", result)
	}

	record, found := recorder.GetRecord("replay-1")
	if !found {
		t.Fatal("expected replay record to remain readable")
	}
	if len(record.Outcomes) != 1 {
		t.Fatalf("expected one replay outcome, got %#v", record.Outcomes)
	}
	outcome := record.Outcomes[0]
	if outcome.TargetRef != "model-a" ||
		outcome.Verdict != string(routerruntime.RouterOutcomeVerdictOverprovisioned) ||
		outcome.Metadata["eval_id"] != "eval-1" {
		t.Fatalf("unexpected replay outcome: %#v", outcome)
	}
}

func TestRouterLearningRuntimeResolvesOmittedModelTargetRefFromReplay(t *testing.T) {
	storage := store.NewMemoryStore(10, 0)
	recorder := routerreplay.NewRecorder(storage)
	rt := newRouterLearningRuntime(nil, recorder, nil)

	if _, err := recorder.AddRecord(routerreplay.RoutingRecord{
		ID:            "replay-1",
		Decision:      "domain_code",
		DecisionTier:  4,
		SelectedModel: "model-a",
	}); err != nil {
		t.Fatalf("add replay record: %v", err)
	}

	result := rt.UpdateOutcome(context.Background(), &routerruntime.RouterOutcome{
		ReplayID: "replay-1",
		Source:   routerruntime.RouterOutcomeSourceUser,
		Target:   routerruntime.RouterOutcomeTargetModel,
		Verdict:  routerruntime.RouterOutcomeVerdictGoodFit,
		Score:    1,
	})
	if result.Updated != 1 || !result.Recorded {
		t.Fatalf("expected omitted target_ref to update and record, got %#v", result)
	}

	exact := rt.experienceSnapshot("domain_code", 4, "model-a")
	if exact.GoodFitCount != 1 {
		t.Fatalf("expected replay-selected model experience update, got %#v", exact)
	}
	record, found := recorder.GetRecord("replay-1")
	if !found || len(record.Outcomes) != 1 {
		t.Fatalf("expected replay outcome, found=%v record=%#v", found, record)
	}
	if got := record.Outcomes[0].TargetRef; got != "model-a" {
		t.Fatalf("expected replay outcome target_ref to be resolved, got %q", got)
	}
}

func TestRouterLearningRuntimeRecordsTelemetryAcrossFallbackKeys(t *testing.T) {
	rt := newRouterLearningRuntime(nil, nil, nil)

	rt.recordModelTelemetry(
		"domain_code",
		4,
		"model-a",
		routerLearningTelemetryObservation{
			LatencySeconds:      0.8,
			LatencyObserved:     true,
			CacheHitRatio:       0.25,
			CacheWritePressure:  0.75,
			CacheObserved:       true,
			InputCostMultiplier: 0.9,
			InputCostObserved:   true,
		},
	)

	exact := rt.experienceSnapshot("domain_code", 4, "model-a")
	if exact.LatencyEWMA != 0.8 ||
		exact.CacheHitEWMA != 0.25 ||
		exact.CacheWriteEWMA != 0.75 ||
		exact.InputCostMultiplierEWMA != 0.9 {
		t.Fatalf("expected exact telemetry update, got %#v", exact)
	}
	tier := rt.experienceSnapshot("other_code", 4, "model-a")
	if tier.LatencyEWMA != 0.8 {
		t.Fatalf("expected tier fallback telemetry update, got %#v", tier)
	}
	global := rt.experienceSnapshot("other_code", 0, "model-a")
	if global.CacheHitEWMA != 0.25 {
		t.Fatalf("expected global fallback telemetry update, got %#v", global)
	}
}

func TestRouterLearningRuntimeTelemetryAllowsZeroCacheObservation(t *testing.T) {
	rt := newRouterLearningRuntime(nil, nil, nil)

	rt.recordModelTelemetry(
		"domain_code",
		4,
		"model-a",
		routerLearningTelemetryObservation{
			CacheHitRatio:      1.0,
			CacheWritePressure: 0.0,
			CacheObserved:      true,
		},
	)
	rt.recordModelTelemetry(
		"domain_code",
		4,
		"model-a",
		routerLearningTelemetryObservation{
			CacheHitRatio:      0.0,
			CacheWritePressure: 1.0,
			CacheObserved:      true,
		},
	)

	exact := rt.experienceSnapshot("domain_code", 4, "model-a")
	if exact.CacheHitEWMA >= 1.0 || exact.CacheHitEWMA <= 0.0 {
		t.Fatalf("expected zero cache-hit observation to decay cache EWMA, got %#v", exact)
	}
	if exact.CacheWriteEWMA <= 0.0 {
		t.Fatalf("expected cache write pressure to update from zero, got %#v", exact)
	}
}

func TestRouterLearningRuntimeRecordsProviderFailureAsReliabilityEvidence(t *testing.T) {
	rt := newRouterLearningRuntime(nil, nil, nil)

	rt.recordModelTelemetry(
		"domain_code",
		4,
		"model-a",
		routerLearningTelemetryObservation{ProviderFailureObserved: true},
	)

	exact := rt.experienceSnapshot("domain_code", 4, "model-a")
	if exact.FailedCount != 1 {
		t.Fatalf("expected provider failure to update reliability evidence, got %#v", exact)
	}
}

func TestObserveRouterLearningUsageTelemetryUsesEffectiveInputCost(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			RouterLearning: config.RouterLearningConfig{Enabled: true},
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"model-a": {
						Pricing: config.ModelPricing{
							PromptPer1M:      10,
							CachedInputPer1M: 2,
						},
					},
				},
			},
		},
	}
	router.observeRouterLearningUsageTelemetry(
		&RequestContext{
			RequestModel:            "model-a",
			VSRSelectedDecisionName: "domain_code",
			VSRSelectedDecision:     &config.Decision{Name: "domain_code", Tier: 4},
		},
		800*time.Millisecond,
		responseUsageMetrics{
			promptTokens:               100,
			cachedPromptTokens:         50,
			cachedPromptTokensReported: true,
			completionTokens:           20,
		},
		routerreplay.UsageCost{},
	)

	exact := router.routerLearningRuntimeState().experienceSnapshot("domain_code", 4, "model-a")
	if exact.LatencyEWMA != 0.8 || exact.CacheHitEWMA != 0.5 || exact.CacheWriteEWMA != 0.5 {
		t.Fatalf("expected latency/cache telemetry update, got %#v", exact)
	}
	if exact.InputCostMultiplierEWMA != 0.6 {
		t.Fatalf("expected cached input cost multiplier 0.6, got %#v", exact)
	}
}
