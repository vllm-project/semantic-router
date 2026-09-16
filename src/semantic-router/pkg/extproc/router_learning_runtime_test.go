package extproc

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestRouterLearningRuntimeUpdateOutcomeUsesTargetRefAndTier(t *testing.T) {
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
	storage := store.NewMemoryStore(10, 0)
	recorder := routerreplay.NewRecorder(storage)
	rt := newRouterLearningRuntime(nil, recorder, nil)
	if _, err := recorder.AddRecord(routerreplay.RoutingRecord{
		ID:            "replay-1",
		Decision:      "domain_code",
		SelectedModel: "model-a",
	}); err != nil {
		t.Fatalf("add replay record: %v", err)
	}

	result := rt.UpdateOutcome(context.Background(), &routerruntime.RouterOutcome{
		ReplayID:  "replay-1",
		Source:    routerruntime.RouterOutcomeSourceEval,
		Target:    routerruntime.RouterOutcomeTargetRoute,
		TargetRef: "domain_code",
		Verdict:   routerruntime.RouterOutcomeVerdictUnderpowered,
	})
	if result.Updated != 0 || !result.Recorded {
		t.Fatalf("expected route outcome to skip online model update but still record, got %#v", result)
	}
}

func TestRouterLearningRuntimeRejectsMissingReplay(t *testing.T) {
	rt := newRouterLearningRuntime(nil, nil, nil)
	result := rt.UpdateOutcome(context.Background(), &routerruntime.RouterOutcome{
		ReplayID:  "missing",
		Source:    routerruntime.RouterOutcomeSourceOperator,
		Target:    routerruntime.RouterOutcomeTargetModel,
		TargetRef: "model-a",
		Verdict:   routerruntime.RouterOutcomeVerdictGoodFit,
	})
	if result.Code != routerruntime.RouterOutcomeCodeReplayNotFound || result.Updated != 0 {
		t.Fatalf("expected replay_not_found, got %#v", result)
	}
}

func TestRouterLearningRuntimeRejectsOwnershipMismatch(t *testing.T) {
	storage := store.NewMemoryStore(10, 0)
	recorder := routerreplay.NewRecorder(storage)
	rt := newRouterLearningRuntime(nil, recorder, nil)
	if _, err := recorder.AddRecord(routerreplay.RoutingRecord{
		ID:            "replay-1",
		SelectedModel: "model-a",
	}); err != nil {
		t.Fatalf("add replay record: %v", err)
	}

	result := rt.UpdateOutcome(context.Background(), &routerruntime.RouterOutcome{
		ReplayID:  "replay-1",
		Source:    routerruntime.RouterOutcomeSourceOperator,
		Target:    routerruntime.RouterOutcomeTargetModel,
		TargetRef: "model-b",
		Verdict:   routerruntime.RouterOutcomeVerdictGoodFit,
	})
	if result.Code != routerruntime.RouterOutcomeCodeOwnershipMismatch || result.Updated != 0 {
		t.Fatalf("expected ownership_mismatch, got %#v", result)
	}
	if exact := rt.experienceSnapshot("", 0, "model-b"); exact.GoodFitCount != 0 {
		t.Fatalf("mismatched ownership must not update experience, got %#v", exact)
	}
}

func TestRouterLearningRuntimeDedupsIdempotencyKey(t *testing.T) {
	storage := store.NewMemoryStore(10, 0)
	recorder := routerreplay.NewRecorder(storage)
	rt := newRouterLearningRuntime(nil, recorder, nil)
	if _, err := recorder.AddRecord(routerreplay.RoutingRecord{
		ID:            "replay-1",
		SelectedModel: "model-a",
		Decision:      "domain_code",
		DecisionTier:  4,
	}); err != nil {
		t.Fatalf("add replay record: %v", err)
	}

	first := rt.UpdateOutcome(context.Background(), &routerruntime.RouterOutcome{
		ReplayID:       "replay-1",
		Source:         routerruntime.RouterOutcomeSourceOperator,
		Target:         routerruntime.RouterOutcomeTargetModel,
		Verdict:        routerruntime.RouterOutcomeVerdictGoodFit,
		Score:          1,
		IdempotencyKey: "idem-1",
	})
	if first.Updated != 1 || !first.Recorded {
		t.Fatalf("expected first ingest to succeed, got %#v", first)
	}
	second := rt.UpdateOutcome(context.Background(), &routerruntime.RouterOutcome{
		ReplayID:       "replay-1",
		Source:         routerruntime.RouterOutcomeSourceOperator,
		Target:         routerruntime.RouterOutcomeTargetModel,
		Verdict:        routerruntime.RouterOutcomeVerdictGoodFit,
		Score:          1,
		IdempotencyKey: "idem-1",
	})
	if second.Code != routerruntime.RouterOutcomeCodeDuplicate || second.Updated != 0 {
		t.Fatalf("expected duplicate, got %#v", second)
	}
	exact := rt.experienceSnapshot("domain_code", 4, "model-a")
	if exact.GoodFitCount != 1 {
		t.Fatalf("duplicate must not double-count experience, got %#v", exact)
	}
}

func TestRouterLearningRuntimeClaimsIdempotencyKeyBeforeConcurrentAppend(t *testing.T) {
	base := store.NewMemoryStore(10, 0)
	storage := &controlledOutcomeStore{
		Storage:       base,
		appendStarted: make(chan struct{}),
		releaseAppend: make(chan struct{}),
	}
	recorder := routerreplay.NewRecorder(storage)
	rt := newRouterLearningRuntime(nil, recorder, nil)
	if _, err := recorder.AddRecord(routerreplay.RoutingRecord{
		ID:            "replay-concurrent",
		SelectedModel: "model-a",
		Decision:      "domain-code",
		DecisionTier:  4,
	}); err != nil {
		t.Fatalf("add replay record: %v", err)
	}

	results := make(chan routerruntime.RouterOutcomeResult, 2)
	update := func() { results <- updateConcurrentRouterOutcome(rt) }
	go update()
	waitForOutcomeAppend(t, storage.appendStarted)
	go update()
	time.Sleep(25 * time.Millisecond)
	if got := storage.appendCalls.Load(); got != 1 {
		t.Fatalf("concurrent duplicate reached storage %d times before release, want 1", got)
	}
	close(storage.releaseAppend)

	first := <-results
	second := <-results
	assertConcurrentOutcomeResults(t, first, second)
	assertSingleConcurrentOutcome(t, recorder, rt)
}

func updateConcurrentRouterOutcome(rt *routerLearningRuntime) routerruntime.RouterOutcomeResult {
	return rt.UpdateOutcome(context.Background(), &routerruntime.RouterOutcome{
		ReplayID:       "replay-concurrent",
		Source:         routerruntime.RouterOutcomeSourceOperator,
		Target:         routerruntime.RouterOutcomeTargetModel,
		Verdict:        routerruntime.RouterOutcomeVerdictGoodFit,
		Score:          1,
		IdempotencyKey: "idem-concurrent",
	})
}

func waitForOutcomeAppend(t *testing.T, appendStarted <-chan struct{}) {
	t.Helper()
	select {
	case <-appendStarted:
	case <-time.After(time.Second):
		t.Fatal("first append did not start")
	}
}

func assertConcurrentOutcomeResults(t *testing.T, first, second routerruntime.RouterOutcomeResult) {
	t.Helper()
	successes := 0
	duplicates := 0
	for _, result := range []routerruntime.RouterOutcomeResult{first, second} {
		if result.Updated == 1 && result.Recorded {
			successes++
		}
		if result.Code == routerruntime.RouterOutcomeCodeDuplicate {
			duplicates++
		}
	}
	if successes != 1 || duplicates != 1 {
		t.Fatalf("concurrent idempotency results = %#v / %#v, want one success and one duplicate", first, second)
	}
}

func assertSingleConcurrentOutcome(t *testing.T, recorder *routerreplay.Recorder, rt *routerLearningRuntime) {
	t.Helper()
	record, found := recorder.GetRecord("replay-concurrent")
	if !found || len(record.Outcomes) != 1 {
		t.Fatalf("stored outcomes = %#v, want exactly one", record.Outcomes)
	}
	if got := rt.experienceSnapshot("domain-code", 4, "model-a").GoodFitCount; got != 1 {
		t.Fatalf("experience count = %d, want 1", got)
	}
}

func TestRouterLearningRuntimeReleasesFailedIdempotencyClaimForRetry(t *testing.T) {
	base := store.NewMemoryStore(10, 0)
	storage := &controlledOutcomeStore{Storage: base}
	storage.failNext.Store(true)
	recorder := routerreplay.NewRecorder(storage)
	rt := newRouterLearningRuntime(nil, recorder, nil)
	if _, err := recorder.AddRecord(routerreplay.RoutingRecord{
		ID:            "replay-retry",
		SelectedModel: "model-a",
	}); err != nil {
		t.Fatalf("add replay record: %v", err)
	}
	outcome := func() *routerruntime.RouterOutcome {
		return &routerruntime.RouterOutcome{
			ReplayID:       "replay-retry",
			Source:         routerruntime.RouterOutcomeSourceOperator,
			Target:         routerruntime.RouterOutcomeTargetModel,
			Verdict:        routerruntime.RouterOutcomeVerdictGoodFit,
			IdempotencyKey: "idem-retry",
		}
	}

	first := rt.UpdateOutcome(context.Background(), outcome())
	if first.Code != routerruntime.RouterOutcomeCodeReplayNotFound {
		t.Fatalf("first result = %#v, want failed append", first)
	}
	second := rt.UpdateOutcome(context.Background(), outcome())
	if second.Code != routerruntime.RouterOutcomeCodeOK || second.Updated != 1 || !second.Recorded {
		t.Fatalf("retry result = %#v, want success", second)
	}
	if got := storage.appendCalls.Load(); got != 2 {
		t.Fatalf("append calls = %d, want 2", got)
	}
}

func TestRouterLearningRuntimePreservesIdempotencyAcrossHotReload(t *testing.T) {
	base := store.NewMemoryStore(10, 0)
	storage := &controlledOutcomeStore{
		Storage:       base,
		appendStarted: make(chan struct{}),
		releaseAppend: make(chan struct{}),
	}
	recorder := routerreplay.NewRecorder(storage)
	if _, err := recorder.AddRecord(routerreplay.RoutingRecord{
		ID:            "replay-reload",
		SelectedModel: "model-a",
		Decision:      "domain-code",
		DecisionTier:  4,
	}); err != nil {
		t.Fatalf("add replay record: %v", err)
	}
	oldRouter := &OpenAIRouter{ReplayRecorder: recorder}
	newRouter := &OpenAIRouter{ReplayRecorder: recorder}
	oldRuntime := oldRouter.routerLearningRuntimeState()

	results := make(chan routerruntime.RouterOutcomeResult, 2)
	update := func(rt *routerLearningRuntime) {
		results <- rt.UpdateOutcome(context.Background(), &routerruntime.RouterOutcome{
			ReplayID:       "replay-reload",
			Source:         routerruntime.RouterOutcomeSourceOperator,
			Target:         routerruntime.RouterOutcomeTargetModel,
			Verdict:        routerruntime.RouterOutcomeVerdictGoodFit,
			Score:          1,
			IdempotencyKey: "idem-reload",
		})
	}
	go update(oldRuntime)
	select {
	case <-storage.appendStarted:
	case <-time.After(time.Second):
		t.Fatal("old generation append did not start")
	}
	inheritRouterLearningState(oldRouter, newRouter)
	newRuntime := newRouter.routerLearningRuntimeState()
	go update(newRuntime)
	time.Sleep(25 * time.Millisecond)
	if got := storage.appendCalls.Load(); got != 1 {
		t.Fatalf("reloaded generation bypassed idempotency claim: append calls = %d", got)
	}
	close(storage.releaseAppend)

	first := <-results
	second := <-results
	successes := 0
	duplicates := 0
	for _, result := range []routerruntime.RouterOutcomeResult{first, second} {
		if result.Updated == 1 && result.Recorded {
			successes++
		}
		if result.Code == routerruntime.RouterOutcomeCodeDuplicate {
			duplicates++
		}
	}
	if successes != 1 || duplicates != 1 {
		t.Fatalf("reload idempotency results = %#v / %#v", first, second)
	}
	if got := newRuntime.experienceSnapshot("domain-code", 4, "model-a").GoodFitCount; got != 1 {
		t.Fatalf("shared experience count = %d, want 1", got)
	}
}

type controlledOutcomeStore struct {
	store.Storage
	appendStarted chan struct{}
	releaseAppend chan struct{}
	startOnce     sync.Once
	appendCalls   atomic.Int32
	failNext      atomic.Bool
}

func (s *controlledOutcomeStore) AppendOutcome(ctx context.Context, id string, outcome store.Outcome) error {
	s.appendCalls.Add(1)
	if s.appendStarted != nil {
		s.startOnce.Do(func() { close(s.appendStarted) })
	}
	if s.releaseAppend != nil {
		select {
		case <-s.releaseAppend:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	if s.failNext.Swap(false) {
		return errors.New("injected append failure")
	}
	return s.Storage.AppendOutcome(ctx, id, outcome)
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

func TestObserveRouterLearningUsageTelemetryUsesReportedCacheWrites(t *testing.T) {
	cacheWriteRate := 12.5
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			RouterLearning: config.RouterLearningConfig{Enabled: true},
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"model-a": {
						Pricing: config.ModelPricing{
							PromptPer1M:      10,
							CachedInputPer1M: 2,
							CacheWritePer1M:  &cacheWriteRate,
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
			cachedPromptTokens:         20,
			cachedPromptTokensReported: true,
			cacheWriteTokens:           30,
			cacheWriteTokensReported:   true,
			completionTokens:           20,
		},
		routerreplay.UsageCost{},
	)

	exact := router.routerLearningRuntimeState().experienceSnapshot("domain_code", 4, "model-a")
	if exact.CacheHitEWMA != 0.2 || exact.CacheWriteEWMA != 0.3 {
		t.Fatalf("expected reported cache telemetry update, got %#v", exact)
	}
	if exact.InputCostMultiplierEWMA != 0.732 {
		t.Fatalf("expected cache-write input cost multiplier 0.732, got %#v", exact)
	}
}
