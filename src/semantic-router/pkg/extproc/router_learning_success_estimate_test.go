package extproc

import (
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestSuccessEstimateSeedOnlyIsInsufficientEvidence(t *testing.T) {
	snap := newRouterLearningRuntime(nil, nil, nil).freezeEvidenceSnapshot("adaptive", 2, []string{"cheap"}, "")
	got := estimateOneCandidateSuccess(snap, "cheap", successEstimateConfig{Now: snap.takenAt})

	if got.Status != successEstimateInsufficient || got.FallbackReason != successFallbackSeedOnly {
		t.Fatalf("expected seed-only insufficient_evidence, got %#v", got)
	}
	if got.Probability != 0 || got.CalibrationVersion != "" {
		t.Fatalf("seed-only estimates must not report a probability, got %#v", got)
	}
}

func TestSuccessEstimateMissingCalibrationIsUnsupported(t *testing.T) {
	rt := newRouterLearningRuntime(nil, nil, nil)
	recordScopedExperience(rt, "adaptive", 2, "frontier", routerLearningOutcomeGoodFit, 10)
	snap := rt.freezeEvidenceSnapshot("adaptive", 2, []string{"frontier"}, "")

	got := estimateOneCandidateSuccess(snap, "frontier", successEstimateConfig{Now: snap.takenAt})
	if got.Status != successEstimateUnsupported || got.FallbackReason != successFallbackMissingCalibration {
		t.Fatalf("expected missing calibration to be unsupported, got %#v", got)
	}
	if got.Probability != 0 || got.SampleCount != 10 || got.EvidenceScope != successEvidenceScopeDecision {
		t.Fatalf("expected outcome counts without a calibrated probability, got %#v", got)
	}
}

func TestSuccessEstimateSparseDecisionFallsBackToBroaderScope(t *testing.T) {
	rt := newRouterLearningRuntime(nil, nil, nil)
	recordScopedExperience(rt, "", 0, "frontier", routerLearningOutcomeGoodFit, 8)
	snap := rt.freezeEvidenceSnapshot("adaptive", 2, []string{"frontier"}, "")

	got := estimateOneCandidateSuccess(snap, "frontier", successEstimateConfig{Now: snap.takenAt})
	if got.Status != successEstimateUnsupported || got.EvidenceScope != successEvidenceScopeGlobal || got.SampleCount != 8 {
		t.Fatalf("expected sparse decision to fall back to global evidence, got %#v", got)
	}
}

func TestSuccessEstimateSparseCohortFallsBackThenInsufficient(t *testing.T) {
	rt := newRouterLearningRuntime(nil, nil, nil)
	snap := rt.freezeEvidenceSnapshot("adaptive", 2, []string{"frontier"}, "cohort-v1")
	got := estimateOneCandidateSuccess(snap, "frontier", successEstimateConfig{Now: snap.takenAt})
	if got.Status != successEstimateInsufficient {
		t.Fatalf("expected empty cohort chain to stay insufficient, got %#v", got)
	}
}

func TestSuccessEstimateStaleEvidence(t *testing.T) {
	rt := newRouterLearningRuntime(nil, nil, nil)
	recordScopedExperience(rt, "adaptive", 2, "frontier", routerLearningOutcomeGoodFit, 4)
	key := modelExperienceKey("adaptive", 2, "frontier")
	rt.shared.mu.Lock()
	rt.shared.experience[key].LastUpdated = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	rt.shared.mu.Unlock()

	now := time.Date(2026, 1, 3, 0, 0, 0, 0, time.UTC)
	snap := rt.freezeEvidenceSnapshot("adaptive", 2, []string{"frontier"}, "")
	got := estimateOneCandidateSuccess(snap, "frontier", successEstimateConfig{
		Now:        now,
		StaleAfter: 24 * time.Hour,
	})
	if got.Status != successEstimateStale || got.FallbackReason != successFallbackStale || got.Probability != 0 {
		t.Fatalf("expected stale evidence, got %#v", got)
	}
}

func TestSuccessEstimateMissingCalibrationRowIsUnsupported(t *testing.T) {
	rt := newRouterLearningRuntime(nil, nil, nil)
	recordScopedExperience(rt, "adaptive", 2, "frontier", routerLearningOutcomeGoodFit, 4)
	snap := rt.freezeEvidenceSnapshot("adaptive", 2, []string{"frontier"}, "")

	got := estimateOneCandidateSuccess(snap, "frontier", successEstimateConfig{
		Now: snap.takenAt,
		Artifact: &successCalibrationArtifact{
			Version: "2026-08-01T00:00:00Z",
			Models: map[string]successCalibrationRow{
				"other": {Probability: 0.9, Coverage: 0.8},
			},
		},
	})
	if got.Status != successEstimateUnsupported || got.FallbackReason != successFallbackMissingModelRow {
		t.Fatalf("expected missing calibration row to be unsupported, got %#v", got)
	}
	if got.Probability != 0 {
		t.Fatalf("missing row must not invent a probability, got %#v", got)
	}
}

func TestSuccessEstimateInjectedArtifactIsCalibratedOnlyForListedModels(t *testing.T) {
	rt := newRouterLearningRuntime(nil, nil, nil)
	recordScopedExperience(rt, "adaptive", 2, "frontier", routerLearningOutcomeGoodFit, 20)
	snap := rt.freezeEvidenceSnapshot("adaptive", 2, []string{"frontier"}, "")

	got := estimateOneCandidateSuccess(snap, "frontier", successEstimateConfig{
		Now: snap.takenAt,
		Artifact: &successCalibrationArtifact{
			Version: "2026-08-01T00:00:00Z",
			Models: map[string]successCalibrationRow{
				"frontier": {Probability: 0.91, Uncertainty: 0.06, Coverage: 0.82},
			},
		},
	})
	if got.Status != successEstimateCalibrated || got.Probability != 0.91 || got.Coverage != 0.82 {
		t.Fatalf("expected injected calibration row, got %#v", got)
	}
}

func TestSuccessEstimateUncalibratedSignalsNeverBecomeProbability(t *testing.T) {
	for _, signal := range []string{"classifier_confidence", "complexity_score", "similarity_score", "quality_score"} {
		got := successEstimateFromUncalibratedSignal("frontier", signal, 0.99)
		if got.Status == successEstimateCalibrated || got.Probability != 0 {
			t.Fatalf("signal %q must never be a calibrated success probability, got %#v", signal, got)
		}
	}
}

func TestSuccessEstimateSnapshotIgnoresConcurrentOutcomeWrites(t *testing.T) {
	rt := newRouterLearningRuntime(nil, nil, nil)
	recordScopedExperience(rt, "adaptive", 2, "frontier", routerLearningOutcomeGoodFit, 5)
	snap := rt.freezeEvidenceSnapshot("adaptive", 2, []string{"frontier"}, "")
	before := estimateOneCandidateSuccess(snap, "frontier", successEstimateConfig{Now: snap.takenAt})

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			recordScopedExperience(rt, "adaptive", 2, "frontier", routerLearningOutcomeGoodFit, 1)
		}()
	}
	wg.Wait()

	after := estimateOneCandidateSuccess(snap, "frontier", successEstimateConfig{Now: snap.takenAt})
	if after.SampleCount != before.SampleCount || after.Status != before.Status {
		t.Fatalf("snapshot estimates must stay immutable, before=%#v after=%#v", before, after)
	}
	live := rt.experienceSnapshot("adaptive", 2, "frontier")
	if live.GoodFitCount <= before.SampleCount {
		t.Fatalf("expected live experience to move past the snapshot, snap=%d live=%d", before.SampleCount, live.GoodFitCount)
	}
}

func TestMergeScopedExperienceReportsConflict(t *testing.T) {
	hits := []scopedExperience{
		{
			scope: successEvidenceScopeDecision,
			found: true,
			exp:   routerLearningModelExperience{GoodFitCount: 20},
		},
		{
			scope: successEvidenceScopeGlobal,
			found: true,
			exp:   routerLearningModelExperience{FailedCount: 20},
		},
	}
	_, _, status, reason := mergeScopedExperience(hits)
	if status != successEstimateConflict || reason != successFallbackConflictingScopes {
		t.Fatalf("expected conflicting scopes, got status=%q reason=%q", status, reason)
	}
}

func TestSuccessEstimateObserveWiringDoesNotChangeSelection(t *testing.T) {
	router, ctx, selCtx, baseResult := successEstimateObserveFixture()
	_, result, selected, applied := router.applyRouterLearning(selCtx, baseResult, &selCtx.CandidateModels[0], ctx)
	if applied || selected == nil || selected.Model != "cheap" || result.SelectedModel != "cheap" {
		t.Fatalf("observe wiring must not change the selected model, result=%#v selected=%#v applied=%v", result, selected, applied)
	}

	policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodAdaptation)
	if !ok || policy.Details.Adaptation == nil {
		t.Fatalf("expected adaptation diagnostics, got %#v", ctx.VSRLearningPolicies)
	}
	assertObserveSuccessEstimates(t, policy)
}

func successEstimateObserveFixture() (*OpenAIRouter, *RequestContext, *selection.SelectionContext, *selection.SelectionResult) {
	router := &OpenAIRouter{Config: routerLearningAdaptationTestConfig()}
	router.routerLearningRuntimeState().recordModelExperience(
		"adaptive",
		2,
		"frontier",
		routerLearningOutcomeGoodFit,
		10,
	)
	ctx := &RequestContext{
		VSRSelectedDecision: &config.Decision{
			Name: "adaptive",
			Tier: 2,
			Adaptations: config.DecisionAdaptationsConfig{
				Mode: config.DecisionAdaptationModeObserve,
			},
		},
	}
	selCtx := &selection.SelectionContext{
		DecisionName: "adaptive",
		CandidateModels: []config.ModelRef{
			{Model: "cheap"},
			{Model: "frontier"},
		},
	}
	baseResult := &selection.SelectionResult{
		SelectedModel: "cheap",
		Score:         1,
		Method:        selection.MethodStatic,
		AllScores:     map[string]float64{"cheap": 1},
	}
	return router, ctx, selCtx, baseResult
}

func assertObserveSuccessEstimates(t *testing.T, policy routerLearningPolicy) {
	t.Helper()
	diag := policy.Details.Adaptation
	if diag.snapshotIdentity == "" || len(diag.successEstimates) != 2 {
		t.Fatalf("expected snapshot identity and per-candidate estimates, got %#v", diag)
	}
	assertConservativeObserveEstimates(t, estimatesByModel(diag.successEstimates))
	assertObserveSuccessReplay(t, policy)
}

func estimatesByModel(estimates []successEstimate) map[string]successEstimate {
	byModel := map[string]successEstimate{}
	for _, estimate := range estimates {
		byModel[estimate.CandidateModel] = estimate
	}
	return byModel
}

func assertConservativeObserveEstimates(t *testing.T, byModel map[string]successEstimate) {
	t.Helper()
	if cheap := byModel["cheap"]; cheap.Status != successEstimateInsufficient || cheap.Probability != 0 {
		t.Fatalf("expected seed-only cheap estimate, got %#v", cheap)
	}
	if frontier := byModel["frontier"]; frontier.Status != successEstimateUnsupported || frontier.Probability != 0 || frontier.SampleCount != 10 {
		t.Fatalf("expected unsupported frontier estimate, got %#v", frontier)
	}
}

func assertObserveSuccessReplay(t *testing.T, policy routerLearningPolicy) {
	t.Helper()
	replay := policy.toReplayAdaptation()
	if replay == nil || replay.SnapshotIdentity == "" || replay.SuccessEstimates["frontier"].Status != string(successEstimateUnsupported) {
		t.Fatalf("expected replay success estimates, got %#v", replay)
	}
	if _, ok := policy.ToMap()["success_estimates"]; ok {
		t.Fatalf("compact policy map must not carry detailed success estimates, got %#v", policy.ToMap())
	}
}

func recordScopedExperience(
	rt *routerLearningRuntime,
	decision string,
	tier int,
	model string,
	verdict routerLearningOutcomeVerdict,
	score float64,
) {
	rt.shared.mu.Lock()
	defer rt.shared.mu.Unlock()
	rt.recordModelExperienceLocked(decision, tier, model, verdict, score)
}
