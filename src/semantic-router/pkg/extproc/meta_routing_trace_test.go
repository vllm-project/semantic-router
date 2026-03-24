package extproc

import (
	"slices"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

func TestRecordMetaRoutingBasePassCapturesAssessmentAndPlan(t *testing.T) {
	router := observedMetaRoutingTestRouter()
	ctx := observedMetaRoutingTestContext()

	router.recordMetaRoutingBasePass(
		ctx,
		signalEvaluationInput{
			evaluationText: "long input",
			compressedText: "short input",
		},
		observedMetaRoutingTestSignals(),
		observedMetaRoutingTestResult(),
		"model-a",
		12*time.Millisecond,
	)

	assertObservedMetaRoutingTrace(t, ctx)
}

func observedMetaRoutingTestRouter() *OpenAIRouter {
	return &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				MetaRouting: config.MetaRoutingConfig{
					Mode:      config.MetaRoutingModeObserve,
					MaxPasses: 2,
					TriggerPolicy: &config.MetaTriggerPolicy{
						DecisionMarginBelow:      metaFloat64Ptr(0.2),
						ProjectionBoundaryWithin: metaFloat64Ptr(0.1),
						PartitionConflict:        metaBoolPtr(true),
						RequiredFamilies: []config.MetaRequiredSignalFamily{{
							Type:          config.SignalTypeEmbedding,
							MinConfidence: metaFloat64Ptr(0.9),
						}},
						FamilyDisagreements: []config.MetaSignalFamilyDisagreement{{
							Cheap:     config.SignalTypeKeyword,
							Expensive: config.SignalTypeEmbedding,
						}},
					},
					AllowedActions: []config.MetaRefinementAction{
						{Type: config.MetaRoutingActionDisableCompression},
						{
							Type:           config.MetaRoutingActionRerunSignalFamilies,
							SignalFamilies: []string{config.SignalTypeEmbedding, config.SignalTypeKeyword, config.SignalTypePII},
						},
					},
				},
			},
		},
	}
}

func observedMetaRoutingTestContext() *RequestContext {
	return &RequestContext{
		VSRSelectedCategory: "math",
		VSRSelectionMethod:  "static",
	}
}

func observedMetaRoutingTestSignals() *classification.SignalResults {
	return &classification.SignalResults{
		MatchedKeywordRules:    []string{"reasoning_request_markers"},
		MatchedEmbeddingRules:  []string{"semantic_match"},
		MatchedProjectionRules: []string{"balance_reasoning"},
		SignalConfidences: map[string]float64{
			"keyword:reasoning_request_markers": 0.95,
			"embedding:semantic_match":          0.62,
			"projection:balance_reasoning":      0.77,
		},
		ProjectionBoundaryDistances: map[string]float64{
			"balance_reasoning": 0.05,
		},
		ProjectionPartitionConflicts: []classification.ProjectionPartitionConflict{{
			Name:       "finance-vs-health",
			SignalType: config.SignalTypeDomain,
			Contenders: []string{"finance", "health"},
		}},
	}
}

func observedMetaRoutingTestResult() *decision.DecisionResult {
	return &decision.DecisionResult{
		Decision:            &config.Decision{Name: "route-a"},
		Confidence:          0.66,
		CandidateCount:      2,
		DecisionMargin:      0.08,
		DecisionWinnerBasis: "confidence",
		RunnerUp: &decision.DecisionCandidate{
			Name:       "route-b",
			Confidence: 0.58,
		},
	}
}

func assertObservedMetaRoutingTrace(t *testing.T, ctx *RequestContext) {
	t.Helper()

	if ctx.MetaRoutingTrace == nil {
		t.Fatal("expected meta routing trace")
	}
	assertObservedMetaRoutingEnvelope(t, ctx.MetaRoutingTrace)
	pass := ctx.MetaRoutingTrace.Passes[0]
	assertObservedMetaRoutingPass(t, pass)
	assertObservedMetaRoutingAssessment(t, pass)
	assertObservedMetaRoutingPlan(t, ctx.MetaRoutingTrace)
}

func assertObservedMetaRoutingEnvelope(t *testing.T, trace *RoutingTrace) {
	t.Helper()

	if trace.Mode != config.MetaRoutingModeObserve {
		t.Fatalf("trace mode = %q, want observe", trace.Mode)
	}
	if trace.PolicyProvider == nil {
		t.Fatal("expected policy provider metadata")
	}
	if trace.PolicyProvider.Kind != "deterministic" {
		t.Fatalf("policy provider kind = %q, want deterministic", trace.PolicyProvider.Kind)
	}
	if trace.PassCount != 1 {
		t.Fatalf("pass_count = %d, want 1", trace.PassCount)
	}
}

func assertObservedMetaRoutingPass(t *testing.T, pass PassTrace) {
	t.Helper()

	if pass.DecisionName != "route-a" {
		t.Fatalf("decision_name = %q, want route-a", pass.DecisionName)
	}
	if pass.DecisionCandidateCount != 2 {
		t.Fatalf("decision_candidate_count = %d, want 2", pass.DecisionCandidateCount)
	}
	if pass.TraceQuality.SignalDominance < 0.33 || pass.TraceQuality.SignalDominance > 0.34 {
		t.Fatalf("signal_dominance = %.3f, want about 0.333", pass.TraceQuality.SignalDominance)
	}
	if pass.TraceQuality.AvgSignalConfidence < 0.77 || pass.TraceQuality.AvgSignalConfidence > 0.79 {
		t.Fatalf("avg_signal_confidence = %.3f, want about 0.78", pass.TraceQuality.AvgSignalConfidence)
	}
	if pass.TraceQuality.ProjectionBoundaryMinDistance == nil || *pass.TraceQuality.ProjectionBoundaryMinDistance != 0.05 {
		t.Fatalf("projection_boundary_min_distance = %+v, want 0.05", pass.TraceQuality.ProjectionBoundaryMinDistance)
	}
	assertContainsString(t, pass.PartitionConflicts, "finance-vs-health")
}

func assertObservedMetaRoutingAssessment(t *testing.T, pass PassTrace) {
	t.Helper()

	if pass.Assessment == nil || !pass.Assessment.NeedsRefine {
		t.Fatalf("assessment = %+v, want needs_refine=true", pass.Assessment)
	}
	assertContainsString(t, pass.Assessment.Triggers, metaRoutingTriggerLowDecisionMargin)
	assertContainsString(t, pass.Assessment.Triggers, metaRoutingTriggerProjectionBoundary)
	assertContainsString(t, pass.Assessment.Triggers, metaRoutingTriggerPartitionConflict)
	assertContainsString(t, pass.Assessment.Triggers, metaRoutingTriggerRequiredFamilyLowConf)
	assertContainsString(t, pass.Assessment.Triggers, metaRoutingTriggerSignalFamilyDisagreement)
	assertContainsString(t, pass.Assessment.RootCauses, metaRoutingCauseCompressionLossRisk)
}

func assertObservedMetaRoutingPlan(t *testing.T, trace *RoutingTrace) {
	t.Helper()

	if trace.FinalPlan == nil {
		t.Fatal("expected final refinement plan")
	}
	if len(trace.FinalPlan.Actions) != 2 {
		t.Fatalf("plan actions = %+v, want 2 actions", trace.FinalPlan.Actions)
	}
	assertContainsString(t, trace.RefinedSignalFamilies, config.SignalTypeEmbedding)
	assertContainsString(t, trace.RefinedSignalFamilies, config.SignalTypeKeyword)
	if slices.Contains(trace.RefinedSignalFamilies, config.SignalTypePII) {
		t.Fatalf("unexpected unrelated refined family in %+v", trace.RefinedSignalFamilies)
	}
}

func TestRecordMetaRoutingBasePassNoopWhenDisabled(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{},
	}
	ctx := &RequestContext{}

	router.recordMetaRoutingBasePass(
		ctx,
		signalEvaluationInput{evaluationText: "x", compressedText: "x"},
		&classification.SignalResults{},
		nil,
		"",
		time.Millisecond,
	)

	if ctx.MetaRoutingTrace != nil {
		t.Fatalf("expected nil trace when meta routing disabled, got %+v", ctx.MetaRoutingTrace)
	}
}

func assertContainsString(t *testing.T, values []string, want string) {
	t.Helper()
	if !slices.Contains(values, want) {
		t.Fatalf("values = %v, want %q", values, want)
	}
}

func metaBoolPtr(v bool) *bool {
	return &v
}

func metaFloat64Ptr(v float64) *float64 {
	return &v
}
