package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestChooseMetaRoutingFinalPassHonorsMode(t *testing.T) {
	base := &metaRoutingPassExecution{
		decisionName:  "base-route",
		selectedModel: "model-a",
		result:        dummyDecisionResult(),
	}
	refined := &metaRoutingPassExecution{
		decisionName:  "refined-route",
		selectedModel: "model-b",
		result:        dummyDecisionResult(),
	}

	if got := chooseMetaRoutingFinalPass(config.MetaRoutingModeShadow, base, refined); got != base {
		t.Fatalf("shadow mode chose %+v, want base", got)
	}
	if got := chooseMetaRoutingFinalPass(config.MetaRoutingModeActive, base, refined); got != refined {
		t.Fatalf("active mode chose %+v, want refined", got)
	}
}

func TestChooseMetaRoutingFinalPassFallsBackWhenRefinementHasNoDecision(t *testing.T) {
	base := &metaRoutingPassExecution{
		decisionName:  "base-route",
		selectedModel: "model-a",
		result:        dummyDecisionResult(),
	}
	refined := &metaRoutingPassExecution{
		selectedModel: "model-b",
	}

	if got := chooseMetaRoutingFinalPass(config.MetaRoutingModeActive, base, refined); got != base {
		t.Fatalf("active mode chose %+v, want base fallback", got)
	}
}

func TestEmitMetaRoutingFeedbackPersistsOnce(t *testing.T) {
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(true, false, 64*1024)
	router := &OpenAIRouter{
		FeedbackRecorder: recorder,
	}

	ctx := &RequestContext{
		RequestID:          "req-123",
		RequestModel:       "MoM",
		RequestQuery:       "test query",
		VSRSelectionMethod: "static",
		VSRMatchedUserFeedback: []string{
			"satisfied",
		},
		MetaRoutingTrace: &RoutingTrace{
			Mode:                    config.MetaRoutingModeActive,
			PassCount:               2,
			TriggerNames:            []string{metaRoutingTriggerLowDecisionMargin},
			RefinedSignalFamilies:   []string{config.SignalTypeEmbedding},
			OverturnedDecision:      true,
			FinalDecisionName:       "route-b",
			FinalDecisionConfidence: 0.84,
			FinalModel:              "model-b",
			FinalPlan: &RefinementPlan{
				Actions: []RefinementActionPlan{{
					Type:           config.MetaRoutingActionRerunSignalFamilies,
					SignalFamilies: []string{config.SignalTypeEmbedding},
				}},
			},
		},
	}

	router.emitMetaRoutingFeedback(ctx, 200)
	router.emitMetaRoutingFeedback(ctx, 200)

	if !ctx.MetaRoutingFeedbackWritten {
		t.Fatal("expected feedback write marker")
	}
	if ctx.MetaRoutingFeedbackID == "" {
		t.Fatal("expected persisted feedback ID")
	}

	records := router.collectMetaRoutingFeedbackRecords()
	if len(records) != 1 {
		t.Fatalf("feedback records = %d, want 1", len(records))
	}
	record := records[0]
	if !record.Action.Executed {
		t.Fatalf("feedback action = %+v, want executed=true", record.Action)
	}
	if record.Outcome.FinalDecisionName != "route-b" {
		t.Fatalf("final decision = %q, want route-b", record.Outcome.FinalDecisionName)
	}
	if len(record.Action.ExecutedSignalFamilies) != 1 || record.Action.ExecutedSignalFamilies[0] != config.SignalTypeEmbedding {
		t.Fatalf("executed signal families = %v, want [embedding]", record.Action.ExecutedSignalFamilies)
	}
}

func dummyDecisionResult() *decision.DecisionResult {
	return &decision.DecisionResult{
		Decision: &config.Decision{Name: "dummy"},
	}
}
