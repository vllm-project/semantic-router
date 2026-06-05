package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/lookuptable"
)

func TestApplyModelSwitchGateEnforcesStay(t *testing.T) {
	router := routerWithModelSwitchGate(selection.ModelSwitchGateModeEnforce)
	selCtx, result := modelSwitchGateSelectionInput()
	selected := &selCtx.CandidateModels[1]

	got, applied := router.applyModelSwitchGate(selCtx, result, selected, &RequestContext{
		RequestID:     "req-1",
		PreviousModel: "current",
	})

	if !applied {
		t.Fatalf("expected gate to apply enforced stay")
	}
	if got.Model != "current" {
		t.Fatalf("expected current model, got %q", got.Model)
	}
}

func TestApplyModelSwitchGateShadowDoesNotOverride(t *testing.T) {
	router := routerWithModelSwitchGate(selection.ModelSwitchGateModeShadow)
	selCtx, result := modelSwitchGateSelectionInput()
	selected := &selCtx.CandidateModels[1]

	got, applied := router.applyModelSwitchGate(selCtx, result, selected, &RequestContext{
		RequestID:     "req-1",
		PreviousModel: "current",
	})

	if applied {
		t.Fatalf("shadow mode must not override selected model")
	}
	if got.Model != "candidate" {
		t.Fatalf("expected candidate model, got %q", got.Model)
	}
}

func TestApplyModelSwitchGateChatCompletionsAuditOnly(t *testing.T) {
	// A first-turn / unresolvable request still has no prior model; the gate must
	// stay audit-only and never override the selector even in enforce.
	router := routerWithModelSwitchGate(selection.ModelSwitchGateModeEnforce)
	selCtx, result := modelSwitchGateSelectionInput()
	selCtx.SessionID = "" // simulate a request with no session derived yet
	selected := &selCtx.CandidateModels[1]

	got, applied := router.applyModelSwitchGate(selCtx, result, selected, &RequestContext{
		RequestID:     "req-cc",
		PreviousModel: "", // no prior model yet (first turn / unresolved session)
	})
	if applied {
		t.Fatalf("enforce must not override when previous_model is missing")
	}
	if got.Model != "candidate" {
		t.Fatalf("expected candidate model unchanged, got %q", got.Model)
	}
}

func TestEstimateGateCacheWarmthMissingHistoryReturnsFalse(t *testing.T) {
	warmth, ok := estimateGateCacheWarmth("model-with-no-history-xyz", time.Now())
	if ok {
		t.Fatalf("expected ok=false with no history, got warmth=%v", warmth)
	}
	if warmth != 0 {
		t.Fatalf("expected warmth=0 when missing, got %v", warmth)
	}
}

func TestEstimateGateCacheWarmthEmptyModel(t *testing.T) {
	if _, ok := estimateGateCacheWarmth("", time.Now()); ok {
		t.Fatalf("empty model must return ok=false")
	}
}

func TestApplyModelSwitchGateHonorsRetentionKeepCurrentModel(t *testing.T) {
	// Even in shadow mode, an explicit EMIT retention { keep_current_model: true }
	// forces a stay on the previous model (the directive is not a heuristic).
	router := routerWithModelSwitchGate(selection.ModelSwitchGateModeShadow)
	selCtx, result := modelSwitchGateSelectionInput()
	selected := &selCtx.CandidateModels[1] // "candidate"

	got, applied := router.applyModelSwitchGate(selCtx, result, selected, &RequestContext{
		RequestID:        "req-keep",
		PreviousModel:    "current",
		EmittedRetention: &config.RetentionDirective{KeepCurrentModel: boolPtr(true)},
	})

	if !applied {
		t.Fatalf("keep_current_model must override the selected model even in shadow mode")
	}
	if got.Model != "current" {
		t.Fatalf("expected current model, got %q", got.Model)
	}
}

func routerWithModelSwitchGate(mode string) *OpenAIRouter {
	lt := lookuptable.NewMemoryStorage()
	_ = lt.Set(lookuptable.HandoffPenaltyKey("current", "candidate"), lookuptable.Entry{Value: 0.05})

	return &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{ModelSelection: config.ModelSelectionConfig{
				ModelSwitchGate: config.ModelSwitchGateConfig{
					Enabled:           true,
					Mode:              mode,
					CacheWarmthWeight: 0.1,
				},
			}},
		},
		LookupTable: lt,
	}
}

func modelSwitchGateSelectionInput() (*selection.SelectionContext, *selection.SelectionResult) {
	return &selection.SelectionContext{
			DecisionName:    "coding-route",
			CategoryName:    "coding",
			SessionID:       "session-1",
			CandidateModels: []config.ModelRef{{Model: "current"}, {Model: "candidate"}},
		}, &selection.SelectionResult{
			SelectedModel: "candidate",
			Score:         0.72,
			AllScores: map[string]float64{
				"current":   0.70,
				"candidate": 0.72,
			},
		}
}
