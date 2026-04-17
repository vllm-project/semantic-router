package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/lookuptable"
)

func TestApplyModelSwitchGateEnforcesStay(t *testing.T) {
	router := routerWithModelSwitchGate(selection.ModelSwitchGateModeEnforce)
	selCtx, result := modelSwitchGateSelectionInput()
	selected := &selCtx.CandidateModels[1]

	got, applied := router.applyModelSwitchGate(selCtx, result, selected, &RequestContext{
		RequestID:           "req-1",
		PreviousModel:       "current",
		CacheWarmthEstimate: 0.8,
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
		RequestID:           "req-1",
		PreviousModel:       "current",
		CacheWarmthEstimate: 0.8,
	})

	if applied {
		t.Fatalf("shadow mode must not override selected model")
	}
	if got.Model != "candidate" {
		t.Fatalf("expected candidate model, got %q", got.Model)
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
