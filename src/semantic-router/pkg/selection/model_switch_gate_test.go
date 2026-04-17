package selection

import (
	"slices"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/lookuptable"
)

func TestModelSwitchGateKeepsCurrentWhenSwitchCostWins(t *testing.T) {
	lt := lookuptable.NewMemoryStorage()
	mustSetLookupEntry(t, lt, lookuptable.HandoffPenaltyKey("current", "candidate"), 0.05)

	gate := NewModelSwitchGate(config.ModelSwitchGateConfig{
		Enabled:            true,
		Mode:               ModelSwitchGateModeEnforce,
		MinSwitchAdvantage: 0,
		CacheWarmthWeight:  0.1,
	}, lt)

	decision := gate.Evaluate(ModelSwitchGateInput{
		SelectionContext: selectionContextForGate(),
		SelectionResult: &SelectionResult{
			SelectedModel: "candidate",
			Score:         0.72,
			AllScores: map[string]float64{
				"current":   0.70,
				"candidate": 0.72,
			},
		},
		CurrentModel:   "current",
		CandidateModel: "candidate",
		CacheWarmth:    0.8,
	})

	if decision.WouldSwitch {
		t.Fatalf("expected gate to keep current model, got switch decision: %+v", decision)
	}
	if !decision.EnforcedStay {
		t.Fatalf("expected enforce mode to keep current model")
	}
	if decision.FinalModel != "current" {
		t.Fatalf("expected final model current, got %q", decision.FinalModel)
	}
}

func TestModelSwitchGateAllowsSwitchWhenAdvantageWins(t *testing.T) {
	lt := lookuptable.NewMemoryStorage()
	mustSetLookupEntry(t, lt, lookuptable.HandoffPenaltyKey("current", "candidate"), 0.03)
	mustSetLookupEntry(t, lt, lookuptable.QualityGapKey("coding", "current", "candidate"), 0.05)

	gate := NewModelSwitchGate(config.ModelSwitchGateConfig{
		Enabled:            true,
		Mode:               ModelSwitchGateModeShadow,
		MinSwitchAdvantage: 0.1,
	}, lt)

	decision := gate.Evaluate(ModelSwitchGateInput{
		SelectionContext: selectionContextForGate(),
		SelectionResult: &SelectionResult{
			SelectedModel: "candidate",
			Score:         0.85,
			AllScores: map[string]float64{
				"current":   0.45,
				"candidate": 0.85,
			},
		},
		CurrentModel:   "current",
		CandidateModel: "candidate",
		CacheWarmth:    0.1,
	})

	if !decision.WouldSwitch {
		t.Fatalf("expected would_switch=true, got decision: %+v", decision)
	}
	if decision.EnforcedStay {
		t.Fatalf("shadow mode must not enforce stay")
	}
	if decision.FinalModel != "candidate" {
		t.Fatalf("expected final model candidate, got %q", decision.FinalModel)
	}
}

func TestModelSwitchGateMissingSignalsFallback(t *testing.T) {
	gate := NewModelSwitchGate(config.ModelSwitchGateConfig{
		Enabled: true,
		Mode:    ModelSwitchGateModeEnforce,
	}, nil)

	decision := gate.Evaluate(ModelSwitchGateInput{
		SelectionContext: &SelectionContext{
			CandidateModels: []config.ModelRef{{Model: "current"}, {Model: "candidate"}},
		},
		SelectionResult: &SelectionResult{SelectedModel: "candidate"},
		CurrentModel:    "current",
		CandidateModel:  "candidate",
	})

	if decision.WouldSwitch || decision.EnforcedStay {
		t.Fatalf("missing signals should fall back without enforcing, got decision: %+v", decision)
	}
	if decision.Reason != "missing_signal_fallback" {
		t.Fatalf("expected missing_signal_fallback, got %q", decision.Reason)
	}
	if !slices.Contains(decision.MissingSignals, "session_id") {
		t.Fatalf("expected missing session_id, got %v", decision.MissingSignals)
	}
}

func selectionContextForGate() *SelectionContext {
	return &SelectionContext{
		DecisionName:    "coding-route",
		CategoryName:    "coding",
		SessionID:       "session-1",
		CandidateModels: []config.ModelRef{{Model: "current"}, {Model: "candidate"}},
	}
}

func mustSetLookupEntry(t *testing.T, lt lookuptable.LookupTableStorage, key lookuptable.Key, value float64) {
	t.Helper()
	if err := lt.Set(key, lookuptable.Entry{
		Value:     value,
		Source:    lookuptable.SourceManual,
		UpdatedAt: time.Now(),
	}); err != nil {
		t.Fatalf("failed to set lookup entry: %v", err)
	}
}
