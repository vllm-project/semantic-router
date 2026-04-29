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
		CacheWarmthOK:  true,
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
	if !decision.NetSwitchAdvantageOK {
		t.Fatalf("expected NetSwitchAdvantageOK=true after evidence collection")
	}
	if decision.SwitchCostEstimate < 0 || decision.SwitchCostEstimate > 1 {
		t.Fatalf("switch_cost_estimate must be clamped to [0,1], got %v", decision.SwitchCostEstimate)
	}
}

func TestModelSwitchGateAllowsSwitchWhenAdvantageWins(t *testing.T) {
	lt := lookuptable.NewMemoryStorage()
	mustSetLookupEntry(t, lt, lookuptable.HandoffPenaltyKey("current", "candidate"), 0.03)
	mustSetLookupEntry(t, lt, lookuptable.QualityGapKey("coding", "current", "candidate"), 0.30)

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
		CacheWarmthOK:  true,
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

func TestModelSwitchGateMissingPreviousModelBlocks(t *testing.T) {
	// previous_model is the only blocking session signal: if it is missing the
	// gate cannot compare stay vs. switch and must fall back to audit-only.
	gate := NewModelSwitchGate(config.ModelSwitchGateConfig{
		Enabled: true,
		Mode:    ModelSwitchGateModeEnforce,
	}, nil)

	decision := gate.Evaluate(ModelSwitchGateInput{
		SelectionContext: &SelectionContext{
			CandidateModels: []config.ModelRef{{Model: "current"}, {Model: "candidate"}},
		},
		SelectionResult: &SelectionResult{SelectedModel: "candidate"},
		CurrentModel:    "", // missing — Chat Completions today
		CandidateModel:  "candidate",
	})

	if decision.WouldSwitch || decision.EnforcedStay {
		t.Fatalf("missing previous_model must fall back without enforcing, got decision: %+v", decision)
	}
	if decision.Reason != "missing_signal_fallback" {
		t.Fatalf("expected missing_signal_fallback, got %q", decision.Reason)
	}
	if !slices.Contains(decision.MissingSignals, "previous_model") {
		t.Fatalf("expected missing previous_model, got %v", decision.MissingSignals)
	}
}

func TestModelSwitchGateSessionIDInformationalOnly(t *testing.T) {
	// session_id missing must NOT block evidence collection: shadow-mode audit
	// of Chat Completions traffic depends on the gate continuing to evaluate.
	lt := lookuptable.NewMemoryStorage()
	mustSetLookupEntry(t, lt, lookuptable.HandoffPenaltyKey("current", "candidate"), 0.03)
	mustSetLookupEntry(t, lt, lookuptable.QualityGapKey("coding", "current", "candidate"), 0.20)

	gate := NewModelSwitchGate(config.ModelSwitchGateConfig{
		Enabled:            true,
		Mode:               ModelSwitchGateModeShadow,
		MinSwitchAdvantage: 0,
	}, lt)

	selCtx := selectionContextForGate()
	selCtx.SessionID = "" // Chat Completions: not derivable at gate time

	decision := gate.Evaluate(ModelSwitchGateInput{
		SelectionContext: selCtx,
		SelectionResult: &SelectionResult{
			SelectedModel: "candidate",
			AllScores: map[string]float64{
				"current":   0.50,
				"candidate": 0.60,
			},
		},
		CurrentModel:   "current",
		CandidateModel: "candidate",
	})

	if decision.Reason == "missing_signal_fallback" {
		t.Fatalf("session_id missing must not block evidence: %+v", decision)
	}
	if !decision.NetSwitchAdvantageOK {
		t.Fatalf("evidence should have been collected, got NetSwitchAdvantageOK=false")
	}
	if !slices.Contains(decision.MissingSignals, "session_id") {
		t.Fatalf("session_id should be recorded as informational missing, got %v", decision.MissingSignals)
	}
	if !slices.Contains(decision.MissingSignals, "cache_warmth") {
		t.Fatalf("cache_warmth should be recorded as missing when CacheWarmthOK=false, got %v", decision.MissingSignals)
	}
}

func TestModelSwitchGateSwitchBenefitPrefersQualityGap(t *testing.T) {
	// When both QualityGap and SelectorScoreDelta are available, switchBenefit
	// must use QualityGap alone (no double-counting). With QualityGap=0.05,
	// HandoffPenalty=0.0, and MinSwitchAdvantage=0.10, the gate should NOT
	// switch even though SelectorScoreDelta=0.40 alone would push advantage
	// above the threshold.
	lt := lookuptable.NewMemoryStorage()
	mustSetLookupEntry(t, lt, lookuptable.QualityGapKey("coding", "current", "candidate"), 0.05)
	mustSetLookupEntry(t, lt, lookuptable.HandoffPenaltyKey("current", "candidate"), 0)

	gate := NewModelSwitchGate(config.ModelSwitchGateConfig{
		Enabled:            true,
		Mode:               ModelSwitchGateModeShadow,
		MinSwitchAdvantage: 0.10,
	}, lt)

	decision := gate.Evaluate(ModelSwitchGateInput{
		SelectionContext: selectionContextForGate(),
		SelectionResult: &SelectionResult{
			SelectedModel: "candidate",
			AllScores: map[string]float64{
				"current":   0.45,
				"candidate": 0.85,
			},
		},
		CurrentModel:   "current",
		CandidateModel: "candidate",
		CacheWarmth:    0,
		CacheWarmthOK:  true,
	})

	if decision.WouldSwitch {
		t.Fatalf("would_switch must be false when QualityGap (preferred) is below threshold; got %+v", decision)
	}
	if decision.NetSwitchAdvantage > 0.06 {
		t.Fatalf("net advantage should equal QualityGap (≈0.05), got %v — possible double-counting with SelectorScoreDelta", decision.NetSwitchAdvantage)
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
