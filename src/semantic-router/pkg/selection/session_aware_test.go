package selection

import (
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestSessionAwareSelectorToolLoopHardLocksCurrentModel(t *testing.T) {
	selector := NewSessionAwareSelector(&SessionAwareConfig{
		FallbackMethod:         MethodStatic,
		ToolLoopHardLock:       true,
		ToolLoopStayBias:       0.35,
		StayBias:               0.10,
		QualityGapMultiplier:   1,
		MaxCacheCostMultiplier: 2.5,
	})
	selector.SetFallbackSelector(stubSessionFallback{
		result: &SelectionResult{
			SelectedModel: "frontier",
			Score:         0.95,
			Method:        MethodHybrid,
			AllScores: map[string]float64{
				"current":  0.40,
				"frontier": 0.95,
			},
		},
	})

	result, err := selector.Select(context.Background(), &SelectionContext{
		CandidateModels: []config.ModelRef{{Model: "current"}, {Model: "frontier"}},
		AgenticSession: &AgenticSessionContext{
			ID:             "sess-1",
			TurnIndex:      3,
			PreviousModel:  "current",
			ActiveToolLoop: true,
			Phase:          AgenticPhaseToolLoop,
		},
	})
	if err != nil {
		t.Fatalf("Select returned error: %v", err)
	}
	if result.SelectedModel != "current" {
		t.Fatalf("expected current model to be hard-locked, got %q", result.SelectedModel)
	}
	if result.SessionPolicy == nil || !result.SessionPolicy.HardLocked {
		t.Fatalf("expected hard-lock policy trace, got %#v", result.SessionPolicy)
	}
	if result.SessionPolicy.HardLockReason != "hard_lock=tool_loop" {
		t.Fatalf("unexpected hard-lock reason: %q", result.SessionPolicy.HardLockReason)
	}
}

func TestSessionAwareSelectorIdleSessionAllowsFallbackSwitch(t *testing.T) {
	selector := NewSessionAwareSelector(&SessionAwareConfig{
		FallbackMethod:         MethodStatic,
		IdleTimeoutSeconds:     60,
		SwitchMargin:           0,
		StayBias:               0,
		ToolLoopHardLock:       true,
		PrefixCacheWeight:      0,
		HandoffPenaltyWeight:   0,
		DefaultHandoffPenalty:  0,
		QualityGapMultiplier:   1,
		MaxCacheCostMultiplier: 2.5,
	})
	selector.SetFallbackSelector(stubSessionFallback{
		result: &SelectionResult{
			SelectedModel: "frontier",
			Score:         0.90,
			Method:        MethodHybrid,
			AllScores: map[string]float64{
				"current":  0.20,
				"frontier": 0.90,
			},
		},
	})

	result, err := selector.Select(context.Background(), &SelectionContext{
		CandidateModels: []config.ModelRef{{Model: "current"}, {Model: "frontier"}},
		AgenticSession: &AgenticSessionContext{
			ID:            "sess-1",
			TurnIndex:     4,
			PreviousModel: "current",
			IdleFor:       2 * time.Minute,
			IdleKnown:     true,
		},
	})
	if err != nil {
		t.Fatalf("Select returned error: %v", err)
	}
	if result.SelectedModel != "frontier" {
		t.Fatalf("expected idle session to accept stronger fallback choice, got %q", result.SelectedModel)
	}
	if result.SessionPolicy == nil || !result.SessionPolicy.IdleExpired {
		t.Fatalf("expected idle-expired policy trace, got %#v", result.SessionPolicy)
	}
	if trace := result.SessionPolicy.CandidateTraces["frontier"]; trace.NetSwitchAdvantage <= 0 {
		t.Fatalf("expected positive frontier switch advantage, got %#v", trace)
	}
}

func TestSessionAwareSelectorIdleSessionClearsContinuityPenalty(t *testing.T) {
	selector := NewSessionAwareSelector(&SessionAwareConfig{
		FallbackMethod:         MethodStatic,
		IdleTimeoutSeconds:     60,
		SwitchMargin:           0.05,
		StayBias:               0.10,
		ToolLoopHardLock:       true,
		PrefixCacheWeight:      0.20,
		HandoffPenaltyWeight:   1,
		DefaultHandoffPenalty:  0.05,
		QualityGapMultiplier:   1,
		MaxCacheCostMultiplier: 2.5,
		SwitchHistoryWeight:    0.04,
	})
	selector.SetFallbackSelector(stubSessionFallback{
		result: &SelectionResult{
			SelectedModel: "frontier",
			Score:         1.0,
			Method:        MethodStatic,
			AllScores: map[string]float64{
				"current":  0.999,
				"frontier": 1.0,
			},
		},
	})

	result, err := selector.Select(context.Background(), &SelectionContext{
		CandidateModels: []config.ModelRef{{Model: "frontier"}, {Model: "current"}},
		AgenticSession: &AgenticSessionContext{
			ID:                "sess-1",
			TurnIndex:         5,
			PreviousModel:     "current",
			MemorySwitchCount: 4,
			HistoryTokens:     4096,
			ContextTokens:     8192,
			CacheWarmth:       1,
			CacheWarmthOK:     true,
			IdleFor:           2 * time.Minute,
			IdleKnown:         true,
		},
	})
	if err != nil {
		t.Fatalf("Select returned error: %v", err)
	}
	if result.SelectedModel != "frontier" {
		t.Fatalf("expected idle boundary to reselect fallback frontier, got %q", result.SelectedModel)
	}

	trace := result.SessionPolicy.CandidateTraces["frontier"]
	if trace.HandoffPenalty != 0 || trace.PrefixCachePenalty != 0 || trace.SwitchHistoryPenalty != 0 {
		t.Fatalf("expected idle boundary to clear continuity penalties, got %#v", trace)
	}
	if trace.NetSwitchAdvantage <= 0 {
		t.Fatalf("expected positive net switch advantage at idle boundary, got %#v", trace)
	}
}

func TestSessionAwareSelectorUsesRouterMemoryTurnCountBeforeFullParsing(t *testing.T) {
	selector := NewSessionAwareSelector(&SessionAwareConfig{
		FallbackMethod:         MethodStatic,
		MinTurnsBeforeSwitch:   1,
		SwitchMargin:           0,
		StayBias:               0,
		ToolLoopHardLock:       true,
		PrefixCacheWeight:      0,
		HandoffPenaltyWeight:   0,
		DefaultHandoffPenalty:  0,
		QualityGapMultiplier:   1,
		MaxCacheCostMultiplier: 2.5,
	})
	selector.SetFallbackSelector(stubSessionFallback{
		result: &SelectionResult{
			SelectedModel: "frontier",
			Score:         0.90,
			Method:        MethodHybrid,
			AllScores: map[string]float64{
				"current":  0.20,
				"frontier": 0.90,
			},
		},
	})

	result, err := selector.Select(context.Background(), &SelectionContext{
		CandidateModels: []config.ModelRef{{Model: "current"}, {Model: "frontier"}},
		AgenticSession: &AgenticSessionContext{
			ID:              "sess-1",
			TurnIndex:       0,
			MemoryTurnCount: 2,
			PreviousModel:   "current",
		},
	})
	if err != nil {
		t.Fatalf("Select returned error: %v", err)
	}
	if result.SelectedModel != "frontier" {
		t.Fatalf("expected memory turn count to satisfy min-turn lock, got %q", result.SelectedModel)
	}
	if result.SessionPolicy == nil || result.SessionPolicy.HardLocked {
		t.Fatalf("expected non-hard-locked policy trace, got %#v", result.SessionPolicy)
	}
}

func TestSessionAwareNetSwitchAdvantageUsesAdjustedCurrentScore(t *testing.T) {
	selector := NewSessionAwareSelector(&SessionAwareConfig{
		FallbackMethod:         MethodStatic,
		SwitchMargin:           0,
		StayBias:               0.10,
		PrefixCacheWeight:      0,
		HandoffPenaltyWeight:   0,
		DefaultHandoffPenalty:  0,
		QualityGapMultiplier:   1,
		MaxCacheCostMultiplier: 1,
	})
	selector.SetFallbackSelector(stubSessionFallback{
		result: &SelectionResult{
			SelectedModel: "frontier",
			Score:         0.62,
			Method:        MethodHybrid,
			AllScores: map[string]float64{
				"current":  0.50,
				"frontier": 0.62,
			},
		},
	})

	result, err := selector.Select(context.Background(), &SelectionContext{
		CandidateModels: []config.ModelRef{{Model: "current"}, {Model: "frontier"}},
		AgenticSession: &AgenticSessionContext{
			ID:            "sess-1",
			TurnIndex:     3,
			PreviousModel: "current",
		},
	})
	if err != nil {
		t.Fatalf("Select returned error: %v", err)
	}

	trace := result.SessionPolicy.CandidateTraces["frontier"]
	if trace.NetSwitchAdvantage < 0.019 || trace.NetSwitchAdvantage > 0.021 {
		t.Fatalf("expected net advantage to subtract adjusted current score, got %#v", trace)
	}
}

func TestSessionAwareContinuationMassUsesRouterMemoryTurns(t *testing.T) {
	got := sessionContinuationMass(&AgenticSessionContext{
		TurnIndex:       0,
		MemoryTurnCount: 4,
	})
	if got <= 0 {
		t.Fatalf("expected router memory turn count to contribute continuation mass, got %.4f", got)
	}
}

func TestSessionAwarePrefixPenaltyScalesWithModelCost(t *testing.T) {
	selector := NewSessionAwareSelector(&SessionAwareConfig{
		PrefixCacheWeight:      0.2,
		QualityGapMultiplier:   1,
		MaxCacheCostMultiplier: 3,
	})
	selector.InitializeFromConfig(map[string]config.ModelParams{
		"cheap": {
			Pricing: config.ModelPricing{PromptPer1M: 0.05, CachedInputPer1M: 0.01, CompletionPer1M: 0.10},
		},
		"frontier": {
			Pricing: config.ModelPricing{PromptPer1M: 10, CachedInputPer1M: 1, CompletionPer1M: 30},
		},
	})
	session := &AgenticSessionContext{
		TurnIndex:     5,
		HistoryTokens: 4096,
		ContextTokens: 8192,
		CacheWarmth:   1,
		CacheWarmthOK: true,
	}

	cheapPenalty := selector.prefixCachePenalty(session, "cheap", "cheap-2", false)
	frontierPenalty := selector.prefixCachePenalty(session, "frontier", "cheap", false)
	if frontierPenalty <= cheapPenalty {
		t.Fatalf("expected frontier switch penalty %.4f > cheap penalty %.4f", frontierPenalty, cheapPenalty)
	}
}

func TestSessionAwareCacheCostPressureUsesPrefixCheckoutDelta(t *testing.T) {
	selector := NewSessionAwareSelector(&SessionAwareConfig{MaxCacheCostMultiplier: 3})
	selector.InitializeFromConfig(map[string]config.ModelParams{
		"uncached-frontier": {
			Pricing: config.ModelPricing{PromptPer1M: 10, CachedInputPer1M: 0, CompletionPer1M: 1},
		},
		"cached-frontier": {
			Pricing: config.ModelPricing{PromptPer1M: 10, CachedInputPer1M: 9, CompletionPer1M: 1},
		},
	})

	uncachedPressure := selector.modelCostPressure("uncached-frontier")
	cachedPressure := selector.modelCostPressure("cached-frontier")
	if cachedPressure >= uncachedPressure {
		t.Fatalf("expected cached-input discount to reduce pressure, got cached %.4f >= uncached %.4f", cachedPressure, uncachedPressure)
	}
}

func TestSessionAwareCacheCostPressureIgnoresCompletionOnlyCost(t *testing.T) {
	selector := NewSessionAwareSelector(&SessionAwareConfig{MaxCacheCostMultiplier: 3})
	selector.InitializeFromConfig(map[string]config.ModelParams{
		"output-heavy": {
			Pricing: config.ModelPricing{PromptPer1M: 0.1, CompletionPer1M: 100},
		},
		"input-heavy": {
			Pricing: config.ModelPricing{PromptPer1M: 10, CachedInputPer1M: 1, CompletionPer1M: 0},
		},
	})

	outputHeavyPressure := selector.modelCostPressure("output-heavy")
	inputHeavyPressure := selector.modelCostPressure("input-heavy")
	if outputHeavyPressure >= inputHeavyPressure {
		t.Fatalf("expected prompt checkout cost to dominate pressure, got output %.4f >= input %.4f", outputHeavyPressure, inputHeavyPressure)
	}
}

func TestSessionAwareRouterMemorySwitchHistoryPenalty(t *testing.T) {
	selector := NewSessionAwareSelector(&SessionAwareConfig{SwitchHistoryWeight: 0.16})
	session := &AgenticSessionContext{MemorySwitchCount: 4}

	if got := selector.switchHistoryPenalty(session); got <= 0 {
		t.Fatalf("expected positive switch-history penalty, got %.4f", got)
	}
}

type stubSessionFallback struct {
	result *SelectionResult
}

func (s stubSessionFallback) Select(context.Context, *SelectionContext) (*SelectionResult, error) {
	copy := *s.result
	copy.AllScores = cloneScores(s.result.AllScores)
	return &copy, nil
}

func (s stubSessionFallback) Method() SelectionMethod { return MethodHybrid }

func (s stubSessionFallback) UpdateFeedback(context.Context, *Feedback) error { return nil }

func (s stubSessionFallback) Tier() AlgorithmTier { return TierSupported }

func (s stubSessionFallback) ExternalDependencies() []Dependency { return nil }
