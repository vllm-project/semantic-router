package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func confidenceScenarioDecisions(tier int) []config.Decision {
	return []config.Decision{
		{
			Name:     "legal_specific",
			Tier:     tier,
			Priority: 180,
			Rules: config.RuleNode{
				Operator: "AND",
				Conditions: []config.RuleNode{
					{Type: "domain", Name: "law"},
					{Type: "complexity", Name: "legal_risk:hard"},
				},
			},
		},
		{
			Name:     "generic_catch",
			Tier:     tier,
			Priority: 60,
			Rules: config.RuleNode{
				Operator: "OR",
				Conditions: []config.RuleNode{
					{Type: "keyword", Name: "question_markers"},
					{Type: "embedding", Name: "general_chat"},
				},
			},
		},
	}
}

func confidenceScenarioSignals() *SignalMatches {
	return &SignalMatches{
		DomainRules:     []string{"law"},
		ComplexityRules: []string{"legal_risk:hard"},
		KeywordRules:    []string{"question_markers"},
		SignalConfidences: map[string]float64{
			"domain:law":                 0.86,
			"complexity:legal_risk:hard": 0.61,
			// keyword signal reports no confidence entry, as in production:
			// evaluateKeywordSignal never writes SignalConfidences.
		},
	}
}

// Within one tier, a keyword leaf with no reported confidence defaults to the
// structural constant 1.0. That constant is not comparable with reported
// scores, so the pool must fall back to priority ordering: legal_specific
// (priority 180) wins over the unscored generic_catch (priority 60) instead
// of losing to its default 1.0.
func TestTieredSelectionUnscoredPoolFallsBackToPriority(t *testing.T) {
	engine := NewDecisionEngine(nil, nil, nil, confidenceScenarioDecisions(1), "priority")

	result, err := engine.EvaluateDecisionsWithSignals(confidenceScenarioSignals())
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals() error = %v", err)
	}
	if result == nil || result.Decision == nil {
		t.Fatal("no decision matched")
	}
	t.Logf("winner=%s confidence=%.3f scored=%v matchedRules=%v",
		result.Decision.Name, result.Confidence, result.ConfidenceScored, result.MatchedRules)
	if result.Decision.Name != "legal_specific" {
		t.Fatalf("winner = %s, want legal_specific via priority fallback in a pool with unscored confidence", result.Decision.Name)
	}
	if !result.ConfidenceScored {
		t.Fatalf("legal_specific should be confidence-scored (domain and complexity both reported)")
	}
}

// The same pool with every competitor reporting a confidence keeps the
// existing within-tier confidence ordering: generic_catch's honestly
// reported 0.95 beats legal_specific's 0.735 mean.
func TestTieredSelectionFullyScoredPoolStillRanksByConfidence(t *testing.T) {
	engine := NewDecisionEngine(nil, nil, nil, confidenceScenarioDecisions(1), "priority")

	signals := confidenceScenarioSignals()
	signals.SignalConfidences["keyword:question_markers"] = 0.95

	result, err := engine.EvaluateDecisionsWithSignals(signals)
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals() error = %v", err)
	}
	if result == nil || result.Decision == nil {
		t.Fatal("no decision matched")
	}
	t.Logf("winner=%s confidence=%.3f", result.Decision.Name, result.Confidence)
	if result.Decision.Name != "generic_catch" {
		t.Fatalf("winner = %s, want generic_catch (reported 0.95 vs 0.735 in a fully scored pool)", result.Decision.Name)
	}
}

// Same decisions and signals without tiers under the default priority
// strategy: priority is compared first, so legal_specific wins regardless.
func TestPriorityStrategyWithoutTiersConsultsPriorityFirst(t *testing.T) {
	engine := NewDecisionEngine(nil, nil, nil, confidenceScenarioDecisions(0), "priority")

	result, err := engine.EvaluateDecisionsWithSignals(confidenceScenarioSignals())
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals() error = %v", err)
	}
	if result == nil || result.Decision == nil {
		t.Fatal("no decision matched")
	}
	t.Logf("winner=%s confidence=%.3f", result.Decision.Name, result.Confidence)
	if result.Decision.Name != "legal_specific" {
		t.Fatalf("winner = %s, want legal_specific under priority strategy", result.Decision.Name)
	}
}

// A catch-all fallback must keep losing to signal-backed decisions inside a
// tier even when the pool is not confidence-comparable — the priority
// fallback must not let a high-priority catch-all outrank real matches.
func TestCatchAllStaysLastInUnscoredTierPool(t *testing.T) {
	engine := NewDecisionEngine(nil, nil, nil, []config.Decision{
		{
			Name:     "keyword_route",
			Tier:     1,
			Priority: 50,
			Rules:    config.RuleNode{Type: "keyword", Name: "urgent"},
		},
		{
			Name:     "fallback_route",
			Tier:     1,
			Priority: 100,
			Rules:    config.RuleNode{Operator: "AND"},
		},
	}, "priority")

	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		KeywordRules: []string{"urgent"},
	})
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals() error = %v", err)
	}
	if result == nil || result.Decision == nil {
		t.Fatal("no decision matched")
	}
	if result.Decision.Name != "keyword_route" {
		t.Fatalf("winner = %s, want keyword_route (catch-all must rank last within the tier)", result.Decision.Name)
	}
}

// An OR that matched only through an unscored branch makes the decision
// unscored even when a scored branch exists but did not match.
func TestORUnscoredWinnerPropagates(t *testing.T) {
	engine := NewDecisionEngine(nil, nil, nil, []config.Decision{
		{
			Name:     "mixed_or",
			Tier:     1,
			Priority: 10,
			Rules: config.RuleNode{
				Operator: "OR",
				Conditions: []config.RuleNode{
					{Type: "embedding", Name: "semantic"},
					{Type: "keyword", Name: "marker"},
				},
			},
		},
		{
			Name:     "scored_route",
			Tier:     1,
			Priority: 20,
			Rules:    config.RuleNode{Type: "domain", Name: "law"},
		},
	}, "priority")

	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		KeywordRules: []string{"marker"},
		DomainRules:  []string{"law"},
		SignalConfidences: map[string]float64{
			"domain:law": 0.9,
		},
	})
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals() error = %v", err)
	}
	if result == nil || result.Decision == nil {
		t.Fatal("no decision matched")
	}
	if result.Decision.Name != "scored_route" {
		t.Fatalf("winner = %s, want scored_route via priority fallback (mixed_or's winning branch is unscored)", result.Decision.Name)
	}
}
