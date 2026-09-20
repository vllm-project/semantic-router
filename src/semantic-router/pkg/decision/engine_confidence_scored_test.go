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

// Within one tier, a keyword leaf reports no measurement at all, so the pool
// falls back to priority ordering: legal_specific (priority 180) wins over
// generic_catch (priority 60) instead of losing to a structural constant.
// legal_specific is itself not comparable here, because its confidence would
// be a mean over two evidence leaves, which moves as leaves are added.
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
	if result.ConfidenceScored {
		t.Fatalf("legal_specific must not be comparable: it aggregates two evidence leaves")
	}
}

// A pool whose members each report one score of the same kind still ranks by
// confidence inside the tier: the reported 0.95 beats the reported 0.86 even
// though the higher priority sits on the other decision.
func TestTieredSelectionSameKindPoolRanksByConfidence(t *testing.T) {
	engine := NewDecisionEngine(nil, nil, nil, []config.Decision{
		{Name: "legal_route", Tier: 1, Priority: 180, Rules: config.RuleNode{Type: "domain", Name: "law"}},
		{Name: "health_route", Tier: 1, Priority: 60, Rules: config.RuleNode{Type: "domain", Name: "health"}},
	}, "priority")

	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		DomainRules:       []string{"law", "health"},
		SignalConfidences: map[string]float64{"domain:law": 0.86, "domain:health": 0.95},
	})
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals() error = %v", err)
	}
	if result == nil || result.Decision == nil {
		t.Fatal("no decision matched")
	}
	if result.Decision.Name != "health_route" {
		t.Fatalf("winner = %s, want health_route (0.95 against 0.86, both probabilities)", result.Decision.Name)
	}
}

// A probability and a similarity are different quantities, so a pool holding
// both ranks by priority even though every member reported a score.
func TestPoolWithDifferentScoreKindsFallsBackToPriority(t *testing.T) {
	engine := NewDecisionEngine(nil, nil, nil, []config.Decision{
		{Name: "domain_route", Tier: 1, Priority: 180, Rules: config.RuleNode{Type: "domain", Name: "law"}},
		{Name: "embedding_route", Tier: 1, Priority: 60, Rules: config.RuleNode{Type: "embedding", Name: "legal_analysis"}},
	}, "priority")

	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		DomainRules:       []string{"law"},
		EmbeddingRules:    []string{"legal_analysis"},
		SignalConfidences: map[string]float64{"domain:law": 0.70, "embedding:legal_analysis": 0.99},
	})
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals() error = %v", err)
	}
	if result == nil || result.Decision == nil {
		t.Fatal("no decision matched")
	}
	if result.Decision.Name != "domain_route" {
		t.Fatalf("winner = %s, want domain_route (priority decides across score kinds)", result.Decision.Name)
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
