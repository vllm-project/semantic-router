package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// The cases below pin the ordering table documented in the decision tutorial
// under docs/tutorials/decision. They describe how ranking behaves today,
// including the two consequences that section calls out.

func rankedWinner(t *testing.T, decisions []config.Decision, strategy config.RoutingStrategy, signals *SignalMatches) *DecisionResult {
	t.Helper()
	result, err := NewDecisionEngine(nil, nil, nil, decisions, strategy).EvaluateDecisionsWithSignals(signals)
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals() error = %v", err)
	}
	if result == nil || result.Decision == nil {
		t.Fatal("no decision matched")
	}
	return result
}

// A decision without a tier is tier 0, so it ranks ahead of every tiered one.
func TestUntieredDecisionOutranksTieredDecision(t *testing.T) {
	untiered := config.Decision{Name: "untiered", Priority: 10, Rules: config.RuleNode{Type: "domain", Name: "law"}}
	tiered := config.Decision{Name: "tiered", Priority: 900, Tier: 1, Rules: config.RuleNode{Type: "domain", Name: "business"}}

	winner := rankedWinner(t, []config.Decision{untiered, tiered}, config.RoutingStrategyPriority, &SignalMatches{
		DomainRules:       []string{"law", "business"},
		SignalConfidences: map[string]float64{"domain:law": 0.60, "domain:business": 0.95},
	})
	if winner.Decision.Name != "untiered" {
		t.Fatalf("winner = %s, want untiered (tier 0 sorts before tier 1)", winner.Decision.Name)
	}
}

// The tiered case never reads routing.strategy, so an unrelated tiered match
// switches an untiered pool from priority ordering to confidence ordering.
func TestMatchedTieredDecisionSwitchesUntieredPoolToConfidence(t *testing.T) {
	highPriority := config.Decision{Name: "high_priority", Priority: 160, Rules: config.RuleNode{Type: "preference", Name: "terse_answers"}}
	highConfidence := config.Decision{Name: "high_confidence", Priority: 135, Rules: config.RuleNode{Type: "language", Name: "es"}}
	tiered := config.Decision{Name: "tiered_fallback", Priority: 200, Tier: 2, Rules: config.RuleNode{Type: "domain", Name: "business"}}

	signals := &SignalMatches{
		PreferenceRules: []string{"terse_answers"},
		LanguageRules:   []string{"es"},
		DomainRules:     []string{"business"},
		SignalConfidences: map[string]float64{
			"preference:terse_answers": 0.70,
			"language:es":              0.99,
			"domain:business":          0.80,
		},
	}

	untieredPool := rankedWinner(t, []config.Decision{highPriority, highConfidence}, config.RoutingStrategyPriority, signals)
	if untieredPool.Decision.Name != "high_priority" {
		t.Fatalf("winner = %s, want high_priority (strategy: priority orders by priority)", untieredPool.Decision.Name)
	}

	withTiered := rankedWinner(t, []config.Decision{highPriority, highConfidence, tiered}, config.RoutingStrategyPriority, signals)
	if withTiered.Decision.Name != "high_confidence" {
		t.Fatalf("winner = %s, want high_confidence (the tiered case ignores strategy: priority)", withTiered.Decision.Name)
	}
}

// A catch-all ranks last everywhere except under strategy: priority.
func TestCatchAllOutranksRealMatchUnderPriorityStrategy(t *testing.T) {
	catchAll := config.Decision{Name: "catch_all", Priority: 500, Rules: config.RuleNode{Operator: "AND"}}
	match := config.Decision{Name: "real_match", Priority: 100, Rules: config.RuleNode{Type: "domain", Name: "law"}}
	signals := &SignalMatches{DomainRules: []string{"law"}, SignalConfidences: map[string]float64{"domain:law": 0.90}}

	underPriority := rankedWinner(t, []config.Decision{catchAll, match}, config.RoutingStrategyPriority, signals)
	if underPriority.Decision.Name != "catch_all" {
		t.Fatalf("winner = %s, want catch_all (strategy: priority has no catch-all rule)", underPriority.Decision.Name)
	}

	underConfidence := rankedWinner(t, []config.Decision{catchAll, match}, config.RoutingStrategyConfidence, signals)
	if underConfidence.Decision.Name != "real_match" {
		t.Fatalf("winner = %s, want real_match (catch-alls rank last here)", underConfidence.Decision.Name)
	}
}

// An OR keeps the winning branch's score only, so an extra keyword match can
// demote a decision that reported a score and drop the pool to priority.
func TestExtraKeywordMatchDropsPoolToPriority(t *testing.T) {
	mixed := config.Decision{Name: "mixed_or", Priority: 10, Tier: 1, Rules: config.RuleNode{
		Operator: "OR",
		Conditions: []config.RuleNode{
			{Type: "embedding", Name: "semantic"},
			{Type: "keyword", Name: "marker"},
		},
	}}
	scored := config.Decision{Name: "scored_route", Priority: 20, Tier: 1, Rules: config.RuleNode{Type: "domain", Name: "law"}}
	decisions := []config.Decision{mixed, scored}
	confidences := map[string]float64{"embedding:semantic": 0.95, "domain:law": 0.80}

	withoutKeyword := rankedWinner(t, decisions, config.RoutingStrategyPriority, &SignalMatches{
		EmbeddingRules: []string{"semantic"}, DomainRules: []string{"law"}, SignalConfidences: confidences,
	})
	if withoutKeyword.Decision.Name != "mixed_or" {
		t.Fatalf("winner = %s, want mixed_or (both members reported scores)", withoutKeyword.Decision.Name)
	}

	withKeyword := rankedWinner(t, decisions, config.RoutingStrategyPriority, &SignalMatches{
		EmbeddingRules: []string{"semantic"}, KeywordRules: []string{"marker"}, DomainRules: []string{"law"}, SignalConfidences: confidences,
	})
	if withKeyword.Decision.Name != "scored_route" {
		t.Fatalf("winner = %s, want scored_route (the keyword branch wins the OR and unscores the pool)", withKeyword.Decision.Name)
	}
}

// AND averages its matched children, so an aggregate falls as a decision
// gains evidence.
func TestANDAggregateFallsAsEvidenceGrows(t *testing.T) {
	twoLeaf := config.Decision{Name: "two_leaf", Priority: 100, Tier: 1, Rules: config.RuleNode{
		Operator:   "AND",
		Conditions: []config.RuleNode{{Type: "embedding", Name: "a"}, {Type: "embedding", Name: "b"}},
	}}
	threeLeaf := config.Decision{Name: "three_leaf", Priority: 200, Tier: 1, Rules: config.RuleNode{
		Operator:   "AND",
		Conditions: []config.RuleNode{{Type: "embedding", Name: "a"}, {Type: "embedding", Name: "b"}, {Type: "domain", Name: "law"}},
	}}

	winner := rankedWinner(t, []config.Decision{twoLeaf, threeLeaf}, config.RoutingStrategyPriority, &SignalMatches{
		EmbeddingRules: []string{"a", "b"},
		DomainRules:    []string{"law"},
		SignalConfidences: map[string]float64{
			"embedding:a": 0.90, "embedding:b": 0.90, "domain:law": 0.88,
		},
	})
	if winner.Decision.Name != "two_leaf" {
		t.Fatalf("winner = %s, want two_leaf (mean of two leaves beats the mean of three)", winner.Decision.Name)
	}
}

// A matched conversation rule reports 1.0 as a score, so a boolean condition
// outranks reported evidence under either strategy.
func TestBooleanConversationScoreOutranksReportedEvidence(t *testing.T) {
	boolean := config.Decision{Name: "boolean_route", Priority: 250, Tier: 2, Rules: config.RuleNode{
		Operator:   "AND",
		Conditions: []config.RuleNode{{Type: "conversation", Name: "has_images"}},
	}}
	measured := config.Decision{Name: "measured_route", Priority: 900, Tier: 2, Rules: config.RuleNode{
		Operator:   "AND",
		Conditions: []config.RuleNode{{Type: "projection", Name: "sensitive"}},
	}}
	decisions := []config.Decision{boolean, measured}
	signals := &SignalMatches{
		ConversationRules: []string{"has_images"},
		ProjectionRules:   []string{"sensitive"},
		SignalConfidences: map[string]float64{"conversation:has_images": 1.0, "projection:sensitive": 0.82},
	}

	for _, strategy := range []config.RoutingStrategy{config.RoutingStrategyPriority, config.RoutingStrategyConfidence} {
		winner := rankedWinner(t, decisions, strategy, signals)
		if winner.Decision.Name != "boolean_route" {
			t.Fatalf("strategy %s: winner = %s, want boolean_route (its reported 1.0 ranks as evidence)", strategy, winner.Decision.Name)
		}
	}
}
