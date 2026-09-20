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
	highPriority := config.Decision{Name: "high_priority", Priority: 160, Rules: config.RuleNode{Type: "domain", Name: "law"}}
	highConfidence := config.Decision{Name: "high_confidence", Priority: 135, Rules: config.RuleNode{Type: "domain", Name: "health"}}
	tiered := config.Decision{Name: "tiered_fallback", Priority: 200, Tier: 2, Rules: config.RuleNode{Type: "keyword", Name: "marker"}}

	signals := &SignalMatches{
		DomainRules:  []string{"law", "health"},
		KeywordRules: []string{"marker"},
		SignalConfidences: map[string]float64{
			"domain:law":    0.70,
			"domain:health": 0.99,
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

// An OR prefers a branch that reported a score, so an extra keyword match adds
// support without removing the evidence the decision already reported.
func TestExtraKeywordMatchKeepsReportedEvidence(t *testing.T) {
	mixed := config.Decision{Name: "mixed_or", Priority: 10, Tier: 1, Rules: config.RuleNode{
		Operator: "OR",
		Conditions: []config.RuleNode{
			{Type: "embedding", Name: "semantic"},
			{Type: "keyword", Name: "marker"},
		},
	}}
	scored := config.Decision{Name: "scored_route", Priority: 20, Tier: 1, Rules: config.RuleNode{Type: "embedding", Name: "support"}}
	decisions := []config.Decision{mixed, scored}
	confidences := map[string]float64{"embedding:semantic": 0.95, "embedding:support": 0.80}

	withoutKeyword := rankedWinner(t, decisions, config.RoutingStrategyPriority, &SignalMatches{
		EmbeddingRules: []string{"semantic", "support"}, SignalConfidences: confidences,
	})
	if withoutKeyword.Decision.Name != "mixed_or" {
		t.Fatalf("winner = %s, want mixed_or (both members reported scores)", withoutKeyword.Decision.Name)
	}

	withKeyword := rankedWinner(t, decisions, config.RoutingStrategyPriority, &SignalMatches{
		EmbeddingRules: []string{"semantic", "support"}, KeywordRules: []string{"marker"}, SignalConfidences: confidences,
	})
	if withKeyword.Decision.Name != "mixed_or" {
		t.Fatalf("winner = %s, want mixed_or (the extra keyword match is support, not a demotion)", withKeyword.Decision.Name)
	}
	if !withKeyword.ConfidenceScored || withKeyword.Confidence != 0.95 {
		t.Fatalf("confidence = %.4f scored = %v, want the reported 0.95 and scored", withKeyword.Confidence, withKeyword.ConfidenceScored)
	}
}

// A conjunction that rests on several evidence leaves is not comparable, so
// the pool ranks by priority instead of by a mean that moves with leaf count.
func TestMultiEvidenceDecisionRanksByPriority(t *testing.T) {
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
	if winner.Decision.Name != "three_leaf" {
		t.Fatalf("winner = %s, want three_leaf (priority decides when neither aggregate is comparable)", winner.Decision.Name)
	}
	if winner.ConfidenceScored {
		t.Fatal("a decision aggregating several evidence leaves must not be comparable")
	}
}

// A conversation predicate and a projection output are policy, so neither
// ranks as evidence and the operator's priority decides under either strategy.
func TestPolicyLeavesLeavePriorityInCharge(t *testing.T) {
	boolean := config.Decision{Name: "boolean_route", Priority: 250, Tier: 2, Rules: config.RuleNode{
		Operator:   "AND",
		Conditions: []config.RuleNode{{Type: "conversation", Name: "has_images"}},
	}}
	measured := config.Decision{Name: "policy_route", Priority: 900, Tier: 2, Rules: config.RuleNode{
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
		if winner.Decision.Name != "policy_route" {
			t.Fatalf("strategy %s: winner = %s, want policy_route (priority 900 against 250)", strategy, winner.Decision.Name)
		}
	}
}

// A KB rule reports the similarity of the matched label, so it cannot be
// ranked against a classifier probability and priority decides.
func TestKnowledgeBaseSimilarityDoesNotOutrankProbability(t *testing.T) {
	kb := config.Decision{Name: "kb_route", Priority: 10, Tier: 1, Rules: config.RuleNode{Type: "kb", Name: "handbook"}}
	domain := config.Decision{Name: "domain_route", Priority: 100, Tier: 1, Rules: config.RuleNode{Type: "domain", Name: "law"}}

	winner := rankedWinner(t, []config.Decision{kb, domain}, config.RoutingStrategyPriority, &SignalMatches{
		KBRules:           []string{"handbook"},
		DomainRules:       []string{"law"},
		SignalConfidences: map[string]float64{"kb:handbook": 0.95, "domain:law": 0.80},
	})
	if winner.Decision.Name != "domain_route" {
		t.Fatalf("winner = %s, want domain_route (a similarity does not rank against a probability)", winner.Decision.Name)
	}
}

// A complexity rule reports a calibrated probability on one backend and the
// magnitude of a prototype margin on another, so its score is not comparable.
func TestComplexityScoreIsNotComparable(t *testing.T) {
	complexity := config.Decision{Name: "complexity_route", Priority: 10, Tier: 1, Rules: config.RuleNode{Type: "complexity", Name: "reasoning:hard"}}
	domain := config.Decision{Name: "domain_route", Priority: 100, Tier: 1, Rules: config.RuleNode{Type: "domain", Name: "law"}}

	winner := rankedWinner(t, []config.Decision{complexity, domain}, config.RoutingStrategyPriority, &SignalMatches{
		ComplexityRules:   []string{"reasoning:hard"},
		DomainRules:       []string{"law"},
		SignalConfidences: map[string]float64{"complexity:reasoning:hard": 0.95, "domain:law": 0.80},
	})
	if winner.Decision.Name != "domain_route" {
		t.Fatalf("winner = %s, want domain_route (the complexity quantity depends on its backend)", winner.Decision.Name)
	}
}

// An OR that can report either kind must not become comparable through the
// branch that happened to match.
func TestMixedKindORIsNotComparable(t *testing.T) {
	mixed := config.Decision{Name: "mixed_or", Priority: 10, Tier: 1, Rules: config.RuleNode{
		Operator: "OR",
		Conditions: []config.RuleNode{
			{Type: "domain", Name: "law"},
			{Type: "embedding", Name: "legal_analysis"},
		},
	}}
	embedding := config.Decision{Name: "embedding_route", Priority: 100, Tier: 1, Rules: config.RuleNode{Type: "embedding", Name: "support"}}

	winner := rankedWinner(t, []config.Decision{mixed, embedding}, config.RoutingStrategyPriority, &SignalMatches{
		DomainRules:    []string{"law"},
		EmbeddingRules: []string{"legal_analysis", "support"},
		SignalConfidences: map[string]float64{
			"domain:law": 0.80, "embedding:legal_analysis": 0.95, "embedding:support": 0.90,
		},
	})
	if winner.Decision.Name != "embedding_route" {
		t.Fatalf("winner = %s, want embedding_route (the OR reports either kind, so priority decides)", winner.Decision.Name)
	}
}

// A classifier that failed and matched through on_error keeps its decision out
// of confidence ranking, the way it did before evidence roles existed.
func TestErrorPolicyMatchIsNotComparable(t *testing.T) {
	guarded := config.Decision{Name: "guarded_route", Priority: 10, Tier: 1, Rules: config.RuleNode{
		Operator: "AND",
		Conditions: []config.RuleNode{
			{Type: "domain", Name: "law"},
			{Type: "classifier", Name: "guard", Label: "safe", OnError: "match"},
		},
	}}
	domain := config.Decision{Name: "domain_route", Priority: 100, Tier: 1, Rules: config.RuleNode{Type: "domain", Name: "health"}}

	winner := rankedWinner(t, []config.Decision{guarded, domain}, config.RoutingStrategyPriority, &SignalMatches{
		DomainRules:       []string{"law", "health"},
		SignalConfidences: map[string]float64{"domain:law": 0.95, "domain:health": 0.90},
		SignalErrors:      map[string]string{"classifier:guard": "classify_failed"},
	})
	if winner.Decision.Name != "domain_route" {
		t.Fatalf("winner = %s, want domain_route (an error-policy match is not evidence)", winner.Decision.Name)
	}
}
