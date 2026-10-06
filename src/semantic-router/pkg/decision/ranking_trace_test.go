package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func rankingTraceFor(t *testing.T, decisions []config.Decision, strategy config.RoutingStrategy, signals *SignalMatches) *RankingTrace {
	t.Helper()
	engine := NewDecisionEngine(nil, nil, nil, decisions, strategy)
	_, diagnostics, err := engine.EvaluateDecisionsWithDiagnostics(signals)
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithDiagnostics() error = %v", err)
	}
	if diagnostics.Ranking == nil {
		t.Fatal("no ranking trace recorded")
	}
	return diagnostics.Ranking
}

func domainDecision(name string, priority, tier int, rule string) config.Decision {
	return config.Decision{Name: name, Priority: priority, Tier: tier, Rules: config.RuleNode{Type: "domain", Name: rule}}
}

// The trace names the key that decided, and reports the pool as comparable
// when confidence ranked it.
func TestRankingTraceReportsConfidenceWinner(t *testing.T) {
	trace := rankingTraceFor(t,
		[]config.Decision{domainDecision("law_route", 180, 1, "law"), domainDecision("health_route", 60, 1, "health")},
		config.RoutingStrategyConfidence,
		&SignalMatches{
			DomainRules:       []string{"law", "health"},
			SignalConfidences: map[string]float64{"domain:law": 0.86, "domain:health": 0.95},
		})

	if trace.Winner != "health_route" || trace.DecidedBy != "confidence" {
		t.Fatalf("trace = %+v, want health_route decided by confidence", trace)
	}
	if !trace.Comparable || trace.Fallback != "" {
		t.Fatalf("trace = %+v, want a comparable pool with no fallback reason", trace)
	}
	if trace.Strategy != string(config.RoutingStrategyConfidence) || trace.ScoreKind != string(config.ScoreKindProbability) {
		t.Fatalf("trace = %+v, want the confidence strategy over probabilities", trace)
	}
	if !trace.Tiered || trace.Tier != 1 || trace.Candidates != 2 {
		t.Fatalf("trace = %+v, want tier 1 of a tiered pool with two candidates", trace)
	}
}

// When the pool mixes kinds the trace says which decision made it
// incomparable and which key decided instead.
func TestRankingTraceReportsMixedKindFallback(t *testing.T) {
	embedding := config.Decision{Name: "embedding_route", Priority: 60, Tier: 1, Rules: config.RuleNode{Type: "embedding", Name: "legal_analysis"}}
	trace := rankingTraceFor(t,
		[]config.Decision{domainDecision("law_route", 180, 1, "law"), embedding},
		config.RoutingStrategyConfidence,
		&SignalMatches{
			DomainRules:       []string{"law"},
			EmbeddingRules:    []string{"legal_analysis"},
			SignalConfidences: map[string]float64{"domain:law": 0.70, "embedding:legal_analysis": 0.99},
		})

	if trace.Comparable {
		t.Fatalf("trace = %+v, want an incomparable pool", trace)
	}
	if trace.Fallback != "embedding_route reported a similarity against a probability" {
		t.Fatalf("fallback = %q, want the decision and the two kinds", trace.Fallback)
	}
	if trace.Winner != "law_route" || trace.DecidedBy != "priority" {
		t.Fatalf("trace = %+v, want law_route decided by priority", trace)
	}
}

// A member that reported no comparable score is named the same way.
func TestRankingTraceReportsUnscoredFallback(t *testing.T) {
	keyword := config.Decision{Name: "keyword_route", Priority: 60, Tier: 1, Rules: config.RuleNode{Type: "keyword", Name: "urgent"}}
	trace := rankingTraceFor(t,
		[]config.Decision{domainDecision("law_route", 180, 1, "law"), keyword},
		config.RoutingStrategyConfidence,
		&SignalMatches{
			DomainRules:       []string{"law"},
			KeywordRules:      []string{"urgent"},
			SignalConfidences: map[string]float64{"domain:law": 0.70},
		})

	if trace.Fallback != "keyword_route reported no comparable score" {
		t.Fatalf("fallback = %q, want the decision that reported nothing", trace.Fallback)
	}
	if trace.DecidedBy != "priority" {
		t.Fatalf("decided_by = %q, want priority", trace.DecidedBy)
	}
}

// Two decisions that differ in nothing but their names are separated by the
// final tie-break, and the trace says so.
func TestRankingTraceReportsNameTieBreak(t *testing.T) {
	trace := rankingTraceFor(t,
		[]config.Decision{domainDecision("beta_route", 100, 1, "law"), domainDecision("alpha_route", 100, 1, "health")},
		config.RoutingStrategyPriority,
		&SignalMatches{
			DomainRules:       []string{"law", "health"},
			SignalConfidences: map[string]float64{"domain:law": 0.80, "domain:health": 0.80},
		})

	if trace.Winner != "alpha_route" || trace.DecidedBy != "name" {
		t.Fatalf("trace = %+v, want alpha_route decided by name", trace)
	}
}

// One matched decision is reported as such rather than as a comparison.
func TestRankingTraceReportsSingleCandidate(t *testing.T) {
	trace := rankingTraceFor(t,
		[]config.Decision{domainDecision("law_route", 180, 0, "law")},
		config.RoutingStrategyPriority,
		&SignalMatches{DomainRules: []string{"law"}, SignalConfidences: map[string]float64{"domain:law": 0.86}})

	if trace.Candidates != 1 || trace.DecidedBy != "only_candidate" || trace.Tiered {
		t.Fatalf("trace = %+v, want one untiered candidate", trace)
	}
}
