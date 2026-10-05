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

// cx is a complexity decision, which reports no comparable score.
func cx(name string, priority int) config.Decision {
	return config.Decision{Name: name, Priority: priority, Rules: config.RuleNode{Type: "complexity", Name: "hard_rule:hard"}}
}

func hardComplexity() *SignalMatches {
	return &SignalMatches{ComplexityRules: []string{"hard_rule:hard"}}
}

func assertRunnerUp(t *testing.T, trace *RankingTrace, runnerUp, reason string) {
	t.Helper()
	if trace.RunnerUp != runnerUp || trace.Reason != reason {
		t.Fatalf("runner_up = %q, reason = %q; want %q, %q", trace.RunnerUp, trace.Reason, runnerUp, reason)
	}
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
	assertRunnerUp(t, trace, "law_route", "confidence 0.95 > 0.86")
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
	assertRunnerUp(t, trace, "embedding_route", "no comparable confidence, priority 180 > 60")
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
	assertRunnerUp(t, trace, "keyword_route", "no comparable confidence, priority 180 > 60")
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
	assertRunnerUp(t, trace, "beta_route", "equal priority 100, equal confidence 0.8, decision name ordering")
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
	assertRunnerUp(t, trace, "", "")
}

// The issue's scenario (#3658): equal priorities and no comparable score
// leave the name to decide, and the reason says so.
func TestRankingTraceReasonNamesSilentNameTieBreak(t *testing.T) {
	trace := rankingTraceFor(t,
		[]config.Decision{cx("escalate-hard", 0), cx("escalate-extreme", 0)},
		config.RoutingStrategyPriority, hardComplexity())

	if trace.Winner != "escalate-extreme" || trace.DecidedBy != "name" {
		t.Fatalf("trace = %+v, want escalate-extreme decided by name", trace)
	}
	assertRunnerUp(t, trace, "escalate-hard", "equal priority 0, no comparable confidence, decision name ordering")
}

func TestRankingTraceReasonStatesPriorities(t *testing.T) {
	trace := rankingTraceFor(t,
		[]config.Decision{cx("escalate-hard", 100), cx("escalate-extreme", 150)},
		config.RoutingStrategyPriority, hardComplexity())

	assertRunnerUp(t, trace, "escalate-hard", "priority 150 > 100")
}

// Under the priority strategy a priority tie falls through to confidence.
func TestRankingTraceReasonStatesConfidenceAfterEqualPriority(t *testing.T) {
	trace := rankingTraceFor(t,
		[]config.Decision{domainDecision("law_route", 7, 0, "law"), domainDecision("health_route", 7, 0, "health")},
		config.RoutingStrategyPriority,
		&SignalMatches{
			DomainRules:       []string{"law", "health"},
			SignalConfidences: map[string]float64{"domain:law": 0.7, "domain:health": 0.9},
		})

	if trace.Winner != "health_route" || trace.DecidedBy != "confidence" || trace.Tiered {
		t.Fatalf("trace = %+v, want untiered health_route decided by confidence", trace)
	}
	assertRunnerUp(t, trace, "law_route", "equal priority 7, confidence 0.9 > 0.7")
}

func TestRankingTraceReasonStatesTiers(t *testing.T) {
	trace := rankingTraceFor(t,
		[]config.Decision{domainDecision("a", 10, 2, "law"), domainDecision("b", 10, 1, "law")},
		config.RoutingStrategyPriority,
		&SignalMatches{DomainRules: []string{"law"}, SignalConfidences: map[string]float64{"domain:law": 0.9}})

	if trace.Winner != "b" || trace.DecidedBy != "tier" {
		t.Fatalf("trace = %+v, want b decided by tier", trace)
	}
	assertRunnerUp(t, trace, "a", "tier 1 ranks before tier 2")
}

func TestRankingTraceReasonStatesCatchAll(t *testing.T) {
	trace := rankingTraceFor(t,
		[]config.Decision{{Name: "default", Priority: 999}, domainDecision("law_route", 1, 0, "law")},
		config.RoutingStrategyPriority,
		&SignalMatches{DomainRules: []string{"law"}, SignalConfidences: map[string]float64{"domain:law": 0.9}})

	if trace.Winner != "law_route" || trace.DecidedBy != "catch_all" {
		t.Fatalf("trace = %+v, want law_route decided by catch_all", trace)
	}
	assertRunnerUp(t, trace, "default", "runner-up is a catch-all")
}

// The runner-up can differ from the decision fallback_reason names, which
// is the first incomparable one in config order.
func TestRankingTraceRunnerUpIndependentOfFallback(t *testing.T) {
	trace := rankingTraceFor(t,
		[]config.Decision{cx("c", 0), cx("a", 0), cx("b", 0)},
		config.RoutingStrategyPriority, hardComplexity())

	if trace.Winner != "a" || trace.Fallback != "c reported no comparable score" {
		t.Fatalf("trace = %+v, want winner a with fallback naming c", trace)
	}
	assertRunnerUp(t, trace, "b", "equal priority 0, no comparable confidence, decision name ordering")
}
