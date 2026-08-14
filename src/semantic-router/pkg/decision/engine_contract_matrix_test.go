package decision_test

import (
	"math"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

type engineContractCase struct {
	name           string
	strategy       config.RoutingStrategy
	decisions      []config.Decision
	signals        *decision.SignalMatches
	wantDecision   string
	wantConfidence float64
	wantRules      []string
	wantTraceKinds []string
}

func TestDecisionEngineContractMatrix(t *testing.T) {
	t.Parallel()
	for _, testCase := range decisionEngineContractCases() {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			engine := decision.NewDecisionEngine(
				nil,
				nil,
				nil,
				testCase.decisions,
				testCase.strategy,
			)
			result, err := engine.EvaluateDecisionsWithSignals(testCase.signals)
			if err != nil {
				t.Fatalf("normal evaluation: %v", err)
			}
			traceResult, traces := engine.EvaluateDecisionsWithTrace(testCase.signals)

			assertContractOutcome(t, "normal", result, testCase)
			assertContractOutcome(t, "trace", traceResult, testCase)
			assertContractTraceInventory(t, traces, testCase.decisions)
			if testCase.wantTraceKinds != nil {
				if got := traceKinds(traces); !slices.Equal(got, testCase.wantTraceKinds) {
					t.Fatalf("trace kinds = %v, want %v", got, testCase.wantTraceKinds)
				}
			}
		})
	}
}

func decisionEngineContractCases() []engineContractCase {
	testCases := logicalContractCases()
	testCases = append(testCases, selectionContractCases()...)
	testCases = append(testCases, predicateAndFallbackContractCases()...)
	return testCases
}

func logicalContractCases() []engineContractCase {
	keywordA := leaf("keyword", "a")
	keywordB := leaf("keyword", "b")
	blocked := leaf("keyword", "blocked")
	return []engineContractCase{
		{
			name:      "and/all-match",
			strategy:  config.RoutingStrategyPriority,
			decisions: []config.Decision{route("and", 0, 0, all(keywordA, leaf("embedding", "b")))},
			signals: matches(
				[]string{"a"},
				[]string{"b"},
				nil,
				nil,
				map[string]float64{"keyword:a": 0.8, "embedding:b": 0.6},
			),
			wantDecision:   "and",
			wantConfidence: 0.7,
			wantRules:      []string{"keyword:a", "embedding:b"},
		},
		{
			name:      "and/missing-child",
			strategy:  config.RoutingStrategyPriority,
			decisions: []config.Decision{route("and", 0, 0, all(keywordA, keywordB))},
			signals:   matches([]string{"a"}, nil, nil, nil, nil),
		},
		{
			name:      "or/best-confidence",
			strategy:  config.RoutingStrategyPriority,
			decisions: []config.Decision{route("or", 0, 0, any(keywordA, leaf("embedding", "b")))},
			signals: matches(
				[]string{"a"},
				[]string{"b"},
				nil,
				nil,
				map[string]float64{"keyword:a": 0.4, "embedding:b": 0.9},
			),
			wantDecision:   "or",
			wantConfidence: 0.9,
			wantRules:      []string{"embedding:b"},
		},
		{
			name:           "not/absent-matches",
			strategy:       config.RoutingStrategyPriority,
			decisions:      []config.Decision{route("not", 0, 0, not(blocked))},
			signals:        matches(nil, nil, nil, nil, nil),
			wantDecision:   "not",
			wantConfidence: 1,
		},
		{
			name:      "not/present-rejects",
			strategy:  config.RoutingStrategyPriority,
			decisions: []config.Decision{route("not", 0, 0, not(blocked))},
			signals:   matches([]string{"blocked"}, nil, nil, nil, nil),
		},
		{
			name:      "nested/all-any-not",
			strategy:  config.RoutingStrategyPriority,
			decisions: []config.Decision{route("nested", 0, 0, all(any(keywordA, keywordB), not(blocked)))},
			signals: matches(
				[]string{"b"},
				nil,
				nil,
				nil,
				map[string]float64{"keyword:b": 0.8},
			),
			wantDecision:   "nested",
			wantConfidence: 0.9,
			wantRules:      []string{"keyword:b"},
			wantTraceKinds: []string{"AND", "OR", "leaf", "leaf", "NOT", "leaf"},
		},
		{
			name:           "fallback/omitted-rules",
			strategy:       config.RoutingStrategyPriority,
			decisions:      []config.Decision{{Name: "fallback"}},
			signals:        matches(nil, nil, nil, nil, nil),
			wantDecision:   "fallback",
			wantConfidence: 0,
			wantTraceKinds: []string{"fallback"},
		},
		{
			name:           "fallback/empty-and",
			strategy:       config.RoutingStrategyPriority,
			decisions:      []config.Decision{route("fallback", 0, 0, all())},
			signals:        matches(nil, nil, nil, nil, nil),
			wantDecision:   "fallback",
			wantConfidence: 0,
			wantTraceKinds: []string{"AND"},
		},
	}
}

func selectionContractCases() []engineContractCase {
	keywordA := leaf("keyword", "a")
	keywordB := leaf("keyword", "b")
	return []engineContractCase{
		{
			name:     "selection/priority",
			strategy: config.RoutingStrategyPriority,
			decisions: []config.Decision{
				route("low", 1, 0, keywordA),
				route("high", 2, 0, keywordA),
			},
			signals:        matches([]string{"a"}, nil, nil, nil, map[string]float64{"keyword:a": 0.3}),
			wantDecision:   "high",
			wantConfidence: 0.3,
			wantRules:      []string{"keyword:a"},
		},
		{
			name:     "selection/confidence",
			strategy: config.RoutingStrategyConfidence,
			decisions: []config.Decision{
				route("low", 0, 0, keywordA),
				route("high", 0, 0, keywordB),
			},
			signals: matches(
				[]string{"a", "b"},
				nil,
				nil,
				nil,
				map[string]float64{"keyword:a": 0.3, "keyword:b": 0.8},
			),
			wantDecision:   "high",
			wantConfidence: 0.8,
			wantRules:      []string{"keyword:b"},
		},
		{
			name:     "selection/lower-tier",
			strategy: config.RoutingStrategyPriority,
			decisions: []config.Decision{
				route("tier-two", 100, 2, keywordA),
				route("tier-one", 1, 1, keywordB),
			},
			signals: matches(
				[]string{"a", "b"},
				nil,
				nil,
				nil,
				map[string]float64{"keyword:a": 0.9, "keyword:b": 0.2},
			),
			wantDecision:   "tier-one",
			wantConfidence: 0.2,
			wantRules:      []string{"keyword:b"},
		},
		{
			name:     "selection/name-tiebreak",
			strategy: config.RoutingStrategyPriority,
			decisions: []config.Decision{
				route("beta", 1, 0, keywordA),
				route("alpha", 1, 0, keywordA),
			},
			signals:        matches([]string{"a"}, nil, nil, nil, nil),
			wantDecision:   "alpha",
			wantConfidence: 1,
			wantRules:      []string{"keyword:a"},
		},
	}
}

func predicateAndFallbackContractCases() []engineContractCase {
	keywordA := leaf("keyword", "a")
	return []engineContractCase{
		predicateContractCase("predicate/gte-boundary", &config.NumericPredicate{GTE: floatPtr(3)}, 3, true),
		predicateContractCase("predicate/gt-boundary", &config.NumericPredicate{GT: floatPtr(3)}, 3, false),
		predicateContractCase("predicate/lte-zero", &config.NumericPredicate{LTE: floatPtr(0)}, 0, true),
		predicateContractCase(
			"predicate/bounded-range",
			&config.NumericPredicate{GT: floatPtr(0), LT: floatPtr(1)},
			0.5,
			true,
		),
		errorContractCase("predicate/on-error-match", "match", true),
		errorContractCase("predicate/on-error-no-match", "no_match", false),
		{
			name:      "projection/output-match",
			strategy:  config.RoutingStrategyPriority,
			decisions: []config.Decision{route("projected", 0, 0, leaf("projection", "high"))},
			signals: matches(
				nil,
				nil,
				nil,
				[]string{"high"},
				map[string]float64{"projection:high": 0.81},
			),
			wantDecision:   "projected",
			wantConfidence: 0.81,
			wantRules:      []string{"projection:high"},
		},
		{
			name:     "selection/confidence-over-fallback",
			strategy: config.RoutingStrategyConfidence,
			decisions: []config.Decision{
				route("specific", 1, 0, keywordA),
				{Name: "fallback", Priority: 100},
			},
			signals:        matches([]string{"a"}, nil, nil, nil, map[string]float64{"keyword:a": 0.72}),
			wantDecision:   "specific",
			wantConfidence: 0.72,
			wantRules:      []string{"keyword:a"},
		},
		{
			name:      "selection/no-match",
			strategy:  config.RoutingStrategyPriority,
			decisions: []config.Decision{route("missing", 0, 0, keywordA)},
			signals:   matches([]string{"b"}, nil, nil, nil, nil),
		},
	}
}

func predicateContractCase(
	name string,
	predicate *config.NumericPredicate,
	value float64,
	wantMatch bool,
) engineContractCase {
	testCase := engineContractCase{
		name:      name,
		strategy:  config.RoutingStrategyPriority,
		decisions: []config.Decision{route("bounded", 0, 0, predicateLeaf(predicate, ""))},
		signals: &decision.SignalMatches{
			SignalValues: map[string]float64{"structure:count": value},
		},
	}
	if wantMatch {
		testCase.wantDecision = "bounded"
		testCase.wantConfidence = 1
		testCase.wantRules = []string{"structure:count"}
	}
	return testCase
}

func errorContractCase(name, onError string, wantMatch bool) engineContractCase {
	testCase := engineContractCase{
		name:      name,
		strategy:  config.RoutingStrategyPriority,
		decisions: []config.Decision{route("guarded", 0, 0, predicateLeaf(&config.NumericPredicate{GTE: floatPtr(0.8)}, onError))},
		signals: &decision.SignalMatches{
			SignalErrors: map[string]string{"structure:count": "unavailable"},
		},
	}
	if wantMatch {
		testCase.wantDecision = "guarded"
		testCase.wantConfidence = 1
		testCase.wantRules = []string{"structure:count"}
	}
	return testCase
}

func predicateLeaf(predicate *config.NumericPredicate, onError string) config.RuleNode {
	return config.RuleNode{
		Type:      "structure",
		Name:      "count",
		Predicate: predicate,
		OnError:   onError,
	}
}

func leaf(signalType, name string) config.RuleNode {
	return config.RuleNode{Type: signalType, Name: name}
}

func all(children ...config.RuleNode) config.RuleNode {
	return config.RuleNode{Operator: "AND", Conditions: children}
}

func any(children ...config.RuleNode) config.RuleNode {
	return config.RuleNode{Operator: "OR", Conditions: children}
}

func not(child config.RuleNode) config.RuleNode {
	return config.RuleNode{Operator: "NOT", Conditions: []config.RuleNode{child}}
}

func route(name string, priority, tier int, rules config.RuleNode) config.Decision {
	return config.Decision{Name: name, Priority: priority, Tier: tier, Rules: rules}
}

func matches(
	keywords, embeddings, structures, projections []string,
	confidences map[string]float64,
) *decision.SignalMatches {
	return &decision.SignalMatches{
		KeywordRules:      keywords,
		EmbeddingRules:    embeddings,
		StructureRules:    structures,
		ProjectionRules:   projections,
		SignalConfidences: confidences,
	}
}

func floatPtr(value float64) *float64 {
	return &value
}

func assertContractOutcome(
	t *testing.T,
	mode string,
	result *decision.DecisionResult,
	want engineContractCase,
) {
	t.Helper()
	gotDecision := ""
	gotConfidence := 0.0
	var gotRules []string
	if result != nil {
		gotDecision = result.Decision.Name
		gotConfidence = result.Confidence
		gotRules = result.MatchedRules
	}
	if gotDecision != want.wantDecision {
		t.Errorf("%s decision = %q, want %q", mode, gotDecision, want.wantDecision)
	}
	if math.Abs(gotConfidence-want.wantConfidence) > 1e-9 {
		t.Errorf("%s confidence = %v, want %v", mode, gotConfidence, want.wantConfidence)
	}
	if !slices.Equal(gotRules, want.wantRules) {
		t.Errorf("%s matched rules = %v, want %v", mode, gotRules, want.wantRules)
	}
}

func assertContractTraceInventory(
	t *testing.T,
	traces []decision.DecisionTrace,
	decisions []config.Decision,
) {
	t.Helper()
	if len(traces) != len(decisions) {
		t.Fatalf("trace count = %d, want %d", len(traces), len(decisions))
	}
	for index, trace := range traces {
		if trace.DecisionName != decisions[index].Name {
			t.Errorf(
				"trace[%d] decision = %q, want %q",
				index,
				trace.DecisionName,
				decisions[index].Name,
			)
		}
		if trace.RootTrace == nil {
			t.Fatalf("trace[%d] has no root", index)
		}
	}
}

func traceKinds(traces []decision.DecisionTrace) []string {
	var result []string
	var walk func(*decision.TraceNode)
	walk = func(node *decision.TraceNode) {
		result = append(result, node.NodeType)
		for _, child := range node.Children {
			walk(child)
		}
	}
	for _, trace := range traces {
		walk(trace.RootTrace)
	}
	return result
}
