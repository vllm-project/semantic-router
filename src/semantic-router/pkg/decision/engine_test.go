package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type evaluateDecisionsCase struct {
	name                  string
	decisions             []config.Decision
	strategy              config.RoutingStrategy
	matchedKeywordRules   []string
	matchedEmbeddingRules []string
	matchedDomainRules    []string
	expectedDecision      string
}

type evaluateSignalsCase struct {
	name             string
	decisions        []config.Decision
	signals          *SignalMatches
	expectedDecision string
}

func ruleDecision(
	name string,
	priority int,
	operator string,
	conditions ...config.RuleCondition,
) config.Decision {
	return config.Decision{
		Name:     name,
		Priority: priority,
		Rules: config.RuleCombination{
			Operator:   operator,
			Conditions: conditions,
		},
		ModelRefs: []config.ModelRef{{Model: "test-model"}},
	}
}

func notCodingDecision() config.Decision {
	return ruleDecision(
		"exclude-coding",
		10,
		"NOT",
		config.RuleCondition{
			Operator: "OR",
			Conditions: []config.RuleCondition{
				{Type: config.SignalTypeKeyword, Name: "programming"},
				{Type: config.SignalTypeDomain, Name: "coding"},
			},
		},
	)
}

func TestDecisionEngine_EvaluateDecisions(t *testing.T) {
	tests := []evaluateDecisionsCase{
		{
			name: "all AND conditions match",
			decisions: []config.Decision{ruleDecision(
				"coding-task", 10, "AND",
				config.RuleCondition{Type: config.SignalTypeKeyword, Name: "programming"},
				config.RuleCondition{Type: config.SignalTypeDomain, Name: "coding"},
			)},
			strategy:            config.RoutingStrategyPriority,
			matchedKeywordRules: []string{"programming"},
			matchedDomainRules:  []string{"coding"},
			expectedDecision:    "coding-task",
		},
		{
			name: "partial AND conditions do not match",
			decisions: []config.Decision{ruleDecision(
				"coding-task", 10, "AND",
				config.RuleCondition{Type: config.SignalTypeKeyword, Name: "programming"},
				config.RuleCondition{Type: config.SignalTypeDomain, Name: "coding"},
			)},
			strategy:            config.RoutingStrategyPriority,
			matchedKeywordRules: []string{"programming"},
		},
		{
			name: "one OR condition matches",
			decisions: []config.Decision{ruleDecision(
				"coding-task", 10, "OR",
				config.RuleCondition{Type: config.SignalTypeKeyword, Name: "programming"},
				config.RuleCondition{Type: config.SignalTypeDomain, Name: "coding"},
			)},
			strategy:            config.RoutingStrategyPriority,
			matchedKeywordRules: []string{"programming"},
			expectedDecision:    "coding-task",
		},
		{
			name: "priority strategy chooses highest priority match",
			decisions: []config.Decision{
				ruleDecision("high-priority-task", 20, "OR", config.RuleCondition{Type: config.SignalTypeKeyword, Name: "urgent"}),
				ruleDecision("low-priority-task", 10, "OR", config.RuleCondition{Type: config.SignalTypeKeyword, Name: "urgent"}),
			},
			strategy:            config.RoutingStrategyPriority,
			matchedKeywordRules: []string{"urgent"},
			expectedDecision:    "high-priority-task",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine := NewDecisionEngine(nil, nil, nil, tt.decisions, tt.strategy)
			result, err := engine.EvaluateDecisions(
				tt.matchedKeywordRules,
				tt.matchedEmbeddingRules,
				tt.matchedDomainRules,
			)
			assertDecisionResult(t, result, err, tt.expectedDecision)
		})
	}
}

func TestDecisionEngine_EvaluateDecisionsWithFactCheck(t *testing.T) {
	tests := []evaluateSignalsCase{
		{
			name: "needs fact check",
			decisions: []config.Decision{ruleDecision(
				"factual-query", 10, "AND",
				config.RuleCondition{Type: config.SignalTypeFactCheck, Name: "needs_fact_check"},
			)},
			signals:          &SignalMatches{FactCheckRules: []string{"needs_fact_check"}},
			expectedDecision: "factual-query",
		},
		{
			name: "no fact check needed",
			decisions: []config.Decision{ruleDecision(
				"creative-query", 10, "AND",
				config.RuleCondition{Type: config.SignalTypeFactCheck, Name: "no_fact_check_needed"},
			)},
			signals:          &SignalMatches{FactCheckRules: []string{"no_fact_check_needed"}},
			expectedDecision: "creative-query",
		},
		{
			name: "fact check and domain",
			decisions: []config.Decision{ruleDecision(
				"factual-science", 10, "AND",
				config.RuleCondition{Type: config.SignalTypeFactCheck, Name: "needs_fact_check"},
				config.RuleCondition{Type: config.SignalTypeDomain, Name: "science"},
			)},
			signals: &SignalMatches{
				DomainRules:    []string{"science"},
				FactCheckRules: []string{"needs_fact_check"},
			},
			expectedDecision: "factual-science",
		},
		{
			name: "fact check does not match",
			decisions: []config.Decision{ruleDecision(
				"factual-query", 10, "AND",
				config.RuleCondition{Type: config.SignalTypeFactCheck, Name: "needs_fact_check"},
			)},
			signals: &SignalMatches{FactCheckRules: []string{"no_fact_check_needed"}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine := NewDecisionEngine(nil, nil, nil, tt.decisions, config.RoutingStrategyPriority)
			result, err := engine.EvaluateDecisionsWithSignals(tt.signals)
			assertDecisionResult(t, result, err, tt.expectedDecision)
		})
	}
}

func TestDecisionEngine_EvaluateDecisionsWithNOTOperator(t *testing.T) {
	tests := []evaluateSignalsCase{
		{
			name:             "matches when no nested OR condition matches",
			decisions:        []config.Decision{notCodingDecision()},
			signals:          &SignalMatches{},
			expectedDecision: "exclude-coding",
		},
		{
			name:      "does not match when one nested OR condition matches",
			decisions: []config.Decision{notCodingDecision()},
			signals:   &SignalMatches{KeywordRules: []string{"programming"}},
		},
		{
			name:      "does not match when all nested OR conditions match",
			decisions: []config.Decision{notCodingDecision()},
			signals: &SignalMatches{
				KeywordRules: []string{"programming"},
				DomainRules:  []string{"coding"},
			},
		},
		{
			name: "matched NOT decision has full confidence",
			decisions: []config.Decision{ruleDecision(
				"non-medical", 10, "NOT",
				config.RuleCondition{Type: config.SignalTypeDomain, Name: "medical"},
			)},
			signals:          &SignalMatches{},
			expectedDecision: "non-medical",
		},
		{
			name: "priority applies to matching NOT decisions",
			decisions: []config.Decision{
				ruleDecision("not-medical-high", 20, "NOT", config.RuleCondition{Type: config.SignalTypeDomain, Name: "medical"}),
				ruleDecision("not-medical-low", 5, "NOT", config.RuleCondition{Type: config.SignalTypeDomain, Name: "medical"}),
			},
			signals:          &SignalMatches{},
			expectedDecision: "not-medical-high",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine := NewDecisionEngine(nil, nil, nil, tt.decisions, config.RoutingStrategyPriority)
			result, err := engine.EvaluateDecisionsWithSignals(tt.signals)
			assertDecisionResult(t, result, err, tt.expectedDecision)
			if result != nil && result.Confidence != 1.0 {
				t.Errorf("expected confidence 1.0 for NOT match, got %f", result.Confidence)
			}
		})
	}
}

func TestDecisionEngine_UnsupportedConditionDoesNotMatch(t *testing.T) {
	engine := NewDecisionEngine(
		nil,
		nil,
		nil,
		[]config.Decision{ruleDecision(
			"unsupported-signal", 10, "AND",
			config.RuleCondition{Type: "unsupported", Name: "unknown"},
		)},
		config.RoutingStrategyPriority,
	)

	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{})
	assertDecisionResult(t, result, err, "")
}
