package services

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A decision rule referencing an unregistered signal type is dropped silently
// by appendSignalToMatchedSignals, so the used-signal report would understate
// what the decision actually depends on.
func TestExtractSignalsFromRuleCombinationIncludesAgenticFacts(t *testing.T) {
	service := &ClassificationService{}
	decision := &config.Decision{
		Name: "reviewer-route",
		Rules: config.RuleCombination{
			Operator: "AND",
			Conditions: []config.RuleCondition{
				{Type: config.SignalTypeAgenticFacts, Name: "reviewer-role"},
			},
		},
	}

	used := service.extractUsedSignalsFromDecision(decision)

	if len(used.AgenticFacts) != 1 || used.AgenticFacts[0] != "reviewer-role" {
		t.Fatalf("used.AgenticFacts = %v, want [reviewer-role]", used.AgenticFacts)
	}
}

func TestBuildMatchedSignalsIncludesAgenticFacts(t *testing.T) {
	matched := buildMatchedSignals(&classification.SignalResults{
		MatchedAgenticFactsRules: []string{"reviewer-role"},
	})

	if len(matched.AgenticFacts) != 1 || matched.AgenticFacts[0] != "reviewer-role" {
		t.Fatalf("matched.AgenticFacts = %v, want [reviewer-role]", matched.AgenticFacts)
	}
}

func TestGetUnmatchedSignalsIncludesAgenticFacts(t *testing.T) {
	reviewer := "reviewer"
	auditor := "auditor"
	classifier := &classification.Classifier{Config: &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{AgenticFactsRules: []config.AgenticFactsRule{
				{
					Name:      "reviewer-role",
					Field:     config.AgenticFactsFieldDelegatedRole,
					Predicate: config.AgenticFactsPredicate{Equals: &reviewer},
				},
				{
					Name:      "auditor-role",
					Field:     config.AgenticFactsFieldDelegatedRole,
					Predicate: config.AgenticFactsPredicate{Equals: &auditor},
				},
			}},
		},
	}}

	unmatched := getUnmatchedSignals(&classification.SignalResults{
		MatchedAgenticFactsRules: []string{"reviewer-role"},
	}, classifier)

	if len(unmatched.AgenticFacts) != 1 || unmatched.AgenticFacts[0] != "auditor-role" {
		t.Fatalf("unmatched.AgenticFacts = %v, want [auditor-role]", unmatched.AgenticFacts)
	}
}
