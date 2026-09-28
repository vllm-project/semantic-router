package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestUsedSignalsExpandComplexityComposerDependencies(t *testing.T) {
	composer := config.RuleCombination{
		Operator: "OR",
		Conditions: []config.RuleNode{
			{Type: config.SignalTypeKeyword, Name: "agentic"},
			{Type: config.SignalTypeEmbedding, Name: "workflow"},
		},
	}
	classifier := &Classifier{Config: &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				ComplexityRules: []config.ComplexityRule{{
					Name:     "delivery",
					Composer: &composer,
				}},
			},
			Decisions: []config.Decision{{
				Name: "complex",
				Rules: config.RuleCombination{
					Type: config.SignalTypeComplexity,
					Name: "delivery:hard",
				},
			}},
		},
	}}

	used := classifier.getUsedSignals()

	if !used["keyword:agentic"] || !used["embedding:workflow"] {
		t.Fatalf("complexity composer dependencies not expanded: %#v", used)
	}
}
