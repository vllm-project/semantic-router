package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestAPersonalDataReplayLimitLeavesThePIISignalsOutOfRouting(t *testing.T) {
	no := false
	classifier := &Classifier{Config: &config.RouterConfig{
		RouterReplay: config.RouterReplayConfig{Enabled: true, CapturePersonalData: &no},
		IntelligentRouting: config.IntelligentRouting{
			Signals:   config.Signals{PIIRules: []config.PIIRule{{Name: "personal_data"}}},
			Decisions: []config.Decision{{Name: "everything", Rules: config.RuleCombination{Operator: "AND"}}},
		},
	}}
	if !classifier.Config.ReplayNeedsPIIEvidence() {
		t.Fatal("the replay data policy reads the PII rules")
	}
	if classifier.getUsedSignals()["pii:personal_data"] {
		t.Fatal("Replay checks the prompt after routing; routing must not evaluate an unreferenced PII signal for it")
	}
}

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
