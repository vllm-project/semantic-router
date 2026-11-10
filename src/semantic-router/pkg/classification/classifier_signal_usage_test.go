package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestAPersonalDataReplayLimitLeavesThePIISignalsOutOfRouting(t *testing.T) {
	cfg := &config.RouterConfig{
		GlobalPlugins: []config.DecisionPlugin{{
			Type:          config.DecisionPluginRouterReplay,
			Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true, "capture_personal_data": false}),
		}},
		IntelligentRouting: config.IntelligentRouting{
			Signals:   config.Signals{PIIRules: []config.PIIRule{{Name: "personal_data"}}},
			Decisions: []config.Decision{{Name: "everything", Rules: config.RuleCombination{Operator: "AND"}}},
		},
	}
	if err := cfg.LayerPlugins(); err != nil {
		t.Fatal(err)
	}
	classifier := &Classifier{Config: cfg}
	if !classifier.Config.ReplayNeedsPIIEvidence() {
		t.Fatal("the replay data policy reads the PII rules")
	}
	if classifier.getUsedSignals()["pii:personal_data"] {
		t.Fatal("Replay checks the prompt after routing; routing must not evaluate an unreferenced PII signal for it")
	}
}
