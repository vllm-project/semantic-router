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

// A request that matches no decision runs the guard of global.plugins, so
// routing evaluates the signals that guard reads even when every decision
// replaces its rules.
func TestTheDefaultGuardsSignalsAreEvaluatedForRouting(t *testing.T) {
	cfg := &config.RouterConfig{
		GlobalPlugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginResponseGuard,
			Configuration: config.MustStructuredPayload(map[string]interface{}{"rules": []interface{}{
				map[string]interface{}{"when": map[string]interface{}{"type": "pii", "name": "personal_data"}, "action": "block"},
			}}),
		}},
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{PIIRules: []config.PIIRule{{Name: "personal_data"}}},
			Decisions: []config.Decision{{Name: "open", Rules: config.RuleCombination{Operator: "AND"}, Plugins: []config.DecisionPlugin{{
				Type: config.DecisionPluginResponseGuard, Configuration: config.MustStructuredPayload(map[string]interface{}{"rules": []interface{}{
					map[string]interface{}{"when": map[string]interface{}{"type": "keyword", "name": "open"}, "action": "none"},
				}}),
			}}}},
		},
	}
	if err := cfg.LayerPlugins(); err != nil {
		t.Fatal(err)
	}
	if !(&Classifier{Config: cfg}).getUsedSignals()["pii:personal_data"] {
		t.Fatal("the default guard reads pii(personal_data), and routing does not evaluate it")
	}
}
