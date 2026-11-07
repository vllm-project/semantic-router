package dsl

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestPluginFieldsResponseGuardKeepsRuleTrees locks in that the AST emitter
// carries a response_guard's rules, condition trees included, and its
// on_unknown, so a runtime config -> AST -> compile round trip loses none of
// them.
func TestPluginFieldsResponseGuardKeepsRuleTrees(t *testing.T) {
	cfg := config.ResponseGuardPluginConfig{
		Rules: []config.ResponseGuardRule{{
			When:   config.RuleNode{Type: config.SignalTypeHallucination, Name: "ungrounded"},
			Action: config.ResponseGuardAnnotate, Details: true,
		}},
		OnUnknown: config.ResponseGuardBlock,
	}
	plugin := pluginFromConfig(t, config.DecisionPluginResponseGuard, cfg)

	fields := pluginConfigToFields(plugin)

	requirePluginStringField(t, "response_guard", "on_unknown", fields, "block")
	rules, ok := fields["rules"].(ArrayValue)
	if !ok || len(rules.Items) != 1 {
		t.Fatalf("rules = %#v, want one rule", fields["rules"])
	}
	rule, ok := rules.Items[0].(ObjectValue)
	if !ok {
		t.Fatalf("rule = %#v, want an object", rules.Items[0])
	}
	requirePluginStringField(t, "response_guard rule", "action", rule.Fields, "annotate")
	requirePluginBoolField(t, "response_guard rule", "details", rule.Fields, true)
	when, ok := rule.Fields["when"].(ObjectValue)
	if !ok {
		t.Fatalf("when = %#v, want the rule's condition", rule.Fields["when"])
	}
	requirePluginStringField(t, "response_guard when", "type", when.Fields, "hallucination")
	requirePluginStringField(t, "response_guard when", "name", when.Fields, "ungrounded")
}

func pluginFromConfig(t *testing.T, pluginType string, cfg interface{}) *config.DecisionPlugin {
	t.Helper()
	payload, err := config.NewStructuredPayload(cfg)
	if err != nil {
		t.Fatalf("NewStructuredPayload: %v", err)
	}
	return &config.DecisionPlugin{Type: pluginType, Configuration: payload}
}

func requirePluginStringField(t *testing.T, plugin, name string, fields map[string]Value, want string) {
	t.Helper()
	v, ok := fields[name]
	if !ok {
		t.Errorf("expected %s plugin AST to carry string field %q", plugin, name)
		return
	}
	sv, ok := v.(StringValue)
	if !ok {
		t.Errorf("%s plugin field %q: expected StringValue, got %T", plugin, name, v)
		return
	}
	if sv.V != want {
		t.Errorf("%s plugin field %q: want %q, got %q", plugin, name, want, sv.V)
	}
}

func requirePluginBoolField(t *testing.T, plugin, name string, fields map[string]Value, want bool) {
	t.Helper()
	v, ok := fields[name]
	if !ok {
		t.Errorf("expected %s plugin AST to carry bool field %q", plugin, name)
		return
	}
	bv, ok := v.(BoolValue)
	if !ok {
		t.Errorf("%s plugin field %q: expected BoolValue, got %T", plugin, name, v)
		return
	}
	if bv.V != want {
		t.Errorf("%s plugin field %q: want %v, got %v", plugin, name, want, bv.V)
	}
}
