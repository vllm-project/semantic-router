package dsl

import (
	"math"
	"strings"
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

// TestPluginFieldsRAGEmitsOnFailure locks in that the AST emitter for
// the RAG plugin surfaces the on_failure knob. The DSL compiler reads it
// in compileRAGPlugin (with documented "warn" / "error" semantics) but
// pluginFieldsRAG used to drop it on the AST path.
func TestPluginFieldsRAGEmitsOnFailure(t *testing.T) {
	topK := 5
	maxContextLength := 2048
	cacheTTL := 60
	similarityThreshold := float32(0.7)
	minConfidenceThreshold := float32(0.4)
	cfg := config.RAGPluginConfig{
		Enabled:                true,
		Backend:                "milvus",
		TopK:                   &topK,
		SimilarityThreshold:    &similarityThreshold,
		MaxContextLength:       &maxContextLength,
		InjectionMode:          "tool_role",
		BackendConfig:          config.MustStructuredPayload(map[string]interface{}{"collection_name": "docs"}),
		OnFailure:              "warn",
		CacheResults:           true,
		CacheTTLSeconds:        &cacheTTL,
		MinConfidenceThreshold: &minConfidenceThreshold,
	}
	plugin := pluginFromConfig(t, config.DecisionPluginRAG, cfg)

	fields := pluginFieldsRAG(plugin)

	requirePluginBoolField(t, "rag", "enabled", fields, true)
	requirePluginStringField(t, "rag", "backend", fields, "milvus")
	requirePluginIntField(t, "rag", "top_k", fields, 5)
	requirePluginFloatField(t, "rag", "similarity_threshold", fields, 0.7)
	requirePluginIntField(t, "rag", "max_context_length", fields, 2048)
	requirePluginStringField(t, "rag", "injection_mode", fields, "tool_role")
	requirePluginObjectField(t, "rag", "backend_config", fields)
	requirePluginStringField(t, "rag", "on_failure", fields, "warn")
	requirePluginBoolField(t, "rag", "cache_results", fields, true)
	requirePluginIntField(t, "rag", "cache_ttl_seconds", fields, 60)
	requirePluginFloatField(t, "rag", "min_confidence_threshold", fields, 0.4)
}

// TestPluginFieldsRAGOmitsOnFailureWhenEmpty preserves the omit-default
// contract for on_failure: an empty string in the runtime config must not
// surface as a noisy field in the AST.
func TestPluginFieldsRAGOmitsOnFailureWhenEmpty(t *testing.T) {
	cfg := config.RAGPluginConfig{Enabled: true, Backend: "milvus"}
	plugin := pluginFromConfig(t, config.DecisionPluginRAG, cfg)

	fields := pluginFieldsRAG(plugin)

	if _, ok := fields["on_failure"]; ok {
		t.Errorf("expected on_failure to be omitted when empty")
	}
}

// TestEmitRAGPluginConfigEmitsOnFailure covers the text-DSL path. Unlike
// the keyword/context AST fix, the RAG on_failure leak affects BOTH the
// AST emitter and the text emitter: emitRAGPluginConfig used to drop it
// too, so Decompile()/Format() round trips through the runtime config
// silently lost on_failure even when authored from DSL.
func TestEmitRAGPluginConfigEmitsOnFailure(t *testing.T) {
	cacheTTL := 60
	minConfidenceThreshold := float32(0.4)
	cfg := config.RAGPluginConfig{
		Enabled:                true,
		Backend:                "milvus",
		BackendConfig:          config.MustStructuredPayload(map[string]interface{}{"collection_name": "docs"}),
		OnFailure:              "error",
		CacheResults:           true,
		CacheTTLSeconds:        &cacheTTL,
		MinConfidenceThreshold: &minConfidenceThreshold,
	}
	plugin := pluginFromConfig(t, config.DecisionPluginRAG, cfg)

	var sb strings.Builder
	emitRAGPluginConfig(&sb, plugin)

	out := sb.String()
	if !strings.Contains(out, `on_failure: "error"`) {
		t.Errorf("expected emitRAGPluginConfig output to contain on_failure, got:\n%s", out)
	}
	for _, want := range []string{
		`backend_config: { collection_name: "docs" }`,
		`cache_results: true`,
		`cache_ttl_seconds: 60`,
		`min_confidence_threshold: 0.4`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("expected emitRAGPluginConfig output to contain %q, got:\n%s", want, out)
		}
	}
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

func requirePluginIntField(t *testing.T, plugin, name string, fields map[string]Value, want int) {
	t.Helper()
	v, ok := fields[name]
	if !ok {
		t.Errorf("expected %s plugin AST to carry int field %q", plugin, name)
		return
	}
	iv, ok := v.(IntValue)
	if !ok {
		t.Errorf("%s plugin field %q: expected IntValue, got %T", plugin, name, v)
		return
	}
	if iv.V != want {
		t.Errorf("%s plugin field %q: want %d, got %d", plugin, name, want, iv.V)
	}
}

func requirePluginFloatField(t *testing.T, plugin, name string, fields map[string]Value, want float64) {
	t.Helper()
	v, ok := fields[name]
	if !ok {
		t.Errorf("expected %s plugin AST to carry float field %q", plugin, name)
		return
	}
	fv, ok := v.(FloatValue)
	if !ok {
		t.Errorf("%s plugin field %q: expected FloatValue, got %T", plugin, name, v)
		return
	}
	if math.Abs(fv.V-want) > 0.000001 {
		t.Errorf("%s plugin field %q: want %v, got %v", plugin, name, want, fv.V)
	}
}

func requirePluginObjectField(t *testing.T, plugin, name string, fields map[string]Value) ObjectValue {
	t.Helper()
	v, ok := fields[name]
	if !ok {
		t.Errorf("expected %s plugin AST to carry object field %q", plugin, name)
		return ObjectValue{}
	}
	ov, ok := v.(ObjectValue)
	if !ok {
		t.Errorf("%s plugin field %q: expected ObjectValue, got %T", plugin, name, v)
		return ObjectValue{}
	}
	return ov
}
