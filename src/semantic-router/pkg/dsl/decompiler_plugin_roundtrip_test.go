package dsl

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestPluginFieldsHallucinationEmitsAllAuthorableFields locks in that the
// AST emitter for the hallucination plugin surfaces every field the DSL
// compiler can read (use_nli, hallucination_action). Without this guard
// the AST converter silently drops those fields while the text decompiler
// emits them, making any runtime config -> AST -> compile round trip
// lossy on those knobs even though the runtime schema and the text DSL
// fully support them.
func TestPluginFieldsHallucinationEmitsAllAuthorableFields(t *testing.T) {
	cfg := config.HallucinationPluginConfig{
		Enabled:             true,
		UseNLI:              true,
		HallucinationAction: "block",
	}
	plugin := pluginFromConfig(t, config.DecisionPluginHallucination, cfg)

	fields := pluginFieldsHallucination(plugin)

	requirePluginBoolField(t, "hallucination", "enabled", fields, true)
	requirePluginBoolField(t, "hallucination", "use_nli", fields, true)
	requirePluginStringField(t, "hallucination", "hallucination_action", fields, "block")
}

// TestPluginFieldsHallucinationOmitsZeroValuedFields documents the
// omit-when-default contract: when the runtime config carries zero / empty
// values, the AST emitter must not introduce noisy default fields.
func TestPluginFieldsHallucinationOmitsZeroValuedFields(t *testing.T) {
	cfg := config.HallucinationPluginConfig{Enabled: true}
	plugin := pluginFromConfig(t, config.DecisionPluginHallucination, cfg)

	fields := pluginFieldsHallucination(plugin)

	if _, ok := fields["use_nli"]; ok {
		t.Errorf("expected use_nli to be omitted when false")
	}
	if _, ok := fields["hallucination_action"]; ok {
		t.Errorf("expected hallucination_action to be omitted when empty")
	}
}

// TestPluginFieldsRAGEmitsOnFailure locks in that the AST emitter for
// the RAG plugin surfaces the on_failure knob. The DSL compiler reads it
// in compileRAGPlugin (with documented "warn" / "error" semantics) but
// pluginFieldsRAG used to drop it on the AST path.
func TestPluginFieldsRAGEmitsOnFailure(t *testing.T) {
	cfg := config.RAGPluginConfig{
		Enabled:   true,
		Backend:   "milvus",
		OnFailure: "warn",
	}
	plugin := pluginFromConfig(t, config.DecisionPluginRAG, cfg)

	fields := pluginFieldsRAG(plugin)

	requirePluginStringField(t, "rag", "on_failure", fields, "warn")
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
	cfg := config.RAGPluginConfig{
		Enabled:   true,
		Backend:   "milvus",
		OnFailure: "error",
	}
	plugin := pluginFromConfig(t, config.DecisionPluginRAG, cfg)

	var sb strings.Builder
	emitRAGPluginConfig(&sb, plugin)

	out := sb.String()
	if !strings.Contains(out, `on_failure: "error"`) {
		t.Errorf("expected emitRAGPluginConfig output to contain on_failure, got:\n%s", out)
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
