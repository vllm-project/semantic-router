package tools

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func boolPtr(b bool) *bool { return &b }

func sampleTool(name string) llmprotocol.Tool {
	return llmprotocol.Tool{
		Name:        name,
		Description: "looks up " + name,
		Strict:      boolPtr(true),
		InputSchema: json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}}}`),
		Cache:       &llmprotocol.CacheDirective{Type: "ephemeral", TTL: "5m"},
	}
}

func TestToolDefinitionFingerprint_Deterministic(t *testing.T) {
	tool := sampleTool("search")
	a := ToolDefinitionFingerprint(tool)
	b := ToolDefinitionFingerprint(tool)
	if a != b {
		t.Fatalf("fingerprint not deterministic: %q vs %q", a, b)
	}
	if a == "" {
		t.Fatal("fingerprint must not be empty")
	}
}

func TestToolDefinitionFingerprint_SchemaKeyOrderInvariant(t *testing.T) {
	a := sampleTool("search")
	a.InputSchema = json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}}}`)
	b := sampleTool("search")
	b.InputSchema = json.RawMessage(`{"properties":{"query":{"type":"string"}},"type":"object"}`)
	if ToolDefinitionFingerprint(a) != ToolDefinitionFingerprint(b) {
		t.Fatal("semantically identical schemas with different key order must fingerprint identically")
	}
}

func TestToolDefinitionFingerprint_SchemaWhitespaceInvariant(t *testing.T) {
	a := sampleTool("search")
	a.InputSchema = json.RawMessage(`{"type":"object"}`)
	b := sampleTool("search")
	b.InputSchema = json.RawMessage("{\n  \"type\" : \"object\"\n}")
	if ToolDefinitionFingerprint(a) != ToolDefinitionFingerprint(b) {
		t.Fatal("semantically identical schemas with different whitespace must fingerprint identically")
	}
}

func TestToolDefinitionFingerprint_SensitiveToSchemaChange(t *testing.T) {
	a := sampleTool("search")
	b := sampleTool("search")
	b.InputSchema = json.RawMessage(`{"type":"object","properties":{"query":{"type":"number"}}}`)
	if ToolDefinitionFingerprint(a) == ToolDefinitionFingerprint(b) {
		t.Fatal("a real schema change must produce a distinct fingerprint")
	}
}

func TestToolDefinitionFingerprint_SensitiveToDescriptionChange(t *testing.T) {
	a := sampleTool("search")
	b := sampleTool("search")
	b.Description = "a completely different description"
	if ToolDefinitionFingerprint(a) == ToolDefinitionFingerprint(b) {
		t.Fatal("a description change must produce a distinct fingerprint")
	}
}

func TestToolDefinitionFingerprint_SensitiveToStrictNilVsFalse(t *testing.T) {
	// nil Strict ("unspecified") and Strict: false are different
	// configurations and must not collide.
	a := sampleTool("search")
	a.Strict = nil
	b := sampleTool("search")
	b.Strict = boolPtr(false)
	if ToolDefinitionFingerprint(a) == ToolDefinitionFingerprint(b) {
		t.Fatal("nil Strict and Strict:false must fingerprint distinctly")
	}
}

func TestToolDefinitionFingerprint_SensitiveToCachePresence(t *testing.T) {
	a := sampleTool("search")
	b := sampleTool("search")
	b.Cache = nil
	if ToolDefinitionFingerprint(a) == ToolDefinitionFingerprint(b) {
		t.Fatal("a present vs absent cache directive must fingerprint distinctly")
	}
}

func TestToolDefinitionFingerprint_NameWhitespaceNormalized(t *testing.T) {
	// sampleTool bakes its argument into Description too ("looks up "+name),
	// which would entangle a name-whitespace difference into the
	// description as well — build both tools directly with an identical
	// Description so only Name varies, isolating what this test checks.
	a := sampleTool("search")
	a.Name = "search"
	b := sampleTool("search")
	b.Name = "  search  "
	if ToolDefinitionFingerprint(a) != ToolDefinitionFingerprint(b) {
		t.Fatal("surrounding whitespace in the name must not change the fingerprint")
	}
}

func TestToolCatalogFingerprint_Deterministic(t *testing.T) {
	catalog := []llmprotocol.Tool{sampleTool("search"), sampleTool("lookup")}
	if ToolCatalogFingerprint(catalog) != ToolCatalogFingerprint(catalog) {
		t.Fatal("catalog fingerprint must be deterministic")
	}
}

func TestToolCatalogFingerprint_OrderInvariant(t *testing.T) {
	forward := []llmprotocol.Tool{sampleTool("alpha"), sampleTool("beta"), sampleTool("gamma")}
	reversed := []llmprotocol.Tool{sampleTool("gamma"), sampleTool("beta"), sampleTool("alpha")}
	shuffled := []llmprotocol.Tool{sampleTool("beta"), sampleTool("alpha"), sampleTool("gamma")}
	f1, f2, f3 := ToolCatalogFingerprint(forward), ToolCatalogFingerprint(reversed), ToolCatalogFingerprint(shuffled)
	if f1 != f2 || f1 != f3 {
		t.Fatalf("catalog fingerprint must be order-invariant: %q %q %q", f1, f2, f3)
	}
}

func TestToolCatalogFingerprint_SensitiveToMembershipChange(t *testing.T) {
	a := []llmprotocol.Tool{sampleTool("alpha"), sampleTool("beta")}
	b := []llmprotocol.Tool{sampleTool("alpha"), sampleTool("gamma")}
	if ToolCatalogFingerprint(a) == ToolCatalogFingerprint(b) {
		t.Fatal("a different catalog membership must produce a distinct fingerprint")
	}
}

func TestToolCatalogFingerprint_EmptyCatalog(t *testing.T) {
	if got := ToolCatalogFingerprint(nil); got == "" {
		t.Fatal("an empty catalog must still produce a non-empty, well-defined fingerprint")
	}
}

func TestToolPolicyFingerprint_Deterministic(t *testing.T) {
	cfg := &config.ToolSelectionPluginConfig{Enabled: true, Mode: "add", TopK: 5}
	if ToolPolicyFingerprint(cfg) != ToolPolicyFingerprint(cfg) {
		t.Fatal("policy fingerprint must be deterministic")
	}
}

func TestToolPolicyFingerprint_NilConfig(t *testing.T) {
	if got := ToolPolicyFingerprint(nil); got == "" {
		t.Fatal("a nil plugin config must still produce a well-defined fingerprint")
	}
}

func TestToolPolicyFingerprint_SensitiveToModeChange(t *testing.T) {
	add := &config.ToolSelectionPluginConfig{Enabled: true, Mode: "add"}
	filter := &config.ToolSelectionPluginConfig{Enabled: true, Mode: "filter"}
	if ToolPolicyFingerprint(add) == ToolPolicyFingerprint(filter) {
		t.Fatal("a mode change must produce a distinct fingerprint")
	}
}

func TestToolPolicyFingerprint_ExplicitDefaultEqualsOmitted(t *testing.T) {
	// StickyToolSelectionConfig{} (present, all zero) and no Sticky block
	// at all both resolve to the same *effective* policy — the fingerprint
	// covers effective policy, not raw YAML presence, so it must match.
	explicitDefault := &config.ToolSelectionPluginConfig{
		Enabled: true, Mode: "add",
		Sticky: &config.StickyToolSelectionConfig{Enabled: true},
	}
	omitted := &config.ToolSelectionPluginConfig{
		Enabled: true, Mode: "add",
		Sticky: &config.StickyToolSelectionConfig{
			Enabled:            true,
			MaxTools:           intPtr(config.StickyToolSelectionDefaultMaxTools),
			MaxNewToolsPerTurn: intPtr(config.StickyToolSelectionDefaultMaxNewToolsPerTurn),
			PinCalledTools:     boolPtr(true),
		},
	}
	if ToolPolicyFingerprint(explicitDefault) != ToolPolicyFingerprint(omitted) {
		t.Fatal("an explicit value equal to the default must fingerprint the same as an omitted field")
	}
}

func TestToolPolicyFingerprint_SensitiveToStickyBoundChange(t *testing.T) {
	base := &config.ToolSelectionPluginConfig{
		Enabled: true, Mode: "add",
		Sticky: &config.StickyToolSelectionConfig{Enabled: true, MaxTools: intPtr(16)},
	}
	changed := &config.ToolSelectionPluginConfig{
		Enabled: true, Mode: "add",
		Sticky: &config.StickyToolSelectionConfig{Enabled: true, MaxTools: intPtr(8)},
	}
	if ToolPolicyFingerprint(base) == ToolPolicyFingerprint(changed) {
		t.Fatal("a sticky bound change must produce a distinct fingerprint")
	}
}

func TestToolPolicyFingerprint_AllowBlockListsOrderInvariant(t *testing.T) {
	forward := &config.ToolSelectionPluginConfig{
		Enabled: true, Mode: "add",
		AdvancedFiltering: &config.AdvancedToolFilteringConfig{
			Enabled: true, AllowTools: []string{"a", "b", "c"},
		},
	}
	reversed := &config.ToolSelectionPluginConfig{
		Enabled: true, Mode: "add",
		AdvancedFiltering: &config.AdvancedToolFilteringConfig{
			Enabled: true, AllowTools: []string{"c", "b", "a"},
		},
	}
	if ToolPolicyFingerprint(forward) != ToolPolicyFingerprint(reversed) {
		t.Fatal("allow_tools order must not change the policy fingerprint — it is an unordered set")
	}
}

func intPtr(v int) *int { return &v }

func TestToolCapabilityFingerprint_Deterministic(t *testing.T) {
	a := ToolCapabilityFingerprint([]string{"tools", "vision"}, "openai_responses")
	b := ToolCapabilityFingerprint([]string{"tools", "vision"}, "openai_responses")
	if a != b {
		t.Fatal("capability fingerprint must be deterministic")
	}
}

func TestToolCapabilityFingerprint_OrderInvariant(t *testing.T) {
	a := ToolCapabilityFingerprint([]string{"tools", "vision"}, "openai_responses")
	b := ToolCapabilityFingerprint([]string{"vision", "tools"}, "openai_responses")
	if a != b {
		t.Fatal("capability list order must not change the fingerprint")
	}
}

func TestToolCapabilityFingerprint_SensitiveToWireFormatChange(t *testing.T) {
	a := ToolCapabilityFingerprint([]string{"tools"}, "openai_responses")
	b := ToolCapabilityFingerprint([]string{"tools"}, "anthropic_messages")
	if a == b {
		t.Fatal("a wire format change must produce a distinct fingerprint")
	}
}

func TestToolCapabilityFingerprint_SensitiveToCapabilityChange(t *testing.T) {
	a := ToolCapabilityFingerprint([]string{"tools"}, "openai_responses")
	b := ToolCapabilityFingerprint([]string{"tools", "vision"}, "openai_responses")
	if a == b {
		t.Fatal("a capability set change must produce a distinct fingerprint")
	}
}

func TestCanonicalizeJSON_EmptyInput(t *testing.T) {
	got, err := canonicalizeJSON(nil)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "null" {
		t.Fatalf("empty input should canonicalize to null, got %q", got)
	}
}

func TestCanonicalizeJSON_InvalidInput(t *testing.T) {
	if _, err := canonicalizeJSON(json.RawMessage(`{not valid json`)); err == nil {
		t.Fatal("expected an error for invalid JSON")
	}
}
