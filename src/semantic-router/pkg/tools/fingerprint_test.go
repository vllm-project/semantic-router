package tools

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func boolPtr(b bool) *bool { return &b }

func intPtr(v int) *int { return &v }

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
	first := ToolCatalogFingerprint(catalog)
	second := ToolCatalogFingerprint(catalog)
	if first != second {
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
	spec := func() *config.ToolsSelectConfig {
		judge := &config.MatchJudge{Preset: "tool_rank", TopK: 5}
		return &config.ToolsSelectConfig{Mode: config.ToolsSelectModeAdd, ToolsDBPath: "tools.json", Match: &config.MatchConfig{Judge: judge}}
	}
	first, second := ToolPolicyFingerprint(spec()), ToolPolicyFingerprint(spec())
	if first != second {
		t.Fatal("same policy must fingerprint identically")
	}
}

func TestToolPolicyFingerprint_NilConfig(t *testing.T) {
	if got := ToolPolicyFingerprint(nil); got == "" {
		t.Fatal("nil policy must still produce a fingerprint")
	}
}

func TestToolPolicyFingerprint_SensitiveToSelection(t *testing.T) {
	base := config.ToolsSelectConfig{Mode: config.ToolsSelectModeAdd}
	need := 0.7
	for name, edit := range map[string]func(*config.ToolsSelectConfig){
		"mode": func(s *config.ToolsSelectConfig) { s.Mode = config.ToolsSelectModeFilter },
		"judge": func(s *config.ToolsSelectConfig) {
			s.Match = &config.MatchConfig{Judge: &config.MatchJudge{Preset: "tool_rank", TopK: 2}}
		},
		"need_threshold": func(s *config.ToolsSelectConfig) { s.NeedThreshold = &need },
		"preserve_count": func(s *config.ToolsSelectConfig) { s.PreserveCount = 1 },
		"history":        func(s *config.ToolsSelectConfig) { s.Input.History = true },
	} {
		changed := base
		edit(&changed)
		if ToolPolicyFingerprint(&base) == ToolPolicyFingerprint(&changed) {
			t.Errorf("a %s change must change the fingerprint", name)
		}
	}
}

func TestToolPolicyFingerprint_ExplicitDefaultEqualsOmitted(t *testing.T) {
	// Defaults written out and defaults omitted are the same effective
	// policy: the judge, the need threshold and the sticky bounds.
	need := config.DefaultToolsNeedThreshold
	explicitDefault := &config.ToolsSelectConfig{
		Mode:          config.ToolsSelectModeAdd,
		Match:         &config.MatchConfig{Judge: config.MatchToolsSelect.Default().Judge},
		NeedThreshold: &need,
		Sticky: &config.StickyToolSelectionConfig{
			MaxTools:           intPtr(config.StickyToolSelectionDefaultMaxTools),
			MaxNewToolsPerTurn: intPtr(config.StickyToolSelectionDefaultMaxNewToolsPerTurn),
			PinCalledTools:     boolPtr(true),
		},
	}
	omitted := &config.ToolsSelectConfig{Mode: config.ToolsSelectModeAdd, Sticky: &config.StickyToolSelectionConfig{}}
	if ToolPolicyFingerprint(explicitDefault) != ToolPolicyFingerprint(omitted) {
		t.Fatal("explicit defaults must fingerprint the same as omitted ones")
	}
}

func TestToolPolicyFingerprint_SensitiveToStickyBoundChange(t *testing.T) {
	base := &config.ToolsSelectConfig{Mode: config.ToolsSelectModeAdd, Sticky: &config.StickyToolSelectionConfig{MaxTools: intPtr(16)}}
	changed := &config.ToolsSelectConfig{Mode: config.ToolsSelectModeAdd, Sticky: &config.StickyToolSelectionConfig{MaxTools: intPtr(8)}}
	if ToolPolicyFingerprint(base) == ToolPolicyFingerprint(changed) {
		t.Fatal("sticky max_tools change must change the fingerprint")
	}
}

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

// schemaWithMaximum builds a minimal tool schema whose only variable is a
// numeric bound, so a fingerprint difference can only come from that number.
func schemaWithMaximum(literal string) json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{"id":{"type":"integer","maximum":` + literal + `}}}`)
}

func TestCanonicalizeJSON_PreservesLargeIntegers(t *testing.T) {
	// Decoding into a plain interface{} routes numbers through float64,
	// whose 53-bit mantissa rounds these literals; each must survive the
	// canonical round trip byte-exact instead.
	for _, literal := range []string{
		"9007199254740993",     // 2^53 + 1, the first integer float64 cannot hold
		"18446744073709551615", // max uint64
		"18446744073709551616", // max uint64 + 1, float64-identical to the line above
		"-9007199254740993",
		"1.0000000000000000001",
	} {
		raw := json.RawMessage(`{"n":` + literal + `}`)
		got, err := canonicalizeJSON(raw)
		if err != nil {
			t.Fatalf("canonicalizeJSON(%s): %v", raw, err)
		}
		if want := `{"n":` + literal + `}`; string(got) != want {
			t.Errorf("number literal not preserved: got %s, want %s", got, want)
		}
	}
}

func TestCanonicalizeJSON_RejectsTrailingData(t *testing.T) {
	// json.Unmarshal rejects trailing bytes on its own; a json.Decoder stops
	// at the first value, so canonicalizeJSON has to re-assert the check --
	// otherwise a schema plus trailing bytes would canonicalize to the
	// schema alone and collide with it.
	for _, raw := range []string{
		`{"type":"object"} {"type":"array"}`,
		`{"type":"object"}trailing`,
		`{"type":"object"}[]`,
	} {
		if _, err := canonicalizeJSON(json.RawMessage(raw)); !errors.Is(err, errTrailingJSON) {
			t.Errorf("canonicalizeJSON(%s) = %v, want errTrailingJSON", raw, err)
		}
	}
}

func TestToolDefinitionFingerprint_SensitiveToLargeIntegerSchemaChange(t *testing.T) {
	// Both bounds are valid, distinct JSON Schema integers that differ by
	// one past float64 precision. Collapsing them would let sticky state
	// survive a real tool-schema change.
	a := sampleTool("search")
	a.InputSchema = schemaWithMaximum("9007199254740992") // 2^53
	b := sampleTool("search")
	b.InputSchema = schemaWithMaximum("9007199254740993") // 2^53 + 1
	if ToolDefinitionFingerprint(a) == ToolDefinitionFingerprint(b) {
		t.Fatal("schema integers beyond float64 precision must fingerprint distinctly")
	}
}

func TestToolDefinitionFingerprint_SensitiveToUint64RangeSchemaChange(t *testing.T) {
	// These two literals round to the same float64 (18446744073709552000),
	// so a float64-based canonicalization produces identical bytes for
	// schemas pinning different account-scale identifiers.
	a := sampleTool("search")
	a.InputSchema = json.RawMessage(`{"type":"object","properties":{"account_id":{"const":18446744073709551615}}}`)
	b := sampleTool("search")
	b.InputSchema = json.RawMessage(`{"type":"object","properties":{"account_id":{"const":18446744073709551616}}}`)
	if ToolDefinitionFingerprint(a) == ToolDefinitionFingerprint(b) {
		t.Fatal("schema constants that share a float64 representation must still fingerprint distinctly")
	}
}

func TestToolCatalogFingerprint_SensitiveToLargeIntegerSchemaChange(t *testing.T) {
	// The catalog fingerprint is the signal sticky state is invalidated on,
	// so the precision guarantee has to hold at that level too.
	a := sampleTool("search")
	a.InputSchema = schemaWithMaximum("9007199254740992")
	b := sampleTool("search")
	b.InputSchema = schemaWithMaximum("9007199254740993")
	forward := []llmprotocol.Tool{sampleTool("lookup"), a}
	changed := []llmprotocol.Tool{sampleTool("lookup"), b}
	if ToolCatalogFingerprint(forward) == ToolCatalogFingerprint(changed) {
		t.Fatal("a large-integer schema change must invalidate the catalog fingerprint")
	}
}
