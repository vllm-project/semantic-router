package editor

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestFullDocumentEditorPreservesCustomRuleBudget(t *testing.T) {
	var rule interface{} = map[string]interface{}{"type": "keyword", "name": "urgent"}
	for i := 1; i < 17; i++ {
		rule = map[string]interface{}{"operator": "NOT", "conditions": []interface{}{rule}}
	}
	base, err := json.Marshal(map[string]interface{}{
		"version": "v0.3",
		"global": map[string]interface{}{"router": map[string]interface{}{
			"config_source": "kubernetes", "decision_rule_limits": map[string]int{"max_depth": 32, "max_nodes": 512},
		}},
		"routing": map[string]interface{}{
			"signals":   map[string]interface{}{"keywords": []interface{}{map[string]interface{}{"name": "urgent", "operator": "OR", "keywords": []string{"urgent"}}}},
			"decisions": []interface{}{map[string]interface{}{"name": "bounded", "rules": rule}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	decompiled := Decompile(string(base))
	if decompiled.Error != "" {
		t.Fatal(decompiled.Error)
	}
	if got := Compile(decompiled.DSL); !strings.Contains(got.Error, "max_depth=16") {
		t.Fatalf("standalone fragment must retain defaults: %+v", got)
	}
	compiled := CompileWithBase(decompiled.DSL, string(base))
	if compiled.Error != "" || compiled.YAML == "" {
		t.Fatalf("full-document compile lost custom limits: %+v", compiled)
	}
	if got := ValidateWithBase(decompiled.DSL, string(base)); got.Error != "" {
		t.Fatal(got.Error)
	}
	if got := ParseWithBase(decompiled.DSL, string(base)); got.Error != "" || got.AST == nil {
		t.Fatalf("visual builder lost custom limits: %+v", got)
	}
	formatted := FormatWithBase(decompiled.DSL, string(base))
	if formatted.Error != "" || formatted.DSL == "" {
		t.Fatalf("formatting lost custom limits: %+v", formatted)
	}
	if got := CompileWithBase(formatted.DSL, string(base)); got.Error != "" {
		t.Fatalf("formatted DSL no longer compiles: %s", got.Error)
	}
}

func TestEditorUsesLowerLimitsBeforeRecursiveAnalysis(t *testing.T) {
	source := `SIGNAL keyword urgent {keywords: ["urgent"] operator: "OR"}
ROUTE bounded {PRIORITY 1 WHEN NOT keyword("urgent") MODEL "test"}`
	base := "global:\n  router:\n    decision_rule_limits:\n      max_nodes: 1\n"
	for _, message := range []string{
		CompileWithBase(source, base).Error,
		ValidateWithBase(source, base).Error,
		ParseWithBase(source, base).Error,
		FormatWithBase(source, base).Error,
	} {
		if !strings.Contains(message, "node count 2 exceeds max_nodes=1") {
			t.Fatalf("expected lower enclosing limit, got %q", message)
		}
	}
}

func TestEditorBareLeafRoundTripUsesOneNode(t *testing.T) {
	base := `version: v0.3
global:
  router:
    config_source: kubernetes
    decision_rule_limits: {max_depth: 1, max_nodes: 1}
routing:
  signals:
    keywords:
      - {name: urgent, operator: OR, keywords: [urgent]}
  decisions:
    - name: bounded
      rules: {type: keyword, name: urgent}
`
	decompiled := Decompile(base)
	if decompiled.Error != "" {
		t.Fatal(decompiled.Error)
	}
	compiled := CompileWithBase(decompiled.DSL, base)
	if compiled.Error != "" {
		t.Fatalf("bare leaf acquired a synthetic rule node: %s", compiled.Error)
	}
}
