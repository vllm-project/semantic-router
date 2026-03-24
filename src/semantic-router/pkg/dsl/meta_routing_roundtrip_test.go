package dsl

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestCompileMetaRoutingBlock(t *testing.T) {
	input := `
SIGNAL domain "general" {
  description: "General requests"
}

META {
  mode: "shadow"
  max_passes: 2
  trigger_policy: {
    decision_margin_below: 0.2,
    required_families: [
      { type: "preference", min_confidence: 0.6, min_matches: 1 }
    ],
    family_disagreements: [
      { cheap: "keyword", expensive: "embedding" }
    ]
  }
  allowed_actions: [
    { type: "disable_compression" },
    { type: "rerun_signal_families", signal_families: ["preference", "jailbreak"] }
  ]
}

ROUTE "general_route" {
  PRIORITY 1
  WHEN domain("general")
  MODEL "qwen3-8b"
}`

	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	if cfg.MetaRouting.Mode != "shadow" {
		t.Fatalf("expected meta mode shadow, got %q", cfg.MetaRouting.Mode)
	}
	if len(cfg.MetaRouting.AllowedActions) != 2 {
		t.Fatalf("expected 2 allowed actions, got %d", len(cfg.MetaRouting.AllowedActions))
	}
}

func TestValidateMetaRoutingBlockRejectsUnknownSignalFamily(t *testing.T) {
	input := `
META {
  mode: "observe"
  trigger_policy: {
    required_families: [
      { type: "unknown_family" }
    ]
  }
}`

	diags, errs := Validate(input)
	if len(errs) > 0 {
		t.Fatalf("unexpected parse errors: %v", errs)
	}
	if len(diags) == 0 {
		t.Fatal("expected META validation diagnostic")
	}
	if !strings.Contains(diags[0].Message, "unknown_family") {
		t.Fatalf("unexpected diagnostic: %#v", diags[0])
	}
}

func TestDecompileRoutingPreservesMetaBlock(t *testing.T) {
	input := `
routing:
  modelCards:
    - name: qwen3-8b
  signals:
    domains:
      - name: general
        description: General requests
  meta:
    mode: active
    max_passes: 2
    trigger_policy:
      decision_margin_below: 0.12
      projection_boundary_within: 0.05
    allowed_actions:
      - type: disable_compression
  decisions:
    - name: general_route
      priority: 1
      rules:
        operator: AND
        conditions:
          - type: domain
            name: general
      modelRefs:
        - model: qwen3-8b
`

	cfg, err := parseRoutingDSLMetaTestConfig(input)
	if err != nil {
		t.Fatalf("ParseYAMLBytes error: %v", err)
	}
	dslText, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting error: %v", err)
	}
	if !strings.Contains(dslText, "META {") {
		t.Fatalf("expected META block in decompiled DSL:\n%s", dslText)
	}
	if !strings.Contains(dslText, `mode: "active"`) {
		t.Fatalf("expected mode to survive decompile:\n%s", dslText)
	}

	ast := DecompileRoutingToAST(cfg)
	if ast.Meta == nil {
		t.Fatal("expected META block in decompiled AST")
	}
}

func parseRoutingDSLMetaTestConfig(yamlText string) (*config.RouterConfig, error) {
	return config.ParseYAMLBytes([]byte("version: v0.3\nlisteners: []\nproviders:\n  defaults:\n    default_model: qwen3-8b\n  models:\n    - name: qwen3-8b\n      backend_refs:\n        - endpoint: 127.0.0.1:8000\n" + yamlText))
}
