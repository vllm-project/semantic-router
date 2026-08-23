package dsl

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestParseTopLevelModelCatalog(t *testing.T) {
	input := `
MODEL "math-small" {
  param_size: "3b"
  capabilities: ["math", "chat"]
  tags: ["local", "fast"]
  modality: "ar"
}

ROUTE math_route {
  PRIORITY 10
  MODEL "math-small"
}`

	prog, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	if len(prog.Models) != 1 {
		t.Fatalf("expected 1 top-level model, got %d", len(prog.Models))
	}
	if prog.Models[0].Name != "math-small" {
		t.Fatalf("unexpected model name %q", prog.Models[0].Name)
	}
}

func TestCompileTopLevelModelCatalog(t *testing.T) {
	input := `
MODEL "math-small" {
  param_size: "3b"
  description: "Math focused local model"
  capabilities: ["math", "chat"]
  loras: ["math-adapter"]
  tags: ["local", "fast"]
  quality_score: 0.91
  modality: "ar"
}

ROUTE math_route {
  PRIORITY 10
  MODEL "math-small"(lora = "math-adapter")
}`

	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	params, ok := cfg.ModelConfig["math-small"]
	if !ok {
		t.Fatal("expected compiled model catalog entry")
	}
	assertCompiledTopLevelModelCatalog(t, params)
}

func assertCompiledTopLevelModelCatalog(t *testing.T, params config.ModelParams) {
	t.Helper()
	assertStringField(t, "reasoning family", "", params.ReasoningFamily)
	assertStringField(t, "param_size", "3b", params.ParamSize)
	assertStringField(t, "description", "Math focused local model", params.Description)
	assertStringField(t, "modality", "ar", params.Modality)
	assertLeadingStringSlice(t, "capabilities", params.Capabilities, 2, "math")
	assertLeadingStringSlice(t, "tags", params.Tags, 2, "local")
	if len(params.LoRAs) != 1 || params.LoRAs[0].Name != "math-adapter" {
		t.Fatalf("loras = %#v", params.LoRAs)
	}
}

func assertStringField(t *testing.T, name, want, got string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s = %q", name, got)
	}
}

func assertLeadingStringSlice(
	t *testing.T,
	name string,
	values []string,
	wantLen int,
	wantFirst string,
) {
	t.Helper()
	if len(values) != wantLen || values[0] != wantFirst {
		t.Fatalf("%s = %#v", name, values)
	}
}

func TestEmitRoutingYAMLFromConfig(t *testing.T) {
	input := `
SIGNAL domain math { description: "math" }

MODEL "math-small" {
  param_size: "3b"
  capabilities: ["math", "chat"]
  loras: ["math-adapter"]
  tags: ["local", "fast"]
}

ROUTE math_route {
  PRIORITY 10
  WHEN domain("math")
  MODEL "math-small"(lora = "math-adapter")
}`

	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	yamlBytes, err := EmitRoutingYAMLFromConfig(cfg)
	if err != nil {
		t.Fatalf("EmitRoutingYAMLFromConfig error: %v", err)
	}

	yamlText := string(yamlBytes)
	if !strings.Contains(yamlText, "document:") {
		t.Fatalf("expected Recipe document fragment, got:\n%s", yamlText)
	}
	if strings.Contains(yamlText, "providers:") || strings.Contains(yamlText, "global:") {
		t.Fatalf("routing fragment leaked static config:\n%s", yamlText)
	}

	var doc struct {
		Document config.CanonicalRouting `yaml:"document"`
	}
	if err := yaml.Unmarshal(yamlBytes, &doc); err != nil {
		t.Fatalf("unmarshal routing fragment: %v", err)
	}
	if len(doc.Document.Decisions) != 1 {
		t.Fatalf("expected one Recipe decision, got %d", len(doc.Document.Decisions))
	}
	if len(doc.Document.Decisions[0].ModelRefs) != 0 {
		t.Fatalf("Recipe document leaked physical Model refs: %#v", doc.Document.Decisions[0].ModelRefs)
	}
}

func TestDecompileRoutingIgnoresStaticCanonicalSections(t *testing.T) {
	cfg := &config.RouterConfig{
		APIServer: config.APIServer{Listeners: []config.Listener{{Name: "main", Address: "0.0.0.0", Port: 8080}}},
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"math-small": {
					ResourceID: "mdl_math", ResourceRevision: 1, ParamSize: "3b",
					Capabilities: []string{"math", "chat"},
					LoRAs:        []config.LoRAAdapter{{Name: "math-adapter"}},
					Tags:         []string{"local", "fast"},
				},
			},
		},
		Recipes: []config.RoutingRecipe{{
			ID: "rcp_math", Revision: 1, Name: "math",
			Profile: config.RoutingProfile{
				Signals:   config.Signals{Categories: []config.Category{{CategoryMetadata: config.CategoryMetadata{Name: "math", Description: "math"}}}},
				Decisions: []config.Decision{{ID: "dec_math", Name: "math_route", Priority: 10, Rules: config.RuleCombination{Operator: "AND", Conditions: []config.RuleCondition{{Type: "domain", Name: "math"}}}}},
				Strategy:  config.RoutingStrategyPriority,
			},
		}},
		Entrypoints: []config.EntrypointMapping{{
			ID: "ep_math", Revision: 1, Name: "math", ModelNames: []string{"math"},
			Rules: []config.EntrypointRule{{
				ID: "rule_math", Name: "default",
				Action: config.EntrypointRuleAction{RecipeID: "rcp_math", Assignments: map[string]config.RoutingAssignmentSet{
					"dec_math": {Models: []config.RoutingModelAssignment{{ModelID: "mdl_math", Priority: 1, Weight: "1", LoRAName: "math-adapter"}}},
				}},
			}},
		}},
	}

	dslText, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("Decompile error: %v", err)
	}

	if !strings.Contains(dslText, `MODEL math-small`) {
		t.Fatalf("expected routing model catalog in DSL:\n%s", dslText)
	}
	if !strings.Contains(dslText, `lora: "math-adapter"`) {
		t.Fatalf("expected Entrypoint LoRA assignment to survive decompile:\n%s", dslText)
	}
	if !strings.Contains(dslText, `loras: ["math-adapter"]`) {
		t.Fatalf("expected LoRA catalog to survive decompile:\n%s", dslText)
	}
	if strings.Contains(dslText, "BACKEND ") || strings.Contains(dslText, "GLOBAL {") {
		t.Fatalf("routing-only decompile leaked static sections:\n%s", dslText)
	}
}

func TestValidateRouteLoRAAgainstModelCatalog(t *testing.T) {
	input := `
MODEL "math-small" {
  loras: ["math-adapter"]
}

ROUTE math_route {
  PRIORITY 10
  MODEL "math-small"(lora = "missing-adapter")
}`

	diags, errs := Validate(input)
	if len(errs) > 0 {
		t.Fatalf("unexpected parse errors: %v", errs)
	}

	for _, diag := range diags {
		if strings.Contains(diag.Message, `LoRA "missing-adapter" is not declared for model "math-small"`) {
			return
		}
	}

	t.Fatalf("expected missing LoRA diagnostic, got %#v", diags)
}
