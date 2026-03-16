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
  loras: [
    { name: "math-adapter", description: "Improves symbolic math responses" },
  ]
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
  loras: [
    { name: "math-adapter", description: "Improves symbolic math responses" },
  ]
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
	if !strings.Contains(yamlText, "routing:") {
		t.Fatalf("expected routing fragment, got:\n%s", yamlText)
	}
	if strings.Contains(yamlText, "providers:") || strings.Contains(yamlText, "global:") {
		t.Fatalf("routing fragment leaked static config:\n%s", yamlText)
	}

	var doc struct {
		Routing config.CanonicalRouting `yaml:"routing"`
	}
	if err := yaml.Unmarshal(yamlBytes, &doc); err != nil {
		t.Fatalf("unmarshal routing fragment: %v", err)
	}
	if len(doc.Routing.ModelCards) != 1 {
		t.Fatalf("expected 1 routing model, got %d", len(doc.Routing.ModelCards))
	}
	if doc.Routing.ModelCards[0].Name != "math-small" {
		t.Fatalf("routing model = %q", doc.Routing.ModelCards[0].Name)
	}
	if len(doc.Routing.ModelCards[0].LoRAs) != 1 || doc.Routing.ModelCards[0].LoRAs[0].Name != "math-adapter" {
		t.Fatalf("routing model loras = %#v", doc.Routing.ModelCards[0].LoRAs)
	}
	if doc.Routing.Decisions[0].ModelRefs[0].LoRAName != "math-adapter" {
		t.Fatalf("decision lora_name = %q", doc.Routing.Decisions[0].ModelRefs[0].LoRAName)
	}
}

func TestDecompileRoutingIgnoresStaticCanonicalSections(t *testing.T) {
	configYAML := `
version: v0.3
listeners:
  - name: main
    address: 0.0.0.0
    port: 8080
providers:
  defaults:
    default_model: math-small
    reasoning_families:
      openai:
        type: effort
        parameter: reasoning.effort
    default_reasoning_effort: medium
  models:
    - name: math-small
      reasoning_family: openai
      provider_model_id: qwen/qwen3
      backend_refs:
        - name: local
          endpoint: http://localhost:8000/v1
          protocol: http
          api_key_env: OPENAI_API_KEY
routing:
  modelCards:
    - name: math-small
      param_size: 3b
      capabilities: [math, chat]
      loras:
        - name: math-adapter
          description: Improves symbolic math responses
      tags: [local, fast]
  signals:
    domains:
      - name: math
        description: math
  decisions:
    - name: math_route
      priority: 10
      rules:
        operator: AND
        conditions:
          - type: domain
            name: math
      modelRefs:
        - model: math-small
          lora_name: math-adapter
global:
  router:
    strategy: priority
`

	cfg, err := config.ParseYAMLBytes([]byte(configYAML))
	if err != nil {
		t.Fatalf("ParseYAMLBytes error: %v", err)
	}

	dslText, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting error: %v", err)
	}

	if !strings.Contains(dslText, `MODEL math-small`) {
		t.Fatalf("expected routing model catalog in DSL:\n%s", dslText)
	}
	if !strings.Contains(dslText, `lora = "math-adapter"`) {
		t.Fatalf("expected LoRA reference to survive decompile:\n%s", dslText)
	}
	if !strings.Contains(dslText, `name: "math-adapter"`) {
		t.Fatalf("expected LoRA catalog to survive decompile:\n%s", dslText)
	}
	if strings.Contains(dslText, "BACKEND ") || strings.Contains(dslText, "GLOBAL {") {
		t.Fatalf("routing-only decompile leaked static sections:\n%s", dslText)
	}
}

func TestValidateRouteLoRAAgainstModelCatalog(t *testing.T) {
	input := `
MODEL "math-small" {
  loras: [
    { name: "math-adapter" },
  ]
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
