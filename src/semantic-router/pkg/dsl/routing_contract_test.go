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
  reasoning_family_ref: "openai"
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
  reasoning_family_ref: "openai"
  param_size: "3b"
  description: "Math focused local model"
  capabilities: ["math", "chat"]
  tags: ["local", "fast"]
  quality_score: 0.91
  modality: "ar"
}

ROUTE math_route {
  PRIORITY 10
  MODEL "math-small"
}`

	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	params, ok := cfg.ModelConfig["math-small"]
	if !ok {
		t.Fatal("expected compiled model catalog entry")
	}
	if params.ReasoningFamily != "openai" {
		t.Fatalf("reasoning family = %q", params.ReasoningFamily)
	}
	if params.ParamSize != "3b" {
		t.Fatalf("param_size = %q", params.ParamSize)
	}
	if params.Description != "Math focused local model" {
		t.Fatalf("description = %q", params.Description)
	}
	if len(params.Capabilities) != 2 || params.Capabilities[0] != "math" {
		t.Fatalf("capabilities = %#v", params.Capabilities)
	}
	if len(params.Tags) != 2 || params.Tags[0] != "local" {
		t.Fatalf("tags = %#v", params.Tags)
	}
	if params.Modality != "ar" {
		t.Fatalf("modality = %q", params.Modality)
	}
}

func TestEmitRoutingYAMLFromConfig(t *testing.T) {
	input := `
SIGNAL domain math { description: "math" }

MODEL "math-small" {
  reasoning_family_ref: "openai"
  param_size: "3b"
  capabilities: ["math", "chat"]
  tags: ["local", "fast"]
}

ROUTE math_route {
  PRIORITY 10
  WHEN domain("math")
  MODEL "math-small"
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
      provider_model_id: qwen/qwen3
      backend_refs:
        - name: local
          endpoint: http://localhost:8000/v1
          protocol: http
          api_key_env: OPENAI_API_KEY
routing:
  modelCards:
    - name: math-small
      reasoning_family_ref: openai
      param_size: 3b
      capabilities: [math, chat]
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
	if strings.Contains(dslText, "BACKEND ") || strings.Contains(dslText, "GLOBAL {") {
		t.Fatalf("routing-only decompile leaked static sections:\n%s", dslText)
	}
}
