package config

import (
	"strings"
	"testing"
)

// withBackend gives model a backend, as providers.models[].backend_refs does.
func withBackend(cfg *RouterConfig, model string) {
	name := model + "-backend"
	cfg.VLLMEndpoints = append(cfg.VLLMEndpoints, VLLMEndpoint{Name: name, Address: "127.0.0.1", Port: 8000, Model: model})
	if cfg.ModelConfig == nil {
		cfg.ModelConfig = map[string]ModelParams{}
	}
	params := cfg.ModelConfig[model]
	params.PreferredEndpoints = []string{name}
	cfg.ModelConfig[model] = params
}

// withDecisionBackends gives every model the decisions call in process a
// backend, for tests about something else.
func withDecisionBackends(cfg *RouterConfig) *RouterConfig {
	for _, decision := range cfg.Decisions {
		for _, model := range decisionCalledModels(decision) {
			withBackend(cfg, model)
		}
	}
	return cfg
}

func looperDecision(algorithm *AlgorithmConfig, models ...string) Decision {
	decision := Decision{Name: "looper", Algorithm: algorithm}
	for _, model := range models {
		decision.ModelRefs = append(decision.ModelRefs, ModelRef{Model: model})
	}
	return decision
}

func TestDecisionCallsNeedABackend(t *testing.T) {
	ratings := &AlgorithmConfig{Type: "ratings"}
	fusion := &AlgorithmConfig{Type: "fusion", Fusion: &FusionAlgorithmConfig{Model: "judge", AnalysisModels: []string{"panel"}}}
	prompt := &AlgorithmConfig{Type: "prompt", Prompt: &PromptSelectionConfig{Model: "helper"}}
	flow := &AlgorithmConfig{Type: "workflows", Workflows: &WorkflowsAlgorithmConfig{
		Mode: "dynamic", Planner: WorkflowPlannerConfig{Model: "planner"},
	}}
	cases := []struct {
		name     string
		decision Decision
		backends []string
		missing  string
	}{
		{"every candidate of a Looper", looperDecision(ratings, "a", "b"), []string{"a"}, "b"},
		{"the Fusion judge", looperDecision(fusion, "a"), []string{"a", "panel"}, "judge"},
		{"the Fusion panel", looperDecision(fusion, "a"), []string{"a", "judge"}, "panel"},
		{"the prompt helper", looperDecision(prompt, "a", "b"), []string{"a", "b"}, "helper"},
		{"the Flow planner", looperDecision(flow, "a"), []string{"a"}, "planner"},
		{"a served Looper", looperDecision(ratings, "a", "b"), []string{"a", "b"}, ""},
		{"a static decision", looperDecision(&AlgorithmConfig{Type: "static"}, "a", "b"), nil, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &RouterConfig{}
			for _, model := range tc.backends {
				withBackend(cfg, model)
			}
			err := validateDecisionLooperBackends(cfg, tc.decision)
			if tc.missing == "" {
				if err != nil {
					t.Fatalf("validate: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), `model "`+tc.missing+`"`) ||
				!strings.Contains(err.Error(), "providers.models[].backend_refs") {
				t.Fatalf("validate = %v, want model %q refused for its missing backend", err, tc.missing)
			}
		})
	}
}

func TestARoutingFragmentNeedsNoBackends(t *testing.T) {
	cfg := &RouterConfig{RoutingFragmentOnly: true}
	if err := validateDecisionLooperBackends(cfg, looperDecision(&AlgorithmConfig{Type: "ratings"}, "a", "b")); err != nil {
		t.Fatalf("a routing fragment declares no providers; got %v", err)
	}
}

func TestALooperModelWithoutBackendRefsFailsTheDocument(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(`version: v0.3
listeners: []
providers:
  defaults:
    model: a
  models:
    - name: a
      backend_refs:
        - endpoint: 127.0.0.1:8000
    - name: b
      provider_model_id: b
routing:
  modelCards:
    - name: a
    - name: b
  decisions:
    - name: rate
      priority: 1
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: a
        - model: b
      algorithm:
        type: ratings
`))
	if err == nil || !strings.Contains(err.Error(), `decision "rate": the Router calls model "b" in process, but it has no backend`) {
		t.Fatalf("parse = %v, want model b refused for its missing backend", err)
	}
}
