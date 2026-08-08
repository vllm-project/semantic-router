package decision

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMultiStepPipelineDecisionFragmentRoutesEachStep(t *testing.T) {
	decisions := loadMultiStepPipelineDecisions(t)
	decisions = append(decisions, config.Decision{
		Name:     "generic_pipeline_step",
		Priority: 10,
		Rules: config.RuleCombination{
			Operator: "OR",
			Conditions: []config.RuleCondition{
				{Type: "keyword", Name: "pipeline_step_summarize"},
				{Type: "keyword", Name: "pipeline_step_extract"},
			},
		},
		ModelRefs: []config.ModelRef{{Model: "qwen2.5:3b"}},
	})

	engine := NewDecisionEngine(nil, nil, nil, decisions, "priority")

	tests := []struct {
		name         string
		matchedRules []string
		wantDecision string
		wantModel    string
	}{
		{
			name:         "summarize step",
			matchedRules: []string{"pipeline_step_summarize"},
			wantDecision: "pipeline_summarize_step",
			wantModel:    "qwen3-8b",
		},
		{
			name:         "extract step",
			matchedRules: []string{"pipeline_step_extract"},
			wantDecision: "pipeline_extract_step",
			wantModel:    "qwen3-32b",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := engine.EvaluateDecisions(tt.matchedRules, nil, nil)
			if err != nil {
				t.Fatalf("EvaluateDecisions() error = %v", err)
			}
			if result == nil {
				t.Fatal("EvaluateDecisions() result is nil")
			}
			if result.Decision.Name != tt.wantDecision {
				t.Fatalf("decision = %q, want %q", result.Decision.Name, tt.wantDecision)
			}
			if len(result.Decision.ModelRefs) != 1 {
				t.Fatalf("modelRefs = %d, want 1", len(result.Decision.ModelRefs))
			}
			if result.Decision.ModelRefs[0].Model != tt.wantModel {
				t.Fatalf("model = %q, want %q", result.Decision.ModelRefs[0].Model, tt.wantModel)
			}
		})
	}
}

func TestMultiStepPipelineDecisionFragmentRejectsAmbiguousMarkers(t *testing.T) {
	engine := NewDecisionEngine(nil, nil, nil, loadMultiStepPipelineDecisions(t), "priority")

	result, err := engine.EvaluateDecisions(
		[]string{"pipeline_step_summarize", "pipeline_step_extract"},
		nil,
		nil,
	)
	if err != nil {
		t.Fatalf("EvaluateDecisions() error = %v", err)
	}
	if result != nil {
		t.Fatalf("EvaluateDecisions() decision = %q, want no match", result.Decision.Name)
	}
}

func loadMultiStepPipelineDecisions(t *testing.T) []config.Decision {
	t.Helper()
	path := filepath.Join(repoRootFromDecisionTest(t), "config", "decision", "composite", "multi-step-pipeline.yaml")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read multi-step pipeline decision fragment: %v", err)
	}

	var fragment struct {
		Routing struct {
			Decisions []config.Decision `yaml:"decisions"`
		} `yaml:"routing"`
	}
	if err := yaml.Unmarshal(data, &fragment); err != nil {
		t.Fatalf("parse multi-step pipeline decision fragment: %v", err)
	}
	if len(fragment.Routing.Decisions) != 2 {
		t.Fatalf("decision fragment decisions = %d, want 2", len(fragment.Routing.Decisions))
	}
	return fragment.Routing.Decisions
}

func repoRootFromDecisionTest(t *testing.T) string {
	t.Helper()
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(filename), "..", "..", "..", ".."))
}
