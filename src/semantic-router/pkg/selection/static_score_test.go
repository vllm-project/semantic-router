package selection

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestStaticSelectorConfiguredScoreBoundary(t *testing.T) {
	tests := []struct {
		name   string
		scores []config.ModelScore
		want   string
		score  float64
	}{
		{"configured one", []config.ModelScore{{Model: "small", Score: 0.2}, {Model: "large", Score: 1}}, "large", 1},
		{"configured below one", []config.ModelScore{{Model: "small", Score: 0.2}, {Model: "large", Score: 0.9}}, "large", 0.9},
		{"configured zero", []config.ModelScore{{Model: "small", Score: -0.2}, {Model: "large", Score: 0}}, "large", 0},
		{"configured tie", []config.ModelScore{{Model: "small", Score: 1}, {Model: "large", Score: 1}}, "small", 1},
		{"unconfigured fallback", nil, "small", 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			selector := NewStaticSelector(nil)
			selector.InitializeFromConfig([]config.Category{{
				CategoryMetadata: config.CategoryMetadata{Name: "business"},
				ModelScores:      tt.scores,
			}})
			result, err := selector.Select(context.Background(), &SelectionContext{
				CategoryName:    "business",
				DecisionName:    "business_route",
				CandidateModels: []config.ModelRef{{Model: "small"}, {Model: "large"}},
			})
			if err != nil {
				t.Fatal(err)
			}
			if result.SelectedModel != tt.want || result.Score != tt.score {
				t.Fatalf("selected %s (%v), want %s (%v)", result.SelectedModel, result.Score, tt.want, tt.score)
			}
		})
	}
}

func TestStaticSelectorPartialScoresKeepOmittedCandidateDefault(t *testing.T) {
	selector := NewStaticSelector(nil)
	selector.InitializeFromConfig([]config.Category{{
		CategoryMetadata: config.CategoryMetadata{Name: "business"},
		ModelScores:      []config.ModelScore{{Model: "small", Score: 0.9}},
	}})
	result, err := selector.Select(context.Background(), &SelectionContext{
		CategoryName:    "business",
		DecisionName:    "business_route",
		CandidateModels: []config.ModelRef{{Model: "small"}, {Model: "large"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.AllScores["small"] != 0.9 || result.AllScores["large"] != 1 {
		t.Fatalf("partial scores = %v, want small=0.9 and omitted large=1", result.AllScores)
	}
	if result.SelectedModel != "large" || result.Score != 1 {
		t.Fatalf("selected %s (%v), want omitted large (1)", result.SelectedModel, result.Score)
	}
}

func TestStaticSelectorConfiguredOneFromCanonicalConfig(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.3
providers:
  defaults:
    model: small
routing:
  modelCards:
    - name: small
    - name: large
  signals:
    domains:
      - name: business
        model_scores:
          - model: small
            score: 0.2
          - model: large
            score: 1.0
  decisions:
    - name: business_route
      rules:
        operator: OR
        conditions:
          - type: domain
            name: business
      algorithm:
        type: static
      modelRefs:
        - model: small
        - model: large
`))
	if err != nil {
		t.Fatal(err)
	}
	selector := NewFactory(nil).WithCategories(cfg.Categories).Create()
	decision := cfg.Decisions[0]
	result, err := selector.Select(context.Background(), &SelectionContext{
		CategoryName:    "business",
		DecisionName:    decision.Name,
		CandidateModels: decision.ModelRefs,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.SelectedModel != "large" || result.Score != 1 {
		t.Fatalf("canonical domain scores selected %s (%v), want large (1)", result.SelectedModel, result.Score)
	}
}
