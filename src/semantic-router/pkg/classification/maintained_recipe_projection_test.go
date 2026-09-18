package classification

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

func TestBalancedObjectiveDistinguishesKnowledgeFromEffort(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "..", "..", "config", "recipes", "multi-objective", "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	recipe, ok := cfg.RecipeByName("balanced")
	if !ok {
		t.Fatal("balanced recipe missing")
	}
	cfg = cfg.ConfigForRecipe(recipe)
	classifier := &Classifier{Config: cfg}
	engine := decision.NewDecisionEngine(nil, nil, nil, cfg.Decisions, config.RoutingStrategyPriority)
	tests := []struct {
		name                                  string
		facts, keywords, difficulty, language []string
		want                                  string
	}{
		{
			name:  "external knowledge alone is not high effort",
			facts: []string{"needs_fact_check"},
			want:  "unified_balance_route",
		},
		{
			name:     "language does not turn knowledge into high effort",
			facts:    []string{"needs_fact_check"},
			language: []string{"ja"},
			want:     "unified_balance_route",
		},
		{
			name:     "explicit verification is deliberate",
			keywords: []string{"unified_balance_verification_markers"},
			want:     "unified_balance_deliberate_route",
		},
		{
			name:       "hard task is deliberate",
			difficulty: []string{"unified_balance_difficulty:hard"},
			want:       "unified_balance_deliberate_route",
		},
		{
			name:     "negated reasoning retains standard path",
			facts:    []string{"needs_fact_check"},
			keywords: []string{"unified_balance_negated_reasoning"},
			want:     "unified_balance_route",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := classifier.applyProjections(&SignalResults{MatchedFactCheckRules: tt.facts, MatchedKeywordRules: tt.keywords, MatchedComplexityRules: tt.difficulty, MatchedLanguageRules: tt.language})
			got, err := engine.EvaluateDecisionsWithSignals(&decision.SignalMatches{FactCheckRules: tt.facts, KeywordRules: tt.keywords, ComplexityRules: tt.difficulty, LanguageRules: tt.language, ProjectionRules: out.MatchedProjectionRules})
			if err != nil {
				t.Fatal(err)
			}
			if got == nil || got.Decision == nil || got.Decision.Name != tt.want {
				t.Fatalf("got %+v, want %s", got, tt.want)
			}
		})
	}
}
