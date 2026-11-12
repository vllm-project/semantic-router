package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

func TestEntrypointRankingDoesNotInheritDefaultRecipeStrategy(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`version: v0.5
global:
  router:
    strategy: priority
recipes:
  - name: default
    strategy: confidence
    signals: &signals
      domains:
        - name: law
        - name: health
    decisions: &decisions
      - name: high-priority
        priority: 100
        rules: {type: domain, name: law}
        modelRefs: [{model: model-a}]
      - name: high-confidence
        priority: 10
        rules: {type: domain, name: health}
        modelRefs: [{model: model-a}]
  - name: independent
    signals: *signals
    decisions: *decisions
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
  - model_names: [public/independent]
    recipe: independent
`))
	if err != nil {
		t.Fatal(err)
	}
	router := &OpenAIRouter{Config: cfg}
	for _, example := range []struct{ model, winner string }{
		{config.DefaultEntrypointModel, "high-confidence"},
		{"public/independent", "high-priority"},
	} {
		ctx := &RequestContext{}
		router.resolveEntrypointForRequest(example.model, ctx)
		profile := cfg.ConfigForRecipe(ctx.Routing.SelectedRecipe())
		engine := decision.NewDecisionEngine(profile.Decisions, profile.Strategy)
		result, err := engine.EvaluateDecisionsWithSignals(&decision.SignalMatches{
			DomainRules:       []string{"law", "health"},
			SignalConfidences: map[string]float64{"domain:law": 0.60, "domain:health": 0.95},
		})
		if err != nil {
			t.Fatal(err)
		}
		if result == nil || result.Decision == nil || result.Decision.Name != example.winner {
			t.Fatalf("request model %q: winner=%+v, want %q", example.model, result, example.winner)
		}
	}
}
