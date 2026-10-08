package extproc

import (
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
)

func TestEntrypointFallbackUsesItsRecipeAndGlobalDefaults(t *testing.T) {
	for _, globalEnabled := range []bool{false, true} {
		name := "without_global_fallback"
		global := ""
		if globalEnabled {
			name = "with_global_fallback"
			global = `global:
  router:
    fallback:
      enabled: true
      max_attempts: 5
      total_timeout: 50s
      per_attempt_timeout: 8s
`
		}
		t.Run(name, func(t *testing.T) {
			cfg, err := config.ParseYAMLBytes([]byte(`version: v0.3
` + global + `routing:
  fallback:
    enabled: true
    max_attempts: 2
    total_timeout: 12s
    per_attempt_timeout: 2s
recipes:
  - name: independent
    routing: {}
entrypoints:
  - model_names: [public/independent]
    recipe: independent
`))
			if err != nil {
				t.Fatal(err)
			}
			components := &routerComponents{cfg: cfg}
			components.buildFallbackRuntime()
			router := &OpenAIRouter{
				Config:                      cfg,
				FallbackOrchestrator:        components.fallbackOrchestrator,
				RecipeFallbackOrchestrators: components.recipeFallbackOrchestrators,
			}
			defaultCtx := &RequestContext{}
			router.resolveEntrypointForRequest(config.DefaultEntrypointModel, defaultCtx)
			if policy := router.fallbackOrchestratorForContext(defaultCtx).Policy(); !policy.Enabled || policy.MaxAttempts != 2 {
				t.Fatalf("default recipe lost its own fallback: %+v", policy)
			}
			for _, model := range []string{"public/independent", "concrete-backend"} {
				ctx := &RequestContext{}
				router.resolveEntrypointForRequest(model, ctx)
				orchestrator := router.fallbackOrchestratorForContext(ctx)
				policy := orchestrator.Policy()
				if policy.Enabled != globalEnabled {
					t.Fatalf("model %q inherited default recipe enablement: %+v", model, policy)
				}
				if globalEnabled && (policy.MaxAttempts != 5 || policy.TotalTimeout != 50*time.Second) {
					t.Fatalf("model %q inherited default recipe limits: %+v", model, policy)
				}
				record := orchestrator.NewExecutionRecord("request", "", "primary")
				result := orchestrator.EvaluateAttempt(context.Background(), record, fallback.AttemptOutcome{
					Model: "primary", Backend: "primary", StatusCode: 503,
				}, fallback.CommitState{BodyReplayable: true})
				if result.CanFallback != globalEnabled {
					t.Fatalf("model %q: fallback on 503=%t, want %t (%+v)", model, result.CanFallback, globalEnabled, result)
				}
			}
		})
	}
}
