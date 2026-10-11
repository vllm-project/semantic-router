package config

import (
	"strings"
	"testing"
	"time"

	"gopkg.in/yaml.v3"
)

// decisionFallbackDoc has a default decision and a recipe decision, each
// with a fallback block (DEFAULT_FALLBACK and RECIPE_FALLBACK).
const decisionFallbackDoc = `
version: v0.3
providers:
  models:
    - name: model-a
      provider_model_id: model-a
      api_format: openai
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
    - name: model-b
      provider_model_id: model-b
      api_format: openai
      backend_refs:
        - endpoint: 127.0.0.1:8001
          provider: vllm
entrypoints:
  - model_names: [support-auto]
    recipe: support
routing:
  modelCards:
    - name: model-a
    - name: model-b
  decisions:
    - name: answer
      priority: 10
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: model-a
        - model: model-b
      fallback: DEFAULT_FALLBACK
recipes:
  - name: support
    routing:
      fallback:
        total_timeout: 20s
      decisions:
        - name: escalate
          priority: 10
          rules:
            operator: AND
            conditions: []
          modelRefs:
            - model: model-a
            - model: model-b
          fallback: RECIPE_FALLBACK
`

func decisionFallbackConfig(defaultFallback, recipeFallback string) string {
	return strings.NewReplacer("DEFAULT_FALLBACK", defaultFallback, "RECIPE_FALLBACK", recipeFallback).
		Replace(decisionFallbackDoc)
}

func TestDecisionFallbackRoundTripsCanonicalConfig(t *testing.T) {
	doc := decisionFallbackConfig(
		"{enabled: true, max_attempts: 2, total_timeout: 15s, per_attempt_timeout: 5s, retryable_status_codes: [429, 503]}",
		"{enabled: false}",
	)
	parsed, err := ParseYAMLBytes([]byte(doc))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	got := parsed.Decisions[0].Fallback
	if got == nil || got.Enabled == nil || !*got.Enabled || got.MaxAttempts != 2 || got.TotalTimeout != 15*time.Second ||
		got.PerAttemptTimeout != 5*time.Second || len(got.RetryableStatusCodes) != 2 {
		t.Fatalf("default decision fallback = %+v", got)
	}
	recipe, ok := parsed.RecipeByName("support")
	if !ok {
		t.Fatal("recipe support is missing")
	}
	if escalate := recipe.Profile.Decisions[0].Fallback; escalate == nil || escalate.Enabled == nil || *escalate.Enabled {
		t.Fatalf("recipe decision fallback = %+v", escalate)
	}
	exported, err := yaml.Marshal(CanonicalConfigFromRouterConfig(parsed))
	if err != nil {
		t.Fatal(err)
	}
	reparsed, err := ParseYAMLBytes(exported)
	if err != nil {
		t.Fatalf("re-parse the exported config: %v\n%s", err, exported)
	}
	if again := reparsed.Decisions[0].Fallback; again == nil || again.TotalTimeout != 15*time.Second || again.MaxAttempts != 2 {
		t.Fatalf("exported decision fallback = %+v", again)
	}
}

func TestDecisionFallbackFailsFast(t *testing.T) {
	for name, tc := range map[string]struct {
		defaultFallback, recipeFallback, want string
	}{
		"the circuit breaker is per backend": {
			"{circuit_breaker: {consecutive_failures: 3}}", "{}", "circuit_breaker",
		},
		"negative attempts": {"{max_attempts: -1}", "{}", "routing.decisions[answer].fallback: fallback max_attempts cannot be negative"},
		"a bad status":      {"{retryable_status_codes: [700]}", "{}", "invalid retryable status code 700"},
		"a bad duration":    {"{total_timeout: soon}", "{}", "total_timeout"},
		"a per-attempt timeout beyond the recipe's total": {
			"{}", "{per_attempt_timeout: 25s}",
			"recipes[support].routing.decisions[escalate].fallback: fallback per_attempt_timeout (25s) cannot exceed total_timeout (20s)",
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := ParseYAMLBytes([]byte(decisionFallbackConfig(tc.defaultFallback, tc.recipeFallback)))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err %v, want %q", err, tc.want)
			}
		})
	}
}
