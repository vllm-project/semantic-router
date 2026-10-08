package gatewayparity

import (
	"net/http"
	"strings"
	"testing"
)

// decisionFallbackConfig is fallbackConfig with RECIPE_FALLBACK as the
// recipe's policy and DECISION_FALLBACK as the decision's override.
const decisionFallbackConfig = `
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    timeout: 30s
providers:
  defaults:
    model: primary-model
  models:
    - name: primary-model
      provider_model_id: primary-model
      api_format: openai
      backend_refs:
        - name: primary
          endpoint: PRIMARY
          protocol: http
          provider: vllm
    - name: fallback-model
      provider_model_id: fallback-model
      api_format: openai
      backend_refs:
        - name: secondary
          endpoint: SECONDARY
          protocol: http
          provider: vllm
routing:
  modelCards:
    - name: primary-model
    - name: fallback-model
  decisions:
    - name: default_route
      priority: 10
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: primary-model
          use_reasoning: false
        - model: fallback-model
          use_reasoning: false
      algorithm:
        type: static
      fallback: DECISION_FALLBACK
  fallback: RECIPE_FALLBACK
`

// A decision's fallback block overrides its recipe's policy field by field,
// and both gateway modes run the policy it resolves to.
func TestDecisionFallbackOverridesTheRecipeInBothModes(t *testing.T) {
	for name, tc := range map[string]struct {
		recipe, decision string
		primaryStatus    int
		wantCandidate    bool
	}{
		"a decision turns fallback on": {
			recipe: "{enabled: false}", decision: "{enabled: true, max_attempts: 2}",
			primaryStatus: http.StatusServiceUnavailable, wantCandidate: true,
		},
		"a decision turns fallback off": {
			recipe: "{enabled: true, max_attempts: 3}", decision: "{enabled: false}",
			primaryStatus: http.StatusServiceUnavailable, wantCandidate: false,
		},
		"a decision's statuses replace the recipe's": {
			recipe: "{enabled: true, max_attempts: 3, retryable_status_codes: [502, 503, 504]}", decision: "{retryable_status_codes: [500]}",
			primaryStatus: http.StatusInternalServerError, wantCandidate: true,
		},
		"a status outside the decision's list does not fall back": {
			recipe: "{enabled: true, max_attempts: 3}", decision: "{retryable_status_codes: [500]}",
			primaryStatus: http.StatusServiceUnavailable, wantCandidate: false,
		},
	} {
		t.Run(name, func(t *testing.T) {
			configYAML := strings.NewReplacer("RECIPE_FALLBACK", tc.recipe, "DECISION_FALLBACK", tc.decision).
				Replace(decisionFallbackConfig)
			for _, native := range []bool{true, false} {
				primary := &scriptedBackend{status: tc.primaryStatus, body: openAIError("primary failed")}
				secondary := &scriptedBackend{status: http.StatusOK, body: chatCompletion("from the fallback model")}
				backends := map[string]http.Handler{"PRIMARY": primary, "SECONDARY": secondary}
				got := postChat(t, gatewayOver(t, configYAML, backends, native))

				if tc.wantCandidate {
					if got.status != http.StatusOK || secondary.hits.Load() != 1 || got.header["x-vsr-selected-model"] != "fallback-model" {
						t.Fatalf("native=%t: %s; candidate hits %d, want the candidate's answer", native, got, secondary.hits.Load())
					}
					continue
				}
				if got.status != tc.primaryStatus || secondary.hits.Load() != 0 {
					t.Fatalf("native=%t: %s; candidate hits %d, want the primary's failure", native, got, secondary.hits.Load())
				}
			}
		})
	}
}
