package config

import (
	"strings"
	"testing"
)

// A decision that names no model of its own falls back to the default model,
// so a configuration where that fallback is not a model cannot serve a
// matched request.
func TestUnservableDecisionsAreRefused(t *testing.T) {
	cases := map[string]string{
		"no model ref and no default": `version: v0.5
providers:
  models:
    - name: general
      backend_refs:
        - name: local
          provider: openai-compatible
          endpoint: 127.0.0.1:8000
recipes:
  - name: default
    decisions:
      - name: chat
`,
		// A decision whose turns resolve through the default model names no
		// model of its own either.
		"candidate iterations only": `version: v0.5
providers:
  models:
    - name: general
      backend_refs:
        - name: local
          provider: openai-compatible
          endpoint: 127.0.0.1:8000
recipes:
  - name: default
    decisions:
      - name: chat
        candidateIterations:
          - models:
              - model: general
`,
		// A decision that turns the inherited fast response off reaches model
		// selection after all.
		"disabled inherited fast response": `version: v0.5
providers:
  models:
    - name: general
      backend_refs:
        - name: local
          provider: openai-compatible
          endpoint: 127.0.0.1:8000
recipes:
  - name: default
    plugins:
      - type: fast_response
        configuration:
          message: Blocked.
    decisions:
      - name: chat
        plugins:
          - type: fast_response
            configuration:
              enabled: false
              message: Blocked.
`,
		"ref with no providers": `version: v0.5
recipes:
  - name: default
    decisions:
      - name: chat
        modelRefs:
          - model: typo-model
`,
	}
	for name, document := range cases {
		_, err := ParseYAMLBytes([]byte(document))
		if name == "ref with no providers" {
			// A routing-only fragment names models the document it lands in
			// declares, so the reference alone is not evidence of a mistake.
			if err != nil {
				t.Errorf("%s: %v", name, err)
			}
			continue
		}
		if err == nil || !strings.Contains(err.Error(), "model") {
			t.Errorf("%s: err = %v, want a decision naming no declared model", name, err)
		}
	}
}

// Decisions that answer without a model of their own need no fallback, and a
// candidate iteration's models are not the decision's own.
func TestDecisionsThatAnswerWithoutAModelNeedNoFallback(t *testing.T) {
	answers := map[string]Decision{
		"fast response": {Plugins: []DecisionPlugin{{
			Type:          DecisionPluginFastResponse,
			Configuration: MustStructuredPayload(FastResponsePluginConfig{Message: "Blocked."}),
		}}},
		"route": {Action: &DecisionAction{Type: DecisionActionRoute, Destination: "general"}},
	}
	for name, decision := range answers {
		if !canonicalDecisionNamesAModel(decision, decision.Plugins) {
			t.Errorf("%s: names no model, want one that needs none", name)
		}
	}
	iterations := Decision{CandidateIterations: []CandidateIterationConfig{{Models: []ModelRef{{Model: "general"}}}}}
	if canonicalDecisionNamesAModel(iterations, iterations.Plugins) {
		t.Error("candidate iterations alone name no model the decision answers with")
	}
	inherited := []DecisionPlugin{{
		Type:          DecisionPluginFastResponse,
		Configuration: MustStructuredPayload(FastResponsePluginConfig{Message: "Blocked."}),
	}}
	if !canonicalDecisionNamesAModel(Decision{}, inherited) {
		t.Error("a fast response its recipe layers in answers without a model")
	}
}

func TestDecisionsThatNameADeclaredModelLoad(t *testing.T) {
	cases := map[string]string{
		"model ref": `version: v0.5
providers:
  models:
    - name: general
      backend_refs:
        - name: local
          provider: openai-compatible
          endpoint: 127.0.0.1:8000
recipes:
  - name: default
    decisions:
      - name: chat
        modelRefs:
          - model: general
`,
		// Decisions that answer without a model of their own need no fallback.
		"fast response refusal": `version: v0.5
providers:
  models:
    - name: general
      backend_refs:
        - name: local
          provider: openai-compatible
          endpoint: 127.0.0.1:8000
recipes:
  - name: default
    decisions:
      - name: chat
        plugins:
          - type: fast_response
            configuration:
              message: Blocked.
`,
		"fusion analysis models": `version: v0.5
providers:
  models:
    - name: general
      backend_refs:
        - name: local
          provider: openai-compatible
          endpoint: 127.0.0.1:8000
recipes:
  - name: default
    decisions:
      - name: chat
        algorithm:
          type: fusion
          fusion:
            model: general
            analysis_models: [general]
`,
		// A recipe's default plugins reach its decisions, and an enabled
		// fast_response among them answers without a model.
		"inherited fast response": `version: v0.5
providers:
  models:
    - name: general
      backend_refs:
        - name: local
          provider: openai-compatible
          endpoint: 127.0.0.1:8000
recipes:
  - name: default
    plugins:
      - type: fast_response
        configuration:
          message: Blocked.
    decisions:
      - name: chat
`,
		"declared default": `version: v0.5
providers:
  defaults:
    model: general
  models:
    - name: general
      backend_refs:
        - name: local
          provider: openai-compatible
          endpoint: 127.0.0.1:8000
recipes:
  - name: default
    decisions:
      - name: chat
`,
	}
	for name, document := range cases {
		if _, err := ParseYAMLBytes([]byte(document)); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
}
