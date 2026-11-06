package config

import (
	"strings"
	"testing"
)

const escalateTestYAML = `
version: v0.3
providers:
  defaults: {model: small}
  models:
    - name: small
      backend_refs: [{name: small, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
    - name: large
      backend_refs: [{name: large, endpoint: "127.0.0.1:8001", protocol: http, weight: 1}]
routing:
  modelCards:
    - {name: small, param_size: 8b}
    - {name: large, param_size: 70b}
  decisions:
    - name: escalate
      priority: 100
      rules: {operator: AND, conditions: []}
      modelRefs: [{model: large}, {model: small}]
      algorithm:
`

func parseEscalateAlgorithm(algorithm string) (*RouterConfig, error) {
	return ParseYAMLBytes([]byte(escalateTestYAML + algorithm))
}

// Each Chat accept check loads; the decision model's answer checks start the
// deployment they ask, and a confidence check needs none.
func TestChatEscalateAcceptChecksLoad(t *testing.T) {
	for _, tc := range []struct {
		name, algorithm string
		asksDecision    bool
	}{
		{"confidence", "        type: escalate\n        escalate:\n          order: size\n          accept: {confidence: {method: margin, gte: 0.6}}\n", false},
		{"answer check", "        type: escalate\n        on_error: fail\n        escalate:\n          accept: {systemone: {gte: 0.7, timeout_ms: 2000}}\n", true},
		{"refusal", "        type: escalate\n        budget: {deadline: 20s, max_calls: 2}\n        escalate:\n          order: cost\n          accept: {refusal: false}\n", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := parseEscalateAlgorithm(tc.algorithm)
			if err != nil {
				t.Fatal(err)
			}
			algorithm := cfg.Decisions[0].Algorithm
			if algorithm.EscalateChecksAnswers() != tc.asksDecision {
				t.Fatalf("EscalateChecksAnswers() = %v, want %v", algorithm.EscalateChecksAnswers(), tc.asksDecision)
			}
			deployment := cfg.EscalateCheckDeployment(algorithm.Escalate.Accept)
			if _, started := ModelRuntimeDeploymentsInUse(cfg)[deployment]; tc.asksDecision && !started {
				t.Fatalf("the answer check's deployment %q is not started", deployment)
			}
		})
	}
}

func TestChatEscalateRejectsChecksItCannotRun(t *testing.T) {
	for _, tc := range []struct{ name, algorithm, want string }{
		{"refusal true", "        type: escalate\n        escalate:\n          accept: {refusal: true}\n", "refusal"},
		{"two checks", "        type: escalate\n        escalate:\n          accept: {confidence: {gte: 0.6}, refusal: false}\n", "one of confidence, systemone or refusal"},
		{"no threshold", "        type: escalate\n        escalate:\n          accept: {systemone: {timeout_ms: 2000}}\n", "gte"},
		{"skip on_error", "        type: escalate\n        on_error: skip\n        escalate:\n          accept: {confidence: {gte: 0.6}}\n", "on_error of escalate"},
		{"unknown deployment", "        type: escalate\n        escalate:\n          accept: {systemone: {gte: 0.7, deployment: missing}}\n", "missing"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := parseEscalateAlgorithm(tc.algorithm); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want one naming %q", err, tc.want)
			}
		})
	}
}

// Escalate on Chat serves the last judged answer unless on_error says fail;
// System One fails, since no stage may serve an answer its rules rejected.
func TestEscalateOnErrorDefaultsByPath(t *testing.T) {
	chat := &AlgorithmConfig{Type: DecisionAlgorithmEscalate, Escalate: &EscalateAlgorithmConfig{}}
	native := &AlgorithmConfig{Type: DecisionAlgorithmEscalate, Escalate: &EscalateAlgorithmConfig{Stages: []EscalateStage{{Model: "kai"}}}}
	if chat.EscalateOnError() != EscalateOnErrorLastAttempt || native.EscalateOnError() != EscalateOnErrorFail {
		t.Fatalf("defaults = %q, %q", chat.EscalateOnError(), native.EscalateOnError())
	}
	chat.OnError = EscalateOnErrorFail
	if chat.EscalateOnError() != EscalateOnErrorFail {
		t.Fatalf("explicit on_error = %q", chat.EscalateOnError())
	}
}
