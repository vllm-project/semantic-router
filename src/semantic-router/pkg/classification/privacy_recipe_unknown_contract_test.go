package classification

import (
	"context"
	"maps"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// The maintained Privacy recipe sends a request to the cloud only when the
// decision model answers that it needs deep reasoning and holds nothing
// private. A rule the model does not answer is unknown, and
// local_privacy_policy matches on an unknown, so the request stays on the
// local model whatever the other rules answered. A blank message is a
// different case: no rule is asked about it, so none is unknown, and it takes
// the default local decision.
func TestPrivacyRecipeKeepsARequestLocalWhenTheDecisionModelCannotAnswer(t *testing.T) {
	const (
		deepPublic = "Compare two public database designs from first principles and recommend one."
		local      = "local/private-qwen"
	)
	cfg := maintainedRecipeConfig(t, "privacy")
	deployment := cfg.SystemOneRuleDeployment(cfg.SystemOneRules[0])
	// The answers that send deepPublic to the cloud.
	answers := func() map[string]modelservice.Answer {
		no, yes := modelservice.Answer{Type: "noul", Noul: 0.01}, modelservice.Answer{Type: "noul", Noul: 0.95}
		return map[string]modelservice.Answer{
			SystemOneQuestionID("attack"): no, SystemOneQuestionID("confidential"): no,
			SystemOneQuestionID("personal_data"): no, SystemOneQuestionID("depth"): yes,
		}
	}
	everyRule := func(reason string) map[string]string {
		errors := map[string]string{}
		for _, rule := range cfg.SystemOneRules {
			errors[config.SignalTypeSystemOne+":"+rule.Name] = systemOneSignalErrorPrefix + reason
		}
		return errors
	}
	unanswered := answers()
	unanswered[SystemOneQuestionID("confidential")] = modelservice.Answer{Type: "noul", Error: "max_length_exceeded"}

	for _, test := range []struct {
		name         string
		text         string
		answers      map[string]modelservice.Answer
		failure      error
		wantDecision string
		wantModel    string
		wantErrors   map[string]string
		wantAsked    bool
	}{
		{"the model answers", deepPublic, answers(), nil, "cloud_frontier_reasoning", "cloud/frontier-reasoning", map[string]string{}, true},
		{"the runtime rejects the request", deepPublic, answers(), modelservice.ErrRejected, "local_privacy_policy", local, everyRule("rejected"), true},
		{"the runtime is unavailable", deepPublic, answers(), modelservice.ErrUnavailable, "local_privacy_policy", local, everyRule("unavailable"), true},
		{"the call times out", deepPublic, answers(), context.DeadlineExceeded, "local_privacy_policy", local, everyRule("timeout"), true},
		{
			"one rule is unanswered", deepPublic, unanswered, nil, "local_privacy_policy", local,
			map[string]string{"systemone:confidential": "systemone_max_length_exceeded"},
			true,
		},
		{"a blank message", " ", answers(), nil, "local_standard", local, map[string]string{}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			decider := newFakeDecider(test.answers)
			if test.failure != nil {
				decider.failures[deployment] = test.failure
			}
			classifier := newApplicationTestClassifier(t, cfg, decider)
			results := newSignalResults()
			classifier.evaluateSystemOneSignals(context.Background(), results, &sync.Mutex{},
				systemOneReading{request: test.text}, usedRules(cfg.SystemOneRules))
			result, err := classifier.EvaluateDecisionWithEngine(results)
			if err != nil || result == nil || result.Decision == nil {
				t.Fatalf("no decision: %+v, %v", result, err)
			}
			if result.Decision.Name != test.wantDecision || result.Decision.ModelRefs[0].Model != test.wantModel {
				t.Errorf("decision %s on %s, want %s on %s",
					result.Decision.Name, result.Decision.ModelRefs[0].Model, test.wantDecision, test.wantModel)
			}
			if !maps.Equal(results.SignalErrors, test.wantErrors) {
				t.Errorf("signal errors %v, want %v", results.SignalErrors, test.wantErrors)
			}
			if asked := decider.calls[deployment] > 0; asked != test.wantAsked {
				t.Errorf("decision model asked = %t, want %t", asked, test.wantAsked)
			}
		})
	}
}

func maintainedRecipeConfig(t *testing.T, recipe string) *config.RouterConfig {
	t.Helper()
	path := filepath.Join("..", "..", "..", "..", "config", "recipes", recipe, "config.yaml")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	return cfg
}
