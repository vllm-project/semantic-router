package classification

import (
	"context"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// Keep the real HTTP/decode boundary in the regression instead of injecting
// an already decoded answer into the classifier.
type setProbabilityRegressionDecider struct {
	client *modelservice.Client
}

func (d setProbabilityRegressionDecider) Decide(ctx context.Context, _ string, request modelservice.Request) (modelservice.Response, error) {
	return d.client.Decide(ctx, request)
}

func TestExplicitSetProbabilityCoverageRegression(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		invalid    bool
		shipping   float64
	}{
		{"incomplete", `{"sets":{"topics":{"selected":[],"probabilities":{"billing":0.9}}}}`, true, 0},
		{"complete_independent_control", `{"sets":{"topics":{"selected":[],"probabilities":{"billing":0.9,"shipping":0.8}}}}`, false, 0.8},
		{"explicit_zero_is_evidence", `{"sets":{"topics":{"selected":[],"probabilities":{"billing":0.9,"shipping":0}}}}`, false, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost || r.URL.Path != "/v1/decisions" {
					t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
					w.WriteHeader(http.StatusNotFound)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(tc.body))
			}))
			defer server.Close()
			client, err := modelservice.NewClient(server.URL)
			if err != nil {
				t.Fatal(err)
			}
			low := 0.2
			rule := config.DecisionSignalRule{
				Name: "topics", Deployment: "mock", Predicate: &config.NumericPredicate{LT: &low},
				Question: config.DecisionQuestion{Type: config.DecisionQuestionSet, Instructions: "Which topics?", Labels: []config.DecisionChoice{{Key: "billing"}, {Key: "shipping"}}},
			}
			classifier := newDecisionTestClassifier(setProbabilityRegressionDecider{client})
			results := newSignalResults()
			classifier.evaluateDecisionDeployment(context.Background(), results, &sync.Mutex{}, "test", "mock", []config.DecisionSignalRule{rule})
			t.Logf("values=%v confidences=%v errors=%v matches=%v", results.SignalValues, results.SignalConfidences, results.SignalErrors, results.MatchedDecisionRules)
			if tc.invalid {
				if results.SignalErrors["decision:topics"] != "decision_invalid_model_output" {
					t.Errorf("expected decision_invalid_model_output, got %q", results.SignalErrors["decision:topics"])
				}
				if len(results.SignalValues) != 0 || len(results.SignalConfidences) != 0 || len(results.MatchedDecisionRules) != 0 {
					t.Errorf("incomplete answer published routing evidence: values=%v confidences=%v matches=%v", results.SignalValues, results.SignalConfidences, results.MatchedDecisionRules)
				}
				if value, exists := results.SignalValues["decision:topics:shipping"]; exists {
					t.Errorf("missing shipping became evidence %g; shipping < 0.2 = %v", value, value < low)
				}
				return
			}
			if len(results.SignalErrors) != 0 {
				t.Errorf("complete probabilities rejected: %v", results.SignalErrors)
			}
			for key, want := range map[string]float64{"billing": 0.9, "shipping": tc.shipping} {
				got, exists := results.SignalValues["decision:topics:"+key]
				if !exists || got != want || results.SignalConfidences["decision:topics:"+key] != want {
					t.Errorf("%s evidence = %g (present=%v), want %g", key, got, exists, want)
				}
			}
			if matched := slices.Contains(results.MatchedDecisionRules, "topics:shipping"); matched != (tc.shipping < low) {
				t.Errorf("shipping < 0.2 matched=%v for explicit probability %g", matched, tc.shipping)
			}
		})
	}
}
