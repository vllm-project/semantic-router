package classification

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// labelEndpoint stands in for a deployed three-class difficulty model,
// answering in the HuggingFace text-classification shape the http_classify
// protocol defines.
func labelEndpoint(t *testing.T, body string) SequenceClassifierBackend {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)

	// The real transport and the real declared mapping, so this exercises the
	// wiring the option builder produces rather than a stand-in.
	backend, err := newHTTPClassifierInference(&config.ExternalModelConfig{
		ModelEndpoint: endpointForTestServer(t, server),
		ModelName:     "difficulty-classifier-svc",
	}, newDeclaredLabelMapping(ComplexityVerdictLabels), time.Second)
	if err != nil {
		t.Fatalf("newHTTPClassifierInference: %v", err)
	}
	t.Cleanup(func() { _ = backend.Close() })
	return backend
}

// The deployment contract for label_distribution.v1, end to end through the
// real transport: the endpoint names its three classes hard, easy and medium,
// and the winning one becomes the verdict a decision matches on.
//
// The unit tests around evaluateComplexityLabels use a fake backend, so none
// of them covers the seam that actually breaks in a deployment - whether the
// names the endpoint reports line up with the verdict vocabulary.
func TestComplexityLabelContract_VerdictFollowsTheWinningLabel(t *testing.T) {
	rules := []config.ComplexityRule{{Name: "needs_reasoning"}}

	cases := map[string]struct {
		body string
		want string
	}{
		"hard wins": {
			`[{"label":"hard","score":0.71},{"label":"easy","score":0.09},{"label":"medium","score":0.20}]`,
			config.ComplexityDifficultyHard,
		},
		"easy wins": {
			`[{"label":"hard","score":0.05},{"label":"easy","score":0.80},{"label":"medium","score":0.15}]`,
			config.ComplexityDifficultyEasy,
		},
		// Reordered on the wire. The transport aligns by name before the
		// argmax is taken positionally, so the endpoint is free to answer in
		// any order - which is what makes the positional read safe.
		"medium wins, reported out of order": {
			`[{"label":"medium","score":0.66},{"label":"hard","score":0.24},{"label":"easy","score":0.10}]`,
			config.ComplexityDifficultyMedium,
		},
	}

	for name, tc := range cases {
		results, err := evaluateComplexityLabels(context.Background(), labelEndpoint(t, tc.body), "text", rules)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		if len(results) != 1 {
			t.Errorf("%s: got %d results for one rule", name, len(results))
			continue
		}
		if results[0].Difficulty != tc.want {
			t.Errorf("%s: verdict = %q, want %q", name, results[0].Difficulty, tc.want)
		}
		// The decision engine matches "<rule>:<verdict>", so this is the
		// string the whole path exists to produce.
		if match := results[0].RuleName + ":" + results[0].Difficulty; match != "needs_reasoning:"+tc.want {
			t.Errorf("%s: match name = %q, want %q", name, match, "needs_reasoning:"+tc.want)
		}
		if !results[0].ConfidenceReported || results[0].Confidence <= 0 {
			t.Errorf("%s: a label distribution reports a real confidence, got %v (reported=%t)",
				name, results[0].Confidence, results[0].ConfidenceReported)
		}
	}
}

// A model whose classes are not named for the verdicts cannot be read, and it
// must say so rather than silently pick a position. This is the failure a
// deployment hits when it points the contract at an off-the-shelf classifier
// that reports LABEL_0/1/2.
func TestComplexityLabelContract_RejectsLabelsThatAreNotVerdicts(t *testing.T) {
	rules := []config.ComplexityRule{{Name: "needs_reasoning"}}

	cases := map[string]string{
		"generic label names": `[{"label":"LABEL_0","score":0.7},{"label":"LABEL_1","score":0.2},{"label":"LABEL_2","score":0.1}]`,
		"one verdict missing": `[{"label":"hard","score":0.7},{"label":"easy","score":0.3}]`,
		"an extra class":      `[{"label":"hard","score":0.5},{"label":"easy","score":0.2},{"label":"medium","score":0.2},{"label":"trivial","score":0.1}]`,
	}

	for name, body := range cases {
		if _, err := evaluateComplexityLabels(context.Background(), labelEndpoint(t, body), "text", rules); err == nil {
			t.Errorf("%s: expected the label contract to be enforced, got no error", name)
		}
	}
}

// One call serves every rule here too - the distribution is a property of the
// request, not of a rule - and each rule keeps its own name so composers can
// gate the same verdict differently.
func TestComplexityLabelContract_OneCallServesEveryRule(t *testing.T) {
	rules := []config.ComplexityRule{{Name: "needs_reasoning"}, {Name: "extreme"}}
	body := `[{"label":"hard","score":0.71},{"label":"easy","score":0.09},{"label":"medium","score":0.20}]`

	results, err := evaluateComplexityLabels(context.Background(), labelEndpoint(t, body), "text", rules)
	if err != nil {
		t.Fatalf("evaluateComplexityLabels: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("got %d results, want one per rule", len(results))
	}
	for i, want := range []string{"needs_reasoning", "extreme"} {
		if results[i].RuleName != want {
			t.Errorf("result %d: rule = %q, want %q", i, results[i].RuleName, want)
		}
		if results[i].Difficulty != config.ComplexityDifficultyHard {
			t.Errorf("result %d: verdict = %q, want hard", i, results[i].Difficulty)
		}
	}
}
