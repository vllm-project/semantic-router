package classification

import (
	"sort"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Each path publishes only the numbers it actually produced. A remote score
// must not appear under ":margin" - it is in the model's own units, not a
// margin - and must not drag along zero-valued margin keys that would read
// as "no evidence either way".
func TestPublishComplexityValuesKeysBySource(t *testing.T) {
	cases := map[string]struct {
		result ComplexityRuleResult
		want   []string
	}{
		"remote score": {
			result: ComplexityRuleResult{RuleName: "r", SignalSource: complexitySignalSourceScore, FusedMargin: 7.3},
			want:   []string{"complexity:r:score"},
		},
		"remote labels": {
			result: ComplexityRuleResult{RuleName: "r", SignalSource: complexitySignalSourceLabels, Confidence: 0.9},
			want:   nil,
		},
		"local text": {
			result: ComplexityRuleResult{RuleName: "r", SignalSource: "text", FusedMargin: 0.2},
			want: []string{
				"complexity:r:image_easy_score", "complexity:r:image_hard_score", "complexity:r:image_margin",
				"complexity:r:margin",
				"complexity:r:text_easy_score", "complexity:r:text_hard_score", "complexity:r:text_margin",
			},
		},
	}

	for name, tc := range cases {
		values := map[string]float64{}
		publishComplexityValues(values, tc.result)
		var got []string
		for key := range values {
			got = append(got, key)
		}
		sort.Strings(got)
		if len(got) != len(tc.want) {
			t.Errorf("%s: keys = %v, want %v", name, got, tc.want)
			continue
		}
		for i := range got {
			if got[i] != tc.want[i] {
				t.Errorf("%s: keys = %v, want %v", name, got, tc.want)
				break
			}
		}
	}

	values := map[string]float64{}
	publishComplexityValues(values, ComplexityRuleResult{RuleName: "r", SignalSource: complexitySignalSourceScore, FusedMargin: 7.3})
	if values["complexity:r:score"] != 7.3 {
		t.Fatalf("score published as %v, want 7.3 in the model's own units", values["complexity:r:score"])
	}
}

// A failed evaluation must reach the decision engine as an error on every
// rule, not as a silently absent signal. The engine reads SignalErrors and
// honours a rule node's on_error, so this is what lets a decision choose to
// fail open or closed.
func TestRecordComplexityFailureMarksEveryRule(t *testing.T) {
	cfg := remoteComplexityConfig(config.RemoteClassifierContractScore)
	cfg.ComplexityRules = append(cfg.ComplexityRules, config.ComplexityRule{
		Name:      "extreme",
		HardAbove: float64Ptr(0.95),
		EasyBelow: float64Ptr(0.10),
	})
	classifier := &Classifier{Config: cfg, complexityScoreBackend: &closableScorer{}}
	results := &SignalResults{}
	var mu sync.Mutex

	classifier.recordComplexityFailure(results, &mu)

	for _, rule := range []string{"needs_reasoning", "extreme"} {
		code, ok := results.SignalErrors["complexity:"+rule]
		if !ok {
			t.Errorf("rule %q: no SignalErrors entry", rule)
			continue
		}
		if code != complexityEvaluationFailedCode {
			t.Errorf("rule %q: code = %q, want %q", rule, code, complexityEvaluationFailedCode)
		}
	}
	if len(results.SignalErrors) != 2 {
		t.Fatalf("SignalErrors = %v, want exactly the two configured rules", results.SignalErrors)
	}
}

func TestComplexitySignalSourceFollowsTheDispatch(t *testing.T) {
	empty := &config.RouterConfig{}
	cases := map[string]struct {
		classifier *Classifier
		want       string
	}{
		"score":  {&Classifier{Config: empty, complexityScoreBackend: &closableScorer{}}, complexitySignalSourceScore},
		"labels": {&Classifier{Config: empty, complexityLabelBackend: &closableSequenceBackend{closableScorer: &closableScorer{}}}, complexitySignalSourceLabels},
		"local":  {&Classifier{Config: empty}, complexitySignalSourceLocal},
	}
	for name, tc := range cases {
		if got := tc.classifier.complexitySignalSource(); got != tc.want {
			t.Errorf("%s: source = %q, want %q", name, got, tc.want)
		}
	}
}
