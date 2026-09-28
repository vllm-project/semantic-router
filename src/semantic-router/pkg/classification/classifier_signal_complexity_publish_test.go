package classification

import (
	"fmt"
	"sort"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
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

// A failed evaluation must be keyed the way a decision names the condition.
// validateComplexityConditionName rejects a bare rule name at config load, so
// every complexity leaf is "<rule>:<verdict>" and a failure recorded under the
// bare rule name would never be found - the outage would read as a clean
// no-match. A failure has no verdict, so all three are marked.
func TestRecordComplexityFailureMarksEveryVerdictOfEveryRule(t *testing.T) {
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
		for _, verdict := range ComplexityVerdictLabels {
			// Built the way decision/engine.go builds its lookup key, so this
			// breaks if either side of the contract moves.
			key := fmt.Sprintf("%s:%s:%s", config.SignalTypeComplexity, rule, verdict)
			code, ok := results.SignalErrors[key]
			if !ok {
				t.Errorf("no SignalErrors entry under %q, the key a decision leaf resolves to", key)
				continue
			}
			if code != complexityEvaluationFailedCode {
				t.Errorf("%s: code = %q, want %q", key, code, complexityEvaluationFailedCode)
			}
		}
	}
	if want := len(ComplexityVerdictLabels) * 2; len(results.SignalErrors) != want {
		t.Fatalf("SignalErrors has %d entries, want %d (every verdict of both rules): %v",
			len(results.SignalErrors), want, results.SignalErrors)
	}
}

// The end-to-end contract: what recordComplexityFailure writes must be what
// the real decision engine reads, so rules.on_unknown can actually govern a
// scorer outage. Asserting the map alone missed this once already.
func TestComplexityFailureReachesTheDecisionEngine(t *testing.T) {
	classifier := &Classifier{
		Config:                 remoteComplexityConfig(config.RemoteClassifierContractScore),
		complexityScoreBackend: &closableScorer{},
	}
	results := &SignalResults{}
	var mu sync.Mutex
	classifier.recordComplexityFailure(results, &mu)

	escalate := func(policy config.UnknownPolicy) config.Decision {
		return config.Decision{
			Name:     "escalate",
			Priority: 100,
			Rules: config.RuleNode{
				Operator:  "AND",
				OnUnknown: policy,
				Conditions: []config.RuleNode{{
					Type: config.SignalTypeComplexity,
					Name: "needs_reasoning:hard",
				}},
			},
		}
	}
	// No verdict was produced, only the failure - exactly the outage shape.
	signals := &decision.SignalMatches{SignalErrors: results.SignalErrors}

	matched := decision.NewDecisionEngine(nil, nil, nil, []config.Decision{escalate(config.RuleOnUnknownMatch)}, "")
	result, err := matched.EvaluateDecisionsWithSignals(signals)
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals with on_unknown=match: %v", err)
	}
	if result == nil || result.Decision == nil {
		t.Fatal("on_unknown=match must resolve a failed complexity signal into a match; the engine saw no failure")
	}

	// Without a policy the failure must stay a no-match, so the default is
	// still to fall through rather than escalate every ungraded request.
	silent := decision.NewDecisionEngine(nil, nil, nil, []config.Decision{escalate("")}, "")
	result, err = silent.EvaluateDecisionsWithSignals(signals)
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals without a policy: %v", err)
	}
	if result != nil && result.Decision != nil {
		t.Fatalf("without on_unknown a failed signal must not match, got %q", result.Decision.Name)
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
