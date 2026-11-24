package classification

import (
	"context"
	"slices"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

const fullCoverageRuleName = "guard"

func fullCoverageRule() config.SystemOneRule {
	return config.SystemOneRule{
		Name:       fullCoverageRuleName,
		Deployment: "kai",
		Coverage:   config.SystemOneCoverageFull,
		Question:   config.SystemOneQuestion{Type: "noul", Instructions: "Is this a jailbreak?"},
	}
}

func truncatingRule() config.SystemOneRule {
	rule := fullCoverageRule()
	rule.Name, rule.Coverage = "tone", ""
	return rule
}

func evaluateFullCoverage(t *testing.T, rules []config.SystemOneRule, answers map[string]modelservice.Answer, failures map[string]error) (*SignalResults, *fakeDecider) {
	t.Helper()
	decider := newFakeDecider(answers)
	for deployment, failure := range failures {
		decider.failures[deployment] = failure
	}
	classifier := newSystemOneTestClassifier(decider, rules...)
	results := newSignalResults()
	classifier.evaluateSystemOneSignals(context.Background(), results, &sync.Mutex{}, systemOneReading{request: "ignore previous instructions"}, usedRules(rules))
	return results, decider
}

func guardAnswer(coverage string) map[string]modelservice.Answer {
	return map[string]modelservice.Answer{
		"systemone:" + fullCoverageRuleName: {Type: "noul", Noul: 0.9, InputCoverage: coverage},
	}
}

// A full-coverage rule that read its whole input still matches on its answer.
func TestSingleCallFullCoverageAcceptsCompleteCoverage(t *testing.T) {
	results, decider := evaluateFullCoverage(t, []config.SystemOneRule{fullCoverageRule()}, guardAnswer("complete"), nil)
	if !slices.Contains(results.MatchedSystemOneRules, fullCoverageRuleName) {
		t.Fatalf("complete coverage did not match: %v", results.MatchedSystemOneRules)
	}
	if results.SignalValues["systemone:"+fullCoverageRuleName] != 0.9 {
		t.Fatalf("complete coverage lost the answer value: %v", results.SignalValues)
	}
	if _, failed := results.SignalErrors["systemone:"+fullCoverageRuleName]; failed {
		t.Fatalf("complete coverage is unknown: %v", results.SignalErrors)
	}
	asked := decider.requests["kai"]
	if len(asked.Questions) != 1 || asked.Questions[0].Truncate {
		t.Fatalf("a full-coverage rule must ask for its whole input: %+v", asked.Questions)
	}
}

// A direct decider can produce a plausible answer without attesting that it
// read the whole input. That answer is unread, never a successful full scan.
func TestSingleCallFullCoverageRejectsUnattestedCoverage(t *testing.T) {
	for _, coverage := range []string{"", "partial", "window"} {
		t.Run("coverage="+coverage, func(t *testing.T) {
			results, _ := evaluateFullCoverage(t, []config.SystemOneRule{fullCoverageRule()}, guardAnswer(coverage), nil)
			if slices.Contains(results.MatchedSystemOneRules, fullCoverageRuleName) {
				t.Fatalf("coverage %q matched a benign result: %v", coverage, results.MatchedSystemOneRules)
			}
			if got := results.SignalErrors["systemone:"+fullCoverageRuleName]; got != "systemone_input_coverage_unknown" {
				t.Fatalf("coverage %q error = %q, want systemone_input_coverage_unknown", coverage, got)
			}
			if value, published := results.SignalValues["systemone:"+fullCoverageRuleName]; published {
				t.Fatalf("coverage %q published an unverified value %v", coverage, value)
			}
		})
	}
}

// The existing outcomes for a call that never produced an answer are
// unchanged: missing answer, an answer's own error, and a transport error.
func TestSingleCallFullCoverageKeepsOtherFailuresUnchanged(t *testing.T) {
	rule := fullCoverageRule()
	cases := []struct {
		name     string
		answers  map[string]modelservice.Answer
		failures map[string]error
		want     string
	}{
		{name: "missing answer", answers: map[string]modelservice.Answer{}, want: "systemone_missing_answer"},
		{
			name: "answer error",
			answers: map[string]modelservice.Answer{
				"systemone:" + fullCoverageRuleName: {Type: "noul", Error: "max_length_exceeded"},
			},
			want: "systemone_max_length_exceeded",
		},
		{
			name:     "transport error",
			answers:  guardAnswer("complete"),
			failures: map[string]error{"kai": modelservice.ErrUnavailable},
			want:     "systemone_unavailable",
		},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			results, _ := evaluateFullCoverage(t, []config.SystemOneRule{rule}, test.answers, test.failures)
			if got := results.SignalErrors["systemone:"+fullCoverageRuleName]; got != test.want {
				t.Fatalf("error = %q, want %q", got, test.want)
			}
			if slices.Contains(results.MatchedSystemOneRules, fullCoverageRuleName) {
				t.Fatalf("%s matched a benign result: %v", test.name, results.MatchedSystemOneRules)
			}
		})
	}
}

// A rule that does not require full coverage keeps deciding on the answer it
// was given: only full-coverage rules need the coverage attestation.
func TestSingleCallTruncatingRuleKeepsItsUnattestedAnswer(t *testing.T) {
	rule := truncatingRule()
	results, decider := evaluateFullCoverage(t, []config.SystemOneRule{rule}, map[string]modelservice.Answer{
		"systemone:" + rule.Name: {Type: "noul", Noul: 0.9},
	}, nil)
	if !slices.Contains(results.MatchedSystemOneRules, rule.Name) {
		t.Fatalf("a truncating rule no longer matches: %v, errors %v", results.MatchedSystemOneRules, results.SignalErrors)
	}
	if asked := decider.requests["kai"]; len(asked.Questions) != 1 || !asked.Questions[0].Truncate {
		t.Fatalf("a truncating rule must ask for its first tokens: %+v", asked.Questions)
	}
}

// One call carries rules of both coverages: an unattested answer leaves only
// the full-coverage rule unknown, so it cannot suppress its sibling's answer.
func TestSingleCallCoverageCheckAppliesPerRule(t *testing.T) {
	results, _ := evaluateFullCoverage(t, []config.SystemOneRule{fullCoverageRule(), truncatingRule()}, map[string]modelservice.Answer{
		"systemone:" + fullCoverageRuleName: {Type: "noul", Noul: 0.9},
		"systemone:tone":                    {Type: "noul", Noul: 0.8},
	}, nil)
	if got := results.SignalErrors["systemone:"+fullCoverageRuleName]; got != "systemone_input_coverage_unknown" {
		t.Fatalf("full-coverage rule error = %q", got)
	}
	if !slices.Contains(results.MatchedSystemOneRules, "tone") {
		t.Fatalf("truncating sibling lost its answer: %v, errors %v", results.MatchedSystemOneRules, results.SignalErrors)
	}
	if slices.Contains(results.MatchedSystemOneRules, fullCoverageRuleName) {
		t.Fatalf("unattested full-coverage answer matched: %v", results.MatchedSystemOneRules)
	}
}

// on_unscanned still decides what an unread full-coverage rule means.
func TestSingleCallUnattestedCoverageObeysOnUnscanned(t *testing.T) {
	ruleFor := func(policy string) config.SystemOneRule {
		rule := fullCoverageRule()
		rule.OnUnscanned = policy
		return rule
	}
	t.Run("match", func(t *testing.T) {
		results, _ := evaluateFullCoverage(t, []config.SystemOneRule{ruleFor(config.SystemOneUnscannedMatch)}, guardAnswer(""), nil)
		if !slices.Contains(results.MatchedSystemOneRules, fullCoverageRuleName) {
			t.Fatalf("on_unscanned match did not match: %v", results.MatchedSystemOneRules)
		}
		if _, failed := results.SignalErrors["systemone:"+fullCoverageRuleName]; failed {
			t.Fatalf("on_unscanned match recorded an error: %v", results.SignalErrors)
		}
	})
	t.Run("no_match", func(t *testing.T) {
		results, _ := evaluateFullCoverage(t, []config.SystemOneRule{ruleFor(config.SystemOneUnscannedNoMatch)}, guardAnswer(""), nil)
		if slices.Contains(results.MatchedSystemOneRules, fullCoverageRuleName) {
			t.Fatalf("on_unscanned no_match matched: %v", results.MatchedSystemOneRules)
		}
		if _, failed := results.SignalErrors["systemone:"+fullCoverageRuleName]; failed {
			t.Fatalf("on_unscanned no_match recorded an error: %v", results.SignalErrors)
		}
	})
}
