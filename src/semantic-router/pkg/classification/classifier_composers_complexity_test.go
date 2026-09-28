package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A scorer failure is recorded for every rule before any composer has been
// evaluated, because the signals a composer reads are computed in parallel.
// Eligibility is only knowable once they are all in, so the failure has to be
// filtered here - otherwise a rule gated on a domain this request is not in
// still carries its failure to the decision engine, and rules.on_unknown:
// match would match a request the rule was never eligible for.
func TestComposerFilterDropsIneligibleComplexityFailures(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.ComplexityRules = []config.ComplexityRule{
		{
			Name:      "medical_depth",
			HardAbove: float64Ptr(0.80),
			EasyBelow: float64Ptr(0.30),
			Composer: &config.RuleCombination{
				Operator:   "AND",
				Conditions: []config.RuleNode{{Type: config.SignalTypeDomain, Name: "medical"}},
			},
		},
		{
			// No composer: always eligible, so its failure must survive.
			Name:      "general_depth",
			HardAbove: float64Ptr(0.80),
			EasyBelow: float64Ptr(0.30),
		},
	}
	classifier := &Classifier{Config: cfg}

	failureFor := func(domains []string) map[string]string {
		results := &SignalResults{
			SignalErrors:       map[string]string{},
			MatchedDomainRules: domains,
		}
		for _, rule := range cfg.ComplexityRules {
			for _, verdict := range ComplexityVerdictLabels {
				key := signalConfidenceKey(config.SignalTypeComplexity, rule.Name+":"+verdict)
				results.SignalErrors[key] = complexityEvaluationFailedCode
			}
		}
		classifier.filterComplexitySignalErrorsByComposer(results)
		return results.SignalErrors
	}

	// The request is not medical: the gated rule's failures must be gone and
	// the ungated rule's must remain.
	errs := failureFor(nil)
	for _, verdict := range ComplexityVerdictLabels {
		if _, present := errs["complexity:medical_depth:"+verdict]; present {
			t.Errorf("medical_depth:%s failure survived for a non-medical request", verdict)
		}
		if _, present := errs["complexity:general_depth:"+verdict]; !present {
			t.Errorf("general_depth:%s failure was dropped although the rule has no composer", verdict)
		}
	}

	// The request is medical: the gated rule is eligible, so its failure has
	// to reach the decision engine or on_unknown could never govern it.
	errs = failureFor([]string{"medical"})
	for _, verdict := range ComplexityVerdictLabels {
		if _, present := errs["complexity:medical_depth:"+verdict]; !present {
			t.Errorf("medical_depth:%s failure was dropped for a medical request", verdict)
		}
	}
}

// Errors belonging to other signals must pass through untouched - the filter
// keys off the complexity prefix, and a jailbreak or generic failure has its
// own lifecycle.
func TestComposerFilterLeavesOtherSignalErrorsAlone(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.ComplexityRules = []config.ComplexityRule{{
		Name:      "gated",
		HardAbove: float64Ptr(0.80),
		EasyBelow: float64Ptr(0.30),
		Composer: &config.RuleCombination{
			Operator:   "AND",
			Conditions: []config.RuleNode{{Type: config.SignalTypeDomain, Name: "medical"}},
		},
	}}
	classifier := &Classifier{Config: cfg}

	results := &SignalResults{SignalErrors: map[string]string{
		"complexity:gated:hard": complexityEvaluationFailedCode,
		"jailbreak:prompt":      "jailbreak_evaluation_failed",
		"classifier:intent":     "generic_evaluation_failed",
	}}
	classifier.filterComplexitySignalErrorsByComposer(results)

	if _, present := results.SignalErrors["complexity:gated:hard"]; present {
		t.Error("the ineligible complexity failure should have been dropped")
	}
	for _, key := range []string{"jailbreak:prompt", "classifier:intent"} {
		if _, present := results.SignalErrors[key]; !present {
			t.Errorf("%q was dropped; the filter must only touch complexity keys", key)
		}
	}
}
