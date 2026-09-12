package classification

import (
	"sync"
	"testing"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

func TestVelaFeedbackKeepsNegativeAndUncertainPredictionsDistinct(t *testing.T) {
	mapping := fourClassMapping()
	mapping.IdxToLabel["4"] = FeedbackLabelNoFeedback
	detector := &FeedbackDetector{mapping: mapping}
	for _, tc := range []struct {
		class      int
		confidence float32
		label      string
		abstained  bool
	}{
		{4, .95, FeedbackLabelNoFeedback, false},
		{4, .4, FeedbackLabelNoFeedback, true},
		{3, .4, FeedbackLabelWantDifferent, true},
		{0, .4, FeedbackLabelSatisfied, true},
		{2, .8, FeedbackLabelWrongAnswer, false},
	} {
		raw := candle.ClassResultWithProbs{Class: tc.class, Confidence: tc.confidence, NumClasses: 5}
		got, err := detector.resultForPrediction(raw, .7)
		if err != nil {
			t.Fatal(err)
		}
		if got.FeedbackType != tc.label || got.Abstained != tc.abstained || got.Confidence != tc.confidence || got.Class != tc.class {
			t.Fatalf("raw prediction was relabeled or its uncertainty hidden: %+v", got)
		}
	}
}

func TestFeedbackRejectsUnmappedClassWithoutInventingSatisfaction(t *testing.T) {
	detector := &FeedbackDetector{mapping: fourClassMapping()}
	for _, class := range []int{-1, 4, 99} {
		got, err := detector.resultForPrediction(candle.ClassResultWithProbs{Class: class, Confidence: .9}, .7)
		if err == nil || got != nil {
			t.Fatalf("unmapped class %d returned %+v, %v", class, got, err)
		}
	}
}

func TestFeedbackAbstentionRespectsDecisionUnknownPolicy(t *testing.T) {
	cfg := &config.RouterConfig{}
	for _, name := range []string{FeedbackLabelSatisfied, FeedbackLabelNeedClarification, FeedbackLabelWrongAnswer, FeedbackLabelWantDifferent} {
		cfg.UserFeedbackRules = append(cfg.UserFeedbackRules, config.UserFeedbackRule{Name: name})
	}
	classifier := &Classifier{Config: cfg}
	for _, abstained := range []bool{false, true} {
		results := newMetricScopeResults()
		var mu sync.Mutex
		classifier.applyUserFeedbackSignalResult(results, &mu, &FeedbackResult{
			FeedbackType: FeedbackLabelNoFeedback, Confidence: .4, Class: 4, Abstained: abstained,
		}, nil)
		if len(results.MatchedUserFeedbackRules) != 0 {
			t.Fatal("non-feedback or uncertainty activated a feedback rule")
		}
		if (len(results.SignalErrors) == 4) != abstained {
			t.Fatalf("negative evidence and unknown result conflated: %v", results.SignalErrors)
		}
		for _, policy := range []config.UnknownPolicy{"", config.RuleOnUnknownMatch} {
			engine := decision.NewDecisionEngine(nil, nil, nil, []config.Decision{{
				Name: "route", Rules: config.RuleNode{
					Operator: "AND", OnUnknown: policy,
					Conditions: []config.RuleNode{{Type: config.SignalTypeUserFeedback, Name: FeedbackLabelSatisfied}},
				},
			}}, "")
			result, err := engine.EvaluateDecisionsWithSignals(&decision.SignalMatches{SignalErrors: results.SignalErrors})
			if err != nil {
				t.Fatal(err)
			}
			matched := result != nil && result.Decision != nil
			if matched != (abstained && policy == config.RuleOnUnknownMatch) {
				t.Fatalf("abstained=%t policy=%q incorrectly matched=%t", abstained, policy, matched)
			}
		}
	}
}
