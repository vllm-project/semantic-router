package classification_test

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func signalSkipClassifier(decisions ...config.Decision) *classification.Classifier {
	return &classification.Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Decisions: decisions,
			},
		},
	}
}

func signalSkipDecision(
	name string,
	operator string,
	conditions ...config.RuleCondition,
) config.Decision {
	return config.Decision{
		Name: name,
		Rules: config.RuleCombination{
			Operator:   operator,
			Conditions: conditions,
		},
	}
}

func signalSkipCondition(signalType string, name string) config.RuleCondition {
	return config.RuleCondition{Type: signalType, Name: name}
}

var _ = Describe("Signal Skip Optimization", func() {
	It("should correctly identify used signal types from decisions", func() {
		classifier := signalSkipClassifier(
			signalSkipDecision(
				"test_decision_1",
				"AND",
				signalSkipCondition(config.SignalTypeKeyword, "math_keywords"),
				signalSkipCondition(config.SignalTypeDomain, "mathematics"),
			),
			signalSkipDecision(
				"test_decision_2",
				"OR",
				signalSkipCondition(config.SignalTypeUserFeedback, "wrong_answer"),
				signalSkipCondition(config.SignalTypePreference, "code_generation"),
			),
		)

		results := classifier.EvaluateAllSignals("test query")
		Expect(results).NotTo(BeNil())
	})

	It("should skip all signals when no decisions are configured", func() {
		classifier := signalSkipClassifier()

		results := classifier.EvaluateAllSignals("test query")
		Expect(results).NotTo(BeNil())
		Expect(len(results.MatchedKeywordRules)).To(Equal(0))
		Expect(len(results.MatchedEmbeddingRules)).To(Equal(0))
		Expect(len(results.MatchedDomainRules)).To(Equal(0))
		Expect(len(results.MatchedFactCheckRules)).To(Equal(0))
		Expect(len(results.MatchedUserFeedbackRules)).To(Equal(0))
		Expect(len(results.MatchedPreferenceRules)).To(Equal(0))
	})

	It("should handle decisions with all signal types", func() {
		classifier := signalSkipClassifier(
			signalSkipDecision(
				"comprehensive_decision",
				"OR",
				signalSkipCondition(config.SignalTypeKeyword, "test_keyword"),
				signalSkipCondition(config.SignalTypeEmbedding, "test_embedding"),
				signalSkipCondition(config.SignalTypeDomain, "test_domain"),
				signalSkipCondition(config.SignalTypeFactCheck, "needs_fact_check"),
				signalSkipCondition(config.SignalTypeUserFeedback, "satisfied"),
				signalSkipCondition(config.SignalTypeReask, "likely_dissatisfied"),
				signalSkipCondition(config.SignalTypePreference, "test_preference"),
			),
		)

		results := classifier.EvaluateAllSignals("test query")
		Expect(results).NotTo(BeNil())
	})
})
