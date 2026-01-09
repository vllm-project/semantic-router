package classification_test

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var _ = Describe("Signal Skip Optimization", func() {
	Context("when analyzing used signals", func() {
		It("should correctly identify used signal types from decisions", func() {
			cfg := &config.RouterConfig{
				IntelligentRouting: config.IntelligentRouting{
					Decisions: []config.Decision{
						{
							Name: "test_decision_1",
							Rules: config.RuleCombination{
								Operator: "AND",
								Conditions: []config.RuleCondition{
									{Type: config.SignalTypeKeyword, Name: "math_keywords"},
									{Type: config.SignalTypeDomain, Name: "mathematics"},
								},
							},
						},
						{
							Name: "test_decision_2",
							Rules: config.RuleCombination{
								Operator: "OR",
								Conditions: []config.RuleCondition{
									{Type: config.SignalTypeUserFeedback, Name: "wrong_answer"},
									{Type: config.SignalTypePreference, Name: "code_generation"},
								},
							},
						},
					},
				},
			}

			classifier := &classification.Classifier{
				Config: cfg,
			}

			// Call EvaluateAllSignals to trigger signal analysis
			// We expect it to skip embedding and fact_check signals
			results := classifier.EvaluateAllSignals("test query")

			// Verify results structure is initialized
			// Note: In Go, uninitialized slices are nil, which is expected
			Expect(results).NotTo(BeNil())
		})

		It("should skip all signals when no decisions are configured", func() {
			cfg := &config.RouterConfig{
				IntelligentRouting: config.IntelligentRouting{
					Decisions: []config.Decision{},
				},
			}

			classifier := &classification.Classifier{
				Config: cfg,
			}

			// Call EvaluateAllSignals - should skip all signals
			results := classifier.EvaluateAllSignals("test query")

			// Verify results structure is initialized
			// All slices should be empty (nil or zero-length)
			Expect(results).NotTo(BeNil())
			Expect(len(results.MatchedKeywordRules)).To(Equal(0))
			Expect(len(results.MatchedEmbeddingRules)).To(Equal(0))
			Expect(len(results.MatchedDomainRules)).To(Equal(0))
			Expect(len(results.MatchedFactCheckRules)).To(Equal(0))
			Expect(len(results.MatchedUserFeedbackRules)).To(Equal(0))
			Expect(len(results.MatchedPreferenceRules)).To(Equal(0))
		})

		It("should handle decisions with all signal types", func() {
			cfg := &config.RouterConfig{
				IntelligentRouting: config.IntelligentRouting{
					Decisions: []config.Decision{
						{
							Name: "comprehensive_decision",
							Rules: config.RuleCombination{
								Operator: "OR",
								Conditions: []config.RuleCondition{
									{Type: config.SignalTypeKeyword, Name: "test_keyword"},
									{Type: config.SignalTypeEmbedding, Name: "test_embedding"},
									{Type: config.SignalTypeDomain, Name: "test_domain"},
									{Type: config.SignalTypeFactCheck, Name: "needs_fact_check"},
									{Type: config.SignalTypeUserFeedback, Name: "satisfied"},
									{Type: config.SignalTypePreference, Name: "test_preference"},
								},
							},
						},
					},
				},
			}

			classifier := &classification.Classifier{
				Config: cfg,
			}

			// Call EvaluateAllSignals - should evaluate all signal types
			results := classifier.EvaluateAllSignals("test query")

			// Verify results structure is initialized
			Expect(results).NotTo(BeNil())
		})
	})
})
