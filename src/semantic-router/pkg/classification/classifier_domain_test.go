package classification

import (
	"errors"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type MockCategoryInference struct {
	classifyResult          candle_binding.ClassResult
	classifyError           error
	classifyWithProbsResult candle_binding.ClassResultWithProbs
	classifyWithProbsError  error
}

func (m *MockCategoryInference) Classify(_ string) (candle_binding.ClassResult, error) {
	return m.classifyResult, m.classifyError
}

func (m *MockCategoryInference) ClassifyWithProbabilities(_ string) (candle_binding.ClassResultWithProbs, error) {
	return m.classifyWithProbsResult, m.classifyWithProbsError
}

var _ CategoryInference = (*MockCategoryInference)(nil)

func domainTestConfig() *config.RouterConfig {
	return &config.RouterConfig{
		InlineModels: config.InlineModels{
			Classifier: config.Classifier{
				CategoryModel: config.CategoryModel{
					ModelID:             "test-model",
					Threshold:           0.3,
					CategoryMappingPath: "test-path",
				},
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name: "test_domain_decision",
					Rules: config.RuleCombination{
						Operator: "AND",
						Conditions: []config.RuleCondition{
							{Type: config.SignalTypeDomain, Name: "economics"},
						},
					},
				},
			},
		},
	}
}

func buildDomainClassifier(mock *MockCategoryInference) *Classifier {
	return &Classifier{
		Config: domainTestConfig(),
		CategoryMapping: &CategoryMapping{
			CategoryToIdx: map[string]int{
				"biology": 0, "business": 1, "chemistry": 2,
				"computer_science": 3, "economics": 4, "engineering": 5,
				"health": 6, "history": 7, "law": 8, "math": 9,
				"other": 10, "philosophy": 11, "physics": 12, "psychology": 13,
			},
			IdxToCategory: map[string]string{
				"0": "biology", "1": "business", "2": "chemistry",
				"3": "computer_science", "4": "economics", "5": "engineering",
				"6": "health", "7": "history", "8": "law", "9": "math",
				"10": "other", "11": "philosophy", "12": "physics", "13": "psychology",
			},
		},
		categoryInference: mock,
	}
}

var _ = Describe("Domain signal: low entropy (confident)", func() {
	It("should return only the top-1 category", func() {
		probs := make([]float32, 14)
		probs[12] = 0.91
		for i := range probs {
			if i != 12 {
				probs[i] = 0.007
			}
		}

		mock := &MockCategoryInference{
			classifyWithProbsResult: candle_binding.ClassResultWithProbs{
				Class: 12, Confidence: 0.91,
				Probabilities: probs, NumClasses: 14,
			},
		}

		classifier := buildDomainClassifier(mock)
		results := classifier.EvaluateAllSignals("What is quantum entanglement?")

		Expect(results.MatchedDomainRules).To(HaveLen(1))
		Expect(results.MatchedDomainRules).To(ContainElement("physics"))
		Expect(results.SignalConfidences).To(HaveKeyWithValue("domain:physics", BeNumerically("~", 0.91, 0.01)))
	})
})

var _ = Describe("Domain signal: high entropy (ambiguous)", func() {
	It("should return multiple categories above threshold", func() {
		probs := make([]float32, 14)
		probs[4] = 0.40
		probs[6] = 0.38
		for i := range probs {
			if i != 4 && i != 6 {
				probs[i] = 0.02
			}
		}

		mock := &MockCategoryInference{
			classifyWithProbsResult: candle_binding.ClassResultWithProbs{
				Class: 4, Confidence: 0.40,
				Probabilities: probs, NumClasses: 14,
			},
		}

		classifier := buildDomainClassifier(mock)
		results := classifier.EvaluateAllSignals("What are the economic impacts of healthcare reform?")

		Expect(results.MatchedDomainRules).To(ContainElement("economics"))
		Expect(results.MatchedDomainRules).To(ContainElement("health"))
		Expect(results.SignalConfidences).To(HaveKey("domain:economics"))
		Expect(results.SignalConfidences).To(HaveKey("domain:health"))
	})
})

var _ = Describe("Domain signal: BERT-base fallback", func() {
	It("should fall back to Classify and return top-1 with SignalConfidences", func() {
		mock := &MockCategoryInference{
			classifyWithProbsError: errors.New("ModernBERT not initialized"),
			classifyResult: candle_binding.ClassResult{
				Class: 4, Confidence: 0.87,
			},
		}

		classifier := buildDomainClassifier(mock)
		results := classifier.EvaluateAllSignals("Explain supply and demand")

		Expect(results.MatchedDomainRules).To(HaveLen(1))
		Expect(results.MatchedDomainRules).To(ContainElement("economics"))
		Expect(results.SignalConfidences).To(HaveKeyWithValue("domain:economics", BeNumerically("~", 0.87, 0.01)))
	})
})

var _ = Describe("Domain signal: no probabilities (mmBERT-32K)", func() {
	It("should use top-1 fallback with SignalConfidences", func() {
		mock := &MockCategoryInference{
			classifyWithProbsResult: candle_binding.ClassResultWithProbs{
				Class: 9, Confidence: 0.91,
			},
		}

		classifier := buildDomainClassifier(mock)
		results := classifier.EvaluateAllSignals("Solve x^2 + 3x - 4 = 0")

		Expect(results.MatchedDomainRules).To(HaveLen(1))
		Expect(results.MatchedDomainRules).To(ContainElement("math"))
		Expect(results.SignalConfidences).To(HaveKeyWithValue("domain:math", BeNumerically("~", 0.91, 0.01)))
	})
})

var _ = Describe("Domain signal: below threshold", func() {
	It("should not match any domain", func() {
		mock := &MockCategoryInference{
			classifyWithProbsResult: candle_binding.ClassResultWithProbs{
				Class: 4, Confidence: 0.15,
			},
		}

		classifier := buildDomainClassifier(mock)
		results := classifier.EvaluateAllSignals("asdfgh jkl")

		Expect(results.MatchedDomainRules).To(BeEmpty())
		Expect(results.SignalConfidences).NotTo(HaveKey("domain:economics"))
	})
})

var _ = Describe("Domain signal: complete classification failure", func() {
	It("should not crash and return empty results", func() {
		mock := &MockCategoryInference{
			classifyWithProbsError: errors.New("ModernBERT not initialized"),
			classifyError:          errors.New("CandleBERT also failed"),
		}

		classifier := buildDomainClassifier(mock)
		results := classifier.EvaluateAllSignals("test query")

		Expect(results.MatchedDomainRules).To(BeEmpty())
		for k := range results.SignalConfidences {
			Expect(k).NotTo(HavePrefix("domain:"))
		}
	})
})
