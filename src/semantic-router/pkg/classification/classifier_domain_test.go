package classification

import (
	"context"
	"errors"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type MockCategoryInference struct {
	classifyResult          tasks.ClassResult
	classifyError           error
	classifyWithProbsResult tasks.ClassResultWithProbs
	classifyWithProbsError  error
}

func (m *MockCategoryInference) Classify(_ context.Context, _ string) (tasks.ClassResult, error) {
	return m.classifyResult, m.classifyError
}

func (m *MockCategoryInference) ClassifyWithProbabilities(_ context.Context, _ string) (tasks.ClassResultWithProbs, error) {
	return m.classifyWithProbsResult, m.classifyWithProbsError
}

var _ CategoryInference = (*MockCategoryInference)(nil)

func domainRule(name string, labels ...string) config.Category {
	return config.Category{CategoryMetadata: config.CategoryMetadata{Name: name, MMLUCategories: labels}}
}

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
			Signals: config.Signals{Categories: []config.Category{
				domainRule("economics", "economics"),
				domainRule("health", "health"),
				domainRule("math", "math"),
				domainRule("physics", "physics"),
			}},
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

func buildDomainClassifier(mock *MockCategoryInference, rules ...config.Category) *Classifier {
	cfg := domainTestConfig()
	if len(rules) > 0 {
		cfg.Categories = rules
	}
	classifier := &Classifier{
		Config: cfg,
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
	classifier.buildCategoryNameMappings()
	return classifier
}

// topProbabilities puts confidence on class top and spreads the rest evenly.
func topProbabilities(top int, confidence float32) []float32 {
	probs := make([]float32, 14)
	for i := range probs {
		probs[i] = (1 - confidence) / 13
	}
	probs[top] = confidence
	return probs
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
			classifyWithProbsResult: tasks.ClassResultWithProbs{
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
			classifyWithProbsResult: tasks.ClassResultWithProbs{
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
			classifyResult: tasks.ClassResult{
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
			classifyWithProbsResult: tasks.ClassResultWithProbs{
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
			classifyWithProbsResult: tasks.ClassResultWithProbs{
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
			classifyError:          errors.New("BERT classifier also failed"),
		}

		classifier := buildDomainClassifier(mock)
		results := classifier.EvaluateAllSignals("test query")

		Expect(results.MatchedDomainRules).To(BeEmpty())
		for k := range results.SignalConfidences {
			Expect(k).NotTo(HavePrefix("domain:"))
		}
	})
})

// The issue's configuration: four declared domains, other the fallback.
var declaredWithOther = []config.Category{
	domainRule("math", "math"),
	domainRule("law", "law"),
	domainRule("health", "health"),
	domainRule("other", "other"),
}

var _ = Describe("Domain signal: only declared rules match", func() {
	It("matches the rule that lists other for a label no rule lists", func() {
		mock := &MockCategoryInference{classifyWithProbsResult: tasks.ClassResultWithProbs{
			Class: 3, Confidence: 0.9, Probabilities: topProbabilities(3, 0.9), NumClasses: 14,
		}}
		results := buildDomainClassifier(mock, declaredWithOther...).
			EvaluateAllSignals("Can you debug this Python function and refactor the algorithm?")

		Expect(results.MatchedDomainRules).To(Equal([]string{"other"}))
		Expect(results.SignalConfidences).To(HaveKeyWithValue("domain:other", BeNumerically("~", 0.9, 0.01)))
		Expect(results.SignalConfidences).NotTo(HaveKey("domain:computer_science"))
	})

	It("matches nothing for a label no rule lists when no rule lists other", func() {
		for _, probs := range [][]float32{topProbabilities(1, 0.9), nil} {
			mock := &MockCategoryInference{classifyWithProbsResult: tasks.ClassResultWithProbs{
				Class: 1, Confidence: 0.9, Probabilities: probs, NumClasses: 14,
			}}
			results := buildDomainClassifier(mock, declaredWithOther[:3]...).
				EvaluateAllSignals("You can reach me at jane.doe@example.com for the report.")

			Expect(results.MatchedDomainRules).To(BeEmpty())
			for key := range results.SignalConfidences {
				Expect(key).NotTo(HavePrefix("domain:"))
			}
		}
	})

	It("matches a rule named after a label without mmlu_categories", func() {
		mock := &MockCategoryInference{classifyWithProbsResult: tasks.ClassResultWithProbs{
			Class: 8, Confidence: 0.9, Probabilities: topProbabilities(8, 0.9), NumClasses: 14,
		}}
		results := buildDomainClassifier(mock, domainRule("law"), domainRule("other", "other")).
			EvaluateAllSignals("Can my landlord evict me without a court order?")

		Expect(results.MatchedDomainRules).To(Equal([]string{"law"}))
	})

	It("counts unlisted labels above the threshold as other when the request is ambiguous", func() {
		probs := make([]float32, 14)
		for i := range probs {
			probs[i] = 0.02
		}
		probs[9], probs[12] = 0.40, 0.36
		mock := &MockCategoryInference{classifyWithProbsResult: tasks.ClassResultWithProbs{
			Class: 9, Confidence: 0.40, Probabilities: probs, NumClasses: 14,
		}}
		results := buildDomainClassifier(mock, declaredWithOther...).
			EvaluateAllSignals("Derive the period of a pendulum from its equation of motion.")

		Expect(results.MatchedDomainRules).To(Equal([]string{"math", "other"}))
		Expect(results.SignalConfidences).To(HaveKeyWithValue("domain:other", BeNumerically("~", 0.36, 0.01)))
		Expect(results.SignalConfidences).NotTo(HaveKey("domain:physics"))
	})

	It("matches a rule once, at its highest label, when several of its labels pass", func() {
		probs := make([]float32, 14)
		for i := range probs {
			probs[i] = 0.02
		}
		probs[1], probs[4] = 0.38, 0.40
		mock := &MockCategoryInference{classifyWithProbsResult: tasks.ClassResultWithProbs{
			Class: 4, Confidence: 0.40, Probabilities: probs, NumClasses: 14,
		}}
		results := buildDomainClassifier(mock, domainRule("commerce", "business", "economics")).
			EvaluateAllSignals("How do tariffs change a retailer's margins?")

		Expect(results.MatchedDomainRules).To(Equal([]string{"commerce"}))
		Expect(results.SignalConfidences).To(HaveKeyWithValue("domain:commerce", BeNumerically("~", 0.40, 0.01)))
	})
})
