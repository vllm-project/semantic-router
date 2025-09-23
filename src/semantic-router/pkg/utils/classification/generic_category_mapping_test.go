package classification

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var _ = Describe("generic category mapping (MMLU-Pro -> generic)", func() {
	var (
		classifier              *Classifier
		mockCategoryInitializer *MockCategoryInitializer
		mockCategoryModel       *MockCategoryInference
	)

	BeforeEach(func() {
		mockCategoryInitializer = &MockCategoryInitializer{InitError: nil}
		mockCategoryModel = &MockCategoryInference{}

		cfg := &config.RouterConfig{}
		cfg.Classifier.CategoryModel.ModelID = "model-id"
		cfg.Classifier.CategoryModel.CategoryMappingPath = "category-mapping-path"
		cfg.Classifier.CategoryModel.Threshold = 0.5

		// Define generic categories with MMLU-Pro mappings
		cfg.Categories = []config.Category{
			{
				Name:            "tech",
				MMLUCategories:  []string{"computer science", "engineering"},
				ModelScores:     []config.ModelScore{{Model: "phi4", Score: 0.9, UseReasoning: config.BoolPtr(false)}},
				ReasoningEffort: "low",
			},
			{
				Name:           "finance",
				MMLUCategories: []string{"economics"},
				ModelScores:    []config.ModelScore{{Model: "gemma3:27b", Score: 0.8, UseReasoning: config.BoolPtr(true)}},
			},
			{
				Name: "politics",
				// No explicit mmlu_categories -> identity fallback when label exists in mapping
				ModelScores: []config.ModelScore{{Model: "gemma3:27b", Score: 0.6, UseReasoning: config.BoolPtr(false)}},
			},
		}

		// Category mapping represents labels coming from the MMLU-Pro model
		categoryMapping := &CategoryMapping{
			CategoryToIdx: map[string]int{
				"computer science": 0,
				"economics":        1,
				"politics":         2,
			},
			IdxToCategory: map[string]string{
				"0": "Computer Science", // different case to assert case-insensitive mapping
				"1": "economics",
				"2": "politics",
			},
		}

		var err error
		classifier, err = newClassifierWithOptions(
			cfg,
			withCategory(categoryMapping, mockCategoryInitializer, mockCategoryModel),
		)
		Expect(err).ToNot(HaveOccurred())
	})

	It("builds expected MMLU<->generic maps", func() {
		Expect(classifier.MMLUToGeneric).To(HaveKeyWithValue("computer science", "tech"))
		Expect(classifier.MMLUToGeneric).To(HaveKeyWithValue("engineering", "tech"))
		Expect(classifier.MMLUToGeneric).To(HaveKeyWithValue("economics", "finance"))
		// identity fallback for a generic name that exists as an MMLU label
		Expect(classifier.MMLUToGeneric).To(HaveKeyWithValue("politics", "politics"))

		Expect(classifier.GenericToMMLU).To(HaveKey("tech"))
		Expect(classifier.GenericToMMLU["tech"]).To(ConsistOf("computer science", "engineering"))
		Expect(classifier.GenericToMMLU).To(HaveKeyWithValue("finance", ConsistOf("economics")))
		Expect(classifier.GenericToMMLU).To(HaveKeyWithValue("politics", ConsistOf("politics")))
	})

	It("translates ClassifyCategory result to generic category", func() {
		// Model returns class index 0 -> "Computer Science" (MMLU) which maps to generic "tech"
		mockCategoryModel.classifyResult = candle_binding.ClassResult{Class: 0, Confidence: 0.92}

		category, score, err := classifier.ClassifyCategory("This text is about GPUs and compilers")
		Expect(err).ToNot(HaveOccurred())
		Expect(category).To(Equal("tech"))
		Expect(score).To(BeNumerically("~", 0.92, 0.001))
	})

	It("translates names in entropy flow and returns generic top category", func() {
		// Probabilities favor index 0 -> generic should be "tech"
		mockCategoryModel.classifyWithProbsResult = candle_binding.ClassResultWithProbs{
			Class:         0,
			Confidence:    0.88,
			Probabilities: []float32{0.7, 0.2, 0.1},
			NumClasses:    3,
		}

		category, confidence, decision, err := classifier.ClassifyCategoryWithEntropy("Economic policies in computer science education")
		Expect(err).ToNot(HaveOccurred())
		Expect(category).To(Equal("tech"))
		Expect(confidence).To(BeNumerically("~", 0.88, 0.001))
		Expect(decision.TopCategories).ToNot(BeEmpty())
		Expect(decision.TopCategories[0].Category).To(Equal("tech"))
	})

	It("falls back to identity when no mapping exists for an MMLU label", func() {
		// index 2 -> "politics" (no explicit mapping provided, but present in MMLU set)
		mockCategoryModel.classifyResult = candle_binding.ClassResult{Class: 2, Confidence: 0.91}

		category, score, err := classifier.ClassifyCategory("This is a political debate")
		Expect(err).ToNot(HaveOccurred())
		Expect(category).To(Equal("politics"))
		Expect(score).To(BeNumerically("~", 0.91, 0.001))
	})
})
