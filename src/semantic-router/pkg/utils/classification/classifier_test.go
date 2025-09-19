package classification

import (
	"errors"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestClassifier(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Classifier Suite")
}

type MockCategoryInference struct {
	classifyResult          candle_binding.ClassResult
	classifyError           error
	classifyWithProbsResult candle_binding.ClassResultWithProbs
	classifyWithProbsError  error
}

func (m *MockCategoryInference) Classify(text string) (candle_binding.ClassResult, error) {
	return m.classifyResult, m.classifyError
}

func (m *MockCategoryInference) ClassifyWithProbabilities(text string) (candle_binding.ClassResultWithProbs, error) {
	return m.classifyWithProbsResult, m.classifyWithProbsError
}

type MockCategoryInitializer struct{ InitError error }

func (m *MockCategoryInitializer) Init(modelID string, useCPU bool, numClasses ...int) error {
	return m.InitError
}

var _ = Describe("category classification and model selection", func() {
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
		classifier, _ = newClassifierWithOptions(cfg,
			withCategory(&CategoryMapping{
				CategoryToIdx: map[string]int{"technology": 0, "sports": 1, "politics": 2},
				IdxToCategory: map[string]string{"0": "technology", "1": "sports", "2": "politics"},
			}, mockCategoryInitializer, mockCategoryModel),
		)
	})

	Describe("initialize category classifier", func() {
		It("should succeed", func() {
			err := classifier.initializeCategoryClassifier()
			Expect(err).ToNot(HaveOccurred())
		})

		Context("when category mapping is not initialized", func() {
			It("should return error", func() {
				classifier.CategoryMapping = nil
				err := classifier.initializeCategoryClassifier()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("category classification is not properly configured"))
			})
		})

		Context("when not enough categories", func() {
			It("should return error", func() {
				classifier.CategoryMapping = &CategoryMapping{
					CategoryToIdx: map[string]int{"technology": 0},
					IdxToCategory: map[string]string{"0": "technology"},
				}
				err := classifier.initializeCategoryClassifier()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("not enough categories for classification"))
			})
		})

		Context("when initialize category classifier fails", func() {
			It("should return error", func() {
				mockCategoryInitializer.InitError = errors.New("initialize category classifier failed")
				err := classifier.initializeCategoryClassifier()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("initialize category classifier failed"))
			})
		})
	})

	Describe("classify category", func() {
		type row struct {
			ModelID             string
			CategoryMappingPath string
			CategoryMapping     *CategoryMapping
		}

		DescribeTable("when category classification is not properly configured",
			func(r row) {
				classifier.Config.Classifier.CategoryModel.ModelID = r.ModelID
				classifier.Config.Classifier.CategoryModel.CategoryMappingPath = r.CategoryMappingPath
				classifier.CategoryMapping = r.CategoryMapping
				_, _, err := classifier.ClassifyCategory("Some text")
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("category classification is not properly configured"))
			},
			Entry("ModelID is empty", row{ModelID: ""}),
			Entry("CategoryMappingPath is empty", row{CategoryMappingPath: ""}),
			Entry("CategoryMapping is nil", row{CategoryMapping: nil}),
		)

		Context("when classification succeeds with high confidence", func() {
			It("should return the correct category", func() {
				mockCategoryModel.classifyResult = candle_binding.ClassResult{
					Class:      2,
					Confidence: 0.95,
				}

				category, score, err := classifier.ClassifyCategory("This is about politics")

				Expect(err).ToNot(HaveOccurred())
				Expect(category).To(Equal("politics"))
				Expect(score).To(BeNumerically("~", 0.95, 0.001))
			})
		})

		Context("when classification confidence is below threshold", func() {
			It("should return empty category", func() {
				mockCategoryModel.classifyResult = candle_binding.ClassResult{
					Class:      0,
					Confidence: 0.3,
				}

				category, score, err := classifier.ClassifyCategory("Ambiguous text")

				Expect(err).ToNot(HaveOccurred())
				Expect(category).To(Equal(""))
				Expect(score).To(BeNumerically("~", 0.3, 0.001))
			})
		})

		Context("when model inference fails", func() {
			It("should return empty category with zero score", func() {
				mockCategoryModel.classifyError = errors.New("model inference failed")

				category, score, err := classifier.ClassifyCategory("Some text")

				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("classification error"))
				Expect(category).To(Equal(""))
				Expect(score).To(BeNumerically("~", 0.0, 0.001))
			})
		})

		Context("when input is empty or invalid", func() {
			It("should handle empty text gracefully", func() {
				mockCategoryModel.classifyResult = candle_binding.ClassResult{
					Class:      0,
					Confidence: 0.8,
				}

				category, score, err := classifier.ClassifyCategory("")

				Expect(err).ToNot(HaveOccurred())
				Expect(category).To(Equal("technology"))
				Expect(score).To(BeNumerically("~", 0.8, 0.001))
			})
		})

		Context("when class index is not found in category mapping", func() {
			It("should handle invalid category mapping gracefully", func() {
				mockCategoryModel.classifyResult = candle_binding.ClassResult{
					Class:      9,
					Confidence: 0.8,
				}

				category, score, err := classifier.ClassifyCategory("Some text")

				Expect(err).ToNot(HaveOccurred())
				Expect(category).To(Equal(""))
				Expect(score).To(BeNumerically("~", 0.8, 0.001))
			})
		})
	})

	Describe("category classification with entropy", func() {
		Context("when category mapping is not initialized", func() {
			It("should return error", func() {
				classifier.CategoryMapping = nil
				_, _, _, err := classifier.ClassifyCategoryWithEntropy("Some text")
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("category classification is not properly configured"))
			})
		})

		Context("when classification succeeds with probabilities", func() {
			It("should return category and entropy decision", func() {
				mockCategoryModel.classifyWithProbsResult = candle_binding.ClassResultWithProbs{
					Class:         2,
					Confidence:    0.95,
					Probabilities: []float32{0.02, 0.03, 0.95},
					NumClasses:    3,
				}

				// Add UseReasoning configuration for the categories
				classifier.Config.Categories = []config.Category{
					{Name: "technology", ModelScores: []config.ModelScore{{Model: "phi4", Score: 0.8, UseReasoning: config.BoolPtr(false)}}},
					{Name: "sports", ModelScores: []config.ModelScore{{Model: "phi4", Score: 0.8, UseReasoning: config.BoolPtr(false)}}},
					{Name: "politics", ModelScores: []config.ModelScore{{Model: "deepseek-v31", Score: 0.9, UseReasoning: config.BoolPtr(true)}}},
				}

				category, confidence, reasoningDecision, err := classifier.ClassifyCategoryWithEntropy("This is about politics")

				Expect(err).ToNot(HaveOccurred())
				Expect(category).To(Equal("politics"))
				Expect(confidence).To(BeNumerically("~", 0.95, 0.001))
				Expect(reasoningDecision.UseReasoning).To(BeTrue()) // Politics uses reasoning
				Expect(len(reasoningDecision.TopCategories)).To(BeNumerically(">", 0))
			})
		})

		Context("when classification confidence is below threshold", func() {
			It("should return empty category but still provide entropy decision", func() {
				mockCategoryModel.classifyWithProbsResult = candle_binding.ClassResultWithProbs{
					Class:         0,
					Confidence:    0.3,
					Probabilities: []float32{0.3, 0.35, 0.35},
					NumClasses:    3,
				}

				classifier.Config.Categories = []config.Category{
					{Name: "technology", ModelScores: []config.ModelScore{{Model: "phi4", Score: 0.8, UseReasoning: config.BoolPtr(false)}}},
					{Name: "sports", ModelScores: []config.ModelScore{{Model: "deepseek-v31", Score: 0.9, UseReasoning: config.BoolPtr(true)}}},
					{Name: "politics", ModelScores: []config.ModelScore{{Model: "deepseek-v31", Score: 0.9, UseReasoning: config.BoolPtr(true)}}},
				}

				category, confidence, reasoningDecision, err := classifier.ClassifyCategoryWithEntropy("Ambiguous text")

				Expect(err).ToNot(HaveOccurred())
				Expect(category).To(Equal(""))
				Expect(confidence).To(BeNumerically("~", 0.3, 0.001))
				Expect(len(reasoningDecision.TopCategories)).To(BeNumerically(">", 0))
			})
		})

		Context("when model inference fails", func() {
			It("should return error", func() {
				mockCategoryModel.classifyWithProbsError = errors.New("model inference failed")

				category, confidence, reasoningDecision, err := classifier.ClassifyCategoryWithEntropy("Some text")

				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("classification error"))
				Expect(category).To(Equal(""))
				Expect(confidence).To(BeNumerically("~", 0.0, 0.001))
				Expect(reasoningDecision.UseReasoning).To(BeFalse())
			})
		})
	})

	BeforeEach(func() {
		classifier.Config.Categories = []config.Category{
			{
				Name: "technology",
				ModelScores: []config.ModelScore{
					{Model: "model-a", Score: 0.9},
					{Model: "model-b", Score: 0.8},
				},
			},
			{
				Name:        "sports",
				ModelScores: []config.ModelScore{},
			},
		}
		classifier.Config.DefaultModel = "default-model"
	})

	Describe("select best model for category", func() {
		It("should return the best model", func() {
			model := classifier.SelectBestModelForCategory("technology")
			Expect(model).To(Equal("model-a"))
		})

		Context("when category is not found", func() {
			It("should return the default model", func() {
				model := classifier.SelectBestModelForCategory("non-existent-category")
				Expect(model).To(Equal("default-model"))
			})
		})

		Context("when no best model is found", func() {
			It("should return the default model", func() {
				model := classifier.SelectBestModelForCategory("sports")
				Expect(model).To(Equal("default-model"))
			})
		})
	})

	Describe("select best model from list", func() {
		It("should return the best model", func() {
			model := classifier.SelectBestModelFromList([]string{"model-a"}, "technology")
			Expect(model).To(Equal("model-a"))
		})

		Context("when candidate models are empty", func() {
			It("should return the default model", func() {
				model := classifier.SelectBestModelFromList([]string{}, "technology")
				Expect(model).To(Equal("default-model"))
			})
		})

		Context("when category is not found", func() {
			It("should return the first candidate model", func() {
				model := classifier.SelectBestModelFromList([]string{"model-a"}, "non-existent-category")
				Expect(model).To(Equal("model-a"))
			})
		})

		Context("when the model is not in the candidate models", func() {
			It("should return the first candidate model", func() {
				model := classifier.SelectBestModelFromList([]string{"model-c"}, "technology")
				Expect(model).To(Equal("model-c"))
			})
		})
	})

	Describe("classify and select best model", func() {
		It("should return the best model", func() {
			mockCategoryModel.classifyResult = candle_binding.ClassResult{
				Class:      0,
				Confidence: 0.9,
			}
			model := classifier.ClassifyAndSelectBestModel("Some text")
			Expect(model).To(Equal("model-a"))
		})

		Context("when the categories are empty", func() {
			It("should return the default model", func() {
				classifier.Config.Categories = nil
				model := classifier.ClassifyAndSelectBestModel("Some text")
				Expect(model).To(Equal("default-model"))
			})
		})

		Context("when the classification fails", func() {
			It("should return the default model", func() {
				mockCategoryModel.classifyError = errors.New("classification failed")
				model := classifier.ClassifyAndSelectBestModel("Some text")
				Expect(model).To(Equal("default-model"))
			})
		})

		Context("when the category name is empty", func() {
			It("should return the default model", func() {
				mockCategoryModel.classifyResult = candle_binding.ClassResult{
					Class:      9,
					Confidence: 0.9,
				}
				model := classifier.ClassifyAndSelectBestModel("Some text")
				Expect(model).To(Equal("default-model"))
			})
		})
	})

	Describe("internal helper methods", func() {
		type row struct {
			query string
			want  *config.Category
		}

		DescribeTable("find category",
			func(r row) {
				cat := classifier.findCategory(r.query)
				if r.want == nil {
					Expect(cat).To(BeNil())
				} else {
					Expect(cat.Name).To(Equal(r.want.Name))
				}
			},
			Entry("should find category case-insensitively", row{query: "TECHNOLOGY", want: &config.Category{Name: "technology"}}),
			Entry("should return nil for non-existent category", row{query: "non-existent", want: nil}),
		)

		Describe("select best model internal", func() {

			It("should select best model without filter", func() {
				cat := &config.Category{
					Name: "test",
					ModelScores: []config.ModelScore{
						{Model: "model-a", Score: 0.9},
						{Model: "model-b", Score: 0.8},
					},
				}

				bestModel, score := classifier.selectBestModelInternal(cat, nil)

				Expect(bestModel).To(Equal("model-a"))
				Expect(score).To(BeNumerically("~", 0.9, 0.001))
			})

			It("should select best model with filter", func() {
				cat := &config.Category{
					Name: "test",
					ModelScores: []config.ModelScore{
						{Model: "model-a", Score: 0.9},
						{Model: "model-b", Score: 0.8},
						{Model: "model-c", Score: 0.7},
					},
				}
				filter := func(model string) bool {
					return model == "model-b" || model == "model-c"
				}

				bestModel, score := classifier.selectBestModelInternal(cat, filter)

				Expect(bestModel).To(Equal("model-b"))
				Expect(score).To(BeNumerically("~", 0.8, 0.001))
			})

			It("should return empty when no models match filter", func() {
				cat := &config.Category{
					Name: "test",
					ModelScores: []config.ModelScore{
						{Model: "model-a", Score: 0.9},
						{Model: "model-b", Score: 0.8},
					},
				}
				filter := func(model string) bool {
					return model == "non-existent-model"
				}

				bestModel, score := classifier.selectBestModelInternal(cat, filter)

				Expect(bestModel).To(Equal(""))
				Expect(score).To(BeNumerically("~", -1.0, 0.001))
			})

			It("should return empty when category has no models", func() {
				cat := &config.Category{
					Name:        "test",
					ModelScores: []config.ModelScore{},
				}

				bestModel, score := classifier.selectBestModelInternal(cat, nil)

				Expect(bestModel).To(Equal(""))
				Expect(score).To(BeNumerically("~", -1.0, 0.001))
			})
		})
	})
})

type MockJailbreakInferenceResponse struct {
	classifyResult candle_binding.ClassResult
	classifyError  error
}

type MockJailbreakInference struct {
	MockJailbreakInferenceResponse
	responseMap map[string]MockJailbreakInferenceResponse
}

func (m *MockJailbreakInference) setMockResponse(text string, class int, confidence float32, err error) {
	m.responseMap[text] = MockJailbreakInferenceResponse{
		classifyResult: candle_binding.ClassResult{
			Class:      class,
			Confidence: confidence,
		},
		classifyError: err,
	}
}

func (m *MockJailbreakInference) Classify(text string) (candle_binding.ClassResult, error) {
	if response, exists := m.responseMap[text]; exists {
		return response.classifyResult, response.classifyError
	}
	return m.classifyResult, m.classifyError
}

type MockJailbreakInitializer struct {
	InitError error
}

func (m *MockJailbreakInitializer) Init(modelID string, useCPU bool, numClasses ...int) error {
	return m.InitError
}

var _ = Describe("jailbreak detection", func() {
	var (
		classifier               *Classifier
		mockJailbreakInitializer *MockJailbreakInitializer
		mockJailbreakModel       *MockJailbreakInference
	)

	BeforeEach(func() {
		mockJailbreakInitializer = &MockJailbreakInitializer{InitError: nil}
		mockJailbreakModel = &MockJailbreakInference{}
		mockJailbreakModel.responseMap = make(map[string]MockJailbreakInferenceResponse)
		cfg := &config.RouterConfig{}
		cfg.PromptGuard.Enabled = true
		cfg.PromptGuard.ModelID = "test-model"
		cfg.PromptGuard.JailbreakMappingPath = "test-mapping"
		cfg.PromptGuard.Threshold = 0.7
		classifier, _ = newClassifierWithOptions(cfg,
			withJailbreak(&JailbreakMapping{
				LabelToIdx: map[string]int{"jailbreak": 0, "benign": 1},
				IdxToLabel: map[string]string{"0": "jailbreak", "1": "benign"},
			}, mockJailbreakInitializer, mockJailbreakModel),
		)
	})

	Describe("initialize jailbreak classifier", func() {
		It("should succeed", func() {
			err := classifier.initializeJailbreakClassifier()
			Expect(err).ToNot(HaveOccurred())
		})

		Context("when jailbreak mapping is not initialized", func() {
			It("should return error", func() {
				classifier.JailbreakMapping = nil
				err := classifier.initializeJailbreakClassifier()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("jailbreak detection is not properly configured"))
			})
		})

		Context("when not enough jailbreak types", func() {
			It("should return error", func() {
				classifier.JailbreakMapping = &JailbreakMapping{
					LabelToIdx: map[string]int{"jailbreak": 0},
					IdxToLabel: map[string]string{"0": "jailbreak"},
				}
				err := classifier.initializeJailbreakClassifier()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("not enough jailbreak types for classification"))
			})
		})

		Context("when initialize jailbreak classifier fails", func() {
			It("should return error", func() {
				mockJailbreakInitializer.InitError = errors.New("initialize jailbreak classifier failed")
				err := classifier.initializeJailbreakClassifier()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("initialize jailbreak classifier failed"))
			})
		})
	})

	Describe("check for jailbreak", func() {
		type row struct {
			Enabled              bool
			ModelID              string
			JailbreakMappingPath string
			JailbreakMapping     *JailbreakMapping
		}

		DescribeTable("when jailbreak detection is not enabled or properly configured",
			func(r row) {
				classifier.Config.PromptGuard.Enabled = r.Enabled
				classifier.Config.PromptGuard.ModelID = r.ModelID
				classifier.Config.PromptGuard.JailbreakMappingPath = r.JailbreakMappingPath
				classifier.JailbreakMapping = r.JailbreakMapping
				isJailbreak, _, _, err := classifier.CheckForJailbreak("Some text")
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("jailbreak detection is not enabled or properly configured"))
				Expect(isJailbreak).To(BeFalse())
			},
			Entry("Enabled is false", row{Enabled: false}),
			Entry("ModelID is empty", row{ModelID: ""}),
			Entry("JailbreakMappingPath is empty", row{JailbreakMappingPath: ""}),
			Entry("JailbreakMapping is nil", row{JailbreakMapping: nil}),
		)

		Context("when text is empty", func() {
			It("should return false", func() {
				isJailbreak, _, _, err := classifier.CheckForJailbreak("")
				Expect(err).ToNot(HaveOccurred())
				Expect(isJailbreak).To(BeFalse())
			})
		})

		Context("when jailbreak is detected with high confidence", func() {
			It("should return true with jailbreak type", func() {
				mockJailbreakModel.classifyResult = candle_binding.ClassResult{
					Class:      0,
					Confidence: 0.9,
				}
				isJailbreak, jailbreakType, confidence, err := classifier.CheckForJailbreak("This is a jailbreak attempt")
				Expect(err).ToNot(HaveOccurred())
				Expect(isJailbreak).To(BeTrue())
				Expect(jailbreakType).To(Equal("jailbreak"))
				Expect(confidence).To(BeNumerically("~", 0.9, 0.001))
			})
		})

		Context("when text is benign with high confidence", func() {
			It("should return false with benign type", func() {
				mockJailbreakModel.classifyResult = candle_binding.ClassResult{
					Class:      1,
					Confidence: 0.9,
				}
				isJailbreak, jailbreakType, confidence, err := classifier.CheckForJailbreak("This is a normal question")
				Expect(err).ToNot(HaveOccurred())
				Expect(isJailbreak).To(BeFalse())
				Expect(jailbreakType).To(Equal("benign"))
				Expect(confidence).To(BeNumerically("~", 0.9, 0.001))
			})
		})

		Context("when jailbreak confidence is below threshold", func() {
			It("should return false even if classified as jailbreak", func() {
				mockJailbreakModel.classifyResult = candle_binding.ClassResult{
					Class:      0,
					Confidence: 0.5,
				}
				isJailbreak, jailbreakType, confidence, err := classifier.CheckForJailbreak("Ambiguous text")
				Expect(err).ToNot(HaveOccurred())
				Expect(isJailbreak).To(BeFalse())
				Expect(jailbreakType).To(Equal("jailbreak"))
				Expect(confidence).To(BeNumerically("~", 0.5, 0.001))
			})
		})

		Context("when model inference fails", func() {
			It("should return error", func() {
				mockJailbreakModel.classifyError = errors.New("model inference failed")
				isJailbreak, jailbreakType, confidence, err := classifier.CheckForJailbreak("Some text")
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("jailbreak classification failed"))
				Expect(isJailbreak).To(BeFalse())
				Expect(jailbreakType).To(Equal(""))
				Expect(confidence).To(BeNumerically("~", 0.0, 0.001))
			})
		})

		Context("when class index is not found in jailbreak mapping", func() {
			It("should return error for unknown class", func() {
				mockJailbreakModel.classifyResult = candle_binding.ClassResult{
					Class:      9,
					Confidence: 0.9,
				}
				isJailbreak, jailbreakType, confidence, err := classifier.CheckForJailbreak("Some text")
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("unknown jailbreak class index"))
				Expect(isJailbreak).To(BeFalse())
				Expect(jailbreakType).To(Equal(""))
				Expect(confidence).To(BeNumerically("~", 0.0, 0.001))
			})
		})
	})

	Describe("analyze content for jailbreak", func() {
		Context("when jailbreak mapping is not initialized", func() {
			It("should return empty list", func() {
				classifier.JailbreakMapping = nil
				hasJailbreak, _, err := classifier.AnalyzeContentForJailbreak([]string{"Some text"})
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("jailbreak detection is not enabled or properly configured"))
				Expect(hasJailbreak).To(BeFalse())
			})
		})

		Context("when 5 texts in total, 1 has jailbreak, 1 has empty text, 1 has model inference failure", func() {
			It("should return 3 results with correct analysis", func() {
				mockJailbreakModel.setMockResponse("text0", 0, 0.9, errors.New("model inference failed"))
				mockJailbreakModel.setMockResponse("text1", 0, 0.3, nil)
				mockJailbreakModel.setMockResponse("text2", 1, 0.9, nil)
				mockJailbreakModel.setMockResponse("text3", 0, 0.9, nil)
				mockJailbreakModel.setMockResponse("", 0, 0.9, nil)
				contentList := []string{"text0", "text1", "text2", "text3", ""}
				hasJailbreak, results, err := classifier.AnalyzeContentForJailbreak(contentList)
				Expect(err).ToNot(HaveOccurred())
				Expect(hasJailbreak).To(BeTrue())
				// only 3 results because the first and the last are skipped because of model inference failure and empty text
				Expect(results).To(HaveLen(3))
				Expect(results[0].IsJailbreak).To(BeFalse())
				Expect(results[0].JailbreakType).To(Equal("jailbreak"))
				Expect(results[0].Confidence).To(BeNumerically("~", 0.3, 0.001))
				Expect(results[1].IsJailbreak).To(BeFalse())
				Expect(results[1].JailbreakType).To(Equal("benign"))
				Expect(results[1].Confidence).To(BeNumerically("~", 0.9, 0.001))
				Expect(results[2].IsJailbreak).To(BeTrue())
				Expect(results[2].JailbreakType).To(Equal("jailbreak"))
				Expect(results[2].Confidence).To(BeNumerically("~", 0.9, 0.001))
			})
		})
	})
})

type MockPIIInitializer struct{ InitError error }

func (m *MockPIIInitializer) Init(modelID string, useCPU bool) error { return m.InitError }

type MockPIIInferenceResponse struct {
	classifyTokensResult candle_binding.TokenClassificationResult
	classifyTokensError  error
}

type MockPIIInference struct {
	MockPIIInferenceResponse
	responseMap map[string]MockPIIInferenceResponse
}

func (m *MockPIIInference) setMockResponse(text string, entities []candle_binding.TokenEntity, err error) {
	m.responseMap[text] = MockPIIInferenceResponse{
		classifyTokensResult: candle_binding.TokenClassificationResult{
			Entities: entities,
		},
		classifyTokensError: err,
	}
}

func (m *MockPIIInference) ClassifyTokens(text string, configPath string) (candle_binding.TokenClassificationResult, error) {
	if response, exists := m.responseMap[text]; exists {
		return response.classifyTokensResult, response.classifyTokensError
	}
	return m.classifyTokensResult, m.classifyTokensError
}

var _ = Describe("PII detection", func() {
	var (
		classifier         *Classifier
		mockPIIInitializer *MockPIIInitializer
		mockPIIModel       *MockPIIInference
	)

	BeforeEach(func() {
		mockPIIInitializer = &MockPIIInitializer{InitError: nil}
		mockPIIModel = &MockPIIInference{}
		mockPIIModel.responseMap = make(map[string]MockPIIInferenceResponse)
		cfg := &config.RouterConfig{}
		cfg.Classifier.PIIModel.ModelID = "test-pii-model"
		cfg.Classifier.PIIModel.PIIMappingPath = "test-pii-mapping-path"
		cfg.Classifier.PIIModel.Threshold = 0.7

		classifier, _ = newClassifierWithOptions(cfg,
			withPII(&PIIMapping{
				LabelToIdx: map[string]int{"PERSON": 0, "EMAIL": 1},
				IdxToLabel: map[string]string{"0": "PERSON", "1": "EMAIL"},
			}, mockPIIInitializer, mockPIIModel),
		)
	})

	Describe("initialize PII classifier", func() {
		It("should succeed", func() {
			err := classifier.initializePIIClassifier()
			Expect(err).ToNot(HaveOccurred())
		})

		Context("when PII mapping is not initialized", func() {
			It("should return error", func() {
				classifier.PIIMapping = nil
				err := classifier.initializePIIClassifier()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("PII detection is not properly configured"))
			})
		})

		Context("when not enough PII types", func() {
			It("should return error", func() {
				classifier.PIIMapping = &PIIMapping{
					LabelToIdx: map[string]int{"PERSON": 0},
					IdxToLabel: map[string]string{"0": "PERSON"},
				}
				err := classifier.initializePIIClassifier()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("not enough PII types for classification"))
			})
		})

		Context("when initialize PII classifier fails", func() {
			It("should return error", func() {
				mockPIIInitializer.InitError = errors.New("initialize PII classifier failed")
				err := classifier.initializePIIClassifier()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("initialize PII classifier failed"))
			})
		})
	})

	Describe("classify PII", func() {
		type row struct {
			ModelID        string
			PIIMappingPath string
			PIIMapping     *PIIMapping
		}

		DescribeTable("when PII detection is not properly configured",
			func(r row) {
				classifier.Config.Classifier.PIIModel.ModelID = r.ModelID
				classifier.Config.Classifier.PIIModel.PIIMappingPath = r.PIIMappingPath
				classifier.PIIMapping = r.PIIMapping
				piiTypes, err := classifier.ClassifyPII("Some text")
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("PII detection is not properly configured"))
				Expect(piiTypes).To(BeEmpty())
			},
			Entry("ModelID is empty", row{ModelID: ""}),
			Entry("PIIMappingPath is empty", row{PIIMappingPath: ""}),
			Entry("PIIMapping is nil", row{PIIMapping: nil}),
		)

		Context("when text is empty", func() {
			It("should return empty list", func() {
				piiTypes, err := classifier.ClassifyPII("")
				Expect(err).ToNot(HaveOccurred())
				Expect(piiTypes).To(BeEmpty())
			})
		})

		Context("when PII entities are detected above threshold", func() {
			It("should return detected PII types", func() {
				mockPIIModel.classifyTokensResult = candle_binding.TokenClassificationResult{
					Entities: []candle_binding.TokenEntity{
						{
							EntityType: "PERSON",
							Text:       "John Doe",
							Start:      0,
							End:        8,
							Confidence: 0.9,
						},
						{
							EntityType: "EMAIL",
							Text:       "john@example.com",
							Start:      9,
							End:        25,
							Confidence: 0.8,
						},
					},
				}

				piiTypes, err := classifier.ClassifyPII("John Doe john@example.com")

				Expect(err).ToNot(HaveOccurred())
				Expect(piiTypes).To(ConsistOf("PERSON", "EMAIL"))
			})
		})

		Context("when PII entities are detected below threshold", func() {
			It("should filter out low confidence entities", func() {
				mockPIIModel.classifyTokensResult = candle_binding.TokenClassificationResult{
					Entities: []candle_binding.TokenEntity{
						{
							EntityType: "PERSON",
							Text:       "John Doe",
							Start:      0,
							End:        8,
							Confidence: 0.9,
						},
						{
							EntityType: "EMAIL",
							Text:       "john@example.com",
							Start:      9,
							End:        25,
							Confidence: 0.5,
						},
					},
				}

				piiTypes, err := classifier.ClassifyPII("John Doe john@example.com")

				Expect(err).ToNot(HaveOccurred())
				Expect(piiTypes).To(ConsistOf("PERSON"))
			})
		})

		Context("when no PII is detected", func() {
			It("should return empty list", func() {
				mockPIIModel.classifyTokensResult = candle_binding.TokenClassificationResult{
					Entities: []candle_binding.TokenEntity{},
				}

				piiTypes, err := classifier.ClassifyPII("Some text")

				Expect(err).ToNot(HaveOccurred())
				Expect(piiTypes).To(BeEmpty())
			})
		})

		Context("when model inference fails", func() {
			It("should return error", func() {
				mockPIIModel.classifyTokensError = errors.New("PII model inference failed")

				piiTypes, err := classifier.ClassifyPII("Some text")

				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("PII token classification error"))
				Expect(piiTypes).To(BeNil())
			})
		})
	})

	Describe("analyze content for PII", func() {
		Context("when PII mapping is not initialized", func() {
			It("should return error", func() {
				classifier.PIIMapping = nil
				hasPII, _, err := classifier.AnalyzeContentForPII([]string{"Some text"})
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("PII detection is not properly configured"))
				Expect(hasPII).To(BeFalse())
			})
		})

		Context("when 5 texts in total, 1 has PII, 1 has empty text, 1 has model inference failure", func() {
			It("should return 3 results with correct analysis", func() {
				mockPIIModel.setMockResponse("Bob", []candle_binding.TokenEntity{}, errors.New("model inference failed"))
				mockPIIModel.setMockResponse("Lisa Smith", []candle_binding.TokenEntity{
					{
						EntityType: "PERSON",
						Text:       "Lisa",
						Start:      0,
						End:        4,
						Confidence: 0.3,
					},
				}, nil)
				mockPIIModel.setMockResponse("Alice Smith", []candle_binding.TokenEntity{
					{
						EntityType: "PERSON",
						Text:       "Alice",
						Start:      0,
						End:        5,
						Confidence: 0.9,
					},
				}, nil)
				mockPIIModel.setMockResponse("No PII here", []candle_binding.TokenEntity{}, nil)
				mockPIIModel.setMockResponse("", []candle_binding.TokenEntity{}, nil)
				contentList := []string{"Bob", "Lisa Smith", "Alice Smith", "No PII here", ""}

				hasPII, results, err := classifier.AnalyzeContentForPII(contentList)

				Expect(err).ToNot(HaveOccurred())
				Expect(hasPII).To(BeTrue())
				// only 3 results because the first and the last are skipped because of model inference failure and empty text
				Expect(results).To(HaveLen(3))
				Expect(results[0].HasPII).To(BeFalse())
				Expect(results[0].Entities).To(BeEmpty())
				Expect(results[1].HasPII).To(BeTrue())
				Expect(results[1].Entities).To(HaveLen(1))
				Expect(results[1].Entities[0].EntityType).To(Equal("PERSON"))
				Expect(results[1].Entities[0].Text).To(Equal("Alice"))
				Expect(results[2].HasPII).To(BeFalse())
				Expect(results[2].Entities).To(BeEmpty())
			})
		})
	})

	Describe("detect PII in content", func() {
		Context("when 5 texts in total, 2 has PII, 1 has empty text, 1 has model inference failure", func() {
			It("should return 2 detected PII types", func() {
				mockPIIModel.setMockResponse("Bob", []candle_binding.TokenEntity{}, errors.New("model inference failed"))
				mockPIIModel.setMockResponse("Lisa Smith", []candle_binding.TokenEntity{
					{
						EntityType: "PERSON",
						Text:       "Lisa",
						Start:      0,
						End:        4,
						Confidence: 0.8,
					},
				}, nil)
				mockPIIModel.setMockResponse("Alice Smith alice@example.com", []candle_binding.TokenEntity{
					{
						EntityType: "PERSON",
						Text:       "Alice",
						Start:      0,
						End:        5,
						Confidence: 0.9,
					}, {
						EntityType: "EMAIL",
						Text:       "alice@example.com",
						Start:      12,
						End:        29,
						Confidence: 0.9,
					},
				}, nil)
				mockPIIModel.setMockResponse("No PII here", []candle_binding.TokenEntity{}, nil)
				mockPIIModel.setMockResponse("", []candle_binding.TokenEntity{}, nil)
				contentList := []string{"Bob", "Lisa Smith", "Alice Smith alice@example.com", "No PII here", ""}

				detectedPII := classifier.DetectPIIInContent(contentList)

				Expect(detectedPII).To(ConsistOf("PERSON", "EMAIL"))
			})
		})
	})
})

var _ = Describe("get models for category", func() {
	var c *Classifier

	BeforeEach(func() {
		c, _ = newClassifierWithOptions(&config.RouterConfig{
			Categories: []config.Category{
				{
					Name: "Toxicity",
					ModelScores: []config.ModelScore{
						{Model: "m1"}, {Model: "m2"},
					},
				},
				{
					Name:        "Toxicity", // duplicate name, should be ignored by "first wins"
					ModelScores: []config.ModelScore{{Model: "mX"}},
				},
				{
					Name:        "Jailbreak",
					ModelScores: []config.ModelScore{{Model: "jb1"}},
				},
			},
		})
	})

	type row struct {
		query string
		want  []string
	}

	DescribeTable("lookup behavior",
		func(r row) {
			got := c.GetModelsForCategory(r.query)
			Expect(got).To(Equal(r.want))
		},

		Entry("case-insensitive match", row{query: "toxicity", want: []string{"m1", "m2"}}),
		Entry("no match returns nil slice", row{query: "NotExist", want: nil}),
		Entry("another category", row{query: "JAILBREAK", want: []string{"jb1"}}),
	)
})

func TestUpdateBestModel(t *testing.T) {

	classifier := &Classifier{}

	bestScore := 0.5
	bestModel := "old-model"

	classifier.updateBestModel(0.8, "new-model", &bestScore, &bestModel)
	if bestScore != 0.8 || bestModel != "new-model" {
		t.Errorf("update: got bestScore=%v, bestModel=%v", bestScore, bestModel)
	}

	classifier.updateBestModel(0.7, "another-model", &bestScore, &bestModel)
	if bestScore != 0.8 || bestModel != "new-model" {
		t.Errorf("not update: got bestScore=%v, bestModel=%v", bestScore, bestModel)
	}
}

func TestForEachModelScore(t *testing.T) {

	c := &Classifier{}
	cat := &config.Category{
		ModelScores: []config.ModelScore{
			{Model: "model-a", Score: 0.9},
			{Model: "model-b", Score: 0.8},
			{Model: "model-c", Score: 0.7},
		},
	}

	var models []string
	c.forEachModelScore(cat, func(ms config.ModelScore) {
		models = append(models, ms.Model)
	})

	expected := []string{"model-a", "model-b", "model-c"}
	if len(models) != len(expected) {
		t.Fatalf("expected %d models, got %d", len(expected), len(models))
	}
	for i, m := range expected {
		if models[i] != m {
			t.Errorf("expected model %s at index %d, got %s", m, i, models[i])
		}
	}
}
