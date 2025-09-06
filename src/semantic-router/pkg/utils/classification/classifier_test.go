package classification

import (
	"errors"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/semantic-router/pkg/config"
)

func TestClassifier(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Classifier Suite")
}

type MockCategoryInference struct {
	classifyResult candle_binding.ClassResult
	classifyError  error
}

func (m *MockCategoryInference) Classify(text string) (candle_binding.ClassResult, error) {
	return m.classifyResult, m.classifyError
}

var _ = Describe("ClassifyCategory", func() {
	var (
		classifier        *Classifier
		mockCategoryModel *MockCategoryInference
	)

	BeforeEach(func() {
		mockCategoryModel = &MockCategoryInference{}
		cfg := &config.RouterConfig{}
		cfg.Classifier.CategoryModel.Threshold = 0.5

		classifier = &Classifier{
			categoryInference: mockCategoryModel,
			Config:            cfg,
			CategoryMapping: &CategoryMapping{
				CategoryToIdx: map[string]int{"technology": 0, "sports": 1, "politics": 2},
				IdxToCategory: map[string]string{"0": "technology", "1": "sports", "2": "politics"},
			},
		}
	})

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

type MockJailbreakInference struct {
	classifyResult candle_binding.ClassResult
	classifyError  error
}

func (m *MockJailbreakInference) Classify(text string) (candle_binding.ClassResult, error) {
	return m.classifyResult, m.classifyError
}

var _ = Describe("CheckForJailbreak", func() {
	var (
		classifier         *Classifier
		mockJailbreakModel *MockJailbreakInference
	)

	BeforeEach(func() {
		mockJailbreakModel = &MockJailbreakInference{}
		cfg := &config.RouterConfig{}
		cfg.PromptGuard.Enabled = true
		cfg.PromptGuard.ModelID = "test-model"
		cfg.PromptGuard.JailbreakMappingPath = "test-mapping"
		cfg.PromptGuard.Threshold = 0.7

		classifier = &Classifier{
			jailbreakInference: mockJailbreakModel,
			Config:             cfg,
			JailbreakMapping: &JailbreakMapping{
				LabelToIdx: map[string]int{"jailbreak": 0, "benign": 1},
				IdxToLabel: map[string]string{"0": "jailbreak", "1": "benign"},
			},
			JailbreakInitialized: true,
		}
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

type PIIInferenceResponse struct {
	classifyTokensResult candle_binding.TokenClassificationResult
	classifyTokensError  error
}

type MockPIIInference struct {
	PIIInferenceResponse
	responseMap map[string]PIIInferenceResponse
}

func (m *MockPIIInference) ClassifyTokens(text string, configPath string) (candle_binding.TokenClassificationResult, error) {
	if response, exists := m.responseMap[text]; exists {
		return response.classifyTokensResult, response.classifyTokensError
	}
	return m.classifyTokensResult, m.classifyTokensError
}

var _ = Describe("PIIClassification", func() {
	var (
		classifier   *Classifier
		mockPIIModel *MockPIIInference
	)

	BeforeEach(func() {
		mockPIIModel = &MockPIIInference{}
		cfg := &config.RouterConfig{}
		cfg.Classifier.PIIModel.ModelID = "test-pii-model"
		cfg.Classifier.PIIModel.Threshold = 0.7

		classifier = &Classifier{
			piiInference: mockPIIModel,
			Config:       cfg,
			PIIMapping: &PIIMapping{
				LabelToIdx: map[string]int{"PERSON": 0, "EMAIL": 1},
				IdxToLabel: map[string]string{"0": "PERSON", "1": "EMAIL"},
			},
		}
	})

	Describe("ClassifyPII", func() {
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

	Describe("AnalyzeContentForPII", func() {
		Context("when some texts contain PII", func() {
			It("should return detailed analysis for each text", func() {
				mockPIIModel.responseMap = make(map[string]PIIInferenceResponse)
				mockPIIModel.responseMap["Alice Smith"] = PIIInferenceResponse{
					classifyTokensResult: candle_binding.TokenClassificationResult{
						Entities: []candle_binding.TokenEntity{
							{
								EntityType: "PERSON",
								Text:       "Alice",
								Start:      0,
								End:        5,
								Confidence: 0.9,
							},
						},
					},
					classifyTokensError: nil,
				}

				mockPIIModel.responseMap["No PII here"] = PIIInferenceResponse{}

				contentList := []string{"Alice Smith", "No PII here"}
				hasPII, results, err := classifier.AnalyzeContentForPII(contentList)

				Expect(err).ToNot(HaveOccurred())
				Expect(hasPII).To(BeTrue())
				Expect(results).To(HaveLen(2))
				Expect(results[0].HasPII).To(BeTrue())
				Expect(results[0].Entities).To(HaveLen(1))
				Expect(results[0].Entities[0].EntityType).To(Equal("PERSON"))
				Expect(results[0].Entities[0].Text).To(Equal("Alice"))
				Expect(results[1].HasPII).To(BeFalse())
				Expect(results[1].Entities).To(BeEmpty())
			})
		})

		Context("when model inference fails", func() {
			It("should return false for hasPII and empty results", func() {
				mockPIIModel.classifyTokensError = errors.New("model failed")

				contentList := []string{"Text 1", "Text 2"}
				hasPII, results, err := classifier.AnalyzeContentForPII(contentList)

				Expect(err).ToNot(HaveOccurred())
				Expect(hasPII).To(BeFalse())
				Expect(results).To(BeEmpty())
			})
		})
	})
})

func TestUpdateBestModel(t *testing.T) {

	classifier := &Classifier{}

	bestScore := 0.5
	bestQuality := 0.5
	bestModel := "old-model"

	classifier.updateBestModel(0.8, 0.9, "new-model", &bestScore, &bestQuality, &bestModel)
	if bestScore != 0.8 || bestQuality != 0.9 || bestModel != "new-model" {
		t.Errorf("update: got bestScore=%v, bestQuality=%v, bestModel=%v", bestScore, bestQuality, bestModel)
	}

	classifier.updateBestModel(0.7, 0.7, "another-model", &bestScore, &bestQuality, &bestModel)
	if bestScore != 0.8 || bestQuality != 0.9 || bestModel != "new-model" {
		t.Errorf("not update: got bestScore=%v, bestQuality=%v, bestModel=%v", bestScore, bestQuality, bestModel)
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
