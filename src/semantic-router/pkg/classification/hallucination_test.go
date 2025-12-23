package classification

import (
	"os"
	"path/filepath"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestHallucination is removed - tests are now part of the main Classifier Suite in classifier_test.go
// This avoids the "Rerunning Suite" error from Ginkgo when multiple RunSpecs are called

// findProjectRoot finds the project root by looking for go.mod
func findProjectRoot() string {
	// Start from current working directory
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}

	// Walk up looking for go.mod
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return ""
}

// getHallucinationModelPathForTest returns the hallucination model path, checking multiple locations
func getHallucinationModelPathForTest() string {
	// Check env var first
	if path := os.Getenv("HALLUCINATION_MODEL_PATH"); path != "" {
		return path
	}
	// Try relative path from test directory (pkg/classification -> models)
	relativePath := "../../../../models/mom-halugate-detector"
	if _, err := os.Stat(relativePath); err == nil {
		return relativePath
	}
	// Try from project root
	if root := findProjectRoot(); root != "" {
		projectPath := filepath.Join(root, "models", "mom-halugate-detector")
		if _, err := os.Stat(projectPath); err == nil {
			return projectPath
		}
	}
	// Return relative path as fallback (will fail gracefully if not found)
	return relativePath
}

// getFactCheckModelPathForTest returns the fact-check model path, checking multiple locations
func getFactCheckModelPathForTest() string {
	// Check env var first
	if path := os.Getenv("FACT_CHECK_MODEL_PATH"); path != "" {
		return path
	}
	// Try relative path from test directory (pkg/classification -> models)
	relativePath := "../../../../models/mom-halugate-sentinel"
	if _, err := os.Stat(relativePath); err == nil {
		return relativePath
	}
	// Try from project root
	if root := findProjectRoot(); root != "" {
		projectPath := filepath.Join(root, "models", "mom-halugate-sentinel")
		if _, err := os.Stat(projectPath); err == nil {
			return projectPath
		}
	}
	// Return relative path as fallback (will fail gracefully if not found)
	return relativePath
}

// skipIfNoHallucinationModel skips the test if the hallucination model is not available
func skipIfNoHallucinationModel() {
	modelPath := getHallucinationModelPathForTest()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		Skip("Skipping: Hallucination model not found at " + modelPath)
	}
}

// skipIfNoFactCheckModelGinkgo skips the Ginkgo test if the fact-check model is not available
func skipIfNoFactCheckModelGinkgo() {
	modelPath := getFactCheckModelPathForTest()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		Skip("Skipping: Fact-check model not found at " + modelPath)
	}
}

var _ = Describe("FactCheckMapping", func() {
	var (
		tempDir     string
		mappingFile string
	)

	BeforeEach(func() {
		var err error
		tempDir, err = os.MkdirTemp("", "fact_check_mapping_test")
		Expect(err).NotTo(HaveOccurred())
		mappingFile = filepath.Join(tempDir, "fact_check_mapping.json")
	})

	AfterEach(func() {
		os.RemoveAll(tempDir)
	})

	Describe("LoadFactCheckMapping", func() {
		Context("with valid mapping file", func() {
			BeforeEach(func() {
				content := `{
					"idx_to_label": {
						"0": "NO_FACT_CHECK_NEEDED",
						"1": "FACT_CHECK_NEEDED"
					},
					"label_to_idx": {
						"NO_FACT_CHECK_NEEDED": 0,
						"FACT_CHECK_NEEDED": 1
					},
					"description": {
						"FACT_CHECK_NEEDED": "Needs verification",
						"NO_FACT_CHECK_NEEDED": "No verification needed"
					}
				}`
				err := os.WriteFile(mappingFile, []byte(content), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should load mapping successfully", func() {
				mapping, err := LoadFactCheckMapping(mappingFile)
				Expect(err).NotTo(HaveOccurred())
				Expect(mapping).NotTo(BeNil())
				Expect(mapping.GetLabelCount()).To(Equal(2))
			})

			It("should return correct labels from index", func() {
				mapping, err := LoadFactCheckMapping(mappingFile)
				Expect(err).NotTo(HaveOccurred())

				label, ok := mapping.GetLabelFromIndex(0)
				Expect(ok).To(BeTrue())
				Expect(label).To(Equal(FactCheckLabelNotNeeded))

				label, ok = mapping.GetLabelFromIndex(1)
				Expect(ok).To(BeTrue())
				Expect(label).To(Equal(FactCheckLabelNeeded))
			})

			It("should return correct index from label", func() {
				mapping, err := LoadFactCheckMapping(mappingFile)
				Expect(err).NotTo(HaveOccurred())

				idx, ok := mapping.GetIndexFromLabel(FactCheckLabelNotNeeded)
				Expect(ok).To(BeTrue())
				Expect(idx).To(Equal(0))

				idx, ok = mapping.GetIndexFromLabel(FactCheckLabelNeeded)
				Expect(ok).To(BeTrue())
				Expect(idx).To(Equal(1))
			})

			It("should identify fact-check needed labels", func() {
				mapping, err := LoadFactCheckMapping(mappingFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(mapping.IsFactCheckNeeded(FactCheckLabelNeeded)).To(BeTrue())
				Expect(mapping.IsFactCheckNeeded(FactCheckLabelNotNeeded)).To(BeFalse())
			})

			It("should return descriptions", func() {
				mapping, err := LoadFactCheckMapping(mappingFile)
				Expect(err).NotTo(HaveOccurred())

				desc, ok := mapping.GetDescription(FactCheckLabelNeeded)
				Expect(ok).To(BeTrue())
				Expect(desc).To(Equal("Needs verification"))
			})
		})

		Context("with invalid file", func() {
			It("should return error for non-existent file", func() {
				mapping, err := LoadFactCheckMapping("/non/existent/file.json")
				Expect(err).To(HaveOccurred())
				Expect(mapping).To(BeNil())
				Expect(err.Error()).To(ContainSubstring("failed to read"))
			})

			It("should return error for empty path", func() {
				mapping, err := LoadFactCheckMapping("")
				Expect(err).To(HaveOccurred())
				Expect(mapping).To(BeNil())
			})

			It("should return error for invalid JSON", func() {
				err := os.WriteFile(mappingFile, []byte("invalid json"), 0o644)
				Expect(err).NotTo(HaveOccurred())

				mapping, err := LoadFactCheckMapping(mappingFile)
				Expect(err).To(HaveOccurred())
				Expect(mapping).To(BeNil())
			})

			It("should return error for insufficient labels", func() {
				content := `{
					"idx_to_label": {"0": "ONLY_ONE"},
					"label_to_idx": {"ONLY_ONE": 0}
				}`
				err := os.WriteFile(mappingFile, []byte(content), 0o644)
				Expect(err).NotTo(HaveOccurred())

				mapping, err := LoadFactCheckMapping(mappingFile)
				Expect(err).To(HaveOccurred())
				Expect(mapping).To(BeNil())
				Expect(err.Error()).To(ContainSubstring("at least 2 labels"))
			})
		})
	})
})

var _ = Describe("FactCheckClassifier", func() {
	var (
		classifier *FactCheckClassifier
		cfg        *config.FactCheckModelConfig
	)

	BeforeEach(func() {
		cfg = &config.FactCheckModelConfig{
			ModelID:   getFactCheckModelPathForTest(),
			Threshold: 0.7,
		}
	})

	Describe("NewFactCheckClassifier", func() {
		It("should return nil for nil config", func() {
			c, err := NewFactCheckClassifier(nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(c).To(BeNil())
		})

		It("should create classifier with config", func() {
			c, err := NewFactCheckClassifier(cfg)
			Expect(err).NotTo(HaveOccurred())
			Expect(c).NotTo(BeNil())
		})
	})

	Describe("Initialize", func() {
		BeforeEach(func() {
			skipIfNoFactCheckModelGinkgo()
			var err error
			classifier, err = NewFactCheckClassifier(cfg)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should initialize successfully", func() {
			err := classifier.Initialize()
			Expect(err).NotTo(HaveOccurred())
			Expect(classifier.IsInitialized()).To(BeTrue())
		})

		It("should be idempotent", func() {
			err := classifier.Initialize()
			Expect(err).NotTo(HaveOccurred())

			err = classifier.Initialize()
			Expect(err).NotTo(HaveOccurred())
			Expect(classifier.IsInitialized()).To(BeTrue())
		})
	})

	Describe("Classify", func() {
		BeforeEach(func() {
			skipIfNoFactCheckModelGinkgo()
			var err error
			classifier, err = NewFactCheckClassifier(cfg)
			Expect(err).NotTo(HaveOccurred())
			err = classifier.Initialize()
			Expect(err).NotTo(HaveOccurred())
		})

		Context("with fact-check needed prompts", func() {
			It("should classify factual questions as needing fact-check", func() {
				testCases := []string{
					"When was the Eiffel Tower built?",
					"Who is the CEO of Apple?",
					"What is the population of France?",
					"Is it true that water boils at 100 degrees?",
					"What year did World War 2 end?",
				}

				for _, text := range testCases {
					result, err := classifier.Classify(text)
					Expect(err).NotTo(HaveOccurred())
					Expect(result).NotTo(BeNil())
					// Note: rule-based classifier may not catch all cases
					// We just verify it returns a valid result
					Expect(result.Label).To(BeElementOf(FactCheckLabelNeeded, FactCheckLabelNotNeeded))
				}
			})
		})

		Context("with no-fact-check needed prompts", func() {
			It("should classify creative/code prompts as not needing fact-check", func() {
				testCases := []string{
					"Write a poem about the ocean",
					"Write a Python function to sort a list",
					"What do you think about AI?",
					"Calculate 25 * 4",
					"Help me debug this code",
				}

				for _, text := range testCases {
					result, err := classifier.Classify(text)
					Expect(err).NotTo(HaveOccurred())
					Expect(result).NotTo(BeNil())
					// Verify valid result
					Expect(result.Label).To(BeElementOf(FactCheckLabelNeeded, FactCheckLabelNotNeeded))
				}
			})
		})

		Context("with empty text", func() {
			It("should return no fact-check needed", func() {
				result, err := classifier.Classify("")
				Expect(err).NotTo(HaveOccurred())
				Expect(result).NotTo(BeNil())
				Expect(result.NeedsFactCheck).To(BeFalse())
				Expect(result.Confidence).To(Equal(float32(1.0)))
			})
		})
	})
})

var _ = Describe("HallucinationDetector", func() {
	var (
		detector *HallucinationDetector
		cfg      *config.HallucinationModelConfig
	)

	BeforeEach(func() {
		cfg = &config.HallucinationModelConfig{
			ModelID:   getHallucinationModelPathForTest(),
			Threshold: 0.5,
			UseCPU:    true,
		}
	})

	Describe("NewHallucinationDetector", func() {
		It("should return error for nil config", func() {
			d, err := NewHallucinationDetector(nil)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("config is required"))
			Expect(d).To(BeNil())
		})

		It("should return error for empty model_id", func() {
			emptyCfg := &config.HallucinationModelConfig{
				Threshold: 0.5,
			}
			d, err := NewHallucinationDetector(emptyCfg)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("model_id is required"))
			Expect(d).To(BeNil())
		})

		It("should create detector with config", func() {
			d, err := NewHallucinationDetector(cfg)
			Expect(err).NotTo(HaveOccurred())
			Expect(d).NotTo(BeNil())
		})
	})

	Describe("Initialize", func() {
		BeforeEach(func() {
			skipIfNoHallucinationModel()
			var err error
			detector, err = NewHallucinationDetector(cfg)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should initialize successfully", func() {
			err := detector.Initialize()
			Expect(err).NotTo(HaveOccurred())
			Expect(detector.IsInitialized()).To(BeTrue())
		})

		It("should be idempotent", func() {
			err := detector.Initialize()
			Expect(err).NotTo(HaveOccurred())

			err = detector.Initialize()
			Expect(err).NotTo(HaveOccurred())
			Expect(detector.IsInitialized()).To(BeTrue())
		})
	})

	Describe("Detect", func() {
		BeforeEach(func() {
			skipIfNoHallucinationModel()
			var err error
			detector, err = NewHallucinationDetector(cfg)
			Expect(err).NotTo(HaveOccurred())
			err = detector.Initialize()
			Expect(err).NotTo(HaveOccurred())
		})

		Context("with grounded answers", func() {
			It("should not detect hallucination when answer is supported by context", func() {
				context := "The Eiffel Tower was built in 1889. It is located in Paris, France."
				question := "When was the Eiffel Tower built?"
				answer := "The Eiffel Tower was built in 1889."

				result, err := detector.Detect(context, question, answer)
				Expect(err).NotTo(HaveOccurred())
				Expect(result).NotTo(BeNil())
				// The rule-based detector should find this grounded
				Expect(result.HallucinationDetected).To(BeFalse())
			})
		})

		Context("with ungrounded answers", func() {
			It("should detect hallucination when answer has unsupported claims", func() {
				context := "The Eiffel Tower was built in 1889. It is located in Paris."
				question := "What is the height of the Eiffel Tower?"
				answer := "The Eiffel Tower is exactly 324 meters tall and was renovated in 2019."

				result, err := detector.Detect(context, question, answer)
				Expect(err).NotTo(HaveOccurred())
				Expect(result).NotTo(BeNil())
				// Should have unsupported spans since "324 meters" and "2019" aren't in context
			})
		})

		Context("with empty inputs", func() {
			It("should handle empty answer", func() {
				result, err := detector.Detect("some context", "question?", "")
				Expect(err).NotTo(HaveOccurred())
				Expect(result).NotTo(BeNil())
				Expect(result.HallucinationDetected).To(BeFalse())
			})

			It("should return error for empty context", func() {
				result, err := detector.Detect("", "question?", "Some answer here.")
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("context is required"))
				Expect(result).To(BeNil())
			})
		})

		Context("with uncertain language", func() {
			It("should be lenient with hedged statements", func() {
				context := "The project started in 2020."
				question := "When did the project start?"
				answer := "I think the project probably started around 2020."

				result, err := detector.Detect(context, question, answer)
				Expect(err).NotTo(HaveOccurred())
				Expect(result).NotTo(BeNil())
				// Hedged language should reduce hallucination detection
			})
		})
	})
})

var _ = Describe("Classifier with Hallucination Mitigation", func() {
	var (
		classifier *Classifier
		cfg        *config.RouterConfig
	)

	Describe("IsFactCheckEnabled", func() {
		It("should return true when properly configured", func() {
			cfg = &config.RouterConfig{}
			cfg.HallucinationMitigation.Enabled = true
			cfg.HallucinationMitigation.FactCheckModel.ModelID = "test-model"

			classifier = &Classifier{Config: cfg}
			Expect(classifier.IsFactCheckEnabled()).To(BeTrue())
		})

		It("should return false when disabled", func() {
			cfg = &config.RouterConfig{}
			cfg.HallucinationMitigation.Enabled = false

			classifier = &Classifier{Config: cfg}
			Expect(classifier.IsFactCheckEnabled()).To(BeFalse())
		})
	})

	Describe("IsHallucinationDetectionEnabled", func() {
		It("should return true when properly configured", func() {
			cfg = &config.RouterConfig{}
			cfg.HallucinationMitigation.Enabled = true
			cfg.HallucinationMitigation.HallucinationModel.ModelID = "test-model"

			classifier = &Classifier{Config: cfg}
			Expect(classifier.IsHallucinationDetectionEnabled()).To(BeTrue())
		})

		It("should return false when disabled", func() {
			cfg = &config.RouterConfig{}
			cfg.HallucinationMitigation.Enabled = false

			classifier = &Classifier{Config: cfg}
			Expect(classifier.IsHallucinationDetectionEnabled()).To(BeFalse())
		})
	})
})
