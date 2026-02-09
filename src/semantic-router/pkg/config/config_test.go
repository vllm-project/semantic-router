package config

import (
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"gopkg.in/yaml.v3"
)

func TestConfig(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Config Suite")
}

var _ = Describe("Config Package", func() {
	var (
		tempDir    string
		configFile string
	)

	BeforeEach(func() {
		var err error
		tempDir, err = os.MkdirTemp("", "config_test")
		Expect(err).NotTo(HaveOccurred())
		configFile = filepath.Join(tempDir, "config.yaml")
	})

	AfterEach(func() {
		os.RemoveAll(tempDir)
		// Reset the singleton config for next test
		ResetConfig()
	})

	Describe("Load", func() {
		Context("with valid YAML configuration", func() {
			BeforeEach(func() {
				validConfig := `
bert_model:
  model_id: "test-bert-model"
  threshold: 0.8
  use_cpu: true

classifier:
  category_model:
    model_id: "test-category-model"
    threshold: 0.7
    use_cpu: false
    use_modernbert: true
    category_mapping_path: "/path/to/category.json"
  pii_model:
    model_id: "test-pii-model"
    threshold: 0.6
    use_cpu: true
    use_modernbert: false
    pii_mapping_path: "/path/to/pii.json"

categories:
  - name: "general"
    description: "General purpose tasks"

decisions:
  - name: "general"
    description: "General purpose decision"
    priority: 100
    rules:
      operator: AND
      conditions:
        - type: keyword
          name: general_keywords
    modelRefs:
      - model: "model-a"
        use_reasoning: true

default_model: "model-b"

semantic_cache:
  enabled: true
  similarity_threshold: 0.9
  max_entries: 1000
  ttl_seconds: 3600

prompt_guard:
  enabled: true
  model_id: "test-jailbreak-model"
  threshold: 0.5
  use_cpu: false
  use_modernbert: true
  jailbreak_mapping_path: "/path/to/jailbreak.json"

vllm_endpoints:
  - name: "endpoint1"
    address: "127.0.0.1"
    port: 8000
    weight: 1
  - name: "endpoint2"
    address: "127.0.0.1"
    port: 8000
    weight: 2

model_config:
  "model-a":
    preferred_endpoints: ["endpoint1"]
  "model-b":
    preferred_endpoints: ["endpoint1", "endpoint2"]

tools:
  enabled: true
  top_k: 5
  similarity_threshold: 0.8
  tools_db_path: "/path/to/tools.json"
  fallback_to_empty: true
`
				err := os.WriteFile(configFile, []byte(validConfig), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should load configuration successfully", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())
				Expect(cfg).NotTo(BeNil())

				// Verify BERT model config
				Expect(cfg.BertModel.ModelID).To(Equal("test-bert-model"))
				Expect(cfg.BertModel.Threshold).To(Equal(float32(0.8)))
				Expect(cfg.BertModel.UseCPU).To(BeTrue())

				// Verify classifier config
				Expect(cfg.Classifier.CategoryModel.ModelID).To(Equal("test-category-model"))
				Expect(cfg.Classifier.CategoryModel.UseModernBERT).To(BeTrue())

				// Verify categories
				Expect(cfg.Categories).To(HaveLen(1))
				Expect(cfg.Categories[0].Name).To(Equal("general"))

				// Verify decisions
				Expect(cfg.Decisions).To(HaveLen(1))
				Expect(cfg.Decisions[0].Name).To(Equal("general"))
				Expect(cfg.Decisions[0].ModelRefs).To(HaveLen(1))
				Expect(cfg.Decisions[0].ModelRefs[0].Model).To(Equal("model-a"))

				// Verify default model
				Expect(cfg.DefaultModel).To(Equal("model-b"))

				// Verify semantic cache (legacy fields)
				Expect(cfg.SemanticCache.Enabled).To(BeTrue())
				Expect(*cfg.SemanticCache.SimilarityThreshold).To(Equal(float32(0.9)))
				Expect(cfg.SemanticCache.MaxEntries).To(Equal(1000))
				Expect(cfg.SemanticCache.TTLSeconds).To(Equal(3600))

				// New fields should have default/zero values when not specified
				Expect(cfg.SemanticCache.BackendType).To(BeEmpty())
				Expect(cfg.SemanticCache.BackendConfigPath).To(BeEmpty())

				// Verify prompt guard
				Expect(cfg.PromptGuard.Enabled).To(BeTrue())
				Expect(cfg.PromptGuard.ModelID).To(Equal("test-jailbreak-model"))
				Expect(cfg.PromptGuard.UseModernBERT).To(BeTrue())

				// Verify model config
				Expect(cfg.ModelConfig).To(HaveKey("model-a"))
				Expect(cfg.ModelConfig["model-a"].PreferredEndpoints).To(ContainElement("endpoint1"))

				// Verify tools config
				Expect(cfg.Tools.Enabled).To(BeTrue())
				Expect(cfg.Tools.TopK).To(Equal(5))
				Expect(*cfg.Tools.SimilarityThreshold).To(Equal(float32(0.8)))
				Expect(cfg.Tools.AdvancedFiltering).To(BeNil())

				// Verify vLLM endpoints config
				Expect(cfg.VLLMEndpoints).To(HaveLen(2))
				Expect(cfg.VLLMEndpoints[0].Name).To(Equal("endpoint1"))
				Expect(cfg.VLLMEndpoints[0].Address).To(Equal("127.0.0.1"))
				Expect(cfg.VLLMEndpoints[0].Port).To(Equal(8000))
				Expect(cfg.VLLMEndpoints[0].Weight).To(Equal(1))

				Expect(cfg.VLLMEndpoints[1].Name).To(Equal("endpoint2"))
				Expect(cfg.VLLMEndpoints[1].Address).To(Equal("127.0.0.1"))
				Expect(cfg.VLLMEndpoints[1].Weight).To(Equal(2))

				// Verify model preferred endpoints
				Expect(cfg.ModelConfig["model-a"].PreferredEndpoints).To(ContainElement("endpoint1"))
				Expect(cfg.ModelConfig["model-b"].PreferredEndpoints).To(ContainElements("endpoint1", "endpoint2"))
			})

			It("should return the same config instance on subsequent calls (singleton)", func() {
				cfg1, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				cfg2, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg1).To(BeIdenticalTo(cfg2))
			})
		})

		Context("with advanced tool filtering configuration", func() {
			BeforeEach(func() {
				configContent := `
bert_model:
  model_id: "test-bert-model"
  threshold: 0.8
  use_cpu: true

decisions:
  - name: "general"
    priority: 100
    rules:
      operator: AND
      conditions:
        - type: keyword
          name: general_keywords
    modelRefs:
      - model: "model-a"
        use_reasoning: true

default_model: "model-b"

tools:
  enabled: true
  top_k: 5
  similarity_threshold: 0.8
  tools_db_path: "/path/to/tools.json"
  fallback_to_empty: true
  advanced_filtering:
    enabled: true
    candidate_pool_size: 20
    min_lexical_overlap: 1
    min_combined_score: 0.35
    weights:
      embed: 0.7
      lexical: 0.2
      tag: 0.05
      name: 0.05
      category: 0.1
    use_category_filter: true
    category_confidence_threshold: 0.6
    allow_tools: ["get_weather"]
    block_tools: ["send_email"]
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should parse advanced tool filtering settings", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())
				Expect(cfg).NotTo(BeNil())

				Expect(cfg.Tools.AdvancedFiltering).NotTo(BeNil())
				advanced := cfg.Tools.AdvancedFiltering
				Expect(advanced.Enabled).To(BeTrue())
				Expect(*advanced.CandidatePoolSize).To(Equal(20))
				Expect(*advanced.MinLexicalOverlap).To(Equal(1))
				Expect(*advanced.MinCombinedScore).To(BeNumerically("~", 0.35, 0.0001))
				Expect(advanced.UseCategoryFilter).NotTo(BeNil())
				Expect(*advanced.UseCategoryFilter).To(BeTrue())
				Expect(*advanced.CategoryConfidenceThreshold).To(BeNumerically("~", 0.6, 0.0001))

				Expect(advanced.Weights.Embed).NotTo(BeNil())
				Expect(*advanced.Weights.Embed).To(BeNumerically("~", 0.7, 0.0001))
				Expect(*advanced.Weights.Lexical).To(BeNumerically("~", 0.2, 0.0001))
				Expect(*advanced.Weights.Tag).To(BeNumerically("~", 0.05, 0.0001))
				Expect(*advanced.Weights.Name).To(BeNumerically("~", 0.05, 0.0001))
				Expect(*advanced.Weights.Category).To(BeNumerically("~", 0.1, 0.0001))

				Expect(advanced.AllowTools).To(ContainElement("get_weather"))
				Expect(advanced.BlockTools).To(ContainElement("send_email"))
			})
		})

		Context("with invalid advanced tool filtering configuration", func() {
			BeforeEach(func() {
				configContent := `
decisions:
  - name: "general"
    priority: 100
    rules:
      operator: AND
      conditions:
        - type: keyword
          name: general_keywords
    modelRefs:
      - model: "model-a"
        use_reasoning: true

default_model: "model-b"

tools:
  enabled: true
  top_k: 5
  tools_db_path: "/path/to/tools.json"
  fallback_to_empty: true
  advanced_filtering:
    enabled: true
    min_combined_score: 1.5
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should reject out-of-range values", func() {
				_, err := Load(configFile)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("min_combined_score must be between 0.0 and 1.0"))
			})
		})

		Context("with missing config file", func() {
			It("should return an error", func() {
				cfg, err := Load("/nonexistent/config.yaml")
				Expect(err).To(HaveOccurred())
				Expect(cfg).To(BeNil())
				Expect(err.Error()).To(ContainSubstring("failed to read config file"))
			})
		})

		Context("with observability metrics configuration", func() {
			It("should default to enabled when metrics block is omitted", func() {
				configContent := `
observability:
  tracing:
    enabled: false
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())
				Expect(cfg.Observability.Metrics.Enabled).To(BeNil())
			})

			It("should honor explicit metrics disable flag", func() {
				configContent := `
observability:
  metrics:
    enabled: false
  tracing:
    enabled: false
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())
				Expect(cfg.Observability.Metrics.Enabled).NotTo(BeNil())
				Expect(*cfg.Observability.Metrics.Enabled).To(BeFalse())
			})

			It("should honor explicit metrics enable flag", func() {
				configContent := `
observability:
  metrics:
    enabled: true
  tracing:
    enabled: false
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())
				Expect(cfg.Observability.Metrics.Enabled).NotTo(BeNil())
				Expect(*cfg.Observability.Metrics.Enabled).To(BeTrue())
			})
		})

		Context("with invalid YAML syntax", func() {
			BeforeEach(func() {
				invalidYAML := `
bert_model:
  model_id: "test-model"
  invalid: [ unclosed array
`
				err := os.WriteFile(configFile, []byte(invalidYAML), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should return a parsing error", func() {
				cfg, err := Load(configFile)
				Expect(err).To(HaveOccurred())
				Expect(cfg).To(BeNil())
				Expect(err.Error()).To(ContainSubstring("failed to parse config file"))
			})
		})

		Context("with empty config file", func() {
			BeforeEach(func() {
				err := os.WriteFile(configFile, []byte(""), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should load successfully with zero values", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())
				Expect(cfg).NotTo(BeNil())
				Expect(cfg.BertModel.ModelID).To(BeEmpty())
				Expect(cfg.DefaultModel).To(BeEmpty())
			})
		})

		Context("concurrent access", func() {
			BeforeEach(func() {
				validConfig := `
bert_model:
  model_id: "test-model"
  threshold: 0.8
default_model: "model-b"
`
				err := os.WriteFile(configFile, []byte(validConfig), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should handle concurrent Load calls safely", func() {
				const numGoroutines = 10
				var wg sync.WaitGroup
				results := make([]*RouterConfig, numGoroutines)
				errors := make([]error, numGoroutines)

				wg.Add(numGoroutines)
				for i := 0; i < numGoroutines; i++ {
					go func(index int) {
						defer wg.Done()
						cfg, err := Load(configFile)
						results[index] = cfg
						errors[index] = err
					}(i)
				}

				wg.Wait()

				// All calls should succeed
				for i := 0; i < numGoroutines; i++ {
					Expect(errors[i]).NotTo(HaveOccurred())
					Expect(results[i]).NotTo(BeNil())
				}

				// All should return the same instance
				for i := 1; i < numGoroutines; i++ {
					Expect(results[i]).To(BeIdenticalTo(results[0]))
				}
			})
		})
	})

	Describe("GetCacheSimilarityThreshold", func() {
		Context("when semantic cache has explicit threshold", func() {
			BeforeEach(func() {
				configContent := `
bert_model:
  threshold: 0.8
semantic_cache:
  similarity_threshold: 0.9
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should return the semantic cache threshold", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				threshold := cfg.GetCacheSimilarityThreshold()
				Expect(threshold).To(Equal(float32(0.9)))
			})
		})

		Context("when semantic cache has no explicit threshold", func() {
			BeforeEach(func() {
				configContent := `
bert_model:
  threshold: 0.8
semantic_cache:
  enabled: true
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should return the BERT model threshold", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				threshold := cfg.GetCacheSimilarityThreshold()
				Expect(threshold).To(Equal(float32(0.8)))
			})
		})
	})

	Describe("GetModelForDecisionIndex", func() {
		BeforeEach(func() {
			configContent := `
decisions:
  - name: "decision1"
    priority: 100
    rules:
      operator: AND
      conditions:
        - type: keyword
          name: rule1
    modelRefs:
      - model: "model1"
        use_reasoning: true
  - name: "decision2"
    priority: 90
    rules:
      operator: OR
      conditions:
        - type: embedding
          name: rule2
    modelRefs:
      - model: "model3"
        use_reasoning: true
default_model: "default-model"
`
			err := os.WriteFile(configFile, []byte(configContent), 0o644)
			Expect(err).NotTo(HaveOccurred())
		})

		Context("with valid decision index", func() {
			It("should return the best model for the decision", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				model := cfg.GetModelForDecisionIndex(0)
				Expect(model).To(Equal("model1"))

				model = cfg.GetModelForDecisionIndex(1)
				Expect(model).To(Equal("model3"))
			})
		})

		Context("with invalid decision index", func() {
			It("should return the default model for negative index", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				model := cfg.GetModelForDecisionIndex(-1)
				Expect(model).To(Equal("default-model"))
			})

			It("should return the default model for index beyond range", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				model := cfg.GetModelForDecisionIndex(10)
				Expect(model).To(Equal("default-model"))
			})
		})

		Context("with decision having no models", func() {
			BeforeEach(func() {
				configContent := `
decisions:
  - name: "empty_decision"
    priority: 50
    rules:
      operator: AND
      conditions:
        - type: keyword
          name: rule1
    modelRefs: []
default_model: "fallback-model"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should return the default model", func() {
				// This should fail validation since decisions must have at least one model
				_, err := Load(configFile)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("has no modelRefs defined"))
			})
		})
	})

	Describe("Feature Enablement Checks", func() {
		Context("PII Classifier", func() {
			It("should return true when properly configured", func() {
				configContent := `
classifier:
  pii_model:
    model_id: "pii-model"
    pii_mapping_path: "/path/to/pii.json"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.IsPIIClassifierEnabled()).To(BeTrue())
			})

			It("should return false when model_id is missing", func() {
				configContent := `
classifier:
  pii_model:
    pii_mapping_path: "/path/to/pii.json"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.IsPIIClassifierEnabled()).To(BeFalse())
			})

			It("should return false when mapping path is missing", func() {
				configContent := `
classifier:
  pii_model:
    model_id: "pii-model"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.IsPIIClassifierEnabled()).To(BeFalse())
			})
		})

		Context("Category Classifier", func() {
			It("should return true when properly configured", func() {
				configContent := `
classifier:
  category_model:
    model_id: "category-model"
    category_mapping_path: "/path/to/category.json"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.IsCategoryClassifierEnabled()).To(BeTrue())
			})

			It("should return false when not configured", func() {
				// Create an empty config file
				err := os.WriteFile(configFile, []byte(""), 0o644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.IsCategoryClassifierEnabled()).To(BeFalse())
			})
		})

		Context("Prompt Guard", func() {
			It("should return true when fully enabled and configured", func() {
				configContent := `
prompt_guard:
  enabled: true
  model_id: "jailbreak-model"
  jailbreak_mapping_path: "/path/to/jailbreak.json"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.IsPromptGuardEnabled()).To(BeTrue())
			})

			It("should return false when disabled", func() {
				configContent := `
prompt_guard:
  enabled: false
  model_id: "jailbreak-model"
  jailbreak_mapping_path: "/path/to/jailbreak.json"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.IsPromptGuardEnabled()).To(BeFalse())
			})

			It("should return false when model_id is missing", func() {
				configContent := `
prompt_guard:
  enabled: true
  jailbreak_mapping_path: "/path/to/jailbreak.json"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.IsPromptGuardEnabled()).To(BeFalse())
			})
		})
	})

	Describe("GetCategoryDescriptions", func() {
		Context("with categories having descriptions", func() {
			BeforeEach(func() {
				configContent := `
categories:
  - name: "category1"
    description: "Description for category 1"
    model_scores:
      - model: "model1"
        score: 0.9
        use_reasoning: true
  - name: "category2"
    description: "Description for category 2"
    model_scores:
      - model: "model2"
        score: 0.8
        use_reasoning: false
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should return all category descriptions", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				descriptions := cfg.GetCategoryDescriptions()
				Expect(descriptions).To(HaveLen(2))
				Expect(descriptions).To(ContainElements(
					"Description for category 1",
					"Description for category 2",
				))
			})
		})

		Context("with categories missing descriptions", func() {
			BeforeEach(func() {
				configContent := `
categories:
  - name: "category1"
    description: "Has description"
    model_scores:
      - model: "model1"
        score: 0.9
        use_reasoning: true
  - name: "category2"
    # No description field
    model_scores:
      - model: "model2"
        score: 0.8
        use_reasoning: false
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should use category name as fallback for missing descriptions", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				descriptions := cfg.GetCategoryDescriptions()
				Expect(descriptions).To(HaveLen(2))
				Expect(descriptions).To(ContainElements(
					"Has description",
					"category2",
				))
			})
		})

		Context("with no categories", func() {
			It("should return empty slice", func() {
				// Create an empty config file
				err := os.WriteFile(configFile, []byte(""), 0o644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				descriptions := cfg.GetCategoryDescriptions()
				Expect(descriptions).To(BeEmpty())
			})
		})
	})

	Describe("Edge Cases and Error Conditions", func() {
		It("should handle configuration with all fields as zero values", func() {
			configContent := `
bert_model:
  threshold: 0
semantic_cache:
  max_entries: 0
  ttl_seconds: 0
`
			err := os.WriteFile(configFile, []byte(configContent), 0o644)
			Expect(err).NotTo(HaveOccurred())

			cfg, err := Load(configFile)
			Expect(err).NotTo(HaveOccurred())
			Expect(cfg.BertModel.Threshold).To(Equal(float32(0)))
			Expect(cfg.SemanticCache.MaxEntries).To(Equal(0))
			Expect(cfg.SemanticCache.TTLSeconds).To(Equal(0))
		})

		It("should handle very large numeric values", func() {
			configContent := `
model_config:
  "large-model":
    preferred_endpoints: ["endpoint1"]
`
			err := os.WriteFile(configFile, []byte(configContent), 0o644)
			Expect(err).NotTo(HaveOccurred())

			cfg, err := Load(configFile)
			Expect(err).NotTo(HaveOccurred())
			Expect(cfg.ModelConfig["large-model"].PreferredEndpoints).To(ContainElement("endpoint1"))
		})

		It("should handle special string values", func() {
			configContent := `
bert_model:
  model_id: "model/with/slashes"
default_model: "model-with-hyphens_and_underscores"
categories:
  - name: "category with spaces"
    description: "Description with special chars: @#$%^&*()"
    model_scores:
      - model: "model-with-hyphens_and_underscores"
        score: 0.9
        use_reasoning: true
`
			err := os.WriteFile(configFile, []byte(configContent), 0o644)
			Expect(err).NotTo(HaveOccurred())

			cfg, err := Load(configFile)
			Expect(err).NotTo(HaveOccurred())
			Expect(cfg.BertModel.ModelID).To(Equal("model/with/slashes"))
			Expect(cfg.DefaultModel).To(Equal("model-with-hyphens_and_underscores"))
			Expect(cfg.Categories[0].Name).To(Equal("category with spaces"))
		})
	})

	Describe("vLLM Endpoints Functions", func() {
		BeforeEach(func() {
			configContent := `
vllm_endpoints:
  - name: "endpoint1"
    address: "127.0.0.1"
    port: 8000
    weight: 1
  - name: "endpoint2"
    address: "127.0.0.1"
    port: 8000
    weight: 2
  - name: "endpoint3"
    address: "127.0.0.1"
    port: 8000
    weight: 1

model_config:
  "model-a":
    preferred_endpoints: ["endpoint1", "endpoint3"]
  "model-b":
    preferred_endpoints: ["endpoint2"]
  "model-c":
    # No preferred endpoints configured

categories:
  - name: "test"
    model_scores:
      - model: "model-a"
        score: 0.9
        use_reasoning: true
      - model: "model-b"
        score: 0.8
        use_reasoning: false

default_model: "model-b"
`
			err := os.WriteFile(configFile, []byte(configContent), 0o644)
			Expect(err).NotTo(HaveOccurred())
		})

		Describe("GetEndpointsForModel", func() {
			It("should return preferred endpoints when configured", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				endpoints := cfg.GetEndpointsForModel("model-a")
				Expect(endpoints).To(HaveLen(2))
				endpointNames := []string{endpoints[0].Name, endpoints[1].Name}
				Expect(endpointNames).To(ContainElements("endpoint1", "endpoint3"))
			})

			It("should return empty slice when no preferred endpoints configured", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				endpoints := cfg.GetEndpointsForModel("model-c")
				Expect(endpoints).To(BeEmpty())
			})

			It("should return empty slice for non-existent model", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				endpoints := cfg.GetEndpointsForModel("non-existent-model")
				Expect(endpoints).To(BeEmpty())
			})

			It("should return only preferred endpoints", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				// model-b has preferred endpoint2
				endpoints := cfg.GetEndpointsForModel("model-b")
				Expect(endpoints).To(HaveLen(1))
				Expect(endpoints[0].Name).To(Equal("endpoint2"))
			})
		})

		Describe("GetEndpointByName", func() {
			It("should return endpoint when it exists", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				endpoint, found := cfg.GetEndpointByName("endpoint1")
				Expect(found).To(BeTrue())
				Expect(endpoint.Name).To(Equal("endpoint1"))
				Expect(endpoint.Address).To(Equal("127.0.0.1"))
				Expect(endpoint.Port).To(Equal(8000))
			})

			It("should return false when endpoint doesn't exist", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				endpoint, found := cfg.GetEndpointByName("non-existent")
				Expect(found).To(BeFalse())
				Expect(endpoint).To(BeNil())
			})
		})

		Describe("GetAllModels", func() {
			It("should return all models from model_config", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				models := cfg.GetAllModels()
				Expect(models).To(HaveLen(3))
				Expect(models).To(ContainElements("model-a", "model-b", "model-c"))
			})
		})

		Describe("SelectBestEndpointForModel", func() {
			It("should select endpoint with highest weight when multiple available", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				// model-a has preferred endpoints: endpoint1 (weight 1) and endpoint3 (weight 1)
				// Since they have the same weight, it should return the first one found
				endpointName, found := cfg.SelectBestEndpointForModel("model-a")
				Expect(found).To(BeTrue())
				Expect(endpointName).To(BeElementOf("endpoint1", "endpoint3"))
			})

			It("should return false for non-existent model", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				endpointName, found := cfg.SelectBestEndpointForModel("non-existent-model")
				Expect(found).To(BeFalse())
				Expect(endpointName).To(BeEmpty())
			})

			It("should return false when model has no preferred endpoints", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				endpointName, found := cfg.SelectBestEndpointForModel("model-c")
				Expect(found).To(BeFalse())
				Expect(endpointName).To(BeEmpty())
			})

			Describe("SelectBestEndpointAddressForModel", func() {
				It("should return endpoint address when model has preferred endpoints", func() {
					cfg, err := Load(configFile)
					Expect(err).NotTo(HaveOccurred())

					// model-a has preferred endpoints
					endpointAddress, found := cfg.SelectBestEndpointAddressForModel("model-a")
					Expect(found).To(BeTrue())
					Expect(endpointAddress).To(MatchRegexp(`127\.0\.0\.1:\d+`))
				})

				It("should return false when model has no preferred endpoints", func() {
					cfg, err := Load(configFile)
					Expect(err).NotTo(HaveOccurred())

					// model-c has no preferred_endpoints configured
					endpointAddress, found := cfg.SelectBestEndpointAddressForModel("model-c")
					Expect(found).To(BeFalse())
					Expect(endpointAddress).To(BeEmpty())
				})

				It("should return false for non-existent model", func() {
					cfg, err := Load(configFile)
					Expect(err).NotTo(HaveOccurred())

					endpointAddress, found := cfg.SelectBestEndpointAddressForModel("non-existent-model")
					Expect(found).To(BeFalse())
					Expect(endpointAddress).To(BeEmpty())
				})
			})
		})

		Describe("ValidateEndpoints", func() {
			It("should pass validation when all models have endpoints", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				err = cfg.ValidateEndpoints()
				Expect(err).NotTo(HaveOccurred())
			})

			It("should fail validation when default model has no endpoints", func() {
				configContent := `
vllm_endpoints:
  - name: "endpoint1"
    address: "127.0.0.1"
    port: 8000
    weight: 1

model_config:
  "existing-model":
    preferred_endpoints: ["endpoint1"]

default_model: "missing-default-model"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				err = cfg.ValidateEndpoints()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("missing-default-model"))
			})
		})

		Describe("vLLM Endpoint Address Validation", func() {
			Context("with valid IP addresses", func() {
				It("should accept IPv4 addresses", func() {
					configContent := `
vllm_endpoints:
  - name: "endpoint1"
    address: "127.0.0.1"
    port: 8000
    weight: 1

model_config:
  "test-model":
    preferred_endpoints: ["endpoint1"]

categories:
  - name: "test"
    model_scores:
      - model: "test-model"
        score: 0.9
        use_reasoning: true

default_model: "test-model"
`
					err := os.WriteFile(configFile, []byte(configContent), 0o644)
					Expect(err).NotTo(HaveOccurred())

					cfg, err := Load(configFile)
					Expect(err).NotTo(HaveOccurred())
					Expect(cfg.VLLMEndpoints[0].Address).To(Equal("127.0.0.1"))
				})

				It("should accept IPv6 addresses", func() {
					configContent := `
vllm_endpoints:
  - name: "endpoint1"
    address: "::1"
    port: 8000
    weight: 1

model_config:
  "test-model":
    preferred_endpoints: ["endpoint1"]

categories:
  - name: "test"
    model_scores:
      - model: "test-model"
        score: 0.9
        use_reasoning: true

default_model: "test-model"
`
					err := os.WriteFile(configFile, []byte(configContent), 0o644)
					Expect(err).NotTo(HaveOccurred())

					cfg, err := Load(configFile)
					Expect(err).NotTo(HaveOccurred())
					Expect(cfg.VLLMEndpoints[0].Address).To(Equal("::1"))
				})

				It("should accept domain names", func() {
					configContent := `
vllm_endpoints:
  - name: "endpoint1"
    address: "example.com"
    port: 8000
    weight: 1

model_config:
  "test-model":
    preferred_endpoints: ["endpoint1"]

categories:
  - name: "test"
    model_scores:
      - model: "test-model"
        score: 0.9
        use_reasoning: true

default_model: "test-model"
`
					err := os.WriteFile(configFile, []byte(configContent), 0o644)
					Expect(err).NotTo(HaveOccurred())

					cfg, err := Load(configFile)
					Expect(err).NotTo(HaveOccurred())
					Expect(cfg.VLLMEndpoints[0].Address).To(Equal("example.com"))
				})
			})
		})
	})

	Describe("Semantic Cache Backend Configuration", func() {
		Context("with memory backend configuration", func() {
			BeforeEach(func() {
				configContent := `
semantic_cache:
  enabled: true
  backend_type: "memory"
  similarity_threshold: 0.85
  max_entries: 2000
  ttl_seconds: 1800
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should parse memory backend configuration correctly", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Enabled).To(BeTrue())
				Expect(cfg.SemanticCache.BackendType).To(Equal("memory"))
				Expect(*cfg.SemanticCache.SimilarityThreshold).To(Equal(float32(0.85)))
				Expect(cfg.SemanticCache.MaxEntries).To(Equal(2000))
				Expect(cfg.SemanticCache.TTLSeconds).To(Equal(1800))
				Expect(cfg.SemanticCache.BackendConfigPath).To(BeEmpty())
			})
		})

		Context("(Deprecated) with file base milvus backend configuration", func() {
			BeforeEach(func() {
				configContent := `
semantic_cache:
  enabled: true
  backend_type: "milvus"
  similarity_threshold: 0.9
  ttl_seconds: 7200
  backend_config_path: "config/semantic-cache/milvus.yaml"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should parse file base milvus backend configuration correctly", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Enabled).To(BeTrue())
				Expect(cfg.SemanticCache.BackendType).To(Equal("milvus"))
				Expect(*cfg.SemanticCache.SimilarityThreshold).To(Equal(float32(0.9)))
				Expect(cfg.SemanticCache.TTLSeconds).To(Equal(7200))
				Expect(cfg.SemanticCache.BackendConfigPath).To(Equal("config/semantic-cache/milvus.yaml"))

				// MaxEntries should be ignored for Milvus backend
				Expect(cfg.SemanticCache.MaxEntries).To(Equal(0))
			})
		})

		Context("with inline milvus backend configuration", func() {
			BeforeEach(func() {
				configContent := `
semantic_cache:
  enabled: true
  backend_type: "milvus"
  similarity_threshold: 0.9
  ttl_seconds: 7200
  milvus:
    connection:
      host: "localhost"
      port: 12345
      database: "semantic_router_cache"
      timeout: 99
      auth:
        enabled: false
        username: "1234"
        password: "1234"
      tls:
        enabled: false
        cert_file: ""
        key_file: ""
        ca_file: ""
    collection:
      name: "semantic_cache"
      description: "test"
      vector_field:
        name: "test"
        dimension: 384
        metric_type: "test"
      index:
        type: "HNSW"
        params:
          M: 16
          efConstruction: 64
    search:
      params:
        ef: 64
      topk: 10
      consistency_level: "Session"
    performance:
      connection_pool:
        max_connections: 10
        max_idle_connections: 5
        acquire_timeout: 5
      batch:
        insert_batch_size: 1000
        timeout: 30
    data_management:
      ttl:
        enabled: true
        timestamp_field: "timestamp"
        cleanup_interval: 3600
      compaction:
        enabled: true
        interval: 86400
    logging:
      level: "info"
      enable_query_log: false
      enable_metrics: true
    development:
      drop_collection_on_startup: true
      auto_create_collection: true
      verbose_errors: true
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should parse inline milvus backend connection configuration correctly", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Milvus).ToNot(BeNil())
				Expect(cfg.SemanticCache.Milvus.Connection).ToNot(BeNil())
				Expect(cfg.SemanticCache.Milvus.Connection.Host).To(Equal("localhost"))
				Expect(cfg.SemanticCache.Milvus.Connection.Port).To(Equal(12345))
				Expect(cfg.SemanticCache.Milvus.Connection.Database).To(Equal("semantic_router_cache"))
				Expect(cfg.SemanticCache.Milvus.Connection.Timeout).To(Equal(99))
				Expect(cfg.SemanticCache.Milvus.Connection.Auth.Enabled).To(BeFalse())
				Expect(cfg.SemanticCache.Milvus.Connection.Auth.Username).To(Equal("1234"))
				Expect(cfg.SemanticCache.Milvus.Connection.Auth.Password).To(Equal("1234"))
				Expect(cfg.SemanticCache.Milvus.Connection.TLS.Enabled).To(BeFalse())
				Expect(cfg.SemanticCache.Milvus.Connection.TLS.CertFile).To(Equal(""))
				Expect(cfg.SemanticCache.Milvus.Connection.TLS.KeyFile).To(Equal(""))
				Expect(cfg.SemanticCache.Milvus.Connection.TLS.CAFile).To(Equal(""))
			})

			It("should parse inline milvus backend collection configuration correctly", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Milvus.Collection).ToNot(BeNil())
				Expect(cfg.SemanticCache.Milvus.Collection.Name).To(Equal("semantic_cache"))
				Expect(cfg.SemanticCache.Milvus.Collection.Description).To(Equal("test"))
				Expect(cfg.SemanticCache.Milvus.Collection.VectorField.Name).To(Equal("test"))
				Expect(cfg.SemanticCache.Milvus.Collection.VectorField.Dimension).To(Equal(384))
				Expect(cfg.SemanticCache.Milvus.Collection.VectorField.MetricType).To(Equal("test"))
				Expect(cfg.SemanticCache.Milvus.Collection.Index.Type).To(Equal("HNSW"))
				Expect(cfg.SemanticCache.Milvus.Collection.Index.Params.M).To(Equal(16))
				Expect(cfg.SemanticCache.Milvus.Collection.Index.Params.EfConstruction).To(Equal(64))
			})

			It("should parse inline milvus backend search configuration correctly", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Milvus.Search).ToNot(BeNil())
				Expect(cfg.SemanticCache.Milvus.Search.Params.Ef).To(Equal(64))
				Expect(cfg.SemanticCache.Milvus.Search.TopK).To(Equal(10))
				Expect(cfg.SemanticCache.Milvus.Search.ConsistencyLevel).To(Equal("Session"))
			})

			It("should parse inline milvus backend performance configuration correctly", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Milvus.Performance).ToNot(BeNil())
				Expect(cfg.SemanticCache.Milvus.Performance.ConnectionPool.MaxConnections).To(Equal(10))
				Expect(cfg.SemanticCache.Milvus.Performance.ConnectionPool.MaxIdleConnections).To(Equal(5))
				Expect(cfg.SemanticCache.Milvus.Performance.ConnectionPool.AcquireTimeout).To(Equal(5))
				Expect(cfg.SemanticCache.Milvus.Performance.Batch.InsertBatchSize).To(Equal(1000))
				Expect(cfg.SemanticCache.Milvus.Performance.Batch.Timeout).To(Equal(30))
			})

			It("should parse inline milvus backend data management configuration correctly", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Milvus.DataManagement).ToNot(BeNil())
				Expect(cfg.SemanticCache.Milvus.DataManagement.TTL.Enabled).To(BeTrue())
				Expect(cfg.SemanticCache.Milvus.DataManagement.TTL.TimestampField).To(Equal("timestamp"))
				Expect(cfg.SemanticCache.Milvus.DataManagement.TTL.CleanupInterval).To(Equal(3600))
				Expect(cfg.SemanticCache.Milvus.DataManagement.Compaction.Enabled).To(BeTrue())
				Expect(cfg.SemanticCache.Milvus.DataManagement.Compaction.Interval).To(Equal(86400))
			})

			It("should parse inline milvus backend logging configuration correctly", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Milvus.Logging.Level).To(Equal("info"))
				Expect(cfg.SemanticCache.Milvus.Logging.EnableQueryLog).To(BeFalse())
				Expect(cfg.SemanticCache.Milvus.Logging.EnableMetrics).To(BeTrue())
			})

			It("should parse inline milvus backend development configuration correctly", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Milvus.Development.DropCollectionOnStartup).To(BeTrue())
				Expect(cfg.SemanticCache.Milvus.Development.AutoCreateCollection).To(BeTrue())
				Expect(cfg.SemanticCache.Milvus.Development.VerboseErrors).To(BeTrue())
			})
		})

		Context("with inline redis backend configuration", func() {
			BeforeEach(func() {
				configContent := `
semantic_cache:
  enabled: true
  backend_type: "milvus"
  similarity_threshold: 0.9
  ttl_seconds: 7200
  redis:
    connection:
      host: "localhost"
      port: 6379
      database: 0
      password: ""
      timeout: 30
      tls:
        enabled: false
        cert_file: ""
        key_file: ""
        ca_file: ""
    index:
      name: "semantic_cache_idx"
      prefix: "doc:"
      vector_field:
        name: "embedding"
        dimension: 384
        metric_type: "COSINE"
      index_type: "HNSW"
      params:
        M: 16
        efConstruction: 64
    search:
      topk: 1
    logging:
      level: "info"
      enable_query_log: false
      enable_metrics: true
    development:
      drop_index_on_startup: true
      auto_create_index: true
      verbose_errors: true
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should parse inline redis backend connection configuration correctly", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Redis).ToNot(BeNil())
				Expect(cfg.SemanticCache.Redis.Connection).ToNot(BeNil())
				Expect(cfg.SemanticCache.Redis.Connection.Host).To(Equal("localhost"))
				Expect(cfg.SemanticCache.Redis.Connection.Port).To(Equal(6379))
				Expect(cfg.SemanticCache.Redis.Connection.Database).To(Equal(0))
				Expect(cfg.SemanticCache.Redis.Connection.Password).To(Equal(""))
				Expect(cfg.SemanticCache.Redis.Connection.Timeout).To(Equal(30))
				Expect(cfg.SemanticCache.Redis.Connection.TLS.Enabled).To(BeFalse())
				Expect(cfg.SemanticCache.Redis.Connection.TLS.CertFile).To(Equal(""))
				Expect(cfg.SemanticCache.Redis.Connection.TLS.KeyFile).To(Equal(""))
				Expect(cfg.SemanticCache.Redis.Connection.TLS.CAFile).To(Equal(""))
			})

			It("should parse inline redis backend index configuration correctly", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Redis.Index).ToNot(BeNil())
				Expect(cfg.SemanticCache.Redis.Index.Name).To(Equal("semantic_cache_idx"))
				Expect(cfg.SemanticCache.Redis.Index.Prefix).To(Equal("doc:"))
				Expect(cfg.SemanticCache.Redis.Index.VectorField.Name).To(Equal("embedding"))
				Expect(cfg.SemanticCache.Redis.Index.VectorField.Dimension).To(Equal(384))
				Expect(cfg.SemanticCache.Redis.Index.VectorField.MetricType).To(Equal("COSINE"))
				Expect(cfg.SemanticCache.Redis.Index.IndexType).To(Equal("HNSW"))
				Expect(cfg.SemanticCache.Redis.Index.Params.M).To(Equal(16))
				Expect(cfg.SemanticCache.Redis.Index.Params.EfConstruction).To(Equal(64))
			})

			It("should parse inline redis backend search configuration correctly", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Redis.Search).ToNot(BeNil())
				Expect(cfg.SemanticCache.Redis.Search.TopK).To(Equal(1))
			})

			It("should parse inline redis backend logging configuration correctly", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Redis.Logging.Level).To(Equal("info"))
				Expect(cfg.SemanticCache.Redis.Logging.EnableQueryLog).To(BeFalse())
				Expect(cfg.SemanticCache.Redis.Logging.EnableMetrics).To(BeTrue())
			})

			It("should parse inline redis backend development configuration correctly", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Redis.Development.DropIndexOnStartup).To(BeTrue())
				Expect(cfg.SemanticCache.Redis.Development.AutoCreateIndex).To(BeTrue())
				Expect(cfg.SemanticCache.Redis.Development.VerboseErrors).To(BeTrue())
			})
		})

		Context("with disabled cache", func() {
			BeforeEach(func() {
				configContent := `
semantic_cache:
  enabled: false
  backend_type: "memory"
  similarity_threshold: 0.8
  max_entries: 1000
  ttl_seconds: 3600
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should preserve configuration even when cache is disabled", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Enabled).To(BeFalse())
				Expect(cfg.SemanticCache.BackendType).To(Equal("memory"))
				Expect(*cfg.SemanticCache.SimilarityThreshold).To(Equal(float32(0.8)))
			})
		})

		Context("with minimal configuration", func() {
			BeforeEach(func() {
				configContent := `
semantic_cache:
  enabled: true
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should handle minimal configuration with default values", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Enabled).To(BeTrue())
				Expect(cfg.SemanticCache.BackendType).To(BeEmpty())       // Should default to empty (memory)
				Expect(cfg.SemanticCache.SimilarityThreshold).To(BeNil()) // Will fallback to BERT threshold
				Expect(cfg.SemanticCache.MaxEntries).To(Equal(0))
				Expect(cfg.SemanticCache.TTLSeconds).To(Equal(0))
				Expect(cfg.SemanticCache.BackendConfigPath).To(BeEmpty())
			})
		})

		Context("with comprehensive configuration", func() {
			BeforeEach(func() {
				configContent := `
bert_model:
  threshold: 0.7

semantic_cache:
  enabled: true
  backend_type: "milvus"
  similarity_threshold: 0.95
  ttl_seconds: 14400
  backend_config_path: "config/cache/production_milvus.yaml"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should parse all semantic cache fields correctly", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Enabled).To(BeTrue())
				Expect(cfg.SemanticCache.BackendType).To(Equal("milvus"))
				Expect(*cfg.SemanticCache.SimilarityThreshold).To(Equal(float32(0.95)))
				Expect(cfg.SemanticCache.TTLSeconds).To(Equal(14400))
				Expect(cfg.SemanticCache.BackendConfigPath).To(Equal("config/cache/production_milvus.yaml"))

				// Verify threshold resolution
				threshold := cfg.GetCacheSimilarityThreshold()
				Expect(threshold).To(Equal(float32(0.95))) // Should use cache threshold, not BERT
			})
		})

		Context("threshold fallback behavior", func() {
			BeforeEach(func() {
				configContent := `
bert_model:
  threshold: 0.75

semantic_cache:
  enabled: true
  backend_type: "memory"
  max_entries: 500
  # No similarity_threshold specified
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should fall back to BERT threshold when cache threshold not specified", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.SimilarityThreshold).To(BeNil())

				// GetCacheSimilarityThreshold should return BERT threshold
				threshold := cfg.GetCacheSimilarityThreshold()
				Expect(threshold).To(Equal(float32(0.75)))
			})
		})

		Context("with edge case values", func() {
			BeforeEach(func() {
				configContent := `
semantic_cache:
  enabled: true
  backend_type: "memory"
  similarity_threshold: 1.0
  max_entries: 0
  ttl_seconds: -1
  backend_config_path: ""
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should handle edge case values correctly", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Enabled).To(BeTrue())
				Expect(cfg.SemanticCache.BackendType).To(Equal("memory"))
				Expect(*cfg.SemanticCache.SimilarityThreshold).To(Equal(float32(1.0)))
				Expect(cfg.SemanticCache.MaxEntries).To(Equal(0))
				Expect(cfg.SemanticCache.TTLSeconds).To(Equal(-1))
				Expect(cfg.SemanticCache.BackendConfigPath).To(BeEmpty())
			})
		})

		Context("with unsupported backend type", func() {
			BeforeEach(func() {
				configContent := `
semantic_cache:
  enabled: true
  backend_type: "redis"
  similarity_threshold: 0.8
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should parse unsupported backend type without error (validation happens at runtime)", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				// Configuration parsing should succeed
				Expect(cfg.SemanticCache.Enabled).To(BeTrue())
				Expect(cfg.SemanticCache.BackendType).To(Equal("redis"))

				// Runtime validation will catch unsupported backend types
			})
		})

		Context("with production-like configuration", func() {
			BeforeEach(func() {
				configContent := `
bert_model:
  model_id: sentence-transformers/all-MiniLM-L12-v2
  threshold: 0.6
  use_cpu: false

semantic_cache:
  enabled: true
  backend_type: "milvus"
  similarity_threshold: 0.85
  ttl_seconds: 86400  # 24 hours
  backend_config_path: "config/semantic-cache/milvus.yaml"

categories:
  - name: "production"
    description: "Production workload"
    model_scores:
      - model: "gpt-4"
        score: 0.95
        use_reasoning: true

default_model: "gpt-4"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should handle production-like configuration correctly", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				// Verify BERT config
				Expect(cfg.BertModel.ModelID).To(Equal("sentence-transformers/all-MiniLM-L12-v2"))
				Expect(cfg.BertModel.Threshold).To(Equal(float32(0.6)))
				Expect(cfg.BertModel.UseCPU).To(BeFalse())

				// Verify semantic cache config
				Expect(cfg.SemanticCache.Enabled).To(BeTrue())
				Expect(cfg.SemanticCache.BackendType).To(Equal("milvus"))
				Expect(*cfg.SemanticCache.SimilarityThreshold).To(Equal(float32(0.85)))
				Expect(cfg.SemanticCache.TTLSeconds).To(Equal(86400))
				Expect(cfg.SemanticCache.BackendConfigPath).To(Equal("config/semantic-cache/milvus.yaml"))

				// Verify threshold resolution
				threshold := cfg.GetCacheSimilarityThreshold()
				Expect(threshold).To(Equal(float32(0.85))) // Should use cache threshold

				// Verify other config is still working
				Expect(cfg.DefaultModel).To(Equal("gpt-4"))
				Expect(cfg.Categories).To(HaveLen(1))
			})
		})

		Context("with multiple backend configurations in comments", func() {
			BeforeEach(func() {
				configContent := `
semantic_cache:
  # Development configuration
  enabled: true
  backend_type: "memory"
  similarity_threshold: 0.8
  max_entries: 1000
  ttl_seconds: 3600

  # Production configuration (commented out)
  # backend_type: "milvus"
  # backend_config_path: "config/semantic-cache/milvus.yaml"
  # max_entries is ignored for Milvus
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should parse active configuration and ignore commented alternatives", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Enabled).To(BeTrue())
				Expect(cfg.SemanticCache.BackendType).To(Equal("memory"))
				Expect(*cfg.SemanticCache.SimilarityThreshold).To(Equal(float32(0.8)))
				Expect(cfg.SemanticCache.MaxEntries).To(Equal(1000))
				Expect(cfg.SemanticCache.TTLSeconds).To(Equal(3600))
				Expect(cfg.SemanticCache.BackendConfigPath).To(BeEmpty()) // Comments are ignored
			})
		})
	})

	Describe("PII Constants", func() {
		It("should have all expected PII type constants defined", func() {
			expectedPIITypes := []string{
				PIITypeAge,
				PIITypeCreditCard,
				PIITypeDateTime,
				PIITypeDomainName,
				PIITypeEmailAddress,
				PIITypeGPE,
				PIITypeIBANCode,
				PIITypeIPAddress,
				PIITypeNoPII,
				PIITypeNRP,
				PIITypeOrganization,
				PIITypePerson,
				PIITypePhoneNumber,
				PIITypeStreetAddress,
				PIITypeUSDriverLicense,
				PIITypeUSSSN,
				PIITypeZipCode,
			}

			// Verify all constants are non-empty strings
			for _, piiType := range expectedPIITypes {
				Expect(piiType).NotTo(BeEmpty())
			}

			// Verify specific values
			Expect(PIITypeNoPII).To(Equal("NO_PII"))
			Expect(PIITypePerson).To(Equal("PERSON"))
			Expect(PIITypeEmailAddress).To(Equal("EMAIL_ADDRESS"))
		})
	})

	// Test batch classification metrics configuration
	Describe("Batch Classification Metrics Configuration", func() {
		It("should parse batch classification metrics configuration correctly", func() {
			yamlContent := `
api:
  batch_classification:
    auto_unified_batching: true
    metrics:
      enabled: true
      detailed_goroutine_tracking: false
      high_resolution_timing: true
      sample_rate: 0.8
      duration_buckets: [0.01, 0.1, 1.0, 10.0]
      size_buckets: [5, 15, 25, 75]
`

			var cfg RouterConfig
			err := yaml.Unmarshal([]byte(yamlContent), &cfg)
			Expect(err).NotTo(HaveOccurred())

			// Verify batch classification configuration (zero-config auto-discovery)
			batchConfig := cfg.API.BatchClassification

			// Verify metrics configuration
			metricsConfig := batchConfig.Metrics
			Expect(metricsConfig.Enabled).To(BeTrue())
			Expect(metricsConfig.DetailedGoroutineTracking).To(BeFalse())
			Expect(metricsConfig.HighResolutionTiming).To(BeTrue())
			Expect(metricsConfig.SampleRate).To(Equal(0.8))

			// Verify custom buckets
			Expect(metricsConfig.DurationBuckets).To(Equal([]float64{0.01, 0.1, 1.0, 10.0}))
			Expect(metricsConfig.SizeBuckets).To(Equal([]float64{5, 15, 25, 75}))
		})

		It("should handle missing metrics configuration with defaults", func() {
			yamlContent := `
api:
  batch_classification:
    auto_unified_batching: false
`

			var cfg RouterConfig
			err := yaml.Unmarshal([]byte(yamlContent), &cfg)
			Expect(err).NotTo(HaveOccurred())

			// Verify that missing metrics configuration doesn't cause errors (zero-config)
			batchConfig := cfg.API.BatchClassification

			// Metrics should have zero values (will be handled by defaults in application)
			metricsConfig := batchConfig.Metrics
			Expect(metricsConfig.Enabled).To(BeFalse())     // Default zero value
			Expect(metricsConfig.SampleRate).To(Equal(0.0)) // Default zero value
		})

		It("should handle partial metrics configuration", func() {
			yamlContent := `
api:
  batch_classification:
    metrics:
      enabled: true
      sample_rate: 0.5
`

			var cfg RouterConfig
			err := yaml.Unmarshal([]byte(yamlContent), &cfg)
			Expect(err).NotTo(HaveOccurred())

			metricsConfig := cfg.API.BatchClassification.Metrics
			Expect(metricsConfig.Enabled).To(BeTrue())
			Expect(metricsConfig.SampleRate).To(Equal(0.5))

			// Other fields should have zero values
			Expect(metricsConfig.DetailedGoroutineTracking).To(BeFalse())
			Expect(metricsConfig.HighResolutionTiming).To(BeFalse())
			Expect(len(metricsConfig.DurationBuckets)).To(Equal(0))
			Expect(len(metricsConfig.SizeBuckets)).To(Equal(0))
		})
	})

	Describe("AutoModelName Configuration", func() {
		Context("GetEffectiveAutoModelName", func() {
			It("should return configured AutoModelName when set", func() {
				cfg := &RouterConfig{
					RouterOptions: RouterOptions{
						AutoModelName: "CustomAuto",
					},
				}
				Expect(cfg.GetEffectiveAutoModelName()).To(Equal("CustomAuto"))
			})

			It("should return default 'MoM' when AutoModelName is not set", func() {
				cfg := &RouterConfig{
					RouterOptions: RouterOptions{
						AutoModelName: "",
					},
				}
				Expect(cfg.GetEffectiveAutoModelName()).To(Equal("MoM"))
			})

			It("should return default 'MoM' for empty RouterConfig", func() {
				cfg := &RouterConfig{}
				Expect(cfg.GetEffectiveAutoModelName()).To(Equal("MoM"))
			})
		})

		Context("IsAutoModelName", func() {
			It("should recognize 'auto' as auto model name for backward compatibility", func() {
				cfg := &RouterConfig{
					RouterOptions: RouterOptions{
						AutoModelName: "MoM",
					},
				}
				Expect(cfg.IsAutoModelName("auto")).To(BeTrue())
			})

			It("should recognize configured AutoModelName", func() {
				cfg := &RouterConfig{
					RouterOptions: RouterOptions{
						AutoModelName: "CustomAuto",
					},
				}
				Expect(cfg.IsAutoModelName("CustomAuto")).To(BeTrue())
			})

			It("should recognize default 'MoM' when AutoModelName is not set", func() {
				cfg := &RouterConfig{
					RouterOptions: RouterOptions{
						AutoModelName: "",
					},
				}
				Expect(cfg.IsAutoModelName("MoM")).To(BeTrue())
			})

			It("should not recognize other model names as auto", func() {
				cfg := &RouterConfig{
					RouterOptions: RouterOptions{
						AutoModelName: "MoM",
					},
				}
				Expect(cfg.IsAutoModelName("gpt-4")).To(BeFalse())
				Expect(cfg.IsAutoModelName("claude")).To(BeFalse())
			})

			It("should support both 'auto' and configured name", func() {
				cfg := &RouterConfig{
					RouterOptions: RouterOptions{
						AutoModelName: "MoM",
					},
				}
				Expect(cfg.IsAutoModelName("auto")).To(BeTrue())
				Expect(cfg.IsAutoModelName("MoM")).To(BeTrue())
				Expect(cfg.IsAutoModelName("other")).To(BeFalse())
			})
		})

		Context("YAML parsing with AutoModelName", func() {
			It("should parse AutoModelName from YAML", func() {
				yamlContent := `
auto_model_name: "CustomRouter"
default_model: "test-model"
`
				var cfg RouterConfig
				err := yaml.Unmarshal([]byte(yamlContent), &cfg)
				Expect(err).NotTo(HaveOccurred())
				Expect(cfg.RouterOptions.AutoModelName).To(Equal("CustomRouter"))
				Expect(cfg.GetEffectiveAutoModelName()).To(Equal("CustomRouter"))
			})

			It("should handle missing AutoModelName in YAML", func() {
				yamlContent := `
default_model: "test-model"
`
				var cfg RouterConfig
				err := yaml.Unmarshal([]byte(yamlContent), &cfg)
				Expect(err).NotTo(HaveOccurred())
				Expect(cfg.RouterOptions.AutoModelName).To(Equal(""))
				Expect(cfg.GetEffectiveAutoModelName()).To(Equal("MoM"))
			})
		})
	})

	Describe("IsJailbreakEnabledForDecision", func() {
		Context("per-decision plugin scoping", func() {
			It("should return false for decision without explicit jailbreak plugin (even if global is enabled)", func() {
				decision := Decision{
					Name:      "test",
					ModelRefs: []ModelRef{{Model: "test"}},
					// No jailbreak plugin configured
				}

				cfg := &RouterConfig{
					InlineModels: InlineModels{
						PromptGuard: PromptGuardConfig{
							Enabled: true, // Global enabled, but should not affect decisions without plugin
						},
					},
					IntelligentRouting: IntelligentRouting{
						Decisions: []Decision{decision},
					},
				}

				// Per-decision scoping: no plugin = no jailbreak detection
				Expect(cfg.IsJailbreakEnabledForDecision("test")).To(BeFalse())
			})

			It("should return false when decision explicitly disables jailbreak", func() {
				decision := Decision{
					Name:      "test",
					ModelRefs: []ModelRef{{Model: "test"}},
					Plugins: []DecisionPlugin{
						{
							Type: "jailbreak",
							Configuration: map[string]interface{}{
								"enabled": false,
							},
						},
					},
				}

				cfg := &RouterConfig{
					InlineModels: InlineModels{
						PromptGuard: PromptGuardConfig{
							Enabled: true,
						},
					},
					IntelligentRouting: IntelligentRouting{
						Decisions: []Decision{decision},
					},
				}

				Expect(cfg.IsJailbreakEnabledForDecision("test")).To(BeFalse())
			})

			It("should return true when decision explicitly enables jailbreak", func() {
				decision := Decision{
					Name:      "test",
					ModelRefs: []ModelRef{{Model: "test"}},
					Plugins: []DecisionPlugin{
						{
							Type: "jailbreak",
							Configuration: map[string]interface{}{
								"enabled": true,
							},
						},
					},
				}

				cfg := &RouterConfig{
					InlineModels: InlineModels{
						PromptGuard: PromptGuardConfig{
							Enabled: true,
						},
					},
					IntelligentRouting: IntelligentRouting{
						Decisions: []Decision{decision},
					},
				}

				Expect(cfg.IsJailbreakEnabledForDecision("test")).To(BeTrue())
			})

			It("should return false for non-existent decision", func() {
				cfg := &RouterConfig{
					InlineModels: InlineModels{
						PromptGuard: PromptGuardConfig{
							Enabled: true,
						},
					},
				}

				Expect(cfg.IsJailbreakEnabledForDecision("non_existent")).To(BeFalse())
			})

			It("should return false for include_history by default", func() {
				decision := Decision{
					Name:      "test",
					ModelRefs: []ModelRef{{Model: "test"}},
					Plugins: []DecisionPlugin{
						{
							Type: "jailbreak",
							Configuration: map[string]interface{}{
								"enabled": true,
							},
						},
					},
				}

				cfg := &RouterConfig{
					IntelligentRouting: IntelligentRouting{
						Decisions: []Decision{decision},
					},
				}

				Expect(cfg.GetJailbreakIncludeHistoryForDecision("test")).To(BeFalse())
			})

			It("should return true when include_history is explicitly enabled", func() {
				decision := Decision{
					Name:      "test",
					ModelRefs: []ModelRef{{Model: "test"}},
					Plugins: []DecisionPlugin{
						{
							Type: "jailbreak",
							Configuration: map[string]interface{}{
								"enabled":         true,
								"include_history": true,
							},
						},
					},
				}

				cfg := &RouterConfig{
					IntelligentRouting: IntelligentRouting{
						Decisions: []Decision{decision},
					},
				}

				Expect(cfg.GetJailbreakIncludeHistoryForDecision("test")).To(BeTrue())
			})
		})
	})

	Describe("IsPIIEnabledForDecision", func() {
		Context("per-decision plugin scoping", func() {
			It("should return false for decision without explicit PII plugin", func() {
				decision := Decision{
					Name:      "test",
					ModelRefs: []ModelRef{{Model: "test"}},
					// No PII plugin configured
				}

				cfg := &RouterConfig{
					IntelligentRouting: IntelligentRouting{
						Decisions: []Decision{decision},
					},
				}

				Expect(cfg.IsPIIEnabledForDecision("test")).To(BeFalse())
			})

			It("should return true when decision explicitly enables PII", func() {
				decision := Decision{
					Name:      "test",
					ModelRefs: []ModelRef{{Model: "test"}},
					Plugins: []DecisionPlugin{
						{
							Type: "pii",
							Configuration: map[string]interface{}{
								"enabled": true,
							},
						},
					},
				}

				cfg := &RouterConfig{
					IntelligentRouting: IntelligentRouting{
						Decisions: []Decision{decision},
					},
				}

				Expect(cfg.IsPIIEnabledForDecision("test")).To(BeTrue())
			})

			It("should return false when decision explicitly disables PII", func() {
				decision := Decision{
					Name:      "test",
					ModelRefs: []ModelRef{{Model: "test"}},
					Plugins: []DecisionPlugin{
						{
							Type: "pii",
							Configuration: map[string]interface{}{
								"enabled": false,
							},
						},
					},
				}

				cfg := &RouterConfig{
					IntelligentRouting: IntelligentRouting{
						Decisions: []Decision{decision},
					},
				}

				Expect(cfg.IsPIIEnabledForDecision("test")).To(BeFalse())
			})

			It("should return false for include_history by default", func() {
				decision := Decision{
					Name:      "test",
					ModelRefs: []ModelRef{{Model: "test"}},
					Plugins: []DecisionPlugin{
						{
							Type: "pii",
							Configuration: map[string]interface{}{
								"enabled": true,
							},
						},
					},
				}

				cfg := &RouterConfig{
					IntelligentRouting: IntelligentRouting{
						Decisions: []Decision{decision},
					},
				}

				Expect(cfg.GetPIIIncludeHistoryForDecision("test")).To(BeFalse())
			})

			It("should return true when include_history is explicitly enabled", func() {
				decision := Decision{
					Name:      "test",
					ModelRefs: []ModelRef{{Model: "test"}},
					Plugins: []DecisionPlugin{
						{
							Type: "pii",
							Configuration: map[string]interface{}{
								"enabled":         true,
								"include_history": true,
							},
						},
					},
				}

				cfg := &RouterConfig{
					IntelligentRouting: IntelligentRouting{
						Decisions: []Decision{decision},
					},
				}

				Expect(cfg.GetPIIIncludeHistoryForDecision("test")).To(BeTrue())
			})
		})
	})

	Describe("IsCacheEnabledForDecision", func() {
		Context("per-decision plugin scoping", func() {
			It("should return false for decision without explicit semantic-cache plugin (even if global is enabled)", func() {
				decision := Decision{
					Name:      "test",
					ModelRefs: []ModelRef{{Model: "test"}},
					// No semantic-cache plugin configured
				}

				cfg := &RouterConfig{
					SemanticCache: SemanticCache{
						Enabled: true, // Global enabled, but should not affect decisions without plugin
					},
					IntelligentRouting: IntelligentRouting{
						Decisions: []Decision{decision},
					},
				}

				// Per-decision scoping: no plugin = no semantic caching
				Expect(cfg.IsCacheEnabledForDecision("test")).To(BeFalse())
			})

			It("should return false when decision explicitly disables semantic-cache", func() {
				decision := Decision{
					Name:      "test",
					ModelRefs: []ModelRef{{Model: "test"}},
					Plugins: []DecisionPlugin{
						{
							Type: "semantic-cache",
							Configuration: map[string]interface{}{
								"enabled": false,
							},
						},
					},
				}

				cfg := &RouterConfig{
					SemanticCache: SemanticCache{
						Enabled: true,
					},
					IntelligentRouting: IntelligentRouting{
						Decisions: []Decision{decision},
					},
				}

				Expect(cfg.IsCacheEnabledForDecision("test")).To(BeFalse())
			})

			It("should return true when decision explicitly enables semantic-cache", func() {
				decision := Decision{
					Name:      "test",
					ModelRefs: []ModelRef{{Model: "test"}},
					Plugins: []DecisionPlugin{
						{
							Type: "semantic-cache",
							Configuration: map[string]interface{}{
								"enabled": true,
							},
						},
					},
				}

				cfg := &RouterConfig{
					SemanticCache: SemanticCache{
						Enabled: false, // Global disabled, but decision enables it
					},
					IntelligentRouting: IntelligentRouting{
						Decisions: []Decision{decision},
					},
				}

				Expect(cfg.IsCacheEnabledForDecision("test")).To(BeTrue())
			})

			It("should return false for non-existent decision", func() {
				cfg := &RouterConfig{
					SemanticCache: SemanticCache{
						Enabled: true,
					},
				}

				Expect(cfg.IsCacheEnabledForDecision("non_existent")).To(BeFalse())
			})

			It("should use decision-level similarity threshold when configured", func() {
				threshold := float32(0.95)
				decision := Decision{
					Name:      "test",
					ModelRefs: []ModelRef{{Model: "test"}},
					Plugins: []DecisionPlugin{
						{
							Type: "semantic-cache",
							Configuration: map[string]interface{}{
								"enabled":              true,
								"similarity_threshold": threshold,
							},
						},
					},
				}

				globalThreshold := float32(0.80)
				cfg := &RouterConfig{
					SemanticCache: SemanticCache{
						Enabled:             true,
						SimilarityThreshold: &globalThreshold,
					},
					IntelligentRouting: IntelligentRouting{
						Decisions: []Decision{decision},
					},
				}

				// Should use decision-level threshold, not global
				Expect(cfg.GetCacheSimilarityThresholdForDecision("test")).To(Equal(threshold))
			})

			It("should use decision-level TTL when configured", func() {
				ttl := 7200
				decision := Decision{
					Name:      "test",
					ModelRefs: []ModelRef{{Model: "test"}},
					Plugins: []DecisionPlugin{
						{
							Type: "semantic-cache",
							Configuration: map[string]interface{}{
								"enabled":     true,
								"ttl_seconds": ttl,
							},
						},
					},
				}

				cfg := &RouterConfig{
					SemanticCache: SemanticCache{
						Enabled:    true,
						TTLSeconds: 3600, // Global TTL
					},
					IntelligentRouting: IntelligentRouting{
						Decisions: []Decision{decision},
					},
				}

				// Should use decision-level TTL, not global
				Expect(cfg.GetCacheTTLSecondsForDecision("test")).To(Equal(ttl))
			})
		})
	})
})

var _ = Describe("MMLU categories in config YAML", func() {
	It("should unmarshal mmlu_categories into Category struct", func() {
		yamlContent := `
categories:
  - name: "tech"
    mmlu_categories: ["computer science", "engineering"]
  - name: "finance"
    mmlu_categories: ["economics"]
  - name: "politics"
`

		var cfg RouterConfig
		Expect(yaml.Unmarshal([]byte(yamlContent), &cfg)).To(Succeed())

		Expect(cfg.Categories).To(HaveLen(3))

		Expect(cfg.Categories[0].Name).To(Equal("tech"))
		Expect(cfg.Categories[0].MMLUCategories).To(ConsistOf("computer science", "engineering"))

		Expect(cfg.Categories[1].Name).To(Equal("finance"))
		Expect(cfg.Categories[1].MMLUCategories).To(ConsistOf("economics"))

		Expect(cfg.Categories[2].Name).To(Equal("politics"))
		Expect(cfg.Categories[2].MMLUCategories).To(BeEmpty())
	})
})

var _ = Describe("ParseConfigFile and ReplaceGlobalConfig", func() {
	var tempDir string

	BeforeEach(func() {
		var err error
		tempDir, err = os.MkdirTemp("", "config_parse_test")
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		os.RemoveAll(tempDir)
		ResetConfig()
	})

	It("should parse configuration via symlink path", func() {
		if runtime.GOOS == "windows" {
			Skip("symlink test is skipped on Windows")
		}

		// Create real config target
		target := filepath.Join(tempDir, "real-config.yaml")
		content := []byte("default_model: test-model\n")
		Expect(os.WriteFile(target, content, 0o644)).To(Succeed())

		// Create symlink pointing to target
		link := filepath.Join(tempDir, "link-config.yaml")
		Expect(os.Symlink(target, link)).To(Succeed())

		cfg, err := Parse(link)
		Expect(err).NotTo(HaveOccurred())
		Expect(cfg).NotTo(BeNil())
		Expect(cfg.DefaultModel).To(Equal("test-model"))
	})

	It("should return error when file does not exist", func() {
		_, err := Parse(filepath.Join(tempDir, "no-such.yaml"))
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("failed to read config file"))
	})

	It("should replace global config and reflect via GetConfig", func() {
		// new config instance
		newCfg := &RouterConfig{
			BackendModels: BackendModels{
				DefaultModel: "new-default",
			},
		}
		Replace(newCfg)
		got := Get()
		Expect(got).To(Equal(newCfg))
		Expect(got.BackendModels.DefaultModel).To(Equal("new-default"))
	})
})

var _ = Describe("IP Address Validation", func() {
	Describe("validateIPAddress", func() {
		Context("with valid IPv4 addresses", func() {
			It("should accept standard IPv4 addresses", func() {
				validIPv4Addresses := []string{
					"127.0.0.1",
					"192.168.1.1",
					"10.0.0.1",
					"172.16.0.1",
					"8.8.8.8",
					"255.255.255.255",
					"0.0.0.0",
				}

				for _, addr := range validIPv4Addresses {
					err := validateIPAddress(addr)
					Expect(err).NotTo(HaveOccurred(), "Expected %s to be valid", addr)
				}
			})
		})

		Context("with valid IPv6 addresses", func() {
			It("should accept standard IPv6 addresses", func() {
				validIPv6Addresses := []string{
					"::1",
					"2001:db8::1",
					"fe80::1",
					"2001:0db8:85a3:0000:0000:8a2e:0370:7334",
					"2001:db8:85a3::8a2e:370:7334",
					"::",
					"::ffff:192.0.2.1",
				}

				for _, addr := range validIPv6Addresses {
					err := validateIPAddress(addr)
					Expect(err).NotTo(HaveOccurred(), "Expected %s to be valid", addr)
				}
			})
		})

		Context("with domain names", func() {
			It("should reject domain names", func() {
				domainNames := []string{
					"example.com",
					"localhost",
					"api.openai.com",
					"subdomain.example.org",
					"test.local",
				}

				for _, domain := range domainNames {
					err := validateIPAddress(domain)
					Expect(err).To(HaveOccurred(), "Expected %s to be rejected", domain)
					Expect(err.Error()).To(ContainSubstring("invalid IP address format"))
				}
			})
		})

		Context("with protocol prefixes", func() {
			It("should reject HTTP/HTTPS prefixes", func() {
				protocolAddresses := []string{
					"http://127.0.0.1",
					"https://192.168.1.1",
					"http://example.com",
					"https://api.openai.com",
				}

				for _, addr := range protocolAddresses {
					err := validateIPAddress(addr)
					Expect(err).To(HaveOccurred(), "Expected %s to be rejected", addr)
					Expect(err.Error()).To(ContainSubstring("protocol prefixes"))
					Expect(err.Error()).To(ContainSubstring("are not supported"))
				}
			})
		})

		Context("with paths", func() {
			It("should reject addresses with paths", func() {
				pathAddresses := []string{
					"127.0.0.1/api",
					"192.168.1.1/health",
					"example.com/v1/api",
					"localhost/status",
				}

				for _, addr := range pathAddresses {
					err := validateIPAddress(addr)
					Expect(err).To(HaveOccurred(), "Expected %s to be rejected", addr)
					Expect(err.Error()).To(ContainSubstring("paths are not supported"))
				}
			})
		})

		Context("with port numbers", func() {
			It("should reject IPv4 addresses with port numbers", func() {
				ipv4PortAddresses := []string{
					"127.0.0.1:8080",
					"192.168.1.1:3000",
					"10.0.0.1:443",
				}

				for _, addr := range ipv4PortAddresses {
					err := validateIPAddress(addr)
					Expect(err).To(HaveOccurred(), "Expected %s to be rejected", addr)
					Expect(err.Error()).To(ContainSubstring("port numbers in address are not supported"))
					Expect(err.Error()).To(ContainSubstring("use 'port' field instead"))
				}
			})

			It("should reject IPv6 addresses with port numbers", func() {
				ipv6PortAddresses := []string{
					"[::1]:8080",
					"[2001:db8::1]:3000",
					"[fe80::1]:443",
				}

				for _, addr := range ipv6PortAddresses {
					err := validateIPAddress(addr)
					Expect(err).To(HaveOccurred(), "Expected %s to be rejected", addr)
					Expect(err.Error()).To(ContainSubstring("port numbers in address are not supported"))
					Expect(err.Error()).To(ContainSubstring("use 'port' field instead"))
				}
			})

			It("should reject domain names with port numbers", func() {
				domainPortAddresses := []string{
					"localhost:8000",
					"example.com:443",
				}

				for _, addr := range domainPortAddresses {
					err := validateIPAddress(addr)
					Expect(err).To(HaveOccurred(), "Expected %s to be rejected", addr)
					// These will be caught by domain detection, not port detection
					Expect(err.Error()).To(ContainSubstring("invalid IP address format"))
				}
			})
		})

		Context("with empty or invalid input", func() {
			It("should reject empty strings", func() {
				emptyInputs := []string{
					"",
					"   ",
					"\t",
					"\n",
				}

				for _, input := range emptyInputs {
					err := validateIPAddress(input)
					Expect(err).To(HaveOccurred(), "Expected '%s' to be rejected", input)
					Expect(err.Error()).To(ContainSubstring("address cannot be empty"))
				}
			})

			It("should reject invalid formats", func() {
				invalidFormats := []string{
					"not-an-ip",
					"256.256.256.256",
					"192.168.1",
					"192.168.1.1.1",
					"gggg::1",
				}

				for _, format := range invalidFormats {
					err := validateIPAddress(format)
					Expect(err).To(HaveOccurred(), "Expected %s to be rejected", format)
					Expect(err.Error()).To(ContainSubstring("invalid IP address format"))
				}
			})
		})
	})

	Describe("helper functions", func() {
		Describe("isValidIPv4", func() {
			It("should correctly identify IPv4 addresses", func() {
				Expect(isValidIPv4("127.0.0.1")).To(BeTrue())
				Expect(isValidIPv4("192.168.1.1")).To(BeTrue())
				Expect(isValidIPv4("::1")).To(BeFalse())
				Expect(isValidIPv4("example.com")).To(BeFalse())
			})
		})

		Describe("isValidIPv6", func() {
			It("should correctly identify IPv6 addresses", func() {
				Expect(isValidIPv6("::1")).To(BeTrue())
				Expect(isValidIPv6("2001:db8::1")).To(BeTrue())
				Expect(isValidIPv6("127.0.0.1")).To(BeFalse())
				Expect(isValidIPv6("example.com")).To(BeFalse())
			})
		})

		Describe("getIPAddressType", func() {
			It("should return correct IP address types", func() {
				Expect(getIPAddressType("127.0.0.1")).To(Equal("IPv4"))
				Expect(getIPAddressType("::1")).To(Equal("IPv6"))
				Expect(getIPAddressType("example.com")).To(Equal("invalid"))
			})
		})
	})
})

var _ = Describe("MCP Configuration Validation", func() {
	Describe("IsMCPCategoryClassifierEnabled", func() {
		var cfg *RouterConfig

		BeforeEach(func() {
			cfg = &RouterConfig{}
		})

		Context("when MCP is fully configured", func() {
			It("should return true", func() {
				cfg.Enabled = true
				cfg.ToolName = "classify_text"

				Expect(cfg.IsMCPCategoryClassifierEnabled()).To(BeTrue())
			})
		})

		Context("when MCP is not enabled", func() {
			It("should return false", func() {
				cfg.Enabled = false
				cfg.ToolName = "classify_text"

				Expect(cfg.IsMCPCategoryClassifierEnabled()).To(BeFalse())
			})
		})

		Context("when MCP tool name is empty", func() {
			It("should return false", func() {
				cfg.Enabled = true
				cfg.ToolName = ""

				Expect(cfg.IsMCPCategoryClassifierEnabled()).To(BeFalse())
			})
		})

		Context("when both enabled and tool name are missing", func() {
			It("should return false", func() {
				cfg.Enabled = false
				cfg.ToolName = ""

				Expect(cfg.IsMCPCategoryClassifierEnabled()).To(BeFalse())
			})
		})
	})

	Describe("MCP Configuration Structure", func() {
		var cfg *RouterConfig

		BeforeEach(func() {
			cfg = &RouterConfig{}
		})

		Context("when configuring stdio transport", func() {
			It("should accept valid stdio configuration", func() {
				cfg.Classifier.MCPCategoryModel.Enabled = true
				cfg.Classifier.MCPCategoryModel.TransportType = "stdio"
				cfg.Classifier.MCPCategoryModel.Command = "python"
				cfg.Classifier.MCPCategoryModel.Args = []string{"server_keyword.py"}
				cfg.Classifier.MCPCategoryModel.ToolName = "classify_text"
				cfg.Classifier.MCPCategoryModel.Threshold = 0.5
				cfg.Classifier.MCPCategoryModel.TimeoutSeconds = 30

				Expect(cfg.Classifier.MCPCategoryModel.Enabled).To(BeTrue())
				Expect(cfg.Classifier.MCPCategoryModel.TransportType).To(Equal("stdio"))
				Expect(cfg.Classifier.MCPCategoryModel.Command).To(Equal("python"))
				Expect(cfg.Classifier.MCPCategoryModel.Args).To(HaveLen(1))
				Expect(cfg.Classifier.MCPCategoryModel.ToolName).To(Equal("classify_text"))
				Expect(cfg.Classifier.MCPCategoryModel.Threshold).To(BeNumerically("==", 0.5))
				Expect(cfg.Classifier.MCPCategoryModel.TimeoutSeconds).To(Equal(30))
			})

			It("should accept environment variables", func() {
				cfg.Env = map[string]string{
					"PYTHONPATH": "/app/lib",
					"LOG_LEVEL":  "debug",
				}

				Expect(cfg.Classifier.MCPCategoryModel.Env).To(HaveLen(2))
				Expect(cfg.Classifier.MCPCategoryModel.Env["PYTHONPATH"]).To(Equal("/app/lib"))
				Expect(cfg.Classifier.MCPCategoryModel.Env["LOG_LEVEL"]).To(Equal("debug"))
			})
		})

		Context("when configuring HTTP transport", func() {
			It("should accept valid HTTP configuration", func() {
				cfg.Enabled = true
				cfg.TransportType = "http"
				cfg.URL = "http://localhost:8080/mcp"
				cfg.ToolName = "classify_text"

				Expect(cfg.Classifier.MCPCategoryModel.TransportType).To(Equal("http"))
				Expect(cfg.Classifier.MCPCategoryModel.URL).To(Equal("http://localhost:8080/mcp"))
			})
		})

		Context("when threshold is not set", func() {
			It("should default to zero", func() {
				cfg.Enabled = true
				cfg.ToolName = "classify_text"

				Expect(cfg.Classifier.MCPCategoryModel.Threshold).To(BeNumerically("==", 0.0))
			})
		})

		Context("when configuring custom threshold", func() {
			It("should accept threshold values between 0 and 1", func() {
				testCases := []float32{0.0, 0.3, 0.5, 0.7, 0.9, 1.0}

				for _, threshold := range testCases {
					cfg.MCPCategoryModel.Threshold = threshold
					Expect(cfg.Classifier.MCPCategoryModel.Threshold).To(BeNumerically("==", threshold))
				}
			})
		})

		Context("when timeout is not set", func() {
			It("should default to zero", func() {
				cfg.Enabled = true
				cfg.ToolName = "classify_text"

				Expect(cfg.Classifier.MCPCategoryModel.TimeoutSeconds).To(Equal(0))
			})
		})
	})

	Describe("MCP vs In-tree Classifier Priority", func() {
		var cfg *RouterConfig

		BeforeEach(func() {
			cfg = &RouterConfig{}
		})

		Context("when both in-tree and MCP are configured", func() {
			It("should have both configurations available", func() {
				// Configure in-tree classifier
				cfg.Classifier.CategoryModel.ModelID = "/path/to/model"
				cfg.Classifier.CategoryModel.CategoryMappingPath = "/path/to/mapping.json"
				cfg.Classifier.CategoryModel.Threshold = 0.7

				// Configure MCP classifier
				cfg.Classifier.MCPCategoryModel.Enabled = true
				cfg.Classifier.MCPCategoryModel.ToolName = "classify_text"
				cfg.Classifier.MCPCategoryModel.Threshold = 0.5

				// Both should be configured
				Expect(cfg.Classifier.CategoryModel.ModelID).ToNot(BeEmpty())
				Expect(cfg.Classifier.MCPCategoryModel.Enabled).To(BeTrue())
			})
		})

		Context("when only in-tree is configured", func() {
			It("should not have MCP enabled", func() {
				cfg.CategoryModel.ModelID = "/path/to/model"
				cfg.CategoryMappingPath = "/path/to/mapping.json"

				Expect(cfg.Classifier.CategoryModel.ModelID).ToNot(BeEmpty())
				Expect(cfg.IsMCPCategoryClassifierEnabled()).To(BeFalse())
			})
		})

		Context("when only MCP is configured", func() {
			It("should have MCP enabled and no in-tree model", func() {
				cfg.Enabled = true
				cfg.ToolName = "classify_text"

				Expect(cfg.IsMCPCategoryClassifierEnabled()).To(BeTrue())
				Expect(cfg.Classifier.CategoryModel.ModelID).To(BeEmpty())
			})
		})

		Context("when neither is configured", func() {
			It("should have neither enabled", func() {
				Expect(cfg.Classifier.CategoryModel.ModelID).To(BeEmpty())
				Expect(cfg.IsMCPCategoryClassifierEnabled()).To(BeFalse())
			})
		})
	})

	Describe("MCP Configuration Fields", func() {
		var cfg *RouterConfig

		BeforeEach(func() {
			cfg = &RouterConfig{}
		})

		It("should support all required fields for stdio transport", func() {
			cfg.Classifier.MCPCategoryModel.Enabled = true
			cfg.Classifier.MCPCategoryModel.TransportType = "stdio"
			cfg.Classifier.MCPCategoryModel.Command = "python3"
			cfg.Classifier.MCPCategoryModel.Args = []string{"-m", "server"}
			cfg.Classifier.MCPCategoryModel.Env = map[string]string{"DEBUG": "1"}
			cfg.Classifier.MCPCategoryModel.ToolName = "classify"
			cfg.Classifier.MCPCategoryModel.Threshold = 0.6
			cfg.Classifier.MCPCategoryModel.TimeoutSeconds = 60

			Expect(cfg.Classifier.MCPCategoryModel.Enabled).To(BeTrue())
			Expect(cfg.Classifier.MCPCategoryModel.TransportType).To(Equal("stdio"))
			Expect(cfg.Classifier.MCPCategoryModel.Command).To(Equal("python3"))
			Expect(cfg.Classifier.MCPCategoryModel.Args).To(Equal([]string{"-m", "server"}))
			Expect(cfg.Classifier.MCPCategoryModel.Env).To(HaveKeyWithValue("DEBUG", "1"))
			Expect(cfg.Classifier.MCPCategoryModel.ToolName).To(Equal("classify"))
			Expect(cfg.Classifier.MCPCategoryModel.Threshold).To(BeNumerically("~", 0.6, 0.01))
			Expect(cfg.Classifier.MCPCategoryModel.TimeoutSeconds).To(Equal(60))
		})

		It("should support all required fields for HTTP transport", func() {
			cfg.Classifier.MCPCategoryModel.Enabled = true
			cfg.Classifier.MCPCategoryModel.TransportType = "http"
			cfg.Classifier.MCPCategoryModel.URL = "https://mcp-server:443/api"
			cfg.Classifier.MCPCategoryModel.ToolName = "classify"
			cfg.Classifier.MCPCategoryModel.Threshold = 0.8
			cfg.Classifier.MCPCategoryModel.TimeoutSeconds = 120

			Expect(cfg.Classifier.MCPCategoryModel.Enabled).To(BeTrue())
			Expect(cfg.Classifier.MCPCategoryModel.TransportType).To(Equal("http"))
			Expect(cfg.Classifier.MCPCategoryModel.URL).To(Equal("https://mcp-server:443/api"))
			Expect(cfg.Classifier.MCPCategoryModel.ToolName).To(Equal("classify"))
			Expect(cfg.Classifier.MCPCategoryModel.Threshold).To(BeNumerically("~", 0.8, 0.01))
			Expect(cfg.Classifier.MCPCategoryModel.TimeoutSeconds).To(Equal(120))
		})

		It("should allow optional fields to be omitted", func() {
			cfg.Enabled = true
			cfg.TransportType = "stdio"
			cfg.Command = "server"
			cfg.ToolName = "classify"

			// Optional fields should have zero values
			Expect(cfg.Classifier.MCPCategoryModel.Args).To(BeNil())
			Expect(cfg.Classifier.MCPCategoryModel.Env).To(BeNil())
			Expect(cfg.Classifier.MCPCategoryModel.URL).To(BeEmpty())
			Expect(cfg.Classifier.MCPCategoryModel.Threshold).To(BeNumerically("==", 0.0))
			Expect(cfg.Classifier.MCPCategoryModel.TimeoutSeconds).To(Equal(0))
		})
	})
})

// ResetConfig resets the singleton config for testing purposes
// This is needed to ensure test isolation
func ResetConfig() {
	configOnce = sync.Once{}
	config = nil
	configErr = nil
}

var _ = Describe("Hallucination Mitigation Configuration", func() {
	var (
		tempDir    string
		configFile string
	)

	BeforeEach(func() {
		var err error
		tempDir, err = os.MkdirTemp("", "hallucination_config_test")
		Expect(err).NotTo(HaveOccurred())
		configFile = filepath.Join(tempDir, "config.yaml")
	})

	AfterEach(func() {
		os.RemoveAll(tempDir)
		ResetConfig()
	})

	Describe("HallucinationMitigationConfig Parsing", func() {
		Context("with full hallucination mitigation configuration", func() {
			BeforeEach(func() {
				configContent := `
hallucination_mitigation:
  enabled: true
  fact_check_model:
    model_id: "models/fact_check_classifier"
    threshold: 0.75
    use_cpu: true
  hallucination_model:
    model_id: "models/hallucination_detect_modernbert"
    threshold: 0.6
    use_cpu: true
  on_hallucination_detected: "block"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should parse hallucination mitigation configuration correctly", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.HallucinationMitigation.Enabled).To(BeTrue())
				Expect(cfg.HallucinationMitigation.FactCheckModel.ModelID).To(Equal("models/fact_check_classifier"))
				Expect(cfg.HallucinationMitigation.FactCheckModel.Threshold).To(Equal(float32(0.75)))
				Expect(cfg.HallucinationMitigation.FactCheckModel.UseCPU).To(BeTrue())
				Expect(cfg.HallucinationMitigation.HallucinationModel.ModelID).To(Equal("models/hallucination_detect_modernbert"))
				Expect(cfg.HallucinationMitigation.HallucinationModel.Threshold).To(Equal(float32(0.6)))
				Expect(cfg.HallucinationMitigation.HallucinationModel.UseCPU).To(BeTrue())
				Expect(cfg.HallucinationMitigation.OnHallucinationDetected).To(Equal("block"))
			})
		})

		Context("with minimal hallucination mitigation configuration", func() {
			BeforeEach(func() {
				configContent := `
hallucination_mitigation:
  enabled: true
  fact_check_model:
    model_id: "models/fact_check"
    mapping_path: "config/mapping.json"
  hallucination_model:
    model_id: "models/hallucination"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should parse with default values for optional fields", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.HallucinationMitigation.Enabled).To(BeTrue())
				Expect(cfg.HallucinationMitigation.FactCheckModel.Threshold).To(Equal(float32(0)))
				Expect(cfg.HallucinationMitigation.HallucinationModel.Threshold).To(Equal(float32(0)))
				Expect(cfg.HallucinationMitigation.OnHallucinationDetected).To(BeEmpty())
			})
		})

		Context("with hallucination mitigation disabled", func() {
			BeforeEach(func() {
				configContent := `
hallucination_mitigation:
  enabled: false
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should have enabled set to false", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.HallucinationMitigation.Enabled).To(BeFalse())
			})
		})

		Context("with missing hallucination_mitigation section", func() {
			BeforeEach(func() {
				configContent := `
default_model: "test-model"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should have hallucination mitigation disabled by default", func() {
				cfg, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.HallucinationMitigation.Enabled).To(BeFalse())
				Expect(cfg.HallucinationMitigation.FactCheckModel.ModelID).To(BeEmpty())
				Expect(cfg.HallucinationMitigation.HallucinationModel.ModelID).To(BeEmpty())
			})
		})
	})

	Describe("IsHallucinationMitigationEnabled", func() {
		It("should return true when enabled is true", func() {
			cfg := &RouterConfig{}
			cfg.HallucinationMitigation.Enabled = true

			Expect(cfg.IsHallucinationMitigationEnabled()).To(BeTrue())
		})

		It("should return false when enabled is false", func() {
			cfg := &RouterConfig{}
			cfg.HallucinationMitigation.Enabled = false

			Expect(cfg.IsHallucinationMitigationEnabled()).To(BeFalse())
		})

		It("should return false for zero-value config", func() {
			cfg := &RouterConfig{}

			Expect(cfg.IsHallucinationMitigationEnabled()).To(BeFalse())
		})
	})

	Describe("IsFactCheckClassifierEnabled", func() {
		It("should return true when fully configured with legacy config", func() {
			cfg := &RouterConfig{}
			cfg.HallucinationMitigation.Enabled = true
			cfg.HallucinationMitigation.FactCheckModel.ModelID = "models/fact_check"

			Expect(cfg.IsFactCheckClassifierEnabled()).To(BeTrue())
		})

		It("should return true when fact_check_rules are configured", func() {
			cfg := &RouterConfig{}
			cfg.FactCheckRules = []FactCheckRule{
				{Name: "needs_fact_check", Description: "Query needs fact verification"},
				{Name: "no_fact_check_needed", Description: "Query does not need fact verification"},
			}
			cfg.HallucinationMitigation.FactCheckModel.ModelID = "models/fact_check"

			Expect(cfg.IsFactCheckClassifierEnabled()).To(BeTrue())
		})

		It("should return false when hallucination mitigation is disabled and no fact_check_rules", func() {
			cfg := &RouterConfig{}
			cfg.HallucinationMitigation.Enabled = false
			cfg.HallucinationMitigation.FactCheckModel.ModelID = "models/fact_check"

			Expect(cfg.IsFactCheckClassifierEnabled()).To(BeFalse())
		})

		It("should return false when model_id is missing", func() {
			cfg := &RouterConfig{}
			cfg.HallucinationMitigation.Enabled = true

			Expect(cfg.IsFactCheckClassifierEnabled()).To(BeFalse())
		})
	})

	Describe("GetFactCheckRules", func() {
		It("should return all configured fact_check_rules", func() {
			cfg := &RouterConfig{}
			cfg.FactCheckRules = []FactCheckRule{
				{Name: "needs_fact_check", Description: "Needs verification"},
				{Name: "no_fact_check_needed", Description: "No verification needed"},
			}

			rules := cfg.GetFactCheckRules()
			Expect(rules).To(HaveLen(2))
			Expect(rules[0].Name).To(Equal("needs_fact_check"))
			Expect(rules[1].Name).To(Equal("no_fact_check_needed"))
		})

		It("should return empty slice when no rules configured", func() {
			cfg := &RouterConfig{}

			rules := cfg.GetFactCheckRules()
			Expect(rules).To(BeEmpty())
		})
	})

	Describe("IsHallucinationModelEnabled", func() {
		It("should return true when fully configured", func() {
			cfg := &RouterConfig{}
			cfg.HallucinationMitigation.Enabled = true
			cfg.HallucinationMitigation.HallucinationModel.ModelID = "models/hallucination"

			Expect(cfg.IsHallucinationModelEnabled()).To(BeTrue())
		})

		It("should return false when hallucination mitigation is disabled", func() {
			cfg := &RouterConfig{}
			cfg.HallucinationMitigation.Enabled = false
			cfg.HallucinationMitigation.HallucinationModel.ModelID = "models/hallucination"

			Expect(cfg.IsHallucinationModelEnabled()).To(BeFalse())
		})

		It("should return false when model_id is missing", func() {
			cfg := &RouterConfig{}
			cfg.HallucinationMitigation.Enabled = true

			Expect(cfg.IsHallucinationModelEnabled()).To(BeFalse())
		})
	})

	Describe("GetFactCheckThreshold", func() {
		It("should return configured threshold when set", func() {
			cfg := &RouterConfig{}
			cfg.HallucinationMitigation.FactCheckModel.Threshold = 0.85

			Expect(cfg.GetFactCheckThreshold()).To(Equal(float32(0.85)))
		})

		It("should return default 0.7 when threshold is not set", func() {
			cfg := &RouterConfig{}

			Expect(cfg.GetFactCheckThreshold()).To(Equal(float32(0.7)))
		})

		It("should return default 0.7 when threshold is zero", func() {
			cfg := &RouterConfig{}
			cfg.HallucinationMitigation.FactCheckModel.Threshold = 0

			Expect(cfg.GetFactCheckThreshold()).To(Equal(float32(0.7)))
		})
	})

	Describe("GetHallucinationModelThreshold", func() {
		It("should return configured threshold when set", func() {
			cfg := &RouterConfig{}
			cfg.HallucinationMitigation.HallucinationModel.Threshold = 0.65

			Expect(cfg.GetHallucinationModelThreshold()).To(Equal(float32(0.65)))
		})

		It("should return default 0.5 when threshold is not set", func() {
			cfg := &RouterConfig{}

			Expect(cfg.GetHallucinationModelThreshold()).To(Equal(float32(0.5)))
		})

		It("should return default 0.5 when threshold is zero", func() {
			cfg := &RouterConfig{}
			cfg.HallucinationMitigation.HallucinationModel.Threshold = 0

			Expect(cfg.GetHallucinationModelThreshold()).To(Equal(float32(0.5)))
		})
	})

	Describe("GetHallucinationAction", func() {
		It("should return 'warn' when action is 'warn'", func() {
			cfg := &RouterConfig{}
			cfg.HallucinationMitigation.OnHallucinationDetected = "warn"

			Expect(cfg.GetHallucinationAction()).To(Equal("warn"))
		})

		It("should return 'warn' when action is 'block' (only warn is supported for global config)", func() {
			cfg := &RouterConfig{}
			cfg.HallucinationMitigation.OnHallucinationDetected = "block"

			// Global config only supports "warn" action
			// Per-decision plugin config supports "header", "body", "none", "block"
			Expect(cfg.GetHallucinationAction()).To(Equal("warn"))
		})

		It("should return default 'warn' when action is empty", func() {
			cfg := &RouterConfig{}

			Expect(cfg.GetHallucinationAction()).To(Equal("warn"))
		})

		It("should return default 'warn' when action is invalid", func() {
			cfg := &RouterConfig{}
			cfg.HallucinationMitigation.OnHallucinationDetected = "invalid"

			Expect(cfg.GetHallucinationAction()).To(Equal("warn"))
		})
	})

	Describe("Decision.GetMemoryConfig", func() {
		It("should return nil when no memory plugin is configured", func() {
			decision := &Decision{
				Name: "test_decision",
				Plugins: []DecisionPlugin{
					{Type: "pii", Configuration: map[string]interface{}{"enabled": true}},
				},
			}

			Expect(decision.GetMemoryConfig()).To(BeNil())
		})

		It("should return memory config when memory plugin is configured", func() {
			decision := &Decision{
				Name: "test_decision",
				Plugins: []DecisionPlugin{
					{
						Type: "memory",
						Configuration: map[string]interface{}{
							"enabled":              true,
							"retrieval_limit":      10,
							"similarity_threshold": 0.75,
							"auto_store":           true,
						},
					},
				},
			}

			memConfig := decision.GetMemoryConfig()
			Expect(memConfig).NotTo(BeNil())
			Expect(memConfig.Enabled).To(BeTrue())
			Expect(*memConfig.RetrievalLimit).To(Equal(10))
			Expect(*memConfig.SimilarityThreshold).To(BeNumerically("~", 0.75, 0.001))
			Expect(*memConfig.AutoStore).To(BeTrue())
		})

		It("should return memory config with partial settings", func() {
			decision := &Decision{
				Name: "test_decision",
				Plugins: []DecisionPlugin{
					{
						Type: "memory",
						Configuration: map[string]interface{}{
							"enabled": false,
							// Only enabled is set, other fields are nil
						},
					},
				},
			}

			memConfig := decision.GetMemoryConfig()
			Expect(memConfig).NotTo(BeNil())
			Expect(memConfig.Enabled).To(BeFalse())
			Expect(memConfig.RetrievalLimit).To(BeNil())
			Expect(memConfig.SimilarityThreshold).To(BeNil())
			Expect(memConfig.AutoStore).To(BeNil())
		})
	})
})
