package config_test

import (
	"os"
	"path/filepath"
	"sync"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
		config.ResetConfig()
	})

	Describe("LoadConfig", func() {
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
    model_scores:
      - model: "model-a"
        score: 0.9
        use_reasoning: true
      - model: "model-b"
        score: 0.8
        use_reasoning: false

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
    pii_policy:
      allow_by_default: false
      pii_types_allowed: ["NO_PII", "ORGANIZATION"]
    preferred_endpoints: ["endpoint1"]
  "model-b":
    pii_policy:
      allow_by_default: true
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
				cfg, err := config.LoadConfig(configFile)
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
				Expect(cfg.Categories[0].ModelScores).To(HaveLen(2))

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
				Expect(cfg.ModelConfig["model-a"].PIIPolicy.AllowByDefault).To(BeFalse())
				Expect(cfg.ModelConfig["model-a"].PIIPolicy.PIITypes).To(ContainElements("NO_PII", "ORGANIZATION"))

				// Verify tools config
				Expect(cfg.Tools.Enabled).To(BeTrue())
				Expect(cfg.Tools.TopK).To(Equal(5))
				Expect(*cfg.Tools.SimilarityThreshold).To(Equal(float32(0.8)))

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
				cfg1, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				cfg2, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg1).To(BeIdenticalTo(cfg2))
			})
		})

		Context("with missing config file", func() {
			It("should return an error", func() {
				cfg, err := config.LoadConfig("/nonexistent/config.yaml")
				Expect(err).To(HaveOccurred())
				Expect(cfg).To(BeNil())
				Expect(err.Error()).To(ContainSubstring("failed to read config file"))
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
				cfg, err := config.LoadConfig(configFile)
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
				cfg, err := config.LoadConfig(configFile)
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

			It("should handle concurrent LoadConfig calls safely", func() {
				const numGoroutines = 10
				var wg sync.WaitGroup
				results := make([]*config.RouterConfig, numGoroutines)
				errors := make([]error, numGoroutines)

				wg.Add(numGoroutines)
				for i := 0; i < numGoroutines; i++ {
					go func(index int) {
						defer wg.Done()
						cfg, err := config.LoadConfig(configFile)
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
				cfg, err := config.LoadConfig(configFile)
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
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				threshold := cfg.GetCacheSimilarityThreshold()
				Expect(threshold).To(Equal(float32(0.8)))
			})
		})
	})

	Describe("GetModelForCategoryIndex", func() {
		BeforeEach(func() {
			configContent := `
categories:
  - name: "category1"
    model_scores:
      - model: "model1"
        score: 0.9
        use_reasoning: true
      - model: "model2"
        score: 0.8
        use_reasoning: false
  - name: "category2"
    model_scores:
      - model: "model3"
        score: 0.95
        use_reasoning: true
default_model: "default-model"
`
			err := os.WriteFile(configFile, []byte(configContent), 0o644)
			Expect(err).NotTo(HaveOccurred())
		})

		Context("with valid category index", func() {
			It("should return the best model for the category", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				model := cfg.GetModelForCategoryIndex(0)
				Expect(model).To(Equal("model1"))

				model = cfg.GetModelForCategoryIndex(1)
				Expect(model).To(Equal("model3"))
			})
		})

		Context("with invalid category index", func() {
			It("should return the default model for negative index", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				model := cfg.GetModelForCategoryIndex(-1)
				Expect(model).To(Equal("default-model"))
			})

			It("should return the default model for index beyond range", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				model := cfg.GetModelForCategoryIndex(10)
				Expect(model).To(Equal("default-model"))
			})
		})

		Context("with category having no models", func() {
			BeforeEach(func() {
				configContent := `
categories:
  - name: "empty_category"
    model_scores:
      - model: "fallback-model"
        score: 0.5
        use_reasoning: false
default_model: "fallback-model"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should return the default model", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				model := cfg.GetModelForCategoryIndex(0)
				Expect(model).To(Equal("fallback-model"))
			})
		})
	})

	Describe("PII Policy Functions", func() {
		BeforeEach(func() {
			configContent := `
model_config:
  "strict-model":
    pii_policy:
      allow_by_default: false
      pii_types_allowed: ["NO_PII", "ORGANIZATION"]
  "permissive-model":
    pii_policy:
      allow_by_default: true
  "unconfigured-model":
    pii_policy:
      allow_by_default: true
`
			err := os.WriteFile(configFile, []byte(configContent), 0o644)
			Expect(err).NotTo(HaveOccurred())
		})

		Describe("GetModelPIIPolicy", func() {
			It("should return configured PII policy for existing model", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				policy := cfg.GetModelPIIPolicy("strict-model")
				Expect(policy.AllowByDefault).To(BeFalse())
				Expect(policy.PIITypes).To(ContainElements("NO_PII", "ORGANIZATION"))

				policy = cfg.GetModelPIIPolicy("permissive-model")
				Expect(policy.AllowByDefault).To(BeTrue())
			})

			It("should return default allow-all policy for non-existent model", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				policy := cfg.GetModelPIIPolicy("non-existent-model")
				Expect(policy.AllowByDefault).To(BeTrue())
				Expect(policy.PIITypes).To(BeEmpty())
			})
		})

		Describe("IsModelAllowedForPIIType", func() {
			It("should allow all PII types when allow_by_default is true", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.IsModelAllowedForPIIType("permissive-model", config.PIITypePerson)).To(BeTrue())
				Expect(cfg.IsModelAllowedForPIIType("permissive-model", config.PIITypeCreditCard)).To(BeTrue())
				Expect(cfg.IsModelAllowedForPIIType("permissive-model", config.PIITypeEmailAddress)).To(BeTrue())
			})

			It("should only allow explicitly permitted PII types when allow_by_default is false", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				// Should allow explicitly listed PII types
				Expect(cfg.IsModelAllowedForPIIType("strict-model", config.PIITypeNoPII)).To(BeTrue())
				Expect(cfg.IsModelAllowedForPIIType("strict-model", config.PIITypeOrganization)).To(BeTrue())

				// Should deny non-listed PII types
				Expect(cfg.IsModelAllowedForPIIType("strict-model", config.PIITypePerson)).To(BeFalse())
				Expect(cfg.IsModelAllowedForPIIType("strict-model", config.PIITypeCreditCard)).To(BeFalse())
				Expect(cfg.IsModelAllowedForPIIType("strict-model", config.PIITypeEmailAddress)).To(BeFalse())
			})

			It("should handle unknown models with default allow-all policy", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.IsModelAllowedForPIIType("unknown-model", config.PIITypePerson)).To(BeTrue())
				Expect(cfg.IsModelAllowedForPIIType("unknown-model", config.PIITypeCreditCard)).To(BeTrue())
			})
		})

		Describe("IsModelAllowedForPIITypes", func() {
			It("should return true when all PII types are allowed", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				piiTypes := []string{config.PIITypeNoPII, config.PIITypeOrganization}
				Expect(cfg.IsModelAllowedForPIITypes("strict-model", piiTypes)).To(BeTrue())
			})

			It("should return false when any PII type is not allowed", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				piiTypes := []string{config.PIITypeNoPII, config.PIITypePerson}
				Expect(cfg.IsModelAllowedForPIITypes("strict-model", piiTypes)).To(BeFalse())
			})

			It("should return true for empty PII types list", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.IsModelAllowedForPIITypes("strict-model", []string{})).To(BeTrue())
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

				cfg, err := config.LoadConfig(configFile)
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

				cfg, err := config.LoadConfig(configFile)
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

				cfg, err := config.LoadConfig(configFile)
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

				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.IsCategoryClassifierEnabled()).To(BeTrue())
			})

			It("should return false when not configured", func() {
				// Create an empty config file
				err := os.WriteFile(configFile, []byte(""), 0o644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := config.LoadConfig(configFile)
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

				cfg, err := config.LoadConfig(configFile)
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

				cfg, err := config.LoadConfig(configFile)
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

				cfg, err := config.LoadConfig(configFile)
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
				cfg, err := config.LoadConfig(configFile)
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
				cfg, err := config.LoadConfig(configFile)
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

				cfg, err := config.LoadConfig(configFile)
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

			cfg, err := config.LoadConfig(configFile)
			Expect(err).NotTo(HaveOccurred())
			Expect(cfg.BertModel.Threshold).To(Equal(float32(0)))
			Expect(cfg.SemanticCache.MaxEntries).To(Equal(0))
			Expect(cfg.SemanticCache.TTLSeconds).To(Equal(0))
		})

		It("should handle very large numeric values", func() {
			configContent := `
model_config:
  "large-model":
    pii_policy:
      allow_by_default: true
`
			err := os.WriteFile(configFile, []byte(configContent), 0o644)
			Expect(err).NotTo(HaveOccurred())

			cfg, err := config.LoadConfig(configFile)
			Expect(err).NotTo(HaveOccurred())
			Expect(cfg.ModelConfig["large-model"].PIIPolicy.AllowByDefault).To(BeTrue())
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

			cfg, err := config.LoadConfig(configFile)
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
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				endpoints := cfg.GetEndpointsForModel("model-a")
				Expect(endpoints).To(HaveLen(2))
				endpointNames := []string{endpoints[0].Name, endpoints[1].Name}
				Expect(endpointNames).To(ContainElements("endpoint1", "endpoint3"))
			})

			It("should return empty slice when no preferred endpoints configured", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				endpoints := cfg.GetEndpointsForModel("model-c")
				Expect(endpoints).To(BeEmpty())
			})

			It("should return empty slice for non-existent model", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				endpoints := cfg.GetEndpointsForModel("non-existent-model")
				Expect(endpoints).To(BeEmpty())
			})

			It("should return only preferred endpoints", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				// model-b has preferred endpoint2
				endpoints := cfg.GetEndpointsForModel("model-b")
				Expect(endpoints).To(HaveLen(1))
				Expect(endpoints[0].Name).To(Equal("endpoint2"))
			})
		})

		Describe("GetEndpointByName", func() {
			It("should return endpoint when it exists", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				endpoint, found := cfg.GetEndpointByName("endpoint1")
				Expect(found).To(BeTrue())
				Expect(endpoint.Name).To(Equal("endpoint1"))
				Expect(endpoint.Address).To(Equal("127.0.0.1"))
				Expect(endpoint.Port).To(Equal(8000))
			})

			It("should return false when endpoint doesn't exist", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				endpoint, found := cfg.GetEndpointByName("non-existent")
				Expect(found).To(BeFalse())
				Expect(endpoint).To(BeNil())
			})
		})

		Describe("GetAllModels", func() {
			It("should return all models from model_config", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				models := cfg.GetAllModels()
				Expect(models).To(HaveLen(3))
				Expect(models).To(ContainElements("model-a", "model-b", "model-c"))
			})
		})

		Describe("SelectBestEndpointForModel", func() {
			It("should select endpoint with highest weight when multiple available", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				// model-a has preferred endpoints: endpoint1 (weight 1) and endpoint3 (weight 1)
				// Since they have the same weight, it should return the first one found
				endpointName, found := cfg.SelectBestEndpointForModel("model-a")
				Expect(found).To(BeTrue())
				Expect(endpointName).To(BeElementOf("endpoint1", "endpoint3"))
			})

			It("should return false for non-existent model", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				endpointName, found := cfg.SelectBestEndpointForModel("non-existent-model")
				Expect(found).To(BeFalse())
				Expect(endpointName).To(BeEmpty())
			})

			It("should return false when model has no preferred endpoints", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				endpointName, found := cfg.SelectBestEndpointForModel("model-c")
				Expect(found).To(BeFalse())
				Expect(endpointName).To(BeEmpty())
			})
		})

		Describe("ValidateEndpoints", func() {
			It("should pass validation when all models have endpoints", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				err = cfg.ValidateEndpoints()
				Expect(err).NotTo(HaveOccurred())
			})

			It("should fail validation when a category model has no endpoints", func() {
				// Add a model to categories that doesn't have preferred_endpoints configured
				configContent := `
vllm_endpoints:
  - name: "endpoint1"
    address: "127.0.0.1"
    port: 8000
    weight: 1

model_config:
  "existing-model":
    preferred_endpoints: ["endpoint1"]

categories:
  - name: "test"
    model_scores:
      - model: "missing-model"
        score: 0.9
        use_reasoning: true

default_model: "existing-model"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				err = cfg.ValidateEndpoints()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("missing-model"))
				Expect(err.Error()).To(ContainSubstring("no available endpoints"))
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

				cfg, err := config.LoadConfig(configFile)
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

					cfg, err := config.LoadConfig(configFile)
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

					cfg, err := config.LoadConfig(configFile)
					Expect(err).NotTo(HaveOccurred())
					Expect(cfg.VLLMEndpoints[0].Address).To(Equal("::1"))
				})
			})

			Context("with invalid address formats", func() {
				It("should reject domain names", func() {
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

					_, err = config.LoadConfig(configFile)
					Expect(err).To(HaveOccurred())
					Expect(err.Error()).To(ContainSubstring("endpoint1"))
					Expect(err.Error()).To(ContainSubstring("address validation failed"))
					Expect(err.Error()).To(ContainSubstring("invalid IP address format"))
				})

				It("should reject protocol prefixes", func() {
					configContent := `
vllm_endpoints:
  - name: "endpoint1"
    address: "http://127.0.0.1"
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

					_, err = config.LoadConfig(configFile)
					Expect(err).To(HaveOccurred())
					Expect(err.Error()).To(ContainSubstring("protocol prefixes"))
					Expect(err.Error()).To(ContainSubstring("are not supported"))
				})

				It("should reject addresses with paths", func() {
					configContent := `
vllm_endpoints:
  - name: "endpoint1"
    address: "127.0.0.1/api"
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

					_, err = config.LoadConfig(configFile)
					Expect(err).To(HaveOccurred())
					Expect(err.Error()).To(ContainSubstring("paths are not supported"))
				})

				It("should reject addresses with port numbers", func() {
					configContent := `
vllm_endpoints:
  - name: "endpoint1"
    address: "127.0.0.1:8080"
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

					_, err = config.LoadConfig(configFile)
					Expect(err).To(HaveOccurred())
					Expect(err.Error()).To(ContainSubstring("port numbers in address are not supported"))
					Expect(err.Error()).To(ContainSubstring("use 'port' field instead"))
				})

				It("should provide comprehensive error messages", func() {
					configContent := `
vllm_endpoints:
  - name: "test-endpoint"
    address: "https://example.com"
    port: 8000
    weight: 1

model_config:
  "test-model":
    preferred_endpoints: ["test-endpoint"]

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

					_, err = config.LoadConfig(configFile)
					Expect(err).To(HaveOccurred())

					errorMsg := err.Error()
					Expect(errorMsg).To(ContainSubstring("test-endpoint"))
					Expect(errorMsg).To(ContainSubstring("Supported formats"))
					Expect(errorMsg).To(ContainSubstring("IPv4: 192.168.1.1"))
					Expect(errorMsg).To(ContainSubstring("IPv6: ::1"))
					Expect(errorMsg).To(ContainSubstring("Unsupported formats"))
					Expect(errorMsg).To(ContainSubstring("Domain names: example.com"))
					Expect(errorMsg).To(ContainSubstring("Protocol prefixes: http://"))
				})
			})

			Context("with multiple endpoints", func() {
				It("should validate all endpoints", func() {
					configContent := `
vllm_endpoints:
  - name: "endpoint1"
    address: "127.0.0.1"
    port: 8000
    weight: 1
  - name: "endpoint2"
    address: "example.com"
    port: 8001
    weight: 1

model_config:
  "test-model1":
    preferred_endpoints: ["endpoint1"]
  "test-model2":
    preferred_endpoints: ["endpoint2"]

categories:
  - name: "test"
    model_scores:
      - model: "test-model1"
        score: 0.9
        use_reasoning: true

default_model: "test-model1"
`
					err := os.WriteFile(configFile, []byte(configContent), 0o644)
					Expect(err).NotTo(HaveOccurred())

					_, err = config.LoadConfig(configFile)
					Expect(err).To(HaveOccurred())
					Expect(err.Error()).To(ContainSubstring("endpoint2"))
					Expect(err.Error()).To(ContainSubstring("invalid IP address format"))
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
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Enabled).To(BeTrue())
				Expect(cfg.SemanticCache.BackendType).To(Equal("memory"))
				Expect(*cfg.SemanticCache.SimilarityThreshold).To(Equal(float32(0.85)))
				Expect(cfg.SemanticCache.MaxEntries).To(Equal(2000))
				Expect(cfg.SemanticCache.TTLSeconds).To(Equal(1800))
				Expect(cfg.SemanticCache.BackendConfigPath).To(BeEmpty())
			})
		})

		Context("with milvus backend configuration", func() {
			BeforeEach(func() {
				configContent := `
semantic_cache:
  enabled: true
  backend_type: "milvus"
  similarity_threshold: 0.9
  ttl_seconds: 7200
  backend_config_path: "config/cache/milvus.yaml"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should parse milvus backend configuration correctly", func() {
				cfg, err := config.LoadConfig(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Enabled).To(BeTrue())
				Expect(cfg.SemanticCache.BackendType).To(Equal("milvus"))
				Expect(*cfg.SemanticCache.SimilarityThreshold).To(Equal(float32(0.9)))
				Expect(cfg.SemanticCache.TTLSeconds).To(Equal(7200))
				Expect(cfg.SemanticCache.BackendConfigPath).To(Equal("config/cache/milvus.yaml"))

				// MaxEntries should be ignored for Milvus backend
				Expect(cfg.SemanticCache.MaxEntries).To(Equal(0))
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
				cfg, err := config.LoadConfig(configFile)
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
				cfg, err := config.LoadConfig(configFile)
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
				cfg, err := config.LoadConfig(configFile)
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
				cfg, err := config.LoadConfig(configFile)
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
				cfg, err := config.LoadConfig(configFile)
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
				cfg, err := config.LoadConfig(configFile)
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
  backend_config_path: "config/cache/milvus.yaml"

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
				cfg, err := config.LoadConfig(configFile)
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
				Expect(cfg.SemanticCache.BackendConfigPath).To(Equal("config/cache/milvus.yaml"))

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
  # backend_config_path: "config/cache/milvus.yaml"
  # max_entries is ignored for Milvus
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should parse active configuration and ignore commented alternatives", func() {
				cfg, err := config.LoadConfig(configFile)
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
				config.PIITypeAge,
				config.PIITypeCreditCard,
				config.PIITypeDateTime,
				config.PIITypeDomainName,
				config.PIITypeEmailAddress,
				config.PIITypeGPE,
				config.PIITypeIBANCode,
				config.PIITypeIPAddress,
				config.PIITypeNoPII,
				config.PIITypeNRP,
				config.PIITypeOrganization,
				config.PIITypePerson,
				config.PIITypePhoneNumber,
				config.PIITypeStreetAddress,
				config.PIITypeUSDriverLicense,
				config.PIITypeUSSSN,
				config.PIITypeZipCode,
			}

			// Verify all constants are non-empty strings
			for _, piiType := range expectedPIITypes {
				Expect(piiType).NotTo(BeEmpty())
			}

			// Verify specific values
			Expect(config.PIITypeNoPII).To(Equal("NO_PII"))
			Expect(config.PIITypePerson).To(Equal("PERSON"))
			Expect(config.PIITypeEmailAddress).To(Equal("EMAIL_ADDRESS"))
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

			var cfg config.RouterConfig
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

			var cfg config.RouterConfig
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

			var cfg config.RouterConfig
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
				cfg := &config.RouterConfig{
					AutoModelName: "CustomAuto",
				}
				Expect(cfg.GetEffectiveAutoModelName()).To(Equal("CustomAuto"))
			})

			It("should return default 'MoM' when AutoModelName is not set", func() {
				cfg := &config.RouterConfig{
					AutoModelName: "",
				}
				Expect(cfg.GetEffectiveAutoModelName()).To(Equal("MoM"))
			})

			It("should return default 'MoM' for empty RouterConfig", func() {
				cfg := &config.RouterConfig{}
				Expect(cfg.GetEffectiveAutoModelName()).To(Equal("MoM"))
			})
		})

		Context("IsAutoModelName", func() {
			It("should recognize 'auto' as auto model name for backward compatibility", func() {
				cfg := &config.RouterConfig{
					AutoModelName: "MoM",
				}
				Expect(cfg.IsAutoModelName("auto")).To(BeTrue())
			})

			It("should recognize configured AutoModelName", func() {
				cfg := &config.RouterConfig{
					AutoModelName: "CustomAuto",
				}
				Expect(cfg.IsAutoModelName("CustomAuto")).To(BeTrue())
			})

			It("should recognize default 'MoM' when AutoModelName is not set", func() {
				cfg := &config.RouterConfig{
					AutoModelName: "",
				}
				Expect(cfg.IsAutoModelName("MoM")).To(BeTrue())
			})

			It("should not recognize other model names as auto", func() {
				cfg := &config.RouterConfig{
					AutoModelName: "MoM",
				}
				Expect(cfg.IsAutoModelName("gpt-4")).To(BeFalse())
				Expect(cfg.IsAutoModelName("claude")).To(BeFalse())
			})

			It("should support both 'auto' and configured name", func() {
				cfg := &config.RouterConfig{
					AutoModelName: "MoM",
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
				var cfg config.RouterConfig
				err := yaml.Unmarshal([]byte(yamlContent), &cfg)
				Expect(err).NotTo(HaveOccurred())
				Expect(cfg.AutoModelName).To(Equal("CustomRouter"))
				Expect(cfg.GetEffectiveAutoModelName()).To(Equal("CustomRouter"))
			})

			It("should handle missing AutoModelName in YAML", func() {
				yamlContent := `
default_model: "test-model"
`
				var cfg config.RouterConfig
				err := yaml.Unmarshal([]byte(yamlContent), &cfg)
				Expect(err).NotTo(HaveOccurred())
				Expect(cfg.AutoModelName).To(Equal(""))
				Expect(cfg.GetEffectiveAutoModelName()).To(Equal("MoM"))
			})
		})
	})
})
