package config

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"gopkg.in/yaml.v3"
)

func loadLegacyRuntimeConfigForTest(configPath string) (*RouterConfig, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}
	return parseLegacyRuntimeConfigForTest(data)
}

func parseLegacyRuntimeConfigForTest(data []byte) (*RouterConfig, error) {
	cfg := &RouterConfig{}
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	if len(cfg.MoMRegistry) == 0 {
		cfg.MoMRegistry = ToLegacyRegistry()
	}
	// Application rules ask the decision model, which a canonical load
	// pre-declares.
	if cfg.ModelDeployments == nil && len(cfg.ApplicationRules()) > 0 {
		cfg.ModelDeployments = DefaultModelDeployments()
	}
	if cfg.VectorStore != nil {
		cfg.VectorStore.ApplyDefaults()
	}
	if err := validateConfigStructure(cfg); err != nil {
		return nil, err
	}
	return cfg, nil
}

func writeCanonicalLoadTestConfig(path string) error {
	return os.WriteFile(path, []byte(`
version: v0.5
listeners:
  - name: http
    address: 0.0.0.0
    port: 8888
providers:
  defaults:
    model: test-model
  models:
    - name: test-model
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
recipes:
  - name: default
    decisions:
      - name: default-route
        priority: 1
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: test-model
            use_reasoning: false
`), 0o644)
}

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
		Expect(os.RemoveAll(tempDir)).To(Succeed())
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

categories:
  - name: "general"
    labels: [other]
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
  similarity_threshold: 0.9
  max_entries: 1000
  ttl_seconds: 3600

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
`
				err := os.WriteFile(configFile, []byte(validConfig), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should load configuration successfully", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())
				Expect(cfg).NotTo(BeNil())

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
				Expect(cfg.SemanticCache.MaxEntries).To(Equal(1000))
				Expect(cfg.SemanticCache.TTLSeconds).To(Equal(3600))

				// New fields should have default/zero values when not specified
				Expect(cfg.SemanticCache.BackendType).To(BeEmpty())

				// Verify model config
				Expect(cfg.ModelConfig).To(HaveKey("model-a"))
				Expect(cfg.ModelConfig["model-a"].PreferredEndpoints).To(ContainElement("endpoint1"))

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
				Expect(writeCanonicalLoadTestConfig(configFile)).To(Succeed())

				cfg1, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				cfg2, err := Load(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg1).To(BeIdenticalTo(cfg2))
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

				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
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

				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
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

				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
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
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
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
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())
				Expect(cfg).NotTo(BeNil())
				Expect(cfg.DefaultModel).To(BeEmpty())
			})
		})

		Context("concurrent access", func() {
			BeforeEach(func() {
				Expect(writeCanonicalLoadTestConfig(configFile)).To(Succeed())
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
        - type: keyword
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
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				model := cfg.GetModelForDecisionIndex(0)
				Expect(model).To(Equal("model1"))

				model = cfg.GetModelForDecisionIndex(1)
				Expect(model).To(Equal("model3"))
			})
		})

		Context("with invalid decision index", func() {
			It("should return the default model for negative index", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				model := cfg.GetModelForDecisionIndex(-1)
				Expect(model).To(Equal("default-model"))
			})

			It("should return the default model for index beyond range", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
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
				// Empty modelRefs is now allowed - decisions without models are valid
				_, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())
			})
		})
	})

	Describe("GetCategoryDescriptions", func() {
		Context("with categories having descriptions", func() {
			BeforeEach(func() {
				configContent := `
categories:
  - name: "category1"
    labels: [other]
    description: "Description for category 1"
    model_scores:
      - model: "model1"
        score: 0.9
        use_reasoning: true
  - name: "category2"
    labels: [other]
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
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
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
    labels: [other]
    description: "Has description"
    model_scores:
      - model: "model1"
        score: 0.9
        use_reasoning: true
  - name: "category2"
    labels: [other]
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
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
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

				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
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

			cfg, err := loadLegacyRuntimeConfigForTest(configFile)
			Expect(err).NotTo(HaveOccurred())
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

			cfg, err := loadLegacyRuntimeConfigForTest(configFile)
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
    labels: [other]
    description: "Description with special chars: @#$%^&*()"
    model_scores:
      - model: "model-with-hyphens_and_underscores"
        score: 0.9
        use_reasoning: true
`
			err := os.WriteFile(configFile, []byte(configContent), 0o644)
			Expect(err).NotTo(HaveOccurred())

			cfg, err := loadLegacyRuntimeConfigForTest(configFile)
			Expect(err).NotTo(HaveOccurred())
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
    labels: [other]
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
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				endpoints := cfg.GetEndpointsForModel("model-a")
				Expect(endpoints).To(HaveLen(2))
				endpointNames := []string{endpoints[0].Name, endpoints[1].Name}
				Expect(endpointNames).To(ContainElements("endpoint1", "endpoint3"))
			})

			It("should return empty slice when no preferred endpoints configured", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				endpoints := cfg.GetEndpointsForModel("model-c")
				Expect(endpoints).To(BeEmpty())
			})

			It("should return empty slice for non-existent model", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				endpoints := cfg.GetEndpointsForModel("non-existent-model")
				Expect(endpoints).To(BeEmpty())
			})

			It("should return only preferred endpoints", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				// model-b has preferred endpoint2
				endpoints := cfg.GetEndpointsForModel("model-b")
				Expect(endpoints).To(HaveLen(1))
				Expect(endpoints[0].Name).To(Equal("endpoint2"))
			})
		})

		Describe("GetEndpointByName", func() {
			It("should return endpoint when it exists", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				endpoint, found := cfg.GetEndpointByName("endpoint1")
				Expect(found).To(BeTrue())
				Expect(endpoint.Name).To(Equal("endpoint1"))
				Expect(endpoint.Address).To(Equal("127.0.0.1"))
				Expect(endpoint.Port).To(Equal(8000))
			})

			It("should return false when endpoint doesn't exist", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				endpoint, found := cfg.GetEndpointByName("non-existent")
				Expect(found).To(BeFalse())
				Expect(endpoint).To(BeNil())
			})
		})

		Describe("GetAllModels", func() {
			It("should return all models from model_config", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				models := cfg.GetAllModels()
				Expect(models).To(HaveLen(3))
				Expect(models).To(ContainElements("model-a", "model-b", "model-c"))
			})
		})

		Describe("ResolvePrimaryBackendForModel", func() {
			It("should return address and endpoint name for model with single endpoint", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				// model-b has a single preferred endpoint: endpoint2
				address, endpointName, found, detailErr := cfg.ResolvePrimaryBackendForModel("model-b")
				Expect(detailErr).NotTo(HaveOccurred())
				Expect(found).To(BeTrue())
				Expect(address).To(Equal("127.0.0.1:8000"))
				Expect(endpointName).To(Equal("endpoint2"))
			})

			It("should return the highest-weight endpoint for model with multiple endpoints", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				// model-a has endpoint1 (weight 1) and endpoint3 (weight 1)
				// Both have the same weight, so we get one of them
				address, endpointName, found, detailErr := cfg.ResolvePrimaryBackendForModel("model-a")
				Expect(detailErr).NotTo(HaveOccurred())
				Expect(found).To(BeTrue())
				Expect(address).To(Equal("127.0.0.1:8000"))
				Expect(endpointName).To(BeElementOf("endpoint1", "endpoint3"))
			})

			It("should return false for non-existent model", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				address, endpointName, found, detailErr := cfg.ResolvePrimaryBackendForModel("non-existent-model")
				Expect(detailErr).NotTo(HaveOccurred())
				Expect(found).To(BeFalse())
				Expect(address).To(BeEmpty())
				Expect(endpointName).To(BeEmpty())
			})

			It("should return false for model with no preferred endpoints", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				// model-c has no preferred endpoints configured
				address, endpointName, found, detailErr := cfg.ResolvePrimaryBackendForModel("model-c")
				Expect(detailErr).NotTo(HaveOccurred())
				Expect(found).To(BeFalse())
				Expect(address).To(BeEmpty())
				Expect(endpointName).To(BeEmpty())
			})

			It("should resolve the highest-weight backend metadata when weights differ", func() {
				configContent := `
vllm_endpoints:
  - name: "ep-low"
    address: "10.0.0.1"
    port: 9000
    weight: 1
  - name: "ep-high"
    address: "10.0.0.2"
    port: 9001
    weight: 10

model_config:
  "weighted-model":
    preferred_endpoints: ["ep-low", "ep-high"]

categories:
  - name: "test"
    labels: [other]
    model_scores:
      - model: "weighted-model"
        score: 0.9

default_model: "weighted-model"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				address, endpointName, found, detailErr := cfg.ResolvePrimaryBackendForModel("weighted-model")
				Expect(detailErr).NotTo(HaveOccurred())
				Expect(found).To(BeTrue())
				Expect(address).To(Equal("10.0.0.2:9001"))
				Expect(endpointName).To(Equal("ep-high"))
			})
		})

		Describe("ResolveExternalModelID", func() {
			It("should resolve external model ID for vllm endpoint type", func() {
				configContent := `
vllm_endpoints:
  - name: "vllm-ep"
    address: "127.0.0.1"
    port: 8000
    type: "vllm"

model_config:
  "my-alias":
    preferred_endpoints: ["vllm-ep"]
    external_model_ids:
      vllm: "Qwen/Qwen2.5-14B-Instruct"

categories:
  - name: "test"
    labels: [other]
    model_scores:
      - model: "my-alias"
        score: 0.9

default_model: "my-alias"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				resolved := cfg.ResolveExternalModelID("my-alias", "vllm-ep")
				Expect(resolved).To(Equal("Qwen/Qwen2.5-14B-Instruct"))
			})

			It("should match aliases to provider model IDs", func() {
				cfg := &RouterConfig{
					BackendModels: BackendModels{
						ModelConfig: map[string]ModelParams{
							"my-alias": {
								ExternalModelIDs: map[string]string{
									"openai": "gpt-5-mini",
								},
							},
						},
					},
				}

				Expect(cfg.ModelNameMatches("my-alias", "gpt-5-mini")).To(BeTrue())
				Expect(cfg.ModelNameMatches("gpt-5-mini", "my-alias")).To(BeTrue())
				Expect(cfg.ModelNameMatches("other", "gpt-5-mini")).To(BeFalse())
			})

			It("should default to vllm type when endpoint has no type set", func() {
				configContent := `
vllm_endpoints:
  - name: "no-type-ep"
    address: "127.0.0.1"
    port: 8000

model_config:
  "my-alias":
    preferred_endpoints: ["no-type-ep"]
    external_model_ids:
      vllm: "Qwen/Qwen2.5-14B-Instruct"

categories:
  - name: "test"
    labels: [other]
    model_scores:
      - model: "my-alias"
        score: 0.9

default_model: "my-alias"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				// Endpoint has no type, so defaults to "vllm"
				resolved := cfg.ResolveExternalModelID("my-alias", "no-type-ep")
				Expect(resolved).To(Equal("Qwen/Qwen2.5-14B-Instruct"))
			})

			It("should default to vllm type when endpoint name does not exist", func() {
				configContent := `
vllm_endpoints:
  - name: "real-ep"
    address: "127.0.0.1"
    port: 8000

model_config:
  "my-alias":
    preferred_endpoints: ["real-ep"]
    external_model_ids:
      vllm: "Qwen/Qwen2.5-14B-Instruct"

categories:
  - name: "test"
    labels: [other]
    model_scores:
      - model: "my-alias"
        score: 0.9

default_model: "my-alias"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				// Non-existent endpoint name falls back to "vllm" type
				resolved := cfg.ResolveExternalModelID("my-alias", "non-existent-ep")
				Expect(resolved).To(Equal("Qwen/Qwen2.5-14B-Instruct"))
			})

			It("should resolve correct type for ollama endpoint", func() {
				configContent := `
vllm_endpoints:
  - name: "ollama-ep"
    address: "127.0.0.1"
    port: 11434
    type: "ollama"

model_config:
  "my-alias":
    preferred_endpoints: ["ollama-ep"]
    external_model_ids:
      vllm: "Qwen/Qwen2.5-14B-Instruct"
      ollama: "qwen2.5:14b"

categories:
  - name: "test"
    labels: [other]
    model_scores:
      - model: "my-alias"
        score: 0.9

default_model: "my-alias"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				resolved := cfg.ResolveExternalModelID("my-alias", "ollama-ep")
				Expect(resolved).To(Equal("qwen2.5:14b"))
			})

			It("should return original model name when no external_model_ids configured", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				// Using the base fixture which has no external_model_ids
				resolved := cfg.ResolveExternalModelID("model-a", "endpoint1")
				Expect(resolved).To(Equal("model-a"))
			})

			It("should return original model name for non-existent model", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				resolved := cfg.ResolveExternalModelID("non-existent-model", "endpoint1")
				Expect(resolved).To(Equal("non-existent-model"))
			})

			It("should return original model name when endpoint type has no mapping", func() {
				configContent := `
vllm_endpoints:
  - name: "custom-ep"
    address: "127.0.0.1"
    port: 8000
    type: "openrouter"

model_config:
  "my-alias":
    preferred_endpoints: ["custom-ep"]
    external_model_ids:
      vllm: "Qwen/Qwen2.5-14B-Instruct"

categories:
  - name: "test"
    labels: [other]
    model_scores:
      - model: "my-alias"
        score: 0.9

default_model: "my-alias"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				// Endpoint type is "openrouter" but external_model_ids only has "vllm"
				resolved := cfg.ResolveExternalModelID("my-alias", "custom-ep")
				Expect(resolved).To(Equal("my-alias"))
			})

			It("should return original model name when config is nil", func() {
				var nilCfg *RouterConfig
				resolved := nilCfg.ResolveExternalModelID("some-model", "some-endpoint")
				Expect(resolved).To(Equal("some-model"))
			})

			It("should return original model name when external_model_ids map is empty", func() {
				configContent := `
vllm_endpoints:
  - name: "ep1"
    address: "127.0.0.1"
    port: 8000

model_config:
  "my-alias":
    preferred_endpoints: ["ep1"]
    external_model_ids: {}

categories:
  - name: "test"
    labels: [other]
    model_scores:
      - model: "my-alias"
        score: 0.9

default_model: "my-alias"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())

				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				resolved := cfg.ResolveExternalModelID("my-alias", "ep1")
				Expect(resolved).To(Equal("my-alias"))
			})
		})

		Describe("ValidateEndpoints", func() {
			It("should pass validation when all models have endpoints", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
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

				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
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
    labels: [other]
    model_scores:
      - model: "test-model"
        score: 0.9
        use_reasoning: true

default_model: "test-model"
`
					err := os.WriteFile(configFile, []byte(configContent), 0o644)
					Expect(err).NotTo(HaveOccurred())

					cfg, err := loadLegacyRuntimeConfigForTest(configFile)
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
    labels: [other]
    model_scores:
      - model: "test-model"
        score: 0.9
        use_reasoning: true

default_model: "test-model"
`
					err := os.WriteFile(configFile, []byte(configContent), 0o644)
					Expect(err).NotTo(HaveOccurred())

					cfg, err := loadLegacyRuntimeConfigForTest(configFile)
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
    labels: [other]
    model_scores:
      - model: "test-model"
        score: 0.9
        use_reasoning: true

default_model: "test-model"
`
					err := os.WriteFile(configFile, []byte(configContent), 0o644)
					Expect(err).NotTo(HaveOccurred())

					cfg, err := loadLegacyRuntimeConfigForTest(configFile)
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
  backend_type: "memory"
  similarity_threshold: 0.85
  max_entries: 2000
  ttl_seconds: 1800
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should parse memory backend configuration correctly", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.BackendType).To(Equal("memory"))
				Expect(cfg.SemanticCache.MaxEntries).To(Equal(2000))
				Expect(cfg.SemanticCache.TTLSeconds).To(Equal(1800))
			})
		})

		Context("with inline milvus backend configuration", func() {
			BeforeEach(func() {
				configContent := `
semantic_cache:
  backend_type: "milvus"
  similarity_threshold: 0.9
  ttl_seconds: 7200
  milvus:
    connection:
      host: "localhost"
      port: 19530
    collection:
      name: "semantic_cache"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should parse inline milvus backend configuration correctly", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.BackendType).To(Equal("milvus"))
				Expect(cfg.SemanticCache.TTLSeconds).To(Equal(7200))
				Expect(cfg.SemanticCache.Milvus).NotTo(BeNil())

				// MaxEntries should be ignored for Milvus backend
				Expect(cfg.SemanticCache.MaxEntries).To(Equal(0))
			})
		})

		Context("with inline milvus backend configuration", func() {
			BeforeEach(func() {
				configContent := `
semantic_cache:
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
    logging:
      level: "info"
    development:
      drop_collection_on_startup: true
      auto_create_collection: true
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should parse inline milvus backend connection configuration correctly", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
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
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
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
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Milvus.Search).ToNot(BeNil())
				Expect(cfg.SemanticCache.Milvus.Search.Params.Ef).To(Equal(64))
				Expect(cfg.SemanticCache.Milvus.Search.TopK).To(Equal(10))
				Expect(cfg.SemanticCache.Milvus.Search.ConsistencyLevel).To(Equal("Session"))
			})

			It("should parse inline milvus backend logging configuration correctly", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Milvus.Logging.Level).To(Equal("info"))
			})

			It("should parse inline milvus backend development configuration correctly", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Milvus.Development.DropCollectionOnStartup).To(BeTrue())
				Expect(cfg.SemanticCache.Milvus.Development.AutoCreateCollection).To(BeTrue())
			})
		})

		Context("with inline redis backend configuration", func() {
			BeforeEach(func() {
				configContent := `
semantic_cache:
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
    development:
      drop_index_on_startup: true
      auto_create_index: true
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should parse inline redis backend connection configuration correctly", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
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
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
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
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Redis.Search).ToNot(BeNil())
				Expect(cfg.SemanticCache.Redis.Search.TopK).To(Equal(1))
			})

			It("should parse inline redis backend logging configuration correctly", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Redis.Logging.Level).To(Equal("info"))
			})

			It("should parse inline redis backend development configuration correctly", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.Redis.Development.DropIndexOnStartup).To(BeTrue())
				Expect(cfg.SemanticCache.Redis.Development.AutoCreateIndex).To(BeTrue())
			})
		})

		Context("with disabled cache", func() {
			BeforeEach(func() {
				configContent := `
semantic_cache:
  backend_type: "memory"
  similarity_threshold: 0.8
  max_entries: 1000
  ttl_seconds: 3600
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should preserve configuration even when cache is disabled", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.BackendType).To(Equal("memory"))
			})
		})

		Context("with minimal configuration", func() {
			BeforeEach(func() {
				configContent := `
semantic_cache:
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should handle minimal configuration with default values", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.BackendType).To(BeEmpty()) // Should default to empty (memory)
				Expect(cfg.SemanticCache.MaxEntries).To(Equal(0))
				Expect(cfg.SemanticCache.TTLSeconds).To(Equal(0))
			})
		})

		Context("with comprehensive configuration", func() {
			BeforeEach(func() {
				configContent := `
bert_model:
  threshold: 0.7

semantic_cache:
  backend_type: "milvus"
  similarity_threshold: 0.95
  ttl_seconds: 14400
  milvus:
    connection:
      host: "milvus"
      port: 19530
    collection:
      name: "production_cache"
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should parse all semantic cache fields correctly", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.BackendType).To(Equal("milvus"))
				Expect(cfg.SemanticCache.TTLSeconds).To(Equal(14400))
				Expect(cfg.SemanticCache.Milvus).NotTo(BeNil())
			})
		})

		Context("threshold fallback behavior", func() {
			BeforeEach(func() {
				configContent := `
semantic_cache:
  backend_type: "memory"
  similarity_threshold: 1.0
  max_entries: 0
  ttl_seconds: -1
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should handle edge case values correctly", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.BackendType).To(Equal("memory"))
				Expect(cfg.SemanticCache.MaxEntries).To(Equal(0))
				Expect(cfg.SemanticCache.TTLSeconds).To(Equal(-1))
			})
		})

		Context("with unsupported backend type", func() {
			BeforeEach(func() {
				configContent := `
semantic_cache:
  backend_type: "redis"
  similarity_threshold: 0.8
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should parse unsupported backend type without error (validation happens at runtime)", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				// Configuration parsing should succeed
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
  backend_type: "milvus"
  similarity_threshold: 0.85
  ttl_seconds: 86400  # 24 hours
  milvus:
    connection:
      host: "milvus"
      port: 19530
    collection:
      name: "semantic_cache"

categories:
  - name: "production"
    labels: [other]
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
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				// Verify semantic cache config
				Expect(cfg.SemanticCache.BackendType).To(Equal("milvus"))
				Expect(cfg.SemanticCache.TTLSeconds).To(Equal(86400))
				Expect(cfg.SemanticCache.Milvus).NotTo(BeNil())

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
  # milvus:
  #   connection:
  #     host: "milvus"
  #     port: 19530
  # max_entries is ignored for Milvus
`
				err := os.WriteFile(configFile, []byte(configContent), 0o644)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should parse active configuration and ignore commented alternatives", func() {
				cfg, err := loadLegacyRuntimeConfigForTest(configFile)
				Expect(err).NotTo(HaveOccurred())

				Expect(cfg.SemanticCache.BackendType).To(Equal("memory"))
				Expect(cfg.SemanticCache.MaxEntries).To(Equal(1000))
				Expect(cfg.SemanticCache.TTLSeconds).To(Equal(3600))
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
    max_batch_size: 64
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
			Expect(batchConfig.MaxBatchSize).To(Equal(64))

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
    max_batch_size: 32
`

			var cfg RouterConfig
			err := yaml.Unmarshal([]byte(yamlContent), &cfg)
			Expect(err).NotTo(HaveOccurred())

			// Verify that missing metrics configuration doesn't cause errors (zero-config)
			batchConfig := cfg.API.BatchClassification
			Expect(batchConfig.MaxBatchSize).To(Equal(32))

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
						InUse: true, // The store is in use, which does not cache a decision without the plugin
					},
					IntelligentRouting: IntelligentRouting{
						Decisions: []Decision{decision},
					},
				}

				// Per-decision scoping: no plugin = no semantic caching
				Expect(cfg.IsCacheEnabledForDecision("test")).To(BeFalse())
			})

			It("should return false when decision explicitly disables response_cache", func() {
				decision := Decision{
					Name:      "test",
					ModelRefs: []ModelRef{{Model: "test"}},
					Plugins: []DecisionPlugin{
						{
							Type: "response_cache",
							Configuration: MustStructuredPayload(map[string]interface{}{
								"enabled": false,
							}),
						},
					},
				}

				cfg := &RouterConfig{
					SemanticCache: SemanticCache{
						InUse: true,
					},
					IntelligentRouting: IntelligentRouting{
						Decisions: []Decision{decision},
					},
				}

				Expect(cfg.IsCacheEnabledForDecision("test")).To(BeFalse())
			})

			It("should return true when decision explicitly enables response_cache", func() {
				decision := Decision{
					Name:      "test",
					ModelRefs: []ModelRef{{Model: "test"}},
					Plugins: []DecisionPlugin{
						{
							Type: "response_cache",
							Configuration: MustStructuredPayload(map[string]interface{}{
								"enabled": true,
							}),
						},
					},
				}

				cfg := &RouterConfig{
					SemanticCache: SemanticCache{
						InUse: false, // The decision's plugin enables caching
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
						InUse: true,
					},
				}

				Expect(cfg.IsCacheEnabledForDecision("non_existent")).To(BeFalse())
			})

			It("should use the decision's semantic match when configured", func() {
				decision := Decision{
					Name:      "test",
					ModelRefs: []ModelRef{{Model: "test"}},
					Plugins: []DecisionPlugin{
						{
							Type: "response_cache",
							Configuration: MustStructuredPayload(map[string]interface{}{
								"enabled": true,
								"semantic": map[string]interface{}{"match": map[string]interface{}{
									"recall": map[string]interface{}{"top_k": 3, "min_similarity": 0.95},
								}},
							}),
						},
					},
				}

				cfg := &RouterConfig{
					SemanticCache: SemanticCache{
						InUse: true,
					},
					IntelligentRouting: IntelligentRouting{
						Decisions: []Decision{decision},
					},
				}

				// The decision's semantic.match bounds its recall.
				block := cfg.ResponseCacheMatch(cfg.GetDecisionByName("test"))
				Expect(block).NotTo(BeNil())
				Expect(*block.Recall.MinSimilarity).To(Equal(0.95))
			})

			It("should use decision-level TTL when configured", func() {
				ttl := 7200
				decision := Decision{
					Name:      "test",
					ModelRefs: []ModelRef{{Model: "test"}},
					Plugins: []DecisionPlugin{
						{
							Type: "response_cache",
							Configuration: MustStructuredPayload(map[string]interface{}{
								"enabled":     true,
								"ttl_seconds": ttl,
							}),
						},
					},
				}

				cfg := &RouterConfig{
					SemanticCache: SemanticCache{
						InUse:      true,
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
})

var _ = Describe("ParseConfigFile and ReplaceGlobalConfig", func() {
	var tempDir string

	BeforeEach(func() {
		var err error
		tempDir, err = os.MkdirTemp("", "config_parse_test")
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		Expect(os.RemoveAll(tempDir)).To(Succeed())
		ResetConfig()
	})

	It("should parse configuration via symlink path", func() {
		if runtime.GOOS == "windows" {
			Skip("symlink test is skipped on Windows")
		}

		// Create real config target
		target := filepath.Join(tempDir, "real-config.yaml")
		content := []byte(`
version: v0.5
listeners: []
providers:
  defaults:
    model: test-model
  models:
    - name: test-model
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
recipes:
  - name: default
    decisions:
      - name: default-route
        priority: 1
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: test-model
`)
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

// ResetConfig resets the singleton config for testing purposes
// This is needed to ensure test isolation
func ResetConfig() {
	configOnce = sync.Once{}
	config = nil
	configErr = nil
}

var _ = Describe("Decision and Global Configuration", func() {
	Describe("RouterConfig.ContextPlugin", func() {
		contextDecision := func(configuration ContextPluginConfig) *Decision {
			return &Decision{Name: "d", Plugins: []DecisionPlugin{{Type: DecisionPluginContext, Configuration: MustStructuredPayload(configuration)}}}
		}

		It("gives a request without a decision the context plugin of global.plugins", func() {
			cfg := &RouterConfig{}
			Expect(cfg.ContextPlugin(nil)).To(BeNil())
			cfg.GlobalPlugins = []DecisionPlugin{{Type: DecisionPluginContext, Configuration: MustStructuredPayload(
				ContextPluginConfig{Enabled: true, Sources: []ContextSource{{Type: ContextSourceMemory}}},
			)}}
			Expect(cfg.LayerPlugins()).To(Succeed())
			plugin := cfg.ContextPlugin(nil)
			Expect(plugin).NotTo(BeNil())
			Expect(plugin.Sources).To(Equal([]ContextSource{{Type: ContextSourceMemory}}))
		})

		It("reads only the sources an enabled context plugin lists", func() {
			cfg := &RouterConfig{}
			cfg.Memory.InUse = true
			plugin := cfg.ContextPlugin(contextDecision(ContextPluginConfig{Enabled: true, Sources: []ContextSource{{Type: ContextSourceRAG, Backend: "vectorstore"}}}))
			Expect(plugin.Source(ContextSourceRAG)).NotTo(BeNil())
			Expect(plugin.Source(ContextSourceMemory)).To(BeNil())
		})

		It("turns context off, global memory included, for a disabled context plugin", func() {
			cfg := &RouterConfig{}
			cfg.Memory.InUse = true
			Expect(cfg.ContextPlugin(contextDecision(ContextPluginConfig{Enabled: false}))).To(BeNil())
		})
	})

	// -----------------------------------------------------------------
	// Provider profiles
	// -----------------------------------------------------------------
	Describe("Provider Profiles", func() {
		Context("YAML parsing", func() {
			It("should parse provider_profiles and endpoint references", func() {
				yamlData := `
provider_profiles:
  openai-prod:
    type: "openai"
    base_url: "https://api.openai.com/v1"
  azure-east:
    type: "azure-openai"
    base_url: "https://myresource.openai.azure.com/openai/deployments/gpt-4o"
    api_version: "2024-10-21"
  anthropic-prod:
    type: "anthropic"
    base_url: "https://api.anthropic.com"
    extra_headers:
      anthropic-version: "2023-06-01"
  bedrock-west:
    type: "bedrock"
    base_url: "https://bedrock-mantle.us-west-2.api.aws/v1"
vllm_endpoints:
  - name: "openai"
    provider_profile: "openai-prod"
  - name: "azure"
    provider_profile: "azure-east"
  - name: "local-vllm"
    address: "127.0.0.1"
    port: 8000
model_config:
  "gpt-4o":
    preferred_endpoints: ["openai", "azure"]
  "Qwen/Qwen2.5-14B":
    preferred_endpoints: ["local-vllm"]
`
				var cfg RouterConfig
				err := yaml.Unmarshal([]byte(yamlData), &cfg)
				Expect(err).NotTo(HaveOccurred())

				// provider_profiles parsed
				Expect(cfg.ProviderProfiles).To(HaveLen(4))
				Expect(cfg.ProviderProfiles["openai-prod"].Type).To(Equal("openai"))
				Expect(cfg.ProviderProfiles["azure-east"].APIVersion).To(Equal("2024-10-21"))
				Expect(cfg.ProviderProfiles["anthropic-prod"].ExtraHeaders).To(HaveKeyWithValue("anthropic-version", "2023-06-01"))

				// endpoint references
				Expect(cfg.VLLMEndpoints).To(HaveLen(3))
				Expect(cfg.VLLMEndpoints[0].ProviderProfileName).To(Equal("openai-prod"))
				Expect(cfg.VLLMEndpoints[2].ProviderProfileName).To(BeEmpty())
			})
		})

		Context("ResolveAddress", func() {
			It("should extract host:port from base_url", func() {
				profiles := map[string]ProviderProfile{
					"openai-prod": {Type: "openai", BaseURL: "https://api.openai.com/v1"},
					"azure-east":  {Type: "azure-openai", BaseURL: "https://myresource.openai.azure.com/openai/deployments/gpt-4o"},
				}

				ep := VLLMEndpoint{Name: "openai", ProviderProfileName: "openai-prod"}
				addr, err := ep.ResolveAddress(profiles)
				Expect(err).NotTo(HaveOccurred())
				Expect(addr).To(Equal("api.openai.com:443"))

				ep2 := VLLMEndpoint{Name: "azure", ProviderProfileName: "azure-east"}
				addr2, err2 := ep2.ResolveAddress(profiles)
				Expect(err2).NotTo(HaveOccurred())
				Expect(addr2).To(Equal("myresource.openai.azure.com:443"))
			})

			It("should use address:port for legacy endpoints (no provider_profile)", func() {
				ep := VLLMEndpoint{Name: "local", Address: "127.0.0.1", Port: 8000}
				addr, err := ep.ResolveAddress(nil)
				Expect(err).NotTo(HaveOccurred())
				Expect(addr).To(Equal("127.0.0.1:8000"))
			})

			It("should error when profile is set but has no base_url", func() {
				profiles := map[string]ProviderProfile{
					"minimal": {Type: "openai"},
				}
				ep := VLLMEndpoint{Name: "ep1", Address: "10.0.0.1", Port: 9000, ProviderProfileName: "minimal"}
				_, err := ep.ResolveAddress(profiles)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("no base_url"))
			})

			It("should error when profile is set but does not exist", func() {
				profiles := map[string]ProviderProfile{}
				ep := VLLMEndpoint{Name: "ep1", ProviderProfileName: "missing"}
				_, err := ep.ResolveAddress(profiles)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("does not exist"))
			})

			It("should error when profile is set but profiles map is nil", func() {
				ep := VLLMEndpoint{Name: "ep1", ProviderProfileName: "some-profile"}
				_, err := ep.ResolveAddress(nil)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("no provider_profiles map"))
			})

			It("should handle explicit port in base_url", func() {
				profiles := map[string]ProviderProfile{
					"custom": {Type: "openai", BaseURL: "http://localhost:8080/v1"},
				}
				ep := VLLMEndpoint{Name: "ep1", ProviderProfileName: "custom"}
				addr, err := ep.ResolveAddress(profiles)
				Expect(err).NotTo(HaveOccurred())
				Expect(addr).To(Equal("localhost:8080"))
			})

			It("should error on unsupported URL scheme", func() {
				profiles := map[string]ProviderProfile{
					"ftp": {Type: "openai", BaseURL: "ftp://files.example.com/v1"},
				}
				ep := VLLMEndpoint{Name: "ep1", ProviderProfileName: "ftp"}
				_, err := ep.ResolveAddress(profiles)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("unsupported scheme"))
			})
		})

		Context("ProviderType", func() {
			It("should accept every catalog provider ID", func() {
				providerTypes := ValidProviderTypes()
				Expect(providerTypes).NotTo(BeEmpty())
				for _, t := range providerTypes {
					pt, err := (&ProviderProfile{Type: t}).ProviderType()
					Expect(err).NotTo(HaveOccurred())
					Expect(pt).To(Equal(t))
				}
			})

			It("should error on unknown type", func() {
				_, err := (&ProviderProfile{Type: "unknown"}).ProviderType()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("unknown provider ID"))
			})

			It("should error on empty type", func() {
				_, err := (&ProviderProfile{}).ProviderType()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("empty type"))
			})

			It("should error on nil profile", func() {
				var nilProfile *ProviderProfile
				_, err := nilProfile.ProviderType()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("nil"))
			})
		})

		Context("ResolveAuthHeader", func() {
			It("should resolve auth metadata for every catalog provider ID", func() {
				for _, providerType := range ValidProviderTypes() {
					_, _, err := (&ProviderProfile{Type: providerType}).ResolveAuthHeader()
					Expect(err).NotTo(HaveOccurred(), providerType)
				}
			})

			It("should return type-specific defaults", func() {
				h, p, err := (&ProviderProfile{Type: "openai"}).ResolveAuthHeader()
				Expect(err).NotTo(HaveOccurred())
				Expect(h).To(Equal("Authorization"))
				Expect(p).To(Equal("Bearer"))

				h, p, err = (&ProviderProfile{Type: "anthropic"}).ResolveAuthHeader()
				Expect(err).NotTo(HaveOccurred())
				Expect(h).To(Equal("x-api-key"))
				Expect(p).To(BeEmpty())

				h, p, err = (&ProviderProfile{Type: "azure-openai"}).ResolveAuthHeader()
				Expect(err).NotTo(HaveOccurred())
				Expect(h).To(Equal("api-key"))
				Expect(p).To(BeEmpty())

				h, p, err = (&ProviderProfile{Type: "bedrock"}).ResolveAuthHeader()
				Expect(err).NotTo(HaveOccurred())
				Expect(h).To(Equal("Authorization"))
				Expect(p).To(Equal("Bearer"))

				h, p, err = (&ProviderProfile{Type: "minimax"}).ResolveAuthHeader()
				Expect(err).NotTo(HaveOccurred())
				Expect(h).To(Equal("Authorization"))
				Expect(p).To(Equal("Bearer"))
			})

			It("should allow explicit overrides", func() {
				profile := &ProviderProfile{
					Type:       "openai",
					AuthHeader: "X-Custom-Auth",
					AuthPrefix: "Token",
				}
				h, p, err := profile.ResolveAuthHeader()
				Expect(err).NotTo(HaveOccurred())
				Expect(h).To(Equal("X-Custom-Auth"))
				Expect(p).To(Equal("Token"))
			})

			It("should error on unknown type", func() {
				_, _, err := (&ProviderProfile{Type: "bogus"}).ResolveAuthHeader()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("unknown provider ID"))
			})
		})

		Context("ResolveReasoningTransport", func() {
			It("should honor a validated provider-model override", func() {
				transport, err := (&ProviderProfile{
					Type: "dashscope", ReasoningTransport: "top_level_boolean",
				}).ResolveReasoningTransport()
				Expect(err).NotTo(HaveOccurred())
				Expect(string(transport)).To(Equal("top_level_boolean"))

				_, err = (&ProviderProfile{
					Type: "dashscope", ReasoningTransport: "invented",
				}).ResolveReasoningTransport()
				Expect(err).To(HaveOccurred())
			})

			It("should return catalog semantics without inspecting endpoint hosts", func() {
				transport, err := (&ProviderProfile{Type: "openai", BaseURL: "https://proxy.example/v1"}).ResolveReasoningTransport()
				Expect(err).NotTo(HaveOccurred())
				Expect(string(transport)).To(Equal("top_level_effort"))

				transport, err = (&ProviderProfile{Type: "deepseek", BaseURL: "https://private.example/v1"}).ResolveReasoningTransport()
				Expect(err).NotTo(HaveOccurred())
				Expect(string(transport)).To(Equal("deepseek_thinking"))

				transport, err = (&ProviderProfile{Type: "zai", BaseURL: "https://private.example/v1"}).ResolveReasoningTransport()
				Expect(err).NotTo(HaveOccurred())
				Expect(string(transport)).To(Equal("thinking_object"))

				transport, err = (&ProviderProfile{Type: "openai-compatible", BaseURL: "https://api.openai.com/v1"}).ResolveReasoningTransport()
				Expect(err).NotTo(HaveOccurred())
				Expect(string(transport)).To(Equal("chat_template_kwargs"))
			})

			It("should reject an unknown provider", func() {
				_, err := (&ProviderProfile{Type: "bogus"}).ResolveReasoningTransport()
				Expect(err).To(HaveOccurred())
			})
		})

		Context("ResolveCreatePath", func() {
			It("should return type-specific default paths", func() {
				path, err := (&ProviderProfile{Type: "openai", BaseURL: "https://api.openai.com/v1"}).ResolveCreatePath("")
				Expect(err).NotTo(HaveOccurred())
				Expect(path).To(Equal("/v1/chat/completions"))

				path, err = (&ProviderProfile{Type: "anthropic", BaseURL: "https://api.anthropic.com"}).ResolveCreatePath("")
				Expect(err).NotTo(HaveOccurred())
				Expect(path).To(Equal("/v1/messages"))

				path, err = (&ProviderProfile{Type: "minimax", BaseURL: "https://api.minimax.io"}).ResolveCreatePath("")
				Expect(err).NotTo(HaveOccurred())
				Expect(path).To(Equal("/v1/chat/completions"))

				path, err = (&ProviderProfile{Type: "openai", BaseURL: "https://api.openai.com/v1"}).ResolveCreatePath("openai/responses@1")
				Expect(err).NotTo(HaveOccurred())
				Expect(path).To(Equal("/v1/responses"))

				path, err = (&ProviderProfile{Type: "deepseek", BaseURL: "https://api.deepseek.com/v1"}).ResolveCreatePath("openai/responses@1")
				Expect(err).NotTo(HaveOccurred())
				Expect(path).To(Equal("/v1/responses"))

				path, err = (&ProviderProfile{Type: "celeris", BaseURL: "https://inference.celeris.ai/celeris-1/v1"}).ResolveCreatePath("openai/responses@1")
				Expect(err).NotTo(HaveOccurred())
				Expect(path).To(Equal("/celeris-1/v1/responses"))
			})

			It("should not double the version segment for a versioned base_url", func() {
				for _, tc := range []struct {
					providerType string
					baseURL      string
					expected     string
				}{
					{"anthropic", "https://api.anthropic.com/v1", "/v1/messages"},
					{"anthropic", "https://api.anthropic.com", "/v1/messages"},
					{"minimax", "https://api.minimax.io/v1", "/v1/chat/completions"},
					{"minimax", "https://api.minimax.io", "/v1/chat/completions"},
					{"openai", "https://api.openai.com/v1", "/v1/chat/completions"},
					{"anthropic", "https://gateway.example.com/openai/v1", "/openai/v1/messages"},
					{"anthropic", "https://gateway.example.com/v1beta", "/v1beta/messages"},
					{"openai", "https://gateway.example.com/v1beta/openai", "/v1beta/openai/chat/completions"},
					{"anthropic", "https://api.anthropic.com/v1/", "/v1/messages"},
				} {
					path, err := (&ProviderProfile{Type: tc.providerType, BaseURL: tc.baseURL}).ResolveCreatePath("")
					Expect(err).NotTo(HaveOccurred(), "%s %s", tc.providerType, tc.baseURL)
					Expect(path).To(Equal(tc.expected), "%s %s", tc.providerType, tc.baseURL)
				}
			})

			It("should append api-version for azure-openai", func() {
				profile := &ProviderProfile{
					Type:       "azure-openai",
					BaseURL:    "https://myresource.openai.azure.com/openai/deployments/gpt-4o",
					APIVersion: "2024-10-21",
				}
				path, err := profile.ResolveCreatePath("")
				Expect(err).NotTo(HaveOccurred())
				Expect(path).To(Equal("/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"))
			})

			It("should resolve the azure-openai responses path from the host root", func() {
				// Azure's v1 Responses route ignores the deployment-scoped base URL and
				// takes no api-version; the deployment name travels in the request body.
				profile := &ProviderProfile{
					Type:       "azure-openai",
					BaseURL:    "https://myresource.openai.azure.com/openai/deployments/astra-prod",
					APIVersion: "2024-10-21",
				}
				path, err := profile.ResolveCreatePath("openai/responses@1")
				Expect(err).NotTo(HaveOccurred())
				Expect(path).To(Equal("/openai/v1/responses"))
			})

			It("should keep the deployment-scoped azure-openai chat path", func() {
				profile := &ProviderProfile{
					Type:       "azure-openai",
					BaseURL:    "https://myresource.openai.azure.com/openai/deployments/astra-prod",
					APIVersion: "2024-10-21",
				}
				path, err := profile.ResolveCreatePath("openai/chat-completions@1")
				Expect(err).NotTo(HaveOccurred())
				Expect(path).To(Equal(
					"/openai/deployments/astra-prod/chat/completions?api-version=2024-10-21"))
			})

			It("should resolve catalog-backed private runtimes", func() {
				path, err := (&ProviderProfile{Type: "vllm"}).ResolveCreatePath("")
				Expect(err).NotTo(HaveOccurred())
				Expect(path).To(Equal("/v1/chat/completions"))
			})

			It("should use explicit ChatPath override", func() {
				profile := &ProviderProfile{Type: "openai", ChatPath: "/custom/path"}
				path, err := profile.ResolveCreatePath("")
				Expect(err).NotTo(HaveOccurred())
				Expect(path).To(Equal("/custom/path"))
			})

			It("should error for nil profile", func() {
				var nilProfile *ProviderProfile
				_, err := nilProfile.ResolveCreatePath("")
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("nil"))
			})
		})

		Context("ExtraHeaders", func() {
			It("should pass through explicit extra_headers only", func() {
				profile := &ProviderProfile{
					Type:         "anthropic",
					ExtraHeaders: map[string]string{"anthropic-version": "2023-06-01", "custom": "value"},
				}
				Expect(profile.ExtraHeaders).To(HaveKeyWithValue("anthropic-version", "2023-06-01"))
				Expect(profile.ExtraHeaders).To(HaveKeyWithValue("custom", "value"))
			})

			It("should be nil when not configured", func() {
				profile := &ProviderProfile{Type: "openai"}
				Expect(profile.ExtraHeaders).To(BeNil())
			})
		})

		Context("GetProviderProfileForEndpoint", func() {
			It("should resolve endpoint to profile", func() {
				cfg := &RouterConfig{
					BackendModels: BackendModels{
						VLLMEndpoints: []VLLMEndpoint{
							{Name: "openai", ProviderProfileName: "openai-prod"},
							{Name: "local", Address: "127.0.0.1", Port: 8000},
						},
						ProviderProfiles: map[string]ProviderProfile{
							"openai-prod": {Type: "openai", BaseURL: "https://api.openai.com/v1"},
						},
					},
				}
				profile, err := cfg.GetProviderProfileForEndpoint("openai")
				Expect(err).NotTo(HaveOccurred())
				Expect(profile).NotTo(BeNil())
				Expect(profile.Type).To(Equal("openai"))

				// Endpoint without profile — valid, returns nil profile and no error
				profile, err = cfg.GetProviderProfileForEndpoint("local")
				Expect(err).NotTo(HaveOccurred())
				Expect(profile).To(BeNil())
			})

			It("should error on non-existent endpoint", func() {
				cfg := &RouterConfig{
					BackendModels: BackendModels{
						VLLMEndpoints: []VLLMEndpoint{
							{Name: "local", Address: "127.0.0.1", Port: 8000},
						},
					},
				}
				_, err := cfg.GetProviderProfileForEndpoint("nonexistent")
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("not found"))
			})

			It("should error on dangling profile reference", func() {
				cfg := &RouterConfig{
					BackendModels: BackendModels{
						VLLMEndpoints: []VLLMEndpoint{
							{Name: "openai", ProviderProfileName: "missing-profile"},
						},
						ProviderProfiles: map[string]ProviderProfile{
							"other-profile": {Type: "openai"},
						},
					},
				}
				_, err := cfg.GetProviderProfileForEndpoint("openai")
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("does not exist"))
				Expect(err.Error()).To(ContainSubstring("missing-profile"))
			})
		})

		Context("ResolvePrimaryBackendForModel with profiles", func() {
			It("should use base_url for address resolution", func() {
				cfg := &RouterConfig{
					BackendModels: BackendModels{
						ModelConfig: map[string]ModelParams{
							"gpt-4o": {PreferredEndpoints: []string{"openai"}},
						},
						VLLMEndpoints: []VLLMEndpoint{
							{Name: "openai", ProviderProfileName: "openai-prod"},
						},
						ProviderProfiles: map[string]ProviderProfile{
							"openai-prod": {Type: "openai", BaseURL: "https://api.openai.com/v1"},
						},
					},
				}

				addr, name, found, detailErr := cfg.ResolvePrimaryBackendForModel("gpt-4o")
				Expect(detailErr).NotTo(HaveOccurred())
				Expect(found).To(BeTrue())
				Expect(name).To(Equal("openai"))
				Expect(addr).To(Equal("api.openai.com:443"))
			})
		})
	})

	// -----------------------------------------------------------------------
	// PromptCompressionConfig
	// -----------------------------------------------------------------------
	Describe("PromptCompressionConfig", func() {
		Context("profiles", func() {
			It("should expose the stable built-in profile list", func() {
				Expect(ValidPromptCompressionProfiles()).To(Equal([]string{
					"default",
					"coding",
					"medical",
					"security",
					"multi_turn",
				}))
			})

			It("should normalize omitted and hyphenated profile names", func() {
				Expect((PromptCompressionConfig{}).NormalizedProfile()).To(Equal("default"))
				Expect((PromptCompressionConfig{Profile: "Multi-Turn"}).NormalizedProfile()).To(Equal("multi_turn"))
			})

			It("should reject unknown profile names", func() {
				err := validateSignalsCompression("signals", &PromptCompressionConfig{Enabled: true, Profile: "finance", MaxTokens: 512})
				Expect(err).To(MatchError(ContainSubstring(`signals.profile: unknown profile "finance"`)))
				Expect(err).To(MatchError(ContainSubstring("default, coding, medical, security, multi_turn")))
			})

			It("should accept the multi-turn profile alias", func() {
				Expect(validateSignalsCompression("signals", &PromptCompressionConfig{Enabled: true, Profile: "multi-turn", MaxTokens: 512})).To(Succeed())
			})
		})

		Context("SkipSignalsSet", func() {
			It("should default to jailbreak and pii when SkipSignals is empty", func() {
				pc := PromptCompressionConfig{Enabled: true, MaxTokens: 512}
				s := pc.SkipSignalsSet()
				Expect(s).To(HaveKey("jailbreak"))
				Expect(s).To(HaveKey("pii"))
				Expect(s).To(HaveLen(2))
			})

			It("should use configured SkipSignals when provided", func() {
				pc := PromptCompressionConfig{
					Enabled:     true,
					MaxTokens:   512,
					SkipSignals: []string{"jailbreak"},
				}
				s := pc.SkipSignalsSet()
				Expect(s).To(HaveKey("jailbreak"))
				Expect(s).NotTo(HaveKey("pii"))
				Expect(s).To(HaveLen(1))
			})

			It("should support adding custom signal types to skip list", func() {
				pc := PromptCompressionConfig{
					Enabled:     true,
					MaxTokens:   512,
					SkipSignals: []string{"jailbreak", "pii", "fact_check"},
				}
				s := pc.SkipSignalsSet()
				Expect(s).To(HaveKey("jailbreak"))
				Expect(s).To(HaveKey("pii"))
				Expect(s).To(HaveKey("fact_check"))
				Expect(s).To(HaveLen(3))
			})
		})

		Context("MinLength", func() {
			It("should be zero by default", func() {
				pc := PromptCompressionConfig{Enabled: true, MaxTokens: 512}
				Expect(pc.MinLength).To(Equal(0))
			})

			It("should parse from YAML", func() {
				yamlStr := `
enabled: true
max_tokens: 512
min_length: 2000
skip_signals:
  - jailbreak
  - pii
`
				var pc PromptCompressionConfig
				err := yaml.Unmarshal([]byte(yamlStr), &pc)
				Expect(err).NotTo(HaveOccurred())
				Expect(pc.Enabled).To(BeTrue())
				Expect(pc.MaxTokens).To(Equal(512))
				Expect(pc.MinLength).To(Equal(2000))
				Expect(pc.SkipSignals).To(Equal([]string{"jailbreak", "pii"}))
			})

			It("should parse custom skip signals from YAML", func() {
				yamlStr := `
enabled: true
max_tokens: 256
min_length: 1000
skip_signals:
  - jailbreak
`
				var pc PromptCompressionConfig
				err := yaml.Unmarshal([]byte(yamlStr), &pc)
				Expect(err).NotTo(HaveOccurred())
				Expect(pc.MinLength).To(Equal(1000))
				Expect(pc.SkipSignals).To(Equal([]string{"jailbreak"}))
				s := pc.SkipSignalsSet()
				Expect(s).To(HaveKey("jailbreak"))
				Expect(s).NotTo(HaveKey("pii"))
			})
		})
	})
})
