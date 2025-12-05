package k8s

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
	k8syaml "k8s.io/apimachinery/pkg/util/yaml"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apis/vllm.ai/v1alpha1"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestConverterWithTestData tests the converter with input/output test data
// This test reads YAML files from testdata/input, converts them, and writes output to testdata/output
func TestConverterWithTestData(t *testing.T) {
	testdataDir := "testdata"
	inputDir := filepath.Join(testdataDir, "input")
	outputDir := filepath.Join(testdataDir, "output")
	baseConfigPath := filepath.Join(testdataDir, "base-config.yaml")

	// Ensure output directory exists
	err := os.MkdirAll(outputDir, 0o755)
	require.NoError(t, err, "Failed to create output directory")

	// Load base config (static parts)
	baseConfigData, err := os.ReadFile(baseConfigPath)
	require.NoError(t, err, "Failed to read base config file: %s", baseConfigPath)

	var baseConfig config.RouterConfig
	err = yaml.Unmarshal(baseConfigData, &baseConfig)
	require.NoError(t, err, "Failed to unmarshal base config")

	// Read all input files
	inputFiles, err := os.ReadDir(inputDir)
	require.NoError(t, err, "Failed to read input directory")

	converter := NewCRDConverter()

	for _, inputFile := range inputFiles {
		if !strings.HasSuffix(inputFile.Name(), ".yaml") && !strings.HasSuffix(inputFile.Name(), ".yml") {
			continue
		}

		t.Run(inputFile.Name(), func(t *testing.T) {
			inputPath := filepath.Join(inputDir, inputFile.Name())
			outputPath := filepath.Join(outputDir, inputFile.Name())

			// Read input file
			inputData, err := os.ReadFile(inputPath)
			require.NoError(t, err, "Failed to read input file: %s", inputPath)

			// Parse YAML documents (pool and route)
			pool, route, err := parseInputYAML(inputData)
			require.NoError(t, err, "Failed to parse input YAML: %s", inputPath)
			require.NotNil(t, pool, "IntelligentPool should not be nil")
			require.NotNil(t, route, "IntelligentRoute should not be nil")

			// Validate CRDs
			err = validateCRDs(pool, route, &baseConfig)
			require.NoError(t, err, "CRD validation failed for %s", inputFile.Name())

			// Convert pool to backend models
			backendModels, err := converter.ConvertIntelligentPool(pool)
			require.NoError(t, err, "Failed to convert IntelligentPool")

			// Convert route to intelligent routing
			intelligentRouting, err := converter.ConvertIntelligentRoute(route)
			require.NoError(t, err, "Failed to convert IntelligentRoute")

			// Merge base config with CRD-derived config
			outputConfig := mergeConfigs(&baseConfig, backendModels, intelligentRouting)

			// Convert plugin configurations from []byte to map for YAML serialization
			normalizePluginConfigurations(outputConfig)

			// Marshal to YAML with 2-space indentation
			var buf strings.Builder
			encoder := yaml.NewEncoder(&buf)
			encoder.SetIndent(2) // Set 2-space indentation to match yamllint config
			err = encoder.Encode(outputConfig)
			require.NoError(t, err, "Failed to marshal output config")
			encoder.Close()

			// Write output file
			err = os.WriteFile(outputPath, []byte(buf.String()), 0o644)
			require.NoError(t, err, "Failed to write output file: %s", outputPath)

			t.Logf("Generated output file: %s", outputPath)

			// Validate the output can be unmarshaled back
			var validateConfig config.RouterConfig
			err = yaml.Unmarshal([]byte(buf.String()), &validateConfig)
			require.NoError(t, err, "Failed to unmarshal generated output")

			// Basic validation
			assert.NotNil(t, validateConfig.BackendModels, "BackendModels should not be nil")
			assert.NotNil(t, validateConfig.IntelligentRouting, "IntelligentRouting should not be nil")
			assert.Len(t, validateConfig.BackendModels.ModelConfig, len(backendModels.ModelConfig), "BackendModels count mismatch")
			assert.Len(t, validateConfig.IntelligentRouting.Decisions, len(intelligentRouting.Decisions), "Decisions count mismatch")
		})
	}
}

// mergeConfigs merges base config with CRD-derived dynamic parts
func mergeConfigs(baseConfig *config.RouterConfig, backendModels *config.BackendModels, intelligentRouting *config.IntelligentRouting) *config.RouterConfig {
	// Start with a copy of base config (contains all static parts)
	merged := *baseConfig

	// Override config source
	merged.ConfigSource = config.ConfigSourceKubernetes

	// Override dynamic parts from CRDs
	merged.BackendModels = *backendModels

	// Merge IntelligentRouting while preserving ReasoningConfig from base
	merged.IntelligentRouting.KeywordRules = intelligentRouting.KeywordRules
	merged.IntelligentRouting.EmbeddingRules = intelligentRouting.EmbeddingRules
	merged.IntelligentRouting.Categories = intelligentRouting.Categories
	merged.IntelligentRouting.Decisions = intelligentRouting.Decisions
	merged.IntelligentRouting.Strategy = intelligentRouting.Strategy
	// Keep ReasoningConfig from base (ReasoningFamilies, DefaultReasoningEffort)

	return &merged
}

// parseInputYAML parses a multi-document YAML file containing IntelligentPool and IntelligentRoute
func parseInputYAML(data []byte) (*v1alpha1.IntelligentPool, *v1alpha1.IntelligentRoute, error) {
	decoder := k8syaml.NewYAMLOrJSONDecoder(strings.NewReader(string(data)), 4096)

	var pool *v1alpha1.IntelligentPool
	var route *v1alpha1.IntelligentRoute

	for {
		var obj map[string]interface{}
		err := decoder.Decode(&obj)
		if err != nil {
			// Check for EOF
			if strings.Contains(err.Error(), "EOF") {
				break
			}
			return nil, nil, err
		}

		if obj == nil {
			continue
		}

		kind, ok := obj["kind"].(string)
		if !ok {
			continue
		}

		// Re-marshal to JSON (runtime.RawExtension expects JSON)
		objData, err := json.Marshal(obj)
		if err != nil {
			return nil, nil, err
		}

		switch kind {
		case "IntelligentPool":
			pool = &v1alpha1.IntelligentPool{}
			if err := json.Unmarshal(objData, pool); err != nil {
				return nil, nil, err
			}
		case "IntelligentRoute":
			route = &v1alpha1.IntelligentRoute{}
			if err := json.Unmarshal(objData, route); err != nil {
				return nil, nil, err
			}
		}
	}

	return pool, route, nil
}

// normalizePluginConfigurations converts plugin configurations from []byte to map[string]interface{}
// This is needed for proper YAML serialization
func normalizePluginConfigurations(cfg *config.RouterConfig) {
	for i := range cfg.IntelligentRouting.Decisions {
		decision := &cfg.IntelligentRouting.Decisions[i]
		for j := range decision.Plugins {
			plugin := &decision.Plugins[j]
			if plugin.Configuration != nil {
				// If configuration is []byte (from Kubernetes RawExtension), convert to map
				if bytes, ok := plugin.Configuration.([]byte); ok {
					var configMap map[string]interface{}
					if err := json.Unmarshal(bytes, &configMap); err == nil {
						plugin.Configuration = configMap
					}
				}
			}
		}
	}
}

// validateCRDs validates IntelligentPool and IntelligentRoute CRDs
// This mirrors the validation logic in controller.go
func validateCRDs(pool *v1alpha1.IntelligentPool, route *v1alpha1.IntelligentRoute, staticConfig *config.RouterConfig) error {
	// Build model map
	modelMap := make(map[string]*v1alpha1.ModelConfig)
	for i := range pool.Spec.Models {
		model := &pool.Spec.Models[i]
		modelMap[model.Name] = model
	}

	// Build signal name sets
	keywordSignalNames := make(map[string]bool)
	embeddingSignalNames := make(map[string]bool)
	domainSignalNames := make(map[string]bool)

	// Check for duplicate keyword signals
	for _, signal := range route.Spec.Signals.Keywords {
		if keywordSignalNames[signal.Name] {
			return fmt.Errorf("duplicate keyword signal name: %s", signal.Name)
		}
		keywordSignalNames[signal.Name] = true
	}

	// Check for duplicate embedding signals
	for _, signal := range route.Spec.Signals.Embeddings {
		if embeddingSignalNames[signal.Name] {
			return fmt.Errorf("duplicate embedding signal name: %s", signal.Name)
		}
		embeddingSignalNames[signal.Name] = true
	}

	// Check for duplicate domain signals
	for _, domain := range route.Spec.Signals.Domains {
		if domainSignalNames[domain.Name] {
			return fmt.Errorf("duplicate domain signal name: %s", domain.Name)
		}
		domainSignalNames[domain.Name] = true
	}

	// Validate decisions
	for _, decision := range route.Spec.Decisions {
		// Validate signal references
		for _, condition := range decision.Signals.Conditions {
			switch condition.Type {
			case "keyword":
				if !keywordSignalNames[condition.Name] {
					return fmt.Errorf("decision %s references unknown keyword signal: %s", decision.Name, condition.Name)
				}
			case "embedding":
				if !embeddingSignalNames[condition.Name] {
					return fmt.Errorf("decision %s references unknown embedding signal: %s", decision.Name, condition.Name)
				}
			case "domain":
				if !domainSignalNames[condition.Name] {
					return fmt.Errorf("decision %s references unknown domain signal: %s", decision.Name, condition.Name)
				}
			}
		}

		// Validate model references
		for _, ms := range decision.ModelRefs {
			model, ok := modelMap[ms.Model]
			if !ok {
				return fmt.Errorf("decision %s references unknown model: %s", decision.Name, ms.Model)
			}

			// Validate LoRA reference
			if ms.LoRAName != "" {
				found := false
				for _, lora := range model.LoRAs {
					if lora.Name == ms.LoRAName {
						found = true
						break
					}
				}
				if !found {
					return fmt.Errorf("decision %s references unknown LoRA %s for model %s", decision.Name, ms.LoRAName, ms.Model)
				}
			}
		}
	}

	// Validate reasoning families
	if staticConfig != nil && staticConfig.ReasoningFamilies != nil {
		for _, model := range pool.Spec.Models {
			if model.ReasoningFamily != "" {
				if _, ok := staticConfig.ReasoningFamilies[model.ReasoningFamily]; !ok {
					return fmt.Errorf("model %s references unknown reasoning family: %s", model.Name, model.ReasoningFamily)
				}
			}
		}
	}

	return nil
}

// TestCRDValidationErrors tests that validation catches various error conditions
func TestCRDValidationErrors(t *testing.T) {
	baseConfig := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			ReasoningConfig: config.ReasoningConfig{
				ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
					"qwen3": {
						Type:      "chat_template_kwargs",
						Parameter: "enable_thinking",
					},
				},
			},
		},
	}

	t.Run("DuplicateKeywordSignal", func(t *testing.T) {
		pool := &v1alpha1.IntelligentPool{
			Spec: v1alpha1.IntelligentPoolSpec{
				DefaultModel: "test-model",
				Models: []v1alpha1.ModelConfig{
					{Name: "test-model"},
				},
			},
		}

		route := &v1alpha1.IntelligentRoute{
			Spec: v1alpha1.IntelligentRouteSpec{
				Signals: v1alpha1.Signals{
					Keywords: []v1alpha1.KeywordSignal{
						{Name: "urgent", Operator: "OR", Keywords: []string{"urgent"}},
						{Name: "urgent", Operator: "OR", Keywords: []string{"critical"}}, // Duplicate!
					},
				},
				Decisions: []v1alpha1.Decision{},
			},
		}

		err := validateCRDs(pool, route, baseConfig)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "duplicate keyword signal name: urgent")
	})

	t.Run("UnknownKeywordSignalReference", func(t *testing.T) {
		pool := &v1alpha1.IntelligentPool{
			Spec: v1alpha1.IntelligentPoolSpec{
				DefaultModel: "test-model",
				Models: []v1alpha1.ModelConfig{
					{Name: "test-model"},
				},
			},
		}

		route := &v1alpha1.IntelligentRoute{
			Spec: v1alpha1.IntelligentRouteSpec{
				Signals: v1alpha1.Signals{
					Keywords: []v1alpha1.KeywordSignal{
						{Name: "urgent", Operator: "OR", Keywords: []string{"urgent"}},
					},
				},
				Decisions: []v1alpha1.Decision{
					{
						Name:     "test-decision",
						Priority: 100,
						Signals: v1alpha1.SignalCombination{
							Operator: "AND",
							Conditions: []v1alpha1.SignalCondition{
								{Type: "keyword", Name: "nonexistent"}, // Unknown signal!
							},
						},
						ModelRefs: []v1alpha1.ModelRef{
							{Model: "test-model"},
						},
					},
				},
			},
		}

		err := validateCRDs(pool, route, baseConfig)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "references unknown keyword signal: nonexistent")
	})

	t.Run("UnknownModelReference", func(t *testing.T) {
		pool := &v1alpha1.IntelligentPool{
			Spec: v1alpha1.IntelligentPoolSpec{
				DefaultModel: "test-model",
				Models: []v1alpha1.ModelConfig{
					{Name: "test-model"},
				},
			},
		}

		route := &v1alpha1.IntelligentRoute{
			Spec: v1alpha1.IntelligentRouteSpec{
				Signals: v1alpha1.Signals{
					Keywords: []v1alpha1.KeywordSignal{
						{Name: "urgent", Operator: "OR", Keywords: []string{"urgent"}},
					},
				},
				Decisions: []v1alpha1.Decision{
					{
						Name:     "test-decision",
						Priority: 100,
						Signals: v1alpha1.SignalCombination{
							Operator: "AND",
							Conditions: []v1alpha1.SignalCondition{
								{Type: "keyword", Name: "urgent"},
							},
						},
						ModelRefs: []v1alpha1.ModelRef{
							{Model: "nonexistent-model"}, // Unknown model!
						},
					},
				},
			},
		}

		err := validateCRDs(pool, route, baseConfig)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "references unknown model: nonexistent-model")
	})

	t.Run("UnknownLoRAReference", func(t *testing.T) {
		pool := &v1alpha1.IntelligentPool{
			Spec: v1alpha1.IntelligentPoolSpec{
				DefaultModel: "test-model",
				Models: []v1alpha1.ModelConfig{
					{
						Name: "test-model",
						LoRAs: []v1alpha1.LoRAConfig{
							{Name: "expert-lora"},
						},
					},
				},
			},
		}

		route := &v1alpha1.IntelligentRoute{
			Spec: v1alpha1.IntelligentRouteSpec{
				Signals: v1alpha1.Signals{
					Keywords: []v1alpha1.KeywordSignal{
						{Name: "urgent", Operator: "OR", Keywords: []string{"urgent"}},
					},
				},
				Decisions: []v1alpha1.Decision{
					{
						Name:     "test-decision",
						Priority: 100,
						Signals: v1alpha1.SignalCombination{
							Operator: "AND",
							Conditions: []v1alpha1.SignalCondition{
								{Type: "keyword", Name: "urgent"},
							},
						},
						ModelRefs: []v1alpha1.ModelRef{
							{Model: "test-model", LoRAName: "nonexistent-lora"}, // Unknown LoRA!
						},
					},
				},
			},
		}

		err := validateCRDs(pool, route, baseConfig)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "references unknown LoRA nonexistent-lora")
	})
}
