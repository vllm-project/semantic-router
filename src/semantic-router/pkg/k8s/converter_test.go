package k8s

import (
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

	baseRouterConfig, err := config.ParseYAMLBytes(baseConfigData)
	require.NoError(t, err, "Failed to parse canonical base config")

	var baseCanonical config.CanonicalConfig
	err = yaml.Unmarshal(baseConfigData, &baseCanonical)
	require.NoError(t, err, "Failed to unmarshal canonical base config")

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
			err = validateCRDs(pool, route, baseRouterConfig)
			require.NoError(t, err, "CRD validation failed for %s", inputFile.Name())

			outputConfig, err := converter.Convert(pool, route, &baseCanonical)
			require.NoError(t, err, "Failed to convert CRDs to canonical config")

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

			decoder := yaml.NewDecoder(strings.NewReader(buf.String()))
			decoder.KnownFields(true)
			var strictCanonical config.CanonicalConfig
			err = decoder.Decode(&strictCanonical)
			require.NoError(t, err, "Failed strict canonical decode")

			validateConfig, err := config.ParseYAMLBytes([]byte(buf.String()))
			require.NoError(t, err, "Failed runtime parse validation")
			assert.Equal(t, config.ConfigSourceKubernetes, validateConfig.ConfigSource, "ConfigSource should remain kubernetes")
			assert.Equal(t, pool.Spec.DefaultModel, validateConfig.DefaultModel, "default model mismatch")
			assert.Len(t, validateConfig.Decisions, len(route.Spec.Decisions), "Decisions count mismatch")
			assert.Len(t, validateConfig.ModelConfig, len(pool.Spec.Models), "Model catalog count mismatch")
		})
	}
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

		switch kind {
		case "IntelligentPool":
			pool = &v1alpha1.IntelligentPool{}
			if err := remarshalYAMLObject(obj, pool); err != nil {
				return nil, nil, err
			}
		case "IntelligentRoute":
			route = &v1alpha1.IntelligentRoute{}
			if err := remarshalYAMLObject(obj, route); err != nil {
				return nil, nil, err
			}
		}
	}

	return pool, route, nil
}

func remarshalYAMLObject(input map[string]interface{}, target interface{}) error {
	data, err := yaml.Marshal(input)
	if err != nil {
		return err
	}
	return yaml.Unmarshal(data, target)
}

// validateCRDs validates IntelligentPool and IntelligentRoute CRDs
// This mirrors the validation logic in controller.go
func validateCRDs(pool *v1alpha1.IntelligentPool, route *v1alpha1.IntelligentRoute, staticConfig *config.RouterConfig) error {
	var reasoningFamilies map[string]config.ReasoningFamilyConfig
	if staticConfig != nil {
		reasoningFamilies = staticConfig.ReasoningFamilies
	}
	return validatePoolRoute(pool, route, reasoningFamilies)
}

func testValidationBaseConfig() *config.RouterConfig {
	return &config.RouterConfig{
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
}

func testPoolWithModels(models ...v1alpha1.ModelConfig) *v1alpha1.IntelligentPool {
	return &v1alpha1.IntelligentPool{
		Spec: v1alpha1.IntelligentPoolSpec{
			DefaultModel: "test-model",
			Models:       models,
		},
	}
}

func testRouteWithKeywords(
	keywords []v1alpha1.KeywordSignal,
	decisions ...v1alpha1.Decision,
) *v1alpha1.IntelligentRoute {
	return &v1alpha1.IntelligentRoute{
		Spec: v1alpha1.IntelligentRouteSpec{
			Signals:   v1alpha1.Signals{Keywords: keywords},
			Decisions: decisions,
		},
	}
}

func testDecision(
	modelRefs []v1alpha1.ModelRef,
	conditions ...v1alpha1.SignalCondition,
) v1alpha1.Decision {
	return v1alpha1.Decision{
		Name:     "test-decision",
		Priority: 100,
		Signals: v1alpha1.SignalCombination{
			Operator:   "AND",
			Conditions: conditions,
		},
		ModelRefs: modelRefs,
	}
}

// TestCRDValidationErrors tests that validation catches various error conditions
func TestCRDValidationErrors(t *testing.T) {
	testCases := []struct {
		name      string
		pool      *v1alpha1.IntelligentPool
		route     *v1alpha1.IntelligentRoute
		wantError string
	}{
		{
			name: "DuplicateKeywordSignal",
			pool: testPoolWithModels(v1alpha1.ModelConfig{Name: "test-model"}),
			route: testRouteWithKeywords(
				[]v1alpha1.KeywordSignal{
					{Name: "urgent", Operator: "OR", Keywords: []string{"urgent"}},
					{Name: "urgent", Operator: "OR", Keywords: []string{"critical"}},
				},
			),
			wantError: "duplicate keyword signal name: urgent",
		},
		{
			name: "UnknownKeywordSignalReference",
			pool: testPoolWithModels(v1alpha1.ModelConfig{Name: "test-model"}),
			route: testRouteWithKeywords(
				[]v1alpha1.KeywordSignal{{Name: "urgent", Operator: "OR", Keywords: []string{"urgent"}}},
				testDecision(
					[]v1alpha1.ModelRef{{Model: "test-model"}},
					v1alpha1.SignalCondition{Type: "keyword", Name: "nonexistent"},
				),
			),
			wantError: "references unknown keyword signal: nonexistent",
		},
		{
			name: "UnknownModelReference",
			pool: testPoolWithModels(v1alpha1.ModelConfig{Name: "test-model"}),
			route: testRouteWithKeywords(
				[]v1alpha1.KeywordSignal{{Name: "urgent", Operator: "OR", Keywords: []string{"urgent"}}},
				testDecision(
					[]v1alpha1.ModelRef{{Model: "nonexistent-model"}},
					v1alpha1.SignalCondition{Type: "keyword", Name: "urgent"},
				),
			),
			wantError: "references unknown model: nonexistent-model",
		},
		{
			name: "UnknownLoRAReference",
			pool: testPoolWithModels(v1alpha1.ModelConfig{
				Name:  "test-model",
				LoRAs: []v1alpha1.LoRAConfig{{Name: "expert-lora"}},
			}),
			route: testRouteWithKeywords(
				[]v1alpha1.KeywordSignal{{Name: "urgent", Operator: "OR", Keywords: []string{"urgent"}}},
				testDecision(
					[]v1alpha1.ModelRef{{Model: "test-model", LoRAName: "nonexistent-lora"}},
					v1alpha1.SignalCondition{Type: "keyword", Name: "urgent"},
				),
			),
			wantError: "references unknown LoRA nonexistent-lora",
		},
	}

	baseConfig := testValidationBaseConfig()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateCRDs(tc.pool, tc.route, baseConfig)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantError)
		})
	}
}
