package modeldownload

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	testFactCheckModel             = "models/test-fact-check"
	testFeedbackModel              = "models/test-feedback"
	testHallucinationDetectorModel = "models/test-hallucination-detector"
)

func TestBuildModelSpecsCoversNamedRecipeSignalsAndPlugins(t *testing.T) {
	cfg := loadGenericMultiRecipeModelNeedsConfig(t)

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}

	want := []string{
		testFactCheckModel,
		testFeedbackModel,
		testHallucinationDetectorModel,
	}
	assertExactModelSpecs(t, specs, want)

	feedbackSpec, ok := findModelSpec(specs, testFeedbackModel)
	if !ok {
		t.Fatalf("feedback model spec %q not found", testFeedbackModel)
	}
	if !slices.Contains(feedbackSpec.RequiredFiles, "feedback_mapping.json") {
		t.Fatalf("feedback model required files = %v, missing mapping", feedbackSpec.RequiredFiles)
	}
}

func TestBuildModelSpecsSkipsLocalHallucinationSnapshotsForEndpointBackend(t *testing.T) {
	cfg := loadGenericMultiRecipeModelNeedsConfig(t)
	// The legacy endpoint form desugars into a remote binding when compiled, so
	// it has to be complete: an endpoint backend without an endpoint is a
	// configuration error, not a local model to download.
	cfg.HallucinationMitigation.HallucinationModel.Backend = config.HallucinationBackendEndpoint
	cfg.HallucinationMitigation.HallucinationModel.Endpoint = "http://127.0.0.1:8077/v1"
	if cfg.HallucinationMitigation.HallucinationModel.ModelID == "" {
		cfg.HallucinationMitigation.HallucinationModel.ModelID = "remote-detector"
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}

	assertExactModelSpecs(t, specs, []string{
		testFactCheckModel,
		testFeedbackModel,
	})
}

func TestBuildModelSpecsPreservesDefaultAPIOnlyModels(t *testing.T) {
	cfg := &config.RouterConfig{
		MoMRegistry: map[string]string{
			testFactCheckModel:             "test/fact-check",
			testFeedbackModel:              "test/feedback",
			testHallucinationDetectorModel: "test/hallucination-detector",
		},
		InlineModels: config.InlineModels{
			HallucinationMitigation: config.HallucinationMitigationConfig{
				Enabled:            true,
				FactCheckModel:     config.FactCheckModelConfig{ModelID: testFactCheckModel},
				HallucinationModel: config.HallucinationModelConfig{ModelID: testHallucinationDetectorModel},
			},
			FeedbackDetector: config.FeedbackDetectorConfig{
				Enabled: true,
				ModelID: testFeedbackModel,
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				FactCheckRules:    []config.FactCheckRule{{Name: "verification-needed"}},
				UserFeedbackRules: []config.UserFeedbackRule{{Name: "correction-needed"}},
			},
		},
	}
	cfg.EmbeddingModels.EmbeddingConfig.Backend = config.EmbeddingBackendOpenAICompatible

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	assertExactModelSpecs(t, specs, []string{
		testFactCheckModel,
		testFeedbackModel,
		testHallucinationDetectorModel,
	})
}

func TestBuildModelSpecsSkipsUnreachableNamedRecipeModels(t *testing.T) {
	cfg := loadGenericMultiRecipeModelNeedsConfig(t)
	entrypoints := cfg.Entrypoints
	cfg.Entrypoints = slices.DeleteFunc(slices.Clone(entrypoints), func(entrypoint config.EntrypointMapping) bool {
		return entrypoint.Recipe == "verification"
	})

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	assertExactModelSpecs(t, specs, []string{testFeedbackModel})

	cfg.Entrypoints = entrypoints
	specs, err = BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() with entrypoint error = %v", err)
	}
	assertExactModelSpecs(t, specs, []string{
		testFactCheckModel,
		testFeedbackModel,
		testHallucinationDetectorModel,
	})
}

func TestBuildModelSpecsAccountsForDefaultAutoReachability(t *testing.T) {
	cfg := parseModelNeedsConfig(t, []byte(`
version: v0.3

routing:
  signals:
    fact_check:
      - name: verification-needed
        description: Requests that require verification.
  decisions:
    - name: default-route
      rules:
        type: fact_check
        name: verification-needed
      plugins:
        - type: hallucination
          configuration:
            enabled: true

global:
  model_catalog:
    system:
      fact_check_classifier: models/test-fact-check
      hallucination_detector: models/test-hallucination-detector
    modules:
      hallucination_mitigation:
        fact_check:
          model_ref: fact_check_classifier
        detector:
          model_ref: hallucination_detector
`))
	want := []string{testFactCheckModel, testHallucinationDetectorModel}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	assertExactModelSpecs(t, specs, want)

	// Renaming the default public entrypoint preserves its model dependencies.
	cfg.Entrypoints = []config.EntrypointMapping{{ModelNames: []string{"special"}, Recipe: config.DefaultRecipeName}}
	specs, err = BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() with an explicit default entrypoint: %v", err)
	}
	assertExactModelSpecs(t, specs, want)
}

func loadGenericMultiRecipeModelNeedsConfig(t *testing.T) *config.RouterConfig {
	t.Helper()
	path := filepath.Join("..", "config", "testdata", "generic-multi-recipe-model-needs.yaml")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read generic multi-recipe fixture: %v", err)
	}
	return parseModelNeedsConfig(t, data)
}

func parseModelNeedsConfig(t *testing.T, data []byte) *config.RouterConfig {
	t.Helper()
	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("parse model needs config: %v", err)
	}

	cfg.EmbeddingModels.EmbeddingConfig.Backend = config.EmbeddingBackendOpenAICompatible
	cfg.MoMRegistry = map[string]string{
		testFactCheckModel:             "test/fact-check",
		testFeedbackModel:              "test/feedback",
		testHallucinationDetectorModel: "test/hallucination-detector",
	}
	return cfg
}

func assertExactModelSpecs(t *testing.T, specs []ModelSpec, want []string) {
	t.Helper()
	if len(specs) != len(want) {
		t.Fatalf("BuildModelSpecs() returned %d specs, want %d: %#v", len(specs), len(want), specs)
	}
	for _, path := range want {
		if _, ok := findModelSpec(specs, path); !ok {
			t.Fatalf("BuildModelSpecs() missing %q: %#v", path, specs)
		}
	}
}

func findModelSpec(specs []ModelSpec, path string) (ModelSpec, bool) {
	for _, spec := range specs {
		if spec.LocalPath == path {
			return spec, true
		}
	}
	return ModelSpec{}, false
}
