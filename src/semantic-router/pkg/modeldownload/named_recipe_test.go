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
	const unreachableModel = "models/unreachable-router"
	cfg := &config.RouterConfig{
		MoMRegistry: map[string]string{
			unreachableModel: "test/unreachable-router",
		},
		Recipes: []config.RoutingRecipe{
			{Name: config.DefaultRecipeName},
			{
				Name: "unmapped",
				Profile: config.RoutingProfile{Decisions: []config.Decision{{
					Name: "unmapped-route",
					Algorithm: &config.AlgorithmConfig{
						GMTRouter: &config.GMTRouterSelectionConfig{ModelPath: unreachableModel},
					},
				}}},
			},
		},
	}
	cfg.EmbeddingModels.EmbeddingConfig.Backend = config.EmbeddingBackendOpenAICompatible

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	if len(specs) != 0 {
		t.Fatalf("unreachable named recipe produced model specs: %#v", specs)
	}

	cfg.Entrypoints = []config.EntrypointMapping{{
		ModelNames: []string{"vllm-sr/unmapped"},
		Recipe:     "unmapped",
	}}
	specs, err = BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() with entrypoint error = %v", err)
	}
	assertExactModelSpecs(t, specs, []string{unreachableModel})
}

func TestBuildModelSpecsAccountsForDefaultAutoReachability(t *testing.T) {
	const defaultModel = "models/default-router"
	cfg := &config.RouterConfig{
		MoMRegistry: map[string]string{
			defaultModel: "test/default-router",
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{
				Name: "default-route",
				Algorithm: &config.AlgorithmConfig{
					GMTRouter: &config.GMTRouterSelectionConfig{ModelPath: defaultModel},
				},
			}},
		},
	}
	cfg.EmbeddingModels.EmbeddingConfig.Backend = config.EmbeddingBackendOpenAICompatible

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	assertExactModelSpecs(t, specs, []string{defaultModel})

	// Renaming the default public entrypoint preserves its model dependencies.
	cfg.Entrypoints = []config.EntrypointMapping{{ModelNames: []string{"special"}, Recipe: config.DefaultRecipeName}}
	specs, err = BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() with an explicit default entrypoint: %v", err)
	}
	assertExactModelSpecs(t, specs, []string{defaultModel})
}

func loadGenericMultiRecipeModelNeedsConfig(t *testing.T) *config.RouterConfig {
	t.Helper()
	path := filepath.Join("..", "config", "testdata", "generic-multi-recipe-model-needs.yaml")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read generic multi-recipe fixture: %v", err)
	}
	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("parse generic multi-recipe fixture: %v", err)
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
