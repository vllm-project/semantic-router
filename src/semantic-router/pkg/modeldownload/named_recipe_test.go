package modeldownload

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	testFactCheckModel              = "models/test-fact-check"
	testFeedbackModel               = "models/test-feedback"
	testHallucinationDetectorModel  = "models/test-hallucination-detector"
	testHallucinationExplainerModel = "models/test-hallucination-explainer"
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
		testHallucinationExplainerModel,
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
	cfg.HallucinationMitigation.HallucinationModel.Backend = config.HallucinationBackendEndpoint

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}

	assertExactModelSpecs(t, specs, []string{
		testFactCheckModel,
		testFeedbackModel,
	})
}

func TestBuildModelSpecsSkipsUnusedHallucinationExplainer(t *testing.T) {
	cfg := loadGenericMultiRecipeModelNeedsConfig(t)
	recipe, ok := cfg.RecipeByName("verification")
	if !ok || len(recipe.Profile.Decisions) != 1 {
		t.Fatal("verification recipe is unavailable")
	}
	recipe.Profile.Decisions[0].Plugins[0].Configuration = config.MustStructuredPayload(map[string]interface{}{
		"enabled": true,
		"use_nli": false,
	})

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

func TestBuildModelSpecsPreservesDefaultAPIOnlyModels(t *testing.T) {
	cfg := &config.RouterConfig{
		MoMRegistry: map[string]string{
			testFactCheckModel:              "test/fact-check",
			testFeedbackModel:               "test/feedback",
			testHallucinationDetectorModel:  "test/hallucination-detector",
			testHallucinationExplainerModel: "test/hallucination-explainer",
		},
		InlineModels: config.InlineModels{
			HallucinationMitigation: config.HallucinationMitigationConfig{
				Enabled:            true,
				FactCheckModel:     config.FactCheckModelConfig{ModelID: testFactCheckModel},
				HallucinationModel: config.HallucinationModelConfig{ModelID: testHallucinationDetectorModel},
				NLIModel:           config.NLIModelConfig{ModelID: testHallucinationExplainerModel},
			},
			FeedbackDetector: config.FeedbackDetectorConfig{
				Enabled: true,
				ModelID: testFeedbackModel,
			},
		},
		Recipes: []config.RoutingRecipe{{
			Name: config.DefaultRecipeName,
			Profile: config.RoutingProfile{Signals: config.Signals{
				FactCheckRules:    []config.FactCheckRule{{Name: "verification-needed"}},
				UserFeedbackRules: []config.UserFeedbackRule{{Name: "correction-needed"}},
			}},
		}},
	}
	cfg.EmbeddingConfig.Backend = config.EmbeddingBackendOpenAICompatible
	if _, reachable := cfg.RecipeForRequestModel("vllm-sr/default"); reachable {
		t.Fatal("an explicit default Recipe became request reachable without an Entrypoint")
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	assertExactModelSpecs(t, specs, []string{
		testFactCheckModel,
		testFeedbackModel,
		testHallucinationDetectorModel,
		testHallucinationExplainerModel,
	})
}

func TestBuildModelSpecsSkipsUnreachableNamedRecipeModels(t *testing.T) {
	const unreachableModel = "models/unreachable-router"
	const authoring = `
version: v0.3
providers:
  models:
    - name: backend
      provider_model_id: backend
      backend_refs: [{provider: vllm, endpoint: http://127.0.0.1:8000}]
routing:
  modelCards:
    - name: backend
recipes:
  - name: default
    routing:
      signals: {}
      decisions:
        - name: default-route
          rules:
            operator: AND
            conditions: []
  - name: unmapped
    routing:
      signals:
        classifiers:
          - name: reachability
            type: local
            model_path: models/unreachable-router
            labels: [NO, YES]
            use_cpu: true
      decisions:
        - name: unmapped-route
          rules:
            operator: AND
            conditions:
              - type: classifier
                name: reachability
                label: YES
                predicate: {gte: 0.5}
`
	parse := func(suffix string) *config.RouterConfig {
		t.Helper()
		cfg, err := modelDownloadAuthoringParser(t).ParseYAMLBytes([]byte(authoring + suffix))
		if err != nil {
			t.Fatalf("parse reachability fixture: %v", err)
		}
		cfg.EmbeddingConfig.Backend = config.EmbeddingBackendOpenAICompatible
		cfg.MoMRegistry = map[string]string{unreachableModel: "test/unreachable-router"}
		return cfg
	}
	cfg := parse(`
entrypoints:
  - model_names: [vllm-sr/default]
    recipe: default
    assignments:
      default-route: {models: [{model: backend}]}
`)

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	if len(specs) != 0 {
		t.Fatalf("unreachable named recipe produced model specs: %#v", specs)
	}

	cfg = parse(`
entrypoints:
  - model_names: [vllm-sr/unmapped]
    recipe: unmapped
    assignments:
      unmapped-route: {models: [{model: backend}]}
`)
	specs, err = BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() with entrypoint error = %v", err)
	}
	assertExactModelSpecs(t, specs, []string{unreachableModel})
}

func TestBuildModelSpecsAccountsForExplicitDefaultEntrypointReachability(t *testing.T) {
	const defaultModel = "models/default-router"
	cfg := &config.RouterConfig{
		MoMRegistry: map[string]string{
			defaultModel: "test/default-router",
		},
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"backend": {ResourceID: "mdl-backend", ResourceRevision: 1},
		}},
		Recipes: []config.RoutingRecipe{{
			ID: "rcp-default", Revision: 1, Name: config.DefaultRecipeName,
			Profile: config.RoutingProfile{Decisions: []config.Decision{{
				ID: "dec-default", Name: "default-route",
				Algorithm: &config.AlgorithmConfig{
					GMTRouter: &config.GMTRouterSelectionConfig{ModelPath: defaultModel},
				},
			}}},
		}},
		Entrypoints: []config.EntrypointMapping{{
			ID: "ep-default", Revision: 1, Name: "default", ModelNames: []string{"router/default"},
			Rules: []config.EntrypointRule{{
				ID: "rule-default", Name: "default",
				Action: config.EntrypointRuleAction{
					RecipeID: "rcp-default", RecipeRevision: 1, Recipe: config.DefaultRecipeName,
					Assignments: map[string]config.RoutingAssignmentSet{
						"dec-default": {Models: []config.RoutingModelAssignment{{
							ModelID: "mdl-backend", ModelRevision: 1, ModelName: "backend", Weight: "1",
						}}},
					},
				},
			}},
		}},
	}
	cfg.EmbeddingConfig.Backend = config.EmbeddingBackendOpenAICompatible
	if err := cfg.PrepareEntrypointRecipes(); err != nil {
		t.Fatalf("PrepareEntrypointRecipes() error = %v", err)
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	assertExactModelSpecs(t, specs, []string{defaultModel})

	cfg.Entrypoints = nil
	specs, err = BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() without Entrypoint error = %v", err)
	}
	if len(specs) != 0 {
		t.Fatalf("unreachable default Recipe produced model specs: %#v", specs)
	}
}

func loadGenericMultiRecipeModelNeedsConfig(t *testing.T) *config.RouterConfig {
	t.Helper()
	path := filepath.Join("..", "config", "testdata", "generic-multi-recipe-model-needs.yaml")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read generic multi-recipe fixture: %v", err)
	}
	cfg, err := modelDownloadAuthoringParser(t).ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("parse generic multi-recipe fixture: %v", err)
	}

	cfg.EmbeddingConfig.Backend = config.EmbeddingBackendOpenAICompatible
	cfg.MoMRegistry = map[string]string{
		testFactCheckModel:              "test/fact-check",
		testFeedbackModel:               "test/feedback",
		testHallucinationDetectorModel:  "test/hallucination-detector",
		testHallucinationExplainerModel: "test/hallucination-explainer",
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
