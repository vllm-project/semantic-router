package modeldownload

import (
	"reflect"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const modalityInventoryArtifact = "models/modality"

func modalityInventoryConfig() *config.RouterConfig {
	cfg := &config.RouterConfig{MoMRegistry: map[string]string{modalityInventoryArtifact: "test/modality"}}
	cfg.ModalityDetector = config.ModalityDetectorConfig{
		Enabled: true,
		ModalityDetectionConfig: config.ModalityDetectionConfig{
			Method:     config.ModalityDetectionClassifier,
			Classifier: &config.ModalityClassifierConfig{ModelPath: modalityInventoryArtifact, UseCPU: true},
		},
	}
	cfg.ModalityRules = []config.ModalityRule{{Name: "AR"}}
	cfg.ModelDeployments = map[string]config.ModelDeployment{
		"native": {Provider: "candle", Artifact: modalityInventoryArtifact, Revision: "main"},
	}
	cfg.ModelBindings = map[string]config.ModelBinding{
		"modality_detector": {Deployment: "native", Contract: "label_distribution.v1", Adapter: "mmbert32k"},
	}
	return cfg
}

func TestModalityInventoryIgnoresUnrelatedRecipe(t *testing.T) {
	cfg := modalityInventoryConfig()
	baseline, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(baseline) != 1 || baseline[0].CheckONNX ||
		!slices.Contains(baseline[0].ExcludePatterns, "*.onnx") || len(baseline[0].RequiredFileGroups) != 1 {
		t.Fatalf("expected native-only modality artifact: %#v", baseline)
	}
	cfg.Recipes = []config.RoutingRecipe{
		{Name: config.DefaultRecipeName, Profile: config.RoutingProfile{Signals: cfg.Signals, ModelBindings: cfg.ModelBindings}},
		{Name: "unrelated"},
	}
	cfg.Entrypoints = []config.EntrypointMapping{{ModelNames: []string{"unrelated"}, Recipe: "unrelated"}}
	actual, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(actual, baseline) {
		t.Fatalf("unrelated recipe changed modality artifact requirements:\nbefore=%#v\nafter=%#v", baseline, actual)
	}
	if cfg.ModalityDetector.Classifier.ModelPath != modalityInventoryArtifact || len(cfg.ModalityRules) != 1 {
		t.Fatal("source modality configuration mutated")
	}
}

func TestModalityInventoryRequiresModelConsumer(t *testing.T) {
	for _, method := range []string{config.ModalityDetectionClassifier, config.ModalityDetectionHybrid, config.ModalityDetectionKeyword} {
		for _, hasRules := range []bool{false, true} {
			name := method + "/without_rules"
			if hasRules {
				name = method + "/with_rules"
			}
			t.Run(name, func(t *testing.T) {
				cfg := modalityInventoryConfig()
				cfg.ModalityDetector.Method = method
				if !hasRules {
					cfg.ModalityRules = nil
				}
				specs, err := BuildModelSpecs(cfg)
				if err != nil {
					t.Fatal(err)
				}
				want := 0
				if hasRules && method != config.ModalityDetectionKeyword {
					want = 1
				}
				if len(specs) != want {
					t.Fatalf("expected %d model artifacts, got %#v", want, specs)
				}
			})
		}
	}
}

func TestModalityInventoryKeepsActualORTRecipeRequirements(t *testing.T) {
	cfg := modalityInventoryConfig()
	cfg.ModelDeployments["onnx"] = config.ModelDeployment{Provider: "ort", Artifact: modalityInventoryArtifact, Revision: "main"}
	cfg.Recipes = []config.RoutingRecipe{
		{Name: config.DefaultRecipeName, Profile: config.RoutingProfile{Signals: cfg.Signals, ModelBindings: cfg.ModelBindings}},
		{Name: "image-route", Profile: config.RoutingProfile{
			Signals: config.Signals{ModalityRules: []config.ModalityRule{{Name: "DIFFUSION"}}},
			ModelBindings: map[string]config.ModelBinding{
				"modality_detector": {Deployment: "onnx", Contract: "label_distribution.v1", Adapter: "mmbert32k"},
			},
		}},
	}
	cfg.Entrypoints = []config.EntrypointMapping{{ModelNames: []string{"image-route"}, Recipe: "image-route"}}
	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(specs) != 1 || !specs[0].CheckONNX || len(specs[0].ExcludePatterns) != 0 || len(specs[0].RequiredFileGroups) != 2 {
		t.Fatalf("real ORT consumer must retain native and ONNX requirements: %#v", specs)
	}
}
