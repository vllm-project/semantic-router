package modeldownload

import (
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestGenericBindingDownloadsOnlySelectedReachableArtifact(t *testing.T) {
	for _, provider := range []string{"candle", "ort", "http"} {
		t.Run(provider, func(t *testing.T) {
			cfg := &config.RouterConfig{MoMRegistry: map[string]string{"models/obsolete": "test/old", "models/new": "test/new", "models/dormant": "test/dormant"}, ExternalModels: []config.ExternalModelConfig{{Name: "selected", ModelRole: config.ModelRoleClassification, ModelEndpoint: config.ClassifierVLLMEndpoint{Address: "localhost", Port: 8080}}}}
			deployment := config.ModelDeployment{Provider: provider, Artifact: "models/new", Revision: "pin"}
			adapter, head := "modernbert", ""
			if provider == "ort" {
				head = "onnx/selected.onnx"
			}
			if provider == "http" {
				deployment.Artifact, deployment.ExternalModel = "", "selected"
				adapter = "http_classify"
			}
			cfg.ModelDeployments = map[string]config.ModelDeployment{"selected": deployment}
			cfg.ClassifierRules = []config.ClassifierSignalRule{{Name: "risk.tenant", Type: "local", ModelPath: "models/obsolete", Labels: []string{"safe", "unsafe"}}}
			cfg.ModelBindings = map[string]config.ModelBinding{"classifier.risk.tenant": {Deployment: "selected", Contract: "label_distribution.v1", Adapter: adapter, Head: head}}
			cfg.Recipes = []config.RoutingRecipe{
				{Name: config.DefaultRecipeName, Profile: config.RoutingProfile{}},
				{Name: "active", Profile: config.RoutingProfile{Signals: cfg.Signals, ModelBindings: cfg.ModelBindings}},
				{Name: "dormant", Profile: config.RoutingProfile{Signals: config.Signals{ClassifierRules: []config.ClassifierSignalRule{{Name: "unused", Type: "local", ModelPath: "models/dormant", Labels: []string{"a", "b"}}}}}},
			}
			cfg.ModelBindings, cfg.ClassifierRules = nil, nil
			cfg.Entrypoints = []config.EntrypointMapping{{ModelNames: []string{"public"}, Recipe: "active"}}
			specs, err := BuildModelSpecs(cfg)
			if err != nil {
				t.Fatal(err)
			}
			if provider == "http" {
				if len(specs) != 0 {
					t.Fatalf("remote override downloaded obsolete artifacts: %#v", specs)
				}
				return
			}
			if len(specs) != 1 || specs[0].LocalPath != "models/new" || specs[0].Revision != "pin" {
				t.Fatalf("wrong inventory: %#v", specs)
			}
			if provider == "ort" && (!specs[0].CheckONNX || !slices.Contains(specs[0].RequiredFiles, head)) {
				t.Fatalf("ORT task graph omitted: %#v", specs[0])
			}
		})
	}
}
