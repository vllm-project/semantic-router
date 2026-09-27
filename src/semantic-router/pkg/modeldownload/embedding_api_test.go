package modeldownload

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestEmbeddingAPIDownloadsGlobalBindingWithoutRecipeDemand(t *testing.T) {
	cfg := &config.RouterConfig{MoMRegistry: map[string]string{"models/global": "test/global", "models/local": "test/local"}}
	cfg.EmbeddingConfig.ModelType = "mmbert"
	cfg.ModelDeployments = map[string]config.ModelDeployment{
		"global": {Artifact: "models/global", Provider: "ort", Device: "cpu", Precision: "native"},
		"local":  {Artifact: "models/local", Provider: "ort", Device: "cpu", Precision: "native"},
	}
	cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "global", Adapter: "mmbert", Contract: "embedding.v1", Head: "model.onnx"}}
	cfg.ModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "local", Adapter: "mmbert", Contract: "embedding.v1", Head: "model.onnx"}}
	for _, enabled := range []bool{false, true} {
		cfg.API.Embeddings.Enabled = enabled
		specs, err := BuildModelSpecs(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if enabled {
			if len(specs) != 1 || specs[0].LocalPath != "models/global" {
				t.Fatalf("API downloaded recipe model or missed global model: %+v", specs)
			}
		} else if len(specs) != 0 {
			t.Fatalf("disabled API downloaded a model: %+v", specs)
		}
	}
}
