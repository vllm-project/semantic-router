package modeldownload

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestGlobalServiceDownloadsIgnoreUnusedDefaultRecipeOverride(t *testing.T) {
	for _, consumer := range []string{"tools", "memory", "vector_store", "response_cache"} {
		t.Run(consumer, func(t *testing.T) {
			cfg := &config.RouterConfig{MoMRegistry: map[string]string{"models/global": "test/global", "models/local": "test/local"}}
			cfg.EmbeddingConfig.ModelType = "mmbert"
			cfg.ModelDeployments = map[string]config.ModelDeployment{
				"global": {Provider: "ort", Artifact: "models/global", Device: "rocm:0"},
				"local":  {Provider: "ort", Artifact: "models/local", Device: "cpu"},
			}
			cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "global", Adapter: "mmbert", Contract: "embedding.v1"}}
			cfg.ModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "local", Adapter: "mmbert", Contract: "embedding.v1"}}
			switch consumer {
			case "tools":
				cfg.Tools.Enabled = true
			case "memory":
				cfg.Memory.Enabled, cfg.Memory.EmbeddingModel = true, "mmbert"
			case "vector_store":
				cfg.VectorStore = &config.VectorStoreConfig{Enabled: true, EmbeddingModel: "mmbert"}
			case "response_cache":
				cfg.SemanticCache.Enabled, cfg.SemanticCache.EmbeddingModel = true, "mmbert"
			}
			specs, err := BuildModelSpecs(cfg)
			if err != nil {
				t.Fatal(err)
			}
			if len(specs) != 1 || specs[0].LocalPath != "models/global" {
				t.Fatalf("service downloaded recipe override: %+v", specs)
			}
			cfg.EmbeddingRules = []config.EmbeddingRule{{Name: "local-use", Candidates: []string{"hello"}}}
			specs, err = BuildModelSpecs(cfg)
			if err != nil || len(specs) != 2 {
				t.Fatalf("actual recipe demand was lost: %+v / %v", specs, err)
			}
		})
	}
}
