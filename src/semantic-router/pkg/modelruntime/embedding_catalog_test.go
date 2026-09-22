package modelruntime

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestEmbeddingCatalogUsesPublishedAdapterAndExecution(t *testing.T) {
	cfg := config.DefaultGlobalConfig()
	for _, alias := range []string{"omni-nano", "omni-mini"} {
		path := config.ResolveModelPath(alias)
		spec := embeddingCatalogSpec(&cfg, config.DefaultRecipeName, "multimodal", path)
		if spec.Binding.Adapter != "vela_omni" || spec.Deployment.Provider != "ort" || spec.Deployment.Device != "cpu" || spec.Deployment.Input.Overflow != "reject" {
			t.Fatalf("catalog contract ignored for %s: %+v", alias, spec)
		}
	}
	legacy := embeddingCatalogSpec(&cfg, config.DefaultRecipeName, "multimodal", "models/mom-embedding-multimodal")
	if legacy.Binding.Adapter != "multimodal" {
		t.Fatal("explicit legacy adapter silently migrated")
	}
	text := embeddingCatalogSpec(&cfg, config.DefaultRecipeName, "mmbert", cfg.MmBertModelPath)
	if text.Binding.Adapter != "mmbert" {
		t.Fatal("pure-text adapter silently migrated")
	}
}
