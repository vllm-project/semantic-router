package modelruntime

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestOwnedRecipeEmbeddingDoesNotProvisionSharedCache(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.SemanticCache.InUse = true
	cfg.GlobalPlugins = []config.DecisionPlugin{{
		Type:          config.DecisionPluginResponseCache,
		Configuration: config.MustStructuredPayload(map[string]any{"enabled": true}),
	}}
	if err := cfg.LayerPlugins(); err != nil {
		t.Fatal(err)
	}
	cfg.EmbeddingDeployment = "shared-cache"
	cfg.ModelDeployments = map[string]config.ModelDeployment{"shared-cache": {Provider: config.ModelRuntimeProvider, Artifact: "/does-not-exist/shared-cache"}}
	prepared, err := PrepareOwnedEmbeddings(context.Background(), cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer prepared.Close()
	if prepared.Ready() {
		t.Fatal("standalone recipe provisioned shared cache model")
	}
	if _, err := PrepareOwnedResponseCacheEmbeddings(context.Background(), cfg, nil); err == nil {
		t.Fatal("service runtime ignored required cache artifact")
	}
}
