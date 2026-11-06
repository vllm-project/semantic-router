package modelruntime

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestOwnedRecipeEmbeddingDoesNotProvisionSharedCache(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.SemanticCache.Enabled = true
	cfg.SemanticCache.EmbeddingModel = "mmbert"
	cfg.MmBertModelPath = "/does-not-exist/shared-cache"
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
