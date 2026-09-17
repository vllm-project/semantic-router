//go:build !windows && cgo && (amd64 || arm64)

package modelruntime

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

func TestEmbeddingAPIAndCacheKeepSeparateConsumersOnSharedORTModel(t *testing.T) {
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("requires real ONNX Runtime")
	}
	artifact, err := filepath.Abs("../../../../onnx-binding/instance/testdata/embedding")
	if err != nil {
		t.Fatal(err)
	}
	cfg := &config.RouterConfig{}
	cfg.API.Embeddings.Enabled = true
	cfg.EmbeddingConfig = config.HNSWConfig{ModelType: "mmbert", TargetDimension: 3, TargetLayer: 1}
	cfg.SemanticCache.Enabled = true
	cfg.SemanticCache.BackendType, cfg.SemanticCache.EmbeddingModel = "redis", "mmbert"
	cfg.ModelDeployments = map[string]config.ModelDeployment{"global": {Artifact: artifact, Provider: "ort", Device: "cpu", Precision: "native", Input: config.ModelInputBudget{MaxTokens: 4096, Overflow: "truncate"}}}
	cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "global", Adapter: "mmbert", Contract: "embedding.v1", Head: "model.onnx"}}
	runtime := native.New(nil)
	cache, err := PrepareOwnedResponseCacheEmbeddings(context.Background(), cfg, runtime)
	if err != nil {
		t.Fatal(err)
	}
	defer cache.Close()
	bindings := runtime.PreparedBindings()
	if len(bindings) != 1 || bindings[0].Identity.Name != "response_cache.embedding" || !cfg.API.Embeddings.Enabled {
		t.Fatalf("cache inherited API demand or changed source configuration: %+v", bindings)
	}
	api, err := PrepareOwnedEmbeddingAPI(context.Background(), cfg, runtime)
	if err != nil {
		t.Fatal(err)
	}
	defer api.Close()
	bindings = runtime.PreparedBindings()
	if len(bindings) != 2 || bindings[0].ResourceID != bindings[1].ResourceID {
		t.Fatalf("separate API/cache handles did not share model: %+v", bindings)
	}
	if closeErr := api.Close(); closeErr != nil {
		t.Fatal(closeErr)
	}
	provider, err := cache.Default()
	if err != nil {
		t.Fatal(err)
	}
	if vector, callErr := provider.Embed(context.Background(), "hello"); callErr != nil || len(vector) != 3 {
		t.Fatalf("API close retired cache model: %v / %v", vector, callErr)
	}
}
