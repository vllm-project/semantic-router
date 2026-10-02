//go:build !windows && cgo && (amd64 || arm64)

package services

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestEmbeddingAPIAndRecipeShareRealORTResource(t *testing.T) {
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
	cfg.ModelDeployments = map[string]config.ModelDeployment{"global": {Artifact: artifact, Provider: "ort", Device: "cpu", Precision: "native", Input: config.ModelInputBudget{MaxTokens: 4096, Overflow: "truncate"}}}
	cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "global", Adapter: "mmbert", Contract: "embedding.v1", Head: "model.onnx"}}
	cfg.EmbeddingRules = []config.EmbeddingRule{{Name: "topic", Candidates: []string{"hello"}}}
	service, err := NewClassificationServiceFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer service.Close()
	bindings, _ := service.PreparedBindings()
	if len(bindings) != 2 || bindings[0].ResourceID == "" || bindings[0].ResourceID != bindings[1].ResourceID {
		t.Fatalf("API and recipe duplicated physical execution: %+v", bindings)
	}
	_, prepared, release, err := service.AcquireEmbeddingAPISnapshot()
	defer release()
	if err != nil {
		t.Fatal(err)
	}
	provider, err := prepared.Default()
	if err != nil {
		release()
		t.Fatal(err)
	}
	longText := strings.Repeat("hello ", 32769)
	if selected, selectErr := prepared.Select(longText, 0.5, 0.5, 3); selectErr != nil || selected != "mmbert" {
		release()
		t.Fatalf("auto selection bypassed configured truncation policy: %q / %v", selected, selectErr)
	}
	vector, err := provider.Embed(context.Background(), longText)
	release()
	if err != nil || len(vector) != 3 {
		t.Fatalf("real API inference failed: %v / %v", vector, err)
	}
	disabled := *cfg
	disabled.API.Embeddings.Enabled = false
	if refreshErr := service.TryRefreshRuntimeConfig(&disabled); refreshErr != nil {
		t.Fatal(refreshErr)
	}
	after, _ := service.PreparedBindings()
	if len(after) != 1 || after[0].ResourceID != bindings[0].ResourceID {
		t.Fatalf("disabling API replaced recipe resource: %+v", after)
	}
	_, recipe, release, err := service.AcquireEmbeddingAPISnapshot()
	defer release()
	if err != nil {
		t.Fatal(err)
	}
	provider, err = recipe.Default()
	if err != nil {
		t.Fatal(err)
	}
	if vector, callErr := provider.Embed(context.Background(), "hello"); callErr != nil || len(vector) != 3 {
		t.Fatalf("API retirement broke recipe inference: %v / %v", vector, callErr)
	}
}
