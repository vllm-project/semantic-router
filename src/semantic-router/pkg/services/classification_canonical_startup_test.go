//go:build !windows && cgo && (amd64 || arm64)

package services

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func TestStandaloneCanonicalStartupAndFirstReloadShareOwnedPool(t *testing.T) {
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("requires real ONNX Runtime")
	}
	artifact, err := filepath.Abs("../../../../onnx-binding/instance/testdata/embedding")
	if err != nil {
		t.Fatal(err)
	}
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig = config.HNSWConfig{ModelType: "mmbert", TargetDimension: 3, TargetLayer: 1}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"global": {Artifact: artifact, Provider: "ort", Device: "cpu", Precision: "native", Input: config.ModelInputBudget{MaxTokens: 4096, Overflow: "truncate"}}}
	cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "global", Adapter: "mmbert", Contract: "embedding.v1", Head: "model.onnx"}}
	cfg.EmbeddingRules = []config.EmbeddingRule{{Name: "topic", Candidates: []string{"hello"}}}
	service, err := NewClassificationServiceFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = service.Close() })
	pool := service.modelPool
	if pool == nil || service.runtimeOwner == nil || service.recipeClassifiers == nil {
		t.Fatal("first API generation did not own canonical runtime")
	}
	before, ok := service.PreparedBindings()
	if !ok || len(before) != 1 || before[0].ResourceID == "" {
		t.Fatalf("initial inventory: %+v", before)
	}
	oldProvider, err := service.classifier.EmbeddingForModel("mmbert", 3, 1)
	if err != nil {
		t.Fatal(err)
	}
	if refreshErr := service.TryRefreshRuntimeConfig(cfg); refreshErr != nil {
		t.Fatal(refreshErr)
	}
	after, ok := service.PreparedBindings()
	if !ok || len(after) != 1 || after[0].ResourceID != before[0].ResourceID || pool != service.modelPool {
		t.Fatalf("first reload replaced pool or execution: %+v", after)
	}
	if _, callErr := oldProvider.Embed(context.Background(), "hello"); !errors.Is(callErr, binding.ErrClosed) {
		t.Fatalf("old generation retained handle: %v", callErr)
	}
	provider, err := service.classifier.EmbeddingForModel("mmbert", 3, 1)
	if err != nil {
		t.Fatal(err)
	}
	if vector, callErr := provider.Embed(context.Background(), "hello"); callErr != nil || len(vector) != 3 {
		t.Fatalf("retiring first generation unloaded replacement: %v / %v", vector, callErr)
	}
	if closeErr := service.Close(); closeErr != nil {
		t.Fatal(closeErr)
	}
	if _, callErr := provider.Embed(context.Background(), "hello"); !errors.Is(callErr, binding.ErrClosed) {
		t.Fatalf("standalone close leaked model: %v", callErr)
	}
}

func TestStandaloneCanonicalStartupRejectsInvalidBindings(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.MoMRegistry = map[string]string{"models/lora_model": "unrelated/discoverable-model"}
	cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "missing", Adapter: "mmbert", Contract: "embedding.v1"}}
	service, err := NewClassificationServiceFromConfig(cfg)
	if err == nil || service != nil {
		if service != nil {
			_ = service.Close()
		}
		t.Fatal("invalid canonical config fell back to an autodiscovered/placeholder service")
	}
	idle, err := NewClassificationServiceWithAutoDiscovery(&config.RouterConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer idle.Close()
	if bindings, ok := idle.PreparedBindings(); !ok || len(bindings) != 0 {
		t.Fatalf("empty canonical config discovered models: %+v", bindings)
	}
}
