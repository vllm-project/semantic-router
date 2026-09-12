//go:build !windows && cgo && (amd64 || arm64)

package native

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

func TestORTEmbeddingPreparesEveryAdvertisedLayerBeforePublication(t *testing.T) {
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("requires the real ONNX Runtime library")
	}
	ctx := context.Background()
	runtime := New(nil)
	spec := embeddingPreparationFixture(t, false)
	prepared, err := runtime.Embedding(ctx, spec, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer prepared.Close()
	if !reflect.DeepEqual(prepared.info.Layers, []int{1, 2}) {
		t.Fatalf("prepared capabilities do not match loaded graphs: %+v", prepared.info)
	}
	if err := prepared.resource.Use(ctx, func(value io.Closer) error {
		info, err := value.(*embeddingEngine).ort.Info()
		if err != nil {
			return err
		}
		if info.CompletedInferences != 2 || len(info.Sessions) != 2 {
			t.Fatalf("ready before each distinct exit executed: %+v", info)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	// The candidate's selected layer works; a different advertised graph has
	// the wrong task output. Preparation must reject it before publication.
	if candidate, err := runtime.Embedding(ctx, embeddingPreparationFixture(t, true), 3, 2); err == nil {
		_ = candidate.Close()
		t.Fatal("candidate published without validating its other advertised layer")
	}
	for _, layer := range []int{1, 2} {
		vector, err := prepared.EmbedWithOptions(ctx, "hello world", embedding.Options{Dimension: 3, Layer: layer})
		if err != nil || len(vector) != 3 {
			t.Fatalf("failed candidate disturbed the previous layer %d: %v", layer, err)
		}
	}
	if _, err := prepared.EmbedWithOptions(ctx, "hello", embedding.Options{Dimension: 3, Layer: 3}); err == nil {
		t.Fatal("unknown layer silently used a different graph")
	}
}

func embeddingPreparationFixture(t *testing.T, invalidExit bool) config.ResolvedModelBinding {
	t.Helper()
	directory := t.TempDir()
	fixtureRoot := filepath.Join("..", "..", "..", "..", "..", "onnx-binding", "instance", "testdata")
	copyFile := func(source, target string) {
		t.Helper()
		data, err := os.ReadFile(source)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(target, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	for _, name := range []string{"config.json", "tokenizer.json"} {
		copyFile(filepath.Join(fixtureRoot, "embedding", name), filepath.Join(directory, name))
	}
	for _, layer := range []string{"layer-1", "layer-2"} {
		kind := "embedding"
		if invalidExit && layer == "layer-1" {
			kind = "sequence"
		}
		copyFile(filepath.Join(fixtureRoot, kind, "model.onnx"), filepath.Join(directory, "onnx", layer, "model.onnx"))
	}
	if err := os.WriteFile(filepath.Join(directory, "onnx", "model_config.json"), []byte(`{"available_layers":[1,2]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	return config.ResolvedModelBinding{
		Recipe: "embedding-preparation", Name: "embedding",
		Binding:    config.ModelBinding{Deployment: "layers", Contract: "embedding.v1", Adapter: "mmbert", Head: "onnx/layer-2/model.onnx"},
		Deployment: config.ModelDeployment{Artifact: directory, Provider: "ort", Device: "cpu", Precision: "native", Input: config.ModelInputBudget{MaxTokens: 8, Overflow: "reject"}},
	}
}
