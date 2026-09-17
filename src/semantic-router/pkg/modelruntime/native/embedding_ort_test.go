//go:build !windows && cgo && (amd64 || arm64)

package native

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	ort "github.com/vllm-project/semantic-router/onnx-binding/instance"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func TestORTEmbeddingPoolPrecisionPreservesRepresentationMath(t *testing.T) {
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("requires the real ONNX Runtime library")
	}
	for _, adapter := range []string{"mmbert", "multimodal"} {
		t.Run(adapter, func(t *testing.T) {
			ctx := context.Background()
			runtime := New(nil)
			spec := embeddingPreparationFixture(t, false)
			view := embedding.Options{Layer: 2, Dimension: 3}
			if adapter == "multimodal" {
				spec.Binding.Adapter, spec.Binding.Head = adapter, ""
				spec.Deployment.Artifact = filepath.Join("..", "..", "..", "..", "..", "onnx-binding", "instance", "testdata", adapter)
				view.Layer = 0
			}
			prepared, err := runtime.ortEmbedding(ctx, spec, view)
			if err != nil {
				t.Fatal(err)
			}
			defer prepared.resource.Close()
			if prepared.identity.Precision != "native" || prepared.capability.Precision != "native" {
				t.Fatalf("ORT resource inherited Candle precision: %+v", prepared.identity)
			}
			// The previous projection borrowed Candle's float32 spelling. Only
			// the physical key changes; load the same graph with the same options.
			previousID := prepared.identity
			previousID.Precision = "float32"
			options, err := ortOptions(spec)
			if err != nil {
				t.Fatal(err)
			}
			budget, gate := resourceAdmission(spec)
			previous, err := runtime.Pool.Acquire(ctx, previousID, budget, gate, func(context.Context) (io.Closer, error) {
				if adapter == "multimodal" {
					model, loadErr := ort.LoadMultiModal(options)
					if loadErr != nil {
						return nil, loadErr
					}
					return &embeddingEngine{multi: model}, nil
				}
				model, loadErr := ort.LoadEmbeddingModel(options)
				if loadErr != nil {
					return nil, loadErr
				}
				return &embeddingEngine{ort: model}, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			defer previous.Close()
			provider, err := runtime.Embedding(ctx, spec, view.Dimension, view.Layer)
			if err != nil {
				t.Fatal(err)
			}
			defer provider.Close()
			var currentOwner io.Closer
			if useErr := prepared.resource.Use(ctx, func(value io.Closer) error { currentOwner = value; return nil }); useErr != nil {
				t.Fatal(useErr)
			}
			var reference tasks.EmbeddingResult
			if useErr := previous.Use(ctx, func(value io.Closer) error {
				if value == currentOwner {
					t.Fatal("different physical precision identities shared an owner")
				}
				var inferErr error
				reference, inferErr = value.(*embeddingEngine).embed("hello world", view)
				return inferErr
			}); useErr != nil {
				t.Fatal(useErr)
			}
			actual, err := provider.text.Call(ctx, string(spec.Recipe), embedding.TextRequest{Text: "hello world", Options: view})
			if err != nil || !reflect.DeepEqual(actual, reference) {
				t.Fatalf("resource key changed inference: %+v != %+v, %v", actual, reference, err)
			}
			previousKey, _ := previousID.Key()
			previousView := &EmbeddingProvider{resource: previous, identity: previousKey, options: view, contentIdentity: prepared.contentIdentity, executionPolicy: provider.executionPolicy, descriptors: make(map[embedding.Options]string)}
			sameCache := provider.CacheIdentity() == previousView.CacheIdentity()
			if sameCache != (adapter == "mmbert") {
				t.Fatal("text content identity must survive, while multimodal physical cache identity must change")
			}
			if closeErr := previous.Close(); closeErr != nil {
				t.Fatal(closeErr)
			}
			if _, callErr := provider.Embed(ctx, "still owned"); callErr != nil {
				t.Fatal("closing the prior physical resource affected the current owner", callErr)
			}
		})
	}
}
