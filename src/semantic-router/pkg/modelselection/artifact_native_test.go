//go:build !windows && cgo && (amd64 || arm64)

package modelselection

import (
	"io"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func loadArtifactSelector(t testing.TB, algorithm, dir string) Selector {
	t.Helper()
	selector, err := NewSelector(&config.MLModelSelectionConfig{Type: algorithm, ModelsPath: dir})
	if err != nil {
		t.Fatalf("load %s artifact from %s: %v", algorithm, dir, err)
	}
	t.Cleanup(func() {
		if err := selector.(io.Closer).Close(); err != nil {
			t.Errorf("close %s selector: %v", algorithm, err)
		}
	})
	return selector
}

func TestModelArtifactsLoadAndSelect(t *testing.T) {
	for _, algorithm := range []string{"knn", "kmeans", "svm", "mlp"} {
		t.Run(algorithm, func(t *testing.T) {
			selector := loadArtifactSelector(t, algorithm, "testdata")
			for _, tc := range []struct{ category, want string }{
				{"math", "model-a"}, {"other", "model-b"}, {"unrecognized-domain", "model-b"},
			} {
				ctx := &SelectionContext{QueryEmbedding: []float64{0.25, 0.25}, CategoryName: tc.category}
				got, err := selector.Select(ctx, testModels)
				if err != nil || got == nil || got.Model != tc.want {
					t.Fatalf("category %q selected %v, %v; want %s", tc.category, got, err, tc.want)
				}
				refs := []config.ModelRef{{Model: "foreign-a"}, {Model: "foreign-b"}}
				if got, err := selector.Select(ctx, refs); err == nil || got != nil {
					t.Fatalf("selection outside candidates must fail: %v, %v", got, err)
				}
			}
			if got, err := selector.Select(&SelectionContext{}, testModels); err == nil || got != nil {
				t.Fatalf("loaded artifact without embedding must fail: %v, %v", got, err)
			}
		})
	}
}

func TestModelArtifactFileLoad(t *testing.T) {
	for _, algorithm := range []string{"knn", "kmeans", "svm"} {
		t.Run(algorithm, func(t *testing.T) {
			selector, err := LoadPretrainedSelector(algorithm, filepath.Join("testdata", algorithm+"_model.json"))
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if closeErr := selector.(io.Closer).Close(); closeErr != nil {
					t.Errorf("close %s selector: %v", algorithm, closeErr)
				}
			})
			ctx := &SelectionContext{QueryEmbedding: []float64{0.25, 0.25}, CategoryName: "math"}
			got, err := selector.Select(ctx, testModels)
			if err != nil || got == nil || got.Model != "model-a" {
				t.Fatalf("file-loaded artifact selected %v, %v; want model-a", got, err)
			}
		})
	}
}

func TestPretrainedModels_LoadAndSelect(t *testing.T) {
	dir, err := pretrainedTestModelsDir()
	if err != nil {
		t.Fatal(err)
	}
	if dir == "" {
		t.Skip("set " + pretrainedTestModelsEnv + " to run published-artifact integration tests")
	}
	refs := []config.ModelRef{{Model: "llama-3.2-1b"}, {Model: "llama-3.2-3b"}, {Model: "codellama-7b"}, {Model: "mistral-7b"}}
	embedding := make([]float64, 1024)
	rng := rand.New(rand.NewSource(1))
	for i := range embedding {
		embedding[i] = rng.Float64()*2 - 1
	}
	for _, algorithm := range []string{"knn", "kmeans", "svm", "mlp"} {
		t.Run(algorithm, func(t *testing.T) {
			selector := loadArtifactSelector(t, algorithm, dir)
			got, err := selector.Select(&SelectionContext{QueryEmbedding: embedding, CategoryName: "math"}, refs)
			if err != nil || got == nil {
				t.Fatalf("published artifact selection: %v, %v", got, err)
			}
		})
	}
}

func BenchmarkArtifactSelection(b *testing.B) {
	for _, algorithm := range []string{"knn", "kmeans", "svm", "mlp"} {
		b.Run(algorithm, func(b *testing.B) {
			selector := loadArtifactSelector(b, algorithm, "testdata")
			ctx := &SelectionContext{QueryEmbedding: []float64{0.25, 0.25}, CategoryName: "math"}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := selector.Select(ctx, testModels); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
