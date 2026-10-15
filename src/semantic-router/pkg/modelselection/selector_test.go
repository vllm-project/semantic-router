/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package modelselection

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var testModels = []config.ModelRef{
	{Model: "model-a"},
	{Model: "model-b"},
}

var algorithms = []string{"knn", "kmeans", "svm", "mlp"}

func TestNewSelector(t *testing.T) {
	for _, algorithm := range algorithms {
		selector, err := NewSelector(&config.MLModelSelectionConfig{Type: algorithm})
		if err != nil || selector.Name() != algorithm {
			t.Fatalf("NewSelector(%s) = %v, %v", algorithm, selector, err)
		}
	}
	for _, cfg := range []*config.MLModelSelectionConfig{nil, {Type: ""}, {Type: "unknown"}} {
		if _, err := NewSelector(cfg); err == nil {
			t.Fatalf("NewSelector(%+v) must fail", cfg)
		}
	}
}

func TestSelectorWithoutArtifact(t *testing.T) {
	for _, algorithm := range algorithms {
		t.Run(algorithm, func(t *testing.T) {
			selector, err := NewSelector(&config.MLModelSelectionConfig{Type: algorithm})
			if err != nil {
				t.Fatal(err)
			}
			if got, err := selector.Select(&SelectionContext{}, nil); err == nil || got != nil {
				t.Fatalf("empty refs: %v, %v", got, err)
			}
			single := []config.ModelRef{{Model: "only-model"}}
			if got, err := selector.Select(&SelectionContext{}, single); err != nil || got.Model != "only-model" {
				t.Fatalf("single candidate: %v, %v", got, err)
			}
			ctx := &SelectionContext{QueryEmbedding: []float64{1, 0}, QueryText: "What is 2+2?"}
			if _, err := selector.Select(ctx, testModels); err == nil || !strings.Contains(err.Error(), "not trained") {
				t.Fatalf("multi-model selection without an artifact must fail: %v", err)
			}
		})
	}
}

func TestArtifactsLoadAndSelect(t *testing.T) {
	for _, algorithm := range algorithms {
		t.Run(algorithm, func(t *testing.T) {
			selector, err := NewSelector(&config.MLModelSelectionConfig{Type: algorithm, ModelsPath: "testdata"})
			if err != nil {
				t.Fatal(err)
			}
			for _, tc := range []struct{ category, want string }{
				{"math", "model-a"}, {"other", "model-b"}, {"unrecognized-domain", "model-b"},
			} {
				ctx := &SelectionContext{QueryEmbedding: []float64{0.25, 0.25}, CategoryName: tc.category}
				got, err := selector.Select(ctx, testModels)
				if err != nil || got == nil || got.Model != tc.want {
					t.Fatalf("category %q selected %v, %v; want %s", tc.category, got, err, tc.want)
				}
				foreign := []config.ModelRef{{Model: "foreign-a"}, {Model: "foreign-b"}}
				if got, err := selector.Select(ctx, foreign); err == nil || got != nil {
					t.Fatalf("selection outside candidates must fail: %v, %v", got, err)
				}
			}
			if got, err := selector.Select(&SelectionContext{}, testModels); err == nil || got != nil {
				t.Fatalf("loaded artifact without embedding must fail: %v, %v", got, err)
			}
			wrongDimension := &SelectionContext{QueryEmbedding: []float64{0.25}, CategoryName: "math"}
			if _, err := selector.Select(wrongDimension, testModels); err == nil {
				t.Fatal("a query of the wrong dimension must fail")
			}
		})
	}
}

func TestSelectMatchesLoRAName(t *testing.T) {
	selector, err := NewSelector(&config.MLModelSelectionConfig{Type: "knn", ModelsPath: "testdata"})
	if err != nil {
		t.Fatal(err)
	}
	refs := []config.ModelRef{{Model: "base", LoRAName: "model-a"}, {Model: "model-b"}}
	ctx := &SelectionContext{QueryEmbedding: []float64{0.25, 0.25}, CategoryName: "math"}
	if got, err := selector.Select(ctx, refs); err != nil || got.LoRAName != "model-a" {
		t.Fatalf("LoRA candidate: %v, %v", got, err)
	}
}

func TestLoadRejectsInvalidArtifacts(t *testing.T) {
	for _, tc := range []struct{ algorithm, artifact string }{
		{"knn", `not json`},
		{"knn", `{"algorithm":"kmeans","trained":true,"k":1,"embeddings":[[1]],"labels":["a"]}`},
		{"knn", `{"algorithm":"knn","trained":false,"k":1,"embeddings":[[1]],"labels":["a"]}`},
		{"knn", `{"algorithm":"knn","trained":true,"k":1,"embeddings":[[1],[1,2]],"labels":["a","b"]}`},
		{"knn", `{"algorithm":"knn","format_version":3,"trained":true,"k":1,"embeddings":[[1]],"labels":["a"]}`},
		{"kmeans", `{"algorithm":"kmeans","trained":true,"centroids":[[1,0]],"cluster_models":[]}`},
		{"svm", `{"algorithm":"svm","trained":true,"model_names":["a","a"],"kernel_type":"Linear","gamma":1}`},
		{"svm", `{"algorithm":"svm","trained":true,"model_names":["a"],"kernel_type":"Poly","gamma":1}`},
		{"svm", `{"algorithm":"svm","format_version":2,"trained":true,"model_names":["a","b"],"kernel_type":"Rbf","gamma":1}`},
		{"svm", `{"algorithm":"svm","trained":true,"feature_dim":1,"model_names":["a","b"],"kernel_type":"Rbf","gamma":1,
			"svc":{"support_vectors":[[1]],"dual_coef":[[1]],"intercept":[0],"n_support":[2,0]}}`},
		{"mlp", `{"algorithm":"mlp","trained":true,"model_names":["a"],"feature_dim":2,
			"layers":[{"type":"linear","in_features":3,"out_features":1,"weight":[[1,1,1]]}]}`},
		{"mlp", `{"algorithm":"mlp","trained":true,"model_names":["a"],"feature_dim":2,"layers":[{"type":"conv"}]}`},
	} {
		selector, err := NewSelector(&config.MLModelSelectionConfig{Type: tc.algorithm})
		if err != nil {
			t.Fatal(err)
		}
		loader := selector.(interface{ LoadFromJSON([]byte) error })
		if err := loader.LoadFromJSON([]byte(tc.artifact)); err == nil {
			t.Errorf("%s artifact %s must be rejected", tc.algorithm, tc.artifact)
		}
	}
}

func TestNewSelectorFailsOnMissingArtifact(t *testing.T) {
	dir := t.TempDir()
	if _, err := NewSelector(&config.MLModelSelectionConfig{Type: "knn", ModelsPath: dir}); err == nil {
		t.Fatal("a missing artifact file must fail")
	}
	if err := os.WriteFile(filepath.Join(dir, "knn_model.json"), []byte(`{"algorithm":"knn"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := NewSelector(&config.MLModelSelectionConfig{Type: "knn", ModelsPath: dir}); err == nil {
		t.Fatal("an invalid artifact file must fail")
	}
}
