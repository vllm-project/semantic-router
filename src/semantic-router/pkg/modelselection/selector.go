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

// Package modelselection chooses a candidate model with a trained KNN,
// KMeans, SVM or MLP artifact over the query embedding and its category.
//
// Training happens in Python (src/training/model_selection/ml_model_selection);
// this package loads the JSON artifacts and runs inference in pure Go with the
// numerics of the bindings it replaces (FusionFactory, arXiv:2507.10540, and
// Avengers-Pro, arXiv:2508.12631).
package modelselection

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Selector chooses the best model from refs based on the selection context.
type Selector interface {
	Select(ctx *SelectionContext, refs []config.ModelRef) (*config.ModelRef, error)
	Name() string
}

// SelectionContext contains information for model selection.
type SelectionContext struct {
	// QueryEmbedding is the embedding vector of the user query.
	QueryEmbedding []float64
	// QueryText is the raw user query text.
	QueryText string
	// CategoryName is the detected category/domain.
	CategoryName string
	// DecisionName is the matched decision name.
	DecisionName string
}

// classifier is a loaded, immutable artifact that names the selected model
// for a feature vector.
type classifier interface {
	classify(features []float64) (string, error)
}

// artifactSelector holds the artifact of one algorithm. Load swaps it
// atomically, so selections never block on a reload.
type artifactSelector struct {
	algorithm string
	parse     func([]byte) (classifier, error)
	model     atomic.Pointer[classifier]
}

// KNNSelector votes over the k nearest training queries, weighted by their
// recorded quality and latency.
type KNNSelector struct{ artifactSelector }

// KMeansSelector routes to the model assigned to the nearest cluster centroid.
type KMeansSelector struct{ artifactSelector }

// SVMSelector runs a trained support-vector classifier (linear or RBF kernel).
type SVMSelector struct{ artifactSelector }

// MLPSelector runs a trained multi-layer perceptron classifier.
type MLPSelector struct{ artifactSelector }

// NewKNNSelector returns a KNN selector without an artifact.
func NewKNNSelector() *KNNSelector {
	return &KNNSelector{artifactSelector{algorithm: "knn", parse: parseKNN}}
}

// NewKMeansSelector returns a KMeans selector without an artifact.
func NewKMeansSelector() *KMeansSelector {
	return &KMeansSelector{artifactSelector{algorithm: "kmeans", parse: parseKMeans}}
}

// NewSVMSelector returns an SVM selector without an artifact.
func NewSVMSelector() *SVMSelector {
	return &SVMSelector{artifactSelector{algorithm: "svm", parse: parseSVM}}
}

// NewMLPSelector returns an MLP selector without an artifact.
func NewMLPSelector() *MLPSelector {
	return &MLPSelector{artifactSelector{algorithm: "mlp", parse: parseMLP}}
}

// NewSelector creates the configured selector. With ModelsPath it loads
// <ModelsPath>/<type>_model.json; otherwise the selector fails every
// multi-candidate selection until an artifact is loaded.
func NewSelector(cfg *config.MLModelSelectionConfig) (Selector, error) {
	if cfg == nil {
		return nil, fmt.Errorf("model selection config is nil")
	}
	var selector interface {
		Selector
		Load(string) error
	}
	switch cfg.Type {
	case "knn":
		selector = NewKNNSelector()
	case "kmeans":
		selector = NewKMeansSelector()
	case "svm":
		selector = NewSVMSelector()
	case "mlp":
		selector = NewMLPSelector()
	default:
		return nil, fmt.Errorf("unknown model selection algorithm: %s (supported: knn, kmeans, svm, mlp)", cfg.Type)
	}
	if cfg.ModelsPath == "" {
		return selector, nil
	}
	path := filepath.Join(cfg.ModelsPath, cfg.Type+"_model.json")
	if err := selector.Load(path); err != nil {
		return nil, err
	}
	return selector, nil
}

// Name returns the algorithm name.
func (s *artifactSelector) Name() string { return s.algorithm }

// Load reads and activates an artifact file.
func (s *artifactSelector) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read %s artifact: %w", s.algorithm, err)
	}
	if err := s.LoadFromJSON(data); err != nil {
		return fmt.Errorf("%w (%s)", err, path)
	}
	logging.ComponentEvent("modelselection", "selector_loaded", map[string]interface{}{
		"algorithm": s.algorithm, "model_path": path,
	})
	return nil
}

// LoadFromJSON validates and activates an artifact.
func (s *artifactSelector) LoadFromJSON(data []byte) error {
	model, err := s.parse(data)
	if err != nil {
		return fmt.Errorf("invalid %s artifact: %w", s.algorithm, err)
	}
	s.model.Store(&model)
	return nil
}

// Select returns the candidate the artifact chooses. A single candidate is
// returned as is; otherwise the artifact and the query embedding are required.
func (s *artifactSelector) Select(ctx *SelectionContext, refs []config.ModelRef) (*config.ModelRef, error) {
	name := strings.ToUpper(s.algorithm)
	if len(refs) == 0 {
		return nil, fmt.Errorf("%s: no model refs provided", name)
	}
	if len(refs) == 1 {
		return &refs[0], nil
	}
	model := s.model.Load()
	if model == nil {
		return nil, fmt.Errorf("%s: model not trained - load pretrained model first", name)
	}
	if ctx == nil || len(ctx.QueryEmbedding) == 0 {
		return nil, fmt.Errorf("%s: no query embedding provided", name)
	}
	selected, err := (*model).classify(CombineEmbeddingWithCategory(ctx.QueryEmbedding, ctx.CategoryName))
	if err != nil {
		return nil, fmt.Errorf("%s selection failed: %w", name, err)
	}
	match := -1
	for i := range refs {
		if candidateName(refs[i]) == selected {
			match = i
		}
	}
	if match < 0 {
		return nil, fmt.Errorf("%s: selected model %s not found in available refs", name, selected)
	}
	logging.Debugf("%s selected model %s", name, selected)
	return &refs[match], nil
}

// candidateName is the name artifacts use for a candidate: its LoRA name when set.
func candidateName(ref config.ModelRef) string {
	if ref.LoRAName != "" {
		return ref.LoRAName
	}
	return ref.Model
}
