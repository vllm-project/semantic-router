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

package selection

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelselection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// MLSelectorAdapter adapts a modelselection.Selector to the selection.Selector interface.
// This bridges the ML-based selectors (KNN, KMeans, SVM) into the router's selection system.
type MLSelectorAdapter struct {
	mlSelector    modelselection.Selector
	method        SelectionMethod
	embeddingFunc func(string) ([]float32, error)
}

// NewMLSelectorAdapter creates a new adapter for an ML selector.
func NewMLSelectorAdapter(mlSelector modelselection.Selector, method SelectionMethod) *MLSelectorAdapter {
	return &MLSelectorAdapter{
		mlSelector: mlSelector,
		method:     method,
	}
}

// SetEmbeddingFunc sets the embedding function for computing query embeddings.
func (a *MLSelectorAdapter) SetEmbeddingFunc(fn func(string) ([]float32, error)) {
	a.embeddingFunc = fn
}

// Select implements selection.Selector interface.
// Converts selection.SelectionContext to modelselection.SelectionContext and calls the ML selector.
func (a *MLSelectorAdapter) Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
	if a.mlSelector == nil {
		return nil, fmt.Errorf("ML selector not initialized")
	}

	// Convert selection.SelectionContext to modelselection.SelectionContext
	mlCtx := &modelselection.SelectionContext{
		QueryText:    selCtx.Query,
		DecisionName: selCtx.DecisionName,
		CategoryName: selCtx.CategoryName,
	}

	// Convert query embedding from []float32 to []float64
	if len(selCtx.QueryEmbedding) > 0 {
		mlCtx.QueryEmbedding = float32ToFloat64(selCtx.QueryEmbedding)
	} else if a.embeddingFunc != nil && selCtx.Query != "" {
		// Compute embedding on demand if not provided
		embedding, err := a.embeddingFunc(selCtx.Query)
		if err != nil {
			logging.Warnf("[MLAdapter] Failed to compute embedding: %v, using empty embedding", err)
		} else {
			mlCtx.QueryEmbedding = float32ToFloat64(embedding)
		}
	}

	// Call the ML selector
	selectedRef, err := a.mlSelector.Select(mlCtx, selCtx.CandidateModels)
	if err != nil {
		return nil, fmt.Errorf("ML selector failed: %w", err)
	}

	if selectedRef == nil {
		// Fallback to first candidate
		if len(selCtx.CandidateModels) > 0 {
			selectedRef = &selCtx.CandidateModels[0]
		} else {
			return nil, fmt.Errorf("no candidates available")
		}
	}

	return &SelectionResult{
		SelectedModel: selectedRef.Model,
		LoRAName:      selectedRef.LoRAName,
		Score:         1.0,
		Confidence:    0.8, // ML selectors provide reasonable confidence
		Method:        a.method,
		Reasoning:     fmt.Sprintf("Selected by %s algorithm", a.method),
	}, nil
}

// Method implements selection.Selector interface.
func (a *MLSelectorAdapter) Method() SelectionMethod {
	return a.method
}

// UpdateFeedback implements selection.Selector interface.
// ML selectors can use feedback for online learning via Train().
func (a *MLSelectorAdapter) UpdateFeedback(ctx context.Context, feedback *Feedback) error {
	// ML selectors support training via Train() method, not direct feedback
	// This could be extended to convert feedback to training records
	logging.Debugf("[MLAdapter] Feedback received for %s, model=%s (training via Train() method)",
		a.method, feedback.WinnerModel)
	return nil
}

// GetMLSelector returns the underlying ML selector for direct access (e.g., for training).
func (a *MLSelectorAdapter) GetMLSelector() modelselection.Selector {
	return a.mlSelector
}

// MLSelectorConfig holds configuration for ML-based selectors.
type MLSelectorConfig struct {
	// ModelsPath is the path to pretrained model files
	ModelsPath string `yaml:"models_path"`

	// EmbeddingDim is the embedding dimension (default: 1024 for Qwen3)
	EmbeddingDim int `yaml:"embedding_dim"`

	// KNN configuration
	KNN *KNNConfig `yaml:"knn,omitempty"`

	// KMeans configuration
	KMeans *KMeansConfig `yaml:"kmeans,omitempty"`

	// SVM configuration
	SVM *SVMConfig `yaml:"svm,omitempty"`
}

// KNNConfig holds KNN-specific configuration.
type KNNConfig struct {
	K              int    `yaml:"k"`
	PretrainedPath string `yaml:"pretrained_path,omitempty"`
}

// KMeansConfig holds KMeans-specific configuration.
type KMeansConfig struct {
	NumClusters      int     `yaml:"num_clusters"`
	EfficiencyWeight float64 `yaml:"efficiency_weight"`
	PretrainedPath   string  `yaml:"pretrained_path,omitempty"`
}

// SVMConfig holds SVM-specific configuration.
type SVMConfig struct {
	Kernel         string  `yaml:"kernel"` // "linear" or "rbf"
	Gamma          float64 `yaml:"gamma"`  // RBF kernel parameter
	PretrainedPath string  `yaml:"pretrained_path,omitempty"`
}

// DefaultMLSelectorConfig returns default ML selector configuration.
func DefaultMLSelectorConfig() *MLSelectorConfig {
	return &MLSelectorConfig{
		ModelsPath:   "src/semantic-router/pkg/modelselection/data/trained_models",
		EmbeddingDim: 1024,
		KNN: &KNNConfig{
			K: 5,
		},
		KMeans: &KMeansConfig{
			NumClusters:      8,
			EfficiencyWeight: 0.1, // 0.9 quality + 0.1 efficiency
		},
		SVM: &SVMConfig{
			Kernel: "rbf",
			Gamma:  1.0,
		},
	}
}

// CreateKNNSelector creates a KNN selector adapter.
func CreateKNNSelector(cfg *MLSelectorConfig, embeddingFunc func(string) ([]float32, error)) (*MLSelectorAdapter, error) {
	knnCfg := cfg.KNN
	if knnCfg == nil {
		knnCfg = &KNNConfig{K: 5}
	}

	mlCfg := &config.MLModelSelectionConfig{
		Type: "knn",
		K:    knnCfg.K,
	}

	mlSelector, err := modelselection.NewSelector(mlCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create KNN selector: %w", err)
	}

	// Try to load pretrained model
	if knnCfg.PretrainedPath != "" {
		if knnSelector, ok := mlSelector.(*modelselection.KNNSelector); ok {
			if err := knnSelector.Load(knnCfg.PretrainedPath); err != nil {
				logging.Warnf("[MLAdapter] Failed to load pretrained KNN model from %s: %v", knnCfg.PretrainedPath, err)
			} else {
				logging.Infof("[MLAdapter] Loaded pretrained KNN model from %s", knnCfg.PretrainedPath)
			}
		}
	}

	adapter := NewMLSelectorAdapter(mlSelector, MethodKNN)
	adapter.SetEmbeddingFunc(embeddingFunc)
	return adapter, nil
}

// CreateKMeansSelector creates a KMeans selector adapter.
func CreateKMeansSelector(cfg *MLSelectorConfig, embeddingFunc func(string) ([]float32, error)) (*MLSelectorAdapter, error) {
	kmeansCfg := cfg.KMeans
	if kmeansCfg == nil {
		kmeansCfg = &KMeansConfig{NumClusters: 8, EfficiencyWeight: 0.1}
	}

	effWeight := kmeansCfg.EfficiencyWeight
	mlCfg := &config.MLModelSelectionConfig{
		Type:             "kmeans",
		NumClusters:      kmeansCfg.NumClusters,
		EfficiencyWeight: &effWeight,
	}

	mlSelector, err := modelselection.NewSelector(mlCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create KMeans selector: %w", err)
	}

	// Try to load pretrained model
	if kmeansCfg.PretrainedPath != "" {
		if kmeansSelector, ok := mlSelector.(*modelselection.KMeansSelector); ok {
			if err := kmeansSelector.Load(kmeansCfg.PretrainedPath); err != nil {
				logging.Warnf("[MLAdapter] Failed to load pretrained KMeans model from %s: %v", kmeansCfg.PretrainedPath, err)
			} else {
				logging.Infof("[MLAdapter] Loaded pretrained KMeans model from %s", kmeansCfg.PretrainedPath)
			}
		}
	}

	adapter := NewMLSelectorAdapter(mlSelector, MethodKMeans)
	adapter.SetEmbeddingFunc(embeddingFunc)
	return adapter, nil
}

// CreateSVMSelector creates an SVM selector adapter.
func CreateSVMSelector(cfg *MLSelectorConfig, embeddingFunc func(string) ([]float32, error)) (*MLSelectorAdapter, error) {
	svmCfg := cfg.SVM
	if svmCfg == nil {
		svmCfg = &SVMConfig{Kernel: "rbf", Gamma: 1.0}
	}

	mlCfg := &config.MLModelSelectionConfig{
		Type:   "svm",
		Kernel: svmCfg.Kernel,
		Gamma:  svmCfg.Gamma,
	}

	mlSelector, err := modelselection.NewSelector(mlCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create SVM selector: %w", err)
	}

	// Try to load pretrained model
	if svmCfg.PretrainedPath != "" {
		if svmSelector, ok := mlSelector.(*modelselection.SVMSelector); ok {
			if err := svmSelector.Load(svmCfg.PretrainedPath); err != nil {
				logging.Warnf("[MLAdapter] Failed to load pretrained SVM model from %s: %v", svmCfg.PretrainedPath, err)
			} else {
				logging.Infof("[MLAdapter] Loaded pretrained SVM model from %s", svmCfg.PretrainedPath)
			}
		}
	}

	adapter := NewMLSelectorAdapter(mlSelector, MethodSVM)
	adapter.SetEmbeddingFunc(embeddingFunc)
	return adapter, nil
}

// float32ToFloat64 converts a slice of float32 to float64.
func float32ToFloat64(f32 []float32) []float64 {
	f64 := make([]float64, len(f32))
	for i, v := range f32 {
		f64[i] = float64(v)
	}
	return f64
}
