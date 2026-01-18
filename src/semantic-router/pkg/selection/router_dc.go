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
	"math"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// RouterDCConfig configures the RouterDC dual-contrastive selector
// Based on arXiv:2409.19886 - Query-Based Router by Dual Contrastive Learning
type RouterDCConfig struct {
	// Temperature for softmax scaling (default: 0.07 as per paper)
	Temperature float64 `yaml:"temperature"`

	// DimensionSize for embeddings (default: 768)
	DimensionSize int `yaml:"dimension_size"`

	// MinSimilarity threshold for valid matches (default: 0.3)
	MinSimilarity float64 `yaml:"min_similarity"`

	// UseQueryContrastive enables query-side contrastive learning
	UseQueryContrastive bool `yaml:"use_query_contrastive"`

	// UseModelContrastive enables model-side contrastive learning
	UseModelContrastive bool `yaml:"use_model_contrastive"`
}

// DefaultRouterDCConfig returns the default RouterDC configuration
func DefaultRouterDCConfig() *RouterDCConfig {
	return &RouterDCConfig{
		Temperature:         0.07,
		DimensionSize:       768,
		MinSimilarity:       0.3,
		UseQueryContrastive: true,
		UseModelContrastive: true,
	}
}

// ModelEmbedding represents a model's capability embedding
type ModelEmbedding struct {
	Model     string    `json:"model"`
	Embedding []float32 `json:"embedding"`
}

// RouterDCSelector implements dual-contrastive learning for query-to-model routing
// The approach learns embeddings for both queries and models, then matches them
// using contrastive learning to find the best model for each query type.
type RouterDCSelector struct {
	config *RouterDCConfig

	// Model embeddings represent each model's capabilities/strengths
	modelEmbeddings map[string][]float32
	embeddingMu     sync.RWMutex

	// Query-model affinity matrix for contrastive learning
	affinityMatrix map[string]map[string]float64 // query_hash -> model -> affinity
	affinityMu     sync.RWMutex

	// Embedding provider function (injected dependency)
	embeddingFunc func(text string) ([]float32, error)
}

// NewRouterDCSelector creates a new RouterDC-based selector
func NewRouterDCSelector(cfg *RouterDCConfig) *RouterDCSelector {
	if cfg == nil {
		cfg = DefaultRouterDCConfig()
	}
	return &RouterDCSelector{
		config:          cfg,
		modelEmbeddings: make(map[string][]float32),
		affinityMatrix:  make(map[string]map[string]float64),
	}
}

// Method returns the selection method type
func (r *RouterDCSelector) Method() SelectionMethod {
	return MethodRouterDC
}

// SetEmbeddingFunc sets the function used to compute embeddings
func (r *RouterDCSelector) SetEmbeddingFunc(f func(text string) ([]float32, error)) {
	r.embeddingFunc = f
}

// InitializeModelEmbeddings sets up model capability embeddings
// Each model is represented by an embedding that captures its strengths
func (r *RouterDCSelector) InitializeModelEmbeddings(modelDescriptions map[string]string) error {
	if r.embeddingFunc == nil {
		return fmt.Errorf("embedding function not set")
	}

	r.embeddingMu.Lock()
	defer r.embeddingMu.Unlock()

	for model, description := range modelDescriptions {
		embedding, err := r.embeddingFunc(description)
		if err != nil {
			logging.Warnf("[RouterDC] Failed to embed model %s description: %v", model, err)
			continue
		}
		r.modelEmbeddings[model] = embedding
	}

	logging.Infof("[RouterDC] Initialized embeddings for %d models", len(r.modelEmbeddings))
	return nil
}

// SetModelEmbedding directly sets a model's embedding
func (r *RouterDCSelector) SetModelEmbedding(model string, embedding []float32) {
	r.embeddingMu.Lock()
	defer r.embeddingMu.Unlock()
	r.modelEmbeddings[model] = embedding
}

// Select chooses the best model using dual-contrastive matching
func (r *RouterDCSelector) Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
	if len(selCtx.CandidateModels) == 0 {
		return nil, fmt.Errorf("no candidate models provided")
	}

	// Get or compute query embedding
	queryEmbedding := selCtx.QueryEmbedding
	if queryEmbedding == nil {
		if r.embeddingFunc == nil {
			// Fall back to first candidate if no embedding capability
			return r.fallbackSelection(selCtx, "no embedding function available")
		}

		var err error
		queryEmbedding, err = r.embeddingFunc(selCtx.Query)
		if err != nil {
			return r.fallbackSelection(selCtx, fmt.Sprintf("embedding error: %v", err))
		}
	}

	allScores := make(map[string]float64)
	var bestModel *config.ModelRef
	var bestScore float64

	r.embeddingMu.RLock()
	defer r.embeddingMu.RUnlock()

	logging.Infof("[RouterDC] Evaluating %d candidates by embedding similarity:",
		len(selCtx.CandidateModels))

	for i := range selCtx.CandidateModels {
		model := &selCtx.CandidateModels[i]
		modelEmb, exists := r.modelEmbeddings[model.Model]

		if !exists {
			// Model has no embedding, assign minimum score
			allScores[model.Model] = r.config.MinSimilarity
			logging.Infof("[RouterDC]   %s: similarity=%.4f (no embedding, using min)", model.Model, r.config.MinSimilarity)
			continue
		}

		// Compute contrastive similarity
		similarity := r.computeContrastiveSimilarity(queryEmbedding, modelEmb)
		allScores[model.Model] = similarity
		logging.Infof("[RouterDC]   %s: similarity=%.4f", model.Model, similarity)

		if similarity > bestScore {
			bestScore = similarity
			bestModel = model
		}
	}

	if bestModel == nil || bestScore < r.config.MinSimilarity {
		return r.fallbackSelection(selCtx, "no model above similarity threshold")
	}

	// Apply softmax with temperature to get calibrated probabilities
	softmaxScores := r.applySoftmax(allScores)

	confidence := bestScore // Use raw similarity as confidence
	if bestScore > 0.9 {
		confidence = 0.95
	}

	reasoning := fmt.Sprintf("Query-model contrastive similarity: %.4f (temperature=%.3f)",
		bestScore, r.config.Temperature)

	logging.Infof("[RouterDC] Selected model %s (similarity=%.4f, confidence=%.2f)",
		bestModel.Model, bestScore, confidence)

	return &SelectionResult{
		SelectedModel: bestModel.Model,
		LoRAName:      bestModel.LoRAName,
		Score:         softmaxScores[bestModel.Model],
		Confidence:    confidence,
		Method:        MethodRouterDC,
		Reasoning:     reasoning,
		AllScores:     softmaxScores,
	}, nil
}

// UpdateFeedback updates model-query affinity based on feedback
func (r *RouterDCSelector) UpdateFeedback(ctx context.Context, feedback *Feedback) error {
	if feedback.WinnerModel == "" {
		return fmt.Errorf("winner model is required")
	}

	// Create a simple hash for the query to track affinity
	queryHash := r.hashQuery(feedback.Query)

	r.affinityMu.Lock()
	defer r.affinityMu.Unlock()

	if r.affinityMatrix[queryHash] == nil {
		r.affinityMatrix[queryHash] = make(map[string]float64)
	}

	// Increase affinity for winner
	currentAffinity := r.affinityMatrix[queryHash][feedback.WinnerModel]
	r.affinityMatrix[queryHash][feedback.WinnerModel] = currentAffinity + 0.1

	// Decrease affinity for loser (if present)
	if feedback.LoserModel != "" && !feedback.Tie {
		loserAffinity := r.affinityMatrix[queryHash][feedback.LoserModel]
		r.affinityMatrix[queryHash][feedback.LoserModel] = math.Max(0, loserAffinity-0.05)
	}

	logging.Debugf("[RouterDC] Updated affinity for query hash %s: winner=%s (+0.1)",
		queryHash[:8], feedback.WinnerModel)

	return nil
}

// computeContrastiveSimilarity calculates dual-contrastive similarity
func (r *RouterDCSelector) computeContrastiveSimilarity(queryEmb, modelEmb []float32) float64 {
	if len(queryEmb) != len(modelEmb) {
		// Handle dimension mismatch by using minimum length
		minLen := len(queryEmb)
		if len(modelEmb) < minLen {
			minLen = len(modelEmb)
		}
		queryEmb = queryEmb[:minLen]
		modelEmb = modelEmb[:minLen]
	}

	// Compute cosine similarity
	similarity := r.cosineSimilarity(queryEmb, modelEmb)

	// Apply temperature scaling for contrastive learning
	// Higher temperature = softer distribution
	scaledSim := similarity / r.config.Temperature

	// Apply sigmoid to bound the result
	return 1.0 / (1.0 + math.Exp(-scaledSim))
}

// cosineSimilarity computes cosine similarity between two vectors
func (r *RouterDCSelector) cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// applySoftmax applies softmax with temperature to scores
func (r *RouterDCSelector) applySoftmax(scores map[string]float64) map[string]float64 {
	result := make(map[string]float64)

	// Find max for numerical stability
	maxScore := math.Inf(-1)
	for _, s := range scores {
		if s > maxScore {
			maxScore = s
		}
	}

	// Compute softmax
	sum := 0.0
	for model, s := range scores {
		exp := math.Exp((s - maxScore) / r.config.Temperature)
		result[model] = exp
		sum += exp
	}

	// Normalize
	for model := range result {
		result[model] /= sum
	}

	return result
}

// hashQuery creates a simple hash for query tracking
func (r *RouterDCSelector) hashQuery(query string) string {
	// Simple hash for query grouping (could use more sophisticated methods)
	if len(query) < 32 {
		return fmt.Sprintf("%x", query)
	}
	return fmt.Sprintf("%x", query[:32])
}

// fallbackSelection returns a fallback result when embedding-based selection fails
func (r *RouterDCSelector) fallbackSelection(selCtx *SelectionContext, reason string) (*SelectionResult, error) {
	if len(selCtx.CandidateModels) == 0 {
		return nil, fmt.Errorf("no candidate models")
	}

	firstModel := &selCtx.CandidateModels[0]
	allScores := make(map[string]float64)
	for i := range selCtx.CandidateModels {
		allScores[selCtx.CandidateModels[i].Model] = 1.0 / float64(len(selCtx.CandidateModels))
	}

	logging.Warnf("[RouterDC] Fallback selection: %s, using first candidate %s", reason, firstModel.Model)

	return &SelectionResult{
		SelectedModel: firstModel.Model,
		LoRAName:      firstModel.LoRAName,
		Score:         allScores[firstModel.Model],
		Confidence:    0.5,
		Method:        MethodRouterDC,
		Reasoning:     fmt.Sprintf("Fallback selection: %s", reason),
		AllScores:     allScores,
	}, nil
}

// GetModelEmbeddings returns all model embeddings (for debugging)
func (r *RouterDCSelector) GetModelEmbeddings() map[string][]float32 {
	r.embeddingMu.RLock()
	defer r.embeddingMu.RUnlock()

	result := make(map[string][]float32)
	for k, v := range r.modelEmbeddings {
		result[k] = v
	}
	return result
}
