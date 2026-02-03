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

// Package selection provides advanced model selection algorithms for intelligent routing.
// It implements multiple selection strategies including Elo rating, RouterDC (dual-contrastive
// learning), AutoMix (POMDP-based), and hybrid approaches that combine multiple techniques.
//
// Reference papers:
//   - Elo: RouteLLM (arXiv:2406.18665) - Weighted Elo using Bradley-Terry model
//   - RouterDC: Query-Based Router by Dual Contrastive Learning (arXiv:2409.19886)
//   - AutoMix: Automatically Mixing Language Models (arXiv:2310.12963)
//   - Hybrid LLM: Cost-Efficient Quality-Aware Query Routing (arXiv:2404.14618)
package selection

import (
	"context"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// SelectionMethod defines the type of model selection algorithm
type SelectionMethod string

const (
	// MethodElo uses Elo rating system with Bradley-Terry model
	// Models are scored based on pairwise comparisons using preference feedback
	MethodElo SelectionMethod = "elo"

	// MethodRouterDC uses dual-contrastive learning for query-to-model routing
	// Learns query embeddings that match well with specific model capabilities
	MethodRouterDC SelectionMethod = "router_dc"

	// MethodAutoMix uses POMDP-based cascaded routing with self-verification
	// Routes to smaller models first, escalates based on self-verification confidence
	MethodAutoMix SelectionMethod = "automix"

	// MethodHybrid combines multiple selection techniques with configurable weights
	// Allows blending Elo, embedding similarity, and cost considerations
	MethodHybrid SelectionMethod = "hybrid"

	// MethodStatic uses static scores from configuration (default behavior)
	MethodStatic SelectionMethod = "static"

	// MethodKNN uses K-Nearest Neighbors for query-based model selection
	// Finds similar historical queries and uses quality-weighted voting
	// Reference: FusionFactory (arXiv:2507.10540) query-level fusion
	MethodKNN SelectionMethod = "knn"

	// MethodKMeans uses KMeans clustering for model selection
	// Clusters queries and assigns models based on quality+latency scores
	// Reference: Avengers-Pro (arXiv:2508.12631) performance-efficiency routing
	MethodKMeans SelectionMethod = "kmeans"

	// MethodSVM uses Support Vector Machine for model classification
	// Learns decision boundaries between model preferences using RBF kernel
	// Reference: FusionFactory (arXiv:2507.10540), Avengers-Pro (arXiv:2508.12631)
	MethodSVM SelectionMethod = "svm"
)

// SelectionContext provides context for model selection decisions
type SelectionContext struct {
	// Query is the user's input query text
	Query string

	// QueryEmbedding is the precomputed embedding vector for the query (optional)
	// If nil, selectors that need embeddings will compute them on demand
	QueryEmbedding []float32

	// ConversationHistory provides prior messages for context-aware selection
	ConversationHistory []string

	// DecisionName is the name of the matched decision for category-specific selection
	DecisionName string

	// CategoryName is the detected domain category (e.g., "physics", "math")
	// Used by ML selectors to create feature vectors with category one-hot encoding
	CategoryName string

	// CandidateModels is the list of models to select from
	CandidateModels []config.ModelRef

	// CostWeight indicates how much to weight cost in selection (0.0-1.0)
	// Higher values prefer cheaper models
	CostWeight float64

	// QualityWeight indicates how much to weight quality/score (0.0-1.0)
	// Higher values prefer higher-quality models
	QualityWeight float64
}

// SelectionResult contains the result of a model selection decision
type SelectionResult struct {
	// SelectedModel is the name of the selected model
	SelectedModel string

	// LoRAName is the LoRA adapter name to use (if applicable)
	LoRAName string

	// Score is the selection score for the chosen model
	Score float64

	// Confidence indicates how confident the selector is in this choice
	Confidence float64

	// Method indicates which selection method was used
	Method SelectionMethod

	// Reasoning provides human-readable explanation for the selection
	Reasoning string

	// AllScores maps each candidate model to its computed score
	AllScores map[string]float64
}

// Selector is the interface for model selection algorithms
type Selector interface {
	// Select chooses the best model from candidates based on the selection context
	Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error)

	// Method returns the selection method type
	Method() SelectionMethod

	// UpdateFeedback allows the selector to learn from user feedback
	// This is primarily used by Elo and learning-based methods
	UpdateFeedback(ctx context.Context, feedback *Feedback) error
}

// Feedback represents user feedback for model comparison
type Feedback struct {
	// Query is the original query that was processed
	Query string

	// WinnerModel is the model that was preferred
	WinnerModel string

	// LoserModel is the model that was not preferred (can be empty for single feedback)
	LoserModel string

	// Tie indicates if both models performed equally
	Tie bool

	// DecisionName is the category/decision context
	DecisionName string

	// Timestamp is when the feedback was recorded
	Timestamp int64
}

// Registry maintains available selection methods and their configurations
type Registry struct {
	selectors map[SelectionMethod]Selector
	mu        sync.RWMutex
}

// NewRegistry creates a new selector registry
func NewRegistry() *Registry {
	return &Registry{
		selectors: make(map[SelectionMethod]Selector),
	}
}

// Register adds a selector to the registry
func (r *Registry) Register(method SelectionMethod, selector Selector) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.selectors[method] = selector
}

// Get retrieves a selector by method type
func (r *Registry) Get(method SelectionMethod) (Selector, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	s, ok := r.selectors[method]
	return s, ok
}

// GlobalRegistry is the default registry for selection methods
var GlobalRegistry = NewRegistry()

// Select uses the specified method to select a model
func Select(ctx context.Context, method SelectionMethod, selCtx *SelectionContext) (*SelectionResult, error) {
	selector, ok := GlobalRegistry.Get(method)
	if !ok {
		// Fall back to static selection
		selector, _ = GlobalRegistry.Get(MethodStatic)
	}
	if selector == nil {
		// Ultimate fallback: return first candidate
		return &SelectionResult{
			SelectedModel: selCtx.CandidateModels[0].Model,
			LoRAName:      selCtx.CandidateModels[0].LoRAName,
			Score:         1.0,
			Confidence:    1.0,
			Method:        MethodStatic,
			Reasoning:     "No selector available, using first candidate",
		}, nil
	}
	return selector.Select(ctx, selCtx)
}
