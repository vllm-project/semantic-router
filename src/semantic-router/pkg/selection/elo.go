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
	"sort"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// DefaultEloRating is the initial Elo rating for new models
const DefaultEloRating = 1500.0

// EloKFactor controls how much ratings change per comparison
// Higher values = faster adaptation but more volatility
const EloKFactor = 32.0

// EloMinRatingFromScore is the base rating when converting static scores (0-1) to Elo
const EloMinRatingFromScore = 1000.0

// EloRatingRange is the rating range for score conversion (scores 0-1 map to 1000-2000)
const EloRatingRange = 1000.0

// EloConfig configures the Elo-based model selector
type EloConfig struct {
	// InitialRating is the starting Elo rating for new models
	InitialRating float64 `yaml:"initial_rating"`

	// KFactor controls rating volatility (higher = more volatile)
	KFactor float64 `yaml:"k_factor"`

	// CategoryWeighted enables per-category Elo ratings
	CategoryWeighted bool `yaml:"category_weighted"`

	// DecayFactor applies time decay to old comparisons (0-1, 0 = no decay)
	DecayFactor float64 `yaml:"decay_factor"`

	// MinComparisons is minimum comparisons before a rating is considered stable
	MinComparisons int `yaml:"min_comparisons"`

	// CostScalingFactor scales cost consideration (0 = ignore cost)
	CostScalingFactor float64 `yaml:"cost_scaling_factor"`

	// StoragePath is the file path for persisting Elo ratings (optional)
	// If set, ratings are loaded on startup and saved after each feedback update
	// Example: "/var/lib/vsr/elo_ratings.json"
	StoragePath string `yaml:"storage_path,omitempty"`

	// AutoSaveInterval is the interval for automatic saves (default: 30s)
	// Only used when StoragePath is set
	AutoSaveInterval string `yaml:"auto_save_interval,omitempty"`
}

// DefaultEloConfig returns the default Elo configuration
func DefaultEloConfig() *EloConfig {
	return &EloConfig{
		InitialRating:     DefaultEloRating,
		KFactor:           EloKFactor,
		CategoryWeighted:  true,
		DecayFactor:       0.0,
		MinComparisons:    5,
		CostScalingFactor: 0.0,
	}
}

// ModelRating stores the Elo rating and metadata for a model
type ModelRating struct {
	Model       string  `json:"model"`
	Rating      float64 `json:"rating"`
	Comparisons int     `json:"comparisons"`
	Wins        int     `json:"wins"`
	Losses      int     `json:"losses"`
	Ties        int     `json:"ties"`
}

// EloSelector implements Elo rating-based model selection
// Based on RouteLLM paper (arXiv:2406.18665) using Bradley-Terry model
type EloSelector struct {
	config *EloConfig

	// Global ratings (not category-specific)
	globalRatings map[string]*ModelRating
	globalMu      sync.RWMutex

	// Category-specific ratings (decision name -> model -> rating)
	categoryRatings map[string]map[string]*ModelRating
	categoryMu      sync.RWMutex

	// Model costs for cost-aware selection (model -> cost per 1M tokens)
	modelCosts map[string]float64
	costMu     sync.RWMutex

	// Storage backend for persisting ratings (optional)
	storage EloStorage
}

// NewEloSelector creates a new Elo-based selector
func NewEloSelector(cfg *EloConfig) *EloSelector {
	if cfg == nil {
		cfg = DefaultEloConfig()
	}
	selector := &EloSelector{
		config:          cfg,
		globalRatings:   make(map[string]*ModelRating),
		categoryRatings: make(map[string]map[string]*ModelRating),
		modelCosts:      make(map[string]float64),
	}

	// Initialize storage if path is configured
	if cfg.StoragePath != "" {
		storage, err := NewFileEloStorage(cfg.StoragePath)
		if err != nil {
			logging.Errorf("[EloSelector] Failed to initialize storage: %v", err)
		} else {
			selector.storage = storage

			// Load existing ratings from storage
			if err := selector.loadFromStorage(); err != nil {
				logging.Warnf("[EloSelector] Failed to load ratings from storage: %v", err)
			}

			// Start auto-save with configurable interval
			interval := 30 * time.Second
			if cfg.AutoSaveInterval != "" {
				if parsed, err := time.ParseDuration(cfg.AutoSaveInterval); err == nil {
					interval = parsed
				}
			}

			storage.StartAutoSave(interval, selector.getAllRatingsForStorage)
			logging.Infof("[EloSelector] Storage initialized with auto-save interval: %v", interval)
		}
	}

	return selector
}

// SetStorage sets a custom storage backend (useful for testing)
func (e *EloSelector) SetStorage(storage EloStorage) {
	e.storage = storage
}

// loadFromStorage loads ratings from the storage backend
func (e *EloSelector) loadFromStorage() error {
	if e.storage == nil {
		return nil
	}

	allRatings, err := e.storage.LoadAllRatings()
	if err != nil {
		return err
	}

	e.globalMu.Lock()
	e.categoryMu.Lock()
	defer e.globalMu.Unlock()
	defer e.categoryMu.Unlock()

	for category, ratings := range allRatings {
		if category == "_global" {
			for model, rating := range ratings {
				e.globalRatings[model] = rating
			}
		} else {
			if e.categoryRatings[category] == nil {
				e.categoryRatings[category] = make(map[string]*ModelRating)
			}
			for model, rating := range ratings {
				e.categoryRatings[category][model] = rating
			}
		}
	}

	logging.Infof("[EloSelector] Loaded %d categories from storage", len(allRatings))
	return nil
}

// getAllRatingsForStorage returns all ratings in a format suitable for storage
func (e *EloSelector) getAllRatingsForStorage() map[string]map[string]*ModelRating {
	e.globalMu.RLock()
	e.categoryMu.RLock()
	defer e.globalMu.RUnlock()
	defer e.categoryMu.RUnlock()

	result := make(map[string]map[string]*ModelRating)

	// Add global ratings
	if len(e.globalRatings) > 0 {
		result["_global"] = make(map[string]*ModelRating)
		for k, v := range e.globalRatings {
			result["_global"][k] = v
		}
	}

	// Add category ratings
	for cat, ratings := range e.categoryRatings {
		result[cat] = make(map[string]*ModelRating)
		for k, v := range ratings {
			result[cat][k] = v
		}
	}

	return result
}

// Close stops storage operations and persists final state
func (e *EloSelector) Close() error {
	if e.storage != nil {
		return e.storage.Close()
	}
	return nil
}

// Method returns the selection method type
func (e *EloSelector) Method() SelectionMethod {
	return MethodElo
}

// SetModelCost sets the cost per 1M tokens for a model
func (e *EloSelector) SetModelCost(model string, costPer1M float64) {
	e.costMu.Lock()
	defer e.costMu.Unlock()
	e.modelCosts[model] = costPer1M
}

// InitializeFromConfig sets up initial ratings from model configuration
func (e *EloSelector) InitializeFromConfig(modelConfig map[string]config.ModelParams, categories []config.Category) {
	e.globalMu.Lock()
	defer e.globalMu.Unlock()

	// Initialize global ratings for all models
	for model := range modelConfig {
		if _, exists := e.globalRatings[model]; !exists {
			e.globalRatings[model] = &ModelRating{
				Model:  model,
				Rating: e.config.InitialRating,
			}
		}
	}

	// Set costs from config
	e.costMu.Lock()
	for model, params := range modelConfig {
		if params.Pricing.PromptPer1M > 0 {
			e.modelCosts[model] = params.Pricing.PromptPer1M
		}
	}
	e.costMu.Unlock()

	// Initialize category ratings from ModelScores if available
	if e.config.CategoryWeighted {
		e.categoryMu.Lock()
		for _, category := range categories {
			if e.categoryRatings[category.Name] == nil {
				e.categoryRatings[category.Name] = make(map[string]*ModelRating)
			}
			for _, ms := range category.ModelScores {
				// Convert static scores to Elo ratings (scale 0-1 -> 1000-2000)
				rating := EloMinRatingFromScore + (ms.Score * EloRatingRange)
				e.categoryRatings[category.Name][ms.Model] = &ModelRating{
					Model:  ms.Model,
					Rating: rating,
				}
			}
		}
		e.categoryMu.Unlock()
	}
}

// Select chooses the best model based on Elo ratings
func (e *EloSelector) Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
	if len(selCtx.CandidateModels) == 0 {
		return nil, fmt.Errorf("no candidate models provided")
	}

	allScores := make(map[string]float64)

	// Get ratings for all candidates
	ratings := e.getRatingsForCandidates(selCtx.DecisionName, selCtx.CandidateModels)

	logging.Infof("[EloSelector] Evaluating %d candidates for category '%s':",
		len(selCtx.CandidateModels), selCtx.DecisionName)
	for _, r := range ratings {
		logging.Infof("[EloSelector]   %s: rating=%.1f (W:%d L:%d T:%d)",
			r.Model, r.Rating, r.Wins, r.Losses, r.Ties)
	}

	// Calculate selection probability using Bradley-Terry model
	// P(model_i wins) = rating_i / sum(all ratings)
	totalRating := 0.0
	for _, r := range ratings {
		totalRating += math.Pow(10, r.Rating/400.0) // Standard Elo probability scale
	}

	if totalRating == 0 {
		// Fallback: uniform distribution
		for _, r := range ratings {
			allScores[r.Model] = 1.0 / float64(len(ratings))
		}
	} else {
		for _, r := range ratings {
			prob := math.Pow(10, r.Rating/400.0) / totalRating
			allScores[r.Model] = prob
		}
	}

	// Apply cost adjustment if enabled
	if e.config.CostScalingFactor > 0 && selCtx.CostWeight > 0 {
		e.applyCostAdjustment(allScores, selCtx.CostWeight)
	}

	// Find best model by combined score
	var bestModel *config.ModelRef
	var bestScore float64
	var bestRating *ModelRating

	for i := range selCtx.CandidateModels {
		model := &selCtx.CandidateModels[i]
		score := allScores[model.Model]

		if score > bestScore || bestModel == nil {
			bestScore = score
			bestModel = model
			for _, r := range ratings {
				if r.Model == model.Model {
					bestRating = r
					break
				}
			}
		}
	}

	if bestModel == nil {
		return nil, fmt.Errorf("could not select a model")
	}

	// Calculate confidence based on rating stability
	confidence := e.calculateConfidence(bestRating)

	reasoning := fmt.Sprintf("Selected based on Elo rating %.1f (win rate: %d/%d)",
		bestRating.Rating,
		bestRating.Wins,
		bestRating.Wins+bestRating.Losses+bestRating.Ties)

	if e.config.CategoryWeighted && selCtx.DecisionName != "" {
		reasoning = fmt.Sprintf("Category '%s': %s", selCtx.DecisionName, reasoning)
	}

	logging.Infof("[EloSelector] Selected model %s (rating=%.1f, score=%.4f, confidence=%.2f)",
		bestModel.Model, bestRating.Rating, bestScore, confidence)

	// Record metrics for all candidate models' current Elo ratings
	for _, r := range ratings {
		category := selCtx.DecisionName
		if category == "" {
			category = "_global"
		}
		RecordEloRating(r.Model, category, r.Rating)
	}

	return &SelectionResult{
		SelectedModel: bestModel.Model,
		LoRAName:      bestModel.LoRAName,
		Score:         bestScore,
		Confidence:    confidence,
		Method:        MethodElo,
		Reasoning:     reasoning,
		AllScores:     allScores,
	}, nil
}

// UpdateFeedback updates Elo ratings based on user preference feedback.
// Supports three modes:
// 1. Comparison: Both WinnerModel and LoserModel set - standard Elo update
// 2. Positive self-feedback: Only WinnerModel set - small rating boost
// 3. Negative self-feedback: Only LoserModel set - small rating penalty
func (e *EloSelector) UpdateFeedback(ctx context.Context, feedback *Feedback) error {
	if feedback.WinnerModel == "" && feedback.LoserModel == "" {
		return fmt.Errorf("either winner_model or loser_model is required")
	}

	// Capture old ratings for metrics
	oldWinnerRating := e.getGlobalRating(feedback.WinnerModel)
	oldLoserRating := e.getGlobalRating(feedback.LoserModel)
	var winnerOldElo, loserOldElo float64
	if oldWinnerRating != nil {
		winnerOldElo = oldWinnerRating.Rating
	} else {
		winnerOldElo = e.config.InitialRating
	}
	if oldLoserRating != nil {
		loserOldElo = oldLoserRating.Rating
	} else {
		loserOldElo = e.config.InitialRating
	}

	// Update global ratings
	e.updateRating(feedback, e.getGlobalRating, e.setGlobalRating)

	// Capture new ratings after update
	newWinnerRating := e.getGlobalRating(feedback.WinnerModel)
	newLoserRating := e.getGlobalRating(feedback.LoserModel)

	// Record metrics for global ratings
	if newWinnerRating != nil {
		RecordFeedbackMetrics(&FeedbackMetrics{
			Winner:       feedback.WinnerModel,
			Loser:        feedback.LoserModel,
			Category:     "_global",
			IsTie:        feedback.Tie,
			WinnerOldElo: winnerOldElo,
			WinnerNewElo: newWinnerRating.Rating,
			LoserOldElo:  loserOldElo,
			LoserNewElo: func() float64 {
				if newLoserRating != nil {
					return newLoserRating.Rating
				}
				return loserOldElo
			}(),
			WinnerStats: *newWinnerRating,
			LoserStats: func() ModelRating {
				if newLoserRating != nil {
					return *newLoserRating
				}
				return ModelRating{}
			}(),
		})
	}

	// Update category ratings if applicable
	if e.config.CategoryWeighted && feedback.DecisionName != "" {
		// Capture old category ratings
		oldCatWinner := e.getCategoryRating(feedback.DecisionName, feedback.WinnerModel)
		oldCatLoser := e.getCategoryRating(feedback.DecisionName, feedback.LoserModel)
		var catWinnerOldElo, catLoserOldElo float64
		if oldCatWinner != nil {
			catWinnerOldElo = oldCatWinner.Rating
		} else {
			catWinnerOldElo = e.config.InitialRating
		}
		if oldCatLoser != nil {
			catLoserOldElo = oldCatLoser.Rating
		} else {
			catLoserOldElo = e.config.InitialRating
		}

		e.updateRating(feedback,
			func(model string) *ModelRating {
				return e.getCategoryRating(feedback.DecisionName, model)
			},
			func(model string, rating *ModelRating) {
				e.setCategoryRating(feedback.DecisionName, model, rating)
			})

		// Capture new category ratings and record metrics
		newCatWinner := e.getCategoryRating(feedback.DecisionName, feedback.WinnerModel)
		newCatLoser := e.getCategoryRating(feedback.DecisionName, feedback.LoserModel)

		if newCatWinner != nil {
			RecordFeedbackMetrics(&FeedbackMetrics{
				Winner:       feedback.WinnerModel,
				Loser:        feedback.LoserModel,
				Category:     feedback.DecisionName,
				IsTie:        feedback.Tie,
				WinnerOldElo: catWinnerOldElo,
				WinnerNewElo: newCatWinner.Rating,
				LoserOldElo:  catLoserOldElo,
				LoserNewElo: func() float64 {
					if newCatLoser != nil {
						return newCatLoser.Rating
					}
					return catLoserOldElo
				}(),
				WinnerStats: *newCatWinner,
				LoserStats: func() ModelRating {
					if newCatLoser != nil {
						return *newCatLoser
					}
					return ModelRating{}
				}(),
			})
		}
	}

	logging.Infof("[EloSelector] Updated ratings: winner=%s, loser=%s, tie=%v",
		feedback.WinnerModel, feedback.LoserModel, feedback.Tie)

	// Mark storage as dirty for auto-save, or save immediately for single updates
	if e.storage != nil {
		if fileStorage, ok := e.storage.(*FileEloStorage); ok {
			fileStorage.MarkDirty()
		} else {
			// For non-file storage, save immediately
			if err := e.storage.SaveAllRatings(e.getAllRatingsForStorage()); err != nil {
				logging.Warnf("[EloSelector] Failed to save ratings to storage: %v", err)
			}
		}
	}

	return nil
}

// updateRating performs the actual Elo rating update
func (e *EloSelector) updateRating(feedback *Feedback,
	getRating func(string) *ModelRating,
	setRating func(string, *ModelRating),
) {
	// Handle self-feedback (single model, no comparison)
	// This is used for automatic signal-based feedback where we only know
	// if one model did well or poorly, not compared to another
	if feedback.LoserModel == "" && feedback.WinnerModel != "" {
		// Positive self-feedback: model did well
		winnerRating := getRating(feedback.WinnerModel)
		if winnerRating == nil {
			winnerRating = &ModelRating{Model: feedback.WinnerModel, Rating: e.config.InitialRating}
		}
		// Apply small positive adjustment (reward)
		winnerRating.Rating += e.config.KFactor * 0.1 // 10% of K-factor for self-feedback
		winnerRating.Comparisons++
		winnerRating.Wins++
		setRating(feedback.WinnerModel, winnerRating)
		return
	}

	if feedback.WinnerModel == "" && feedback.LoserModel != "" {
		// Negative self-feedback: model did poorly
		loserRating := getRating(feedback.LoserModel)
		if loserRating == nil {
			loserRating = &ModelRating{Model: feedback.LoserModel, Rating: e.config.InitialRating}
		}
		// Apply small negative adjustment (penalty)
		loserRating.Rating -= e.config.KFactor * 0.1 // 10% of K-factor for self-feedback
		loserRating.Comparisons++
		loserRating.Losses++
		setRating(feedback.LoserModel, loserRating)
		return
	}

	// Standard two-model comparison
	if feedback.WinnerModel == "" {
		return // No valid feedback
	}

	winnerRating := getRating(feedback.WinnerModel)
	if winnerRating == nil {
		winnerRating = &ModelRating{Model: feedback.WinnerModel, Rating: e.config.InitialRating}
	}

	loserRating := getRating(feedback.LoserModel)
	if loserRating == nil {
		loserRating = &ModelRating{Model: feedback.LoserModel, Rating: e.config.InitialRating}
	}

	// Calculate expected scores using Bradley-Terry model
	// E_a = 1 / (1 + 10^((R_b - R_a) / 400))
	expectedWinner := 1.0 / (1.0 + math.Pow(10, (loserRating.Rating-winnerRating.Rating)/400.0))
	expectedLoser := 1.0 - expectedWinner

	// Determine actual scores
	var actualWinner, actualLoser float64
	if feedback.Tie {
		actualWinner = 0.5
		actualLoser = 0.5
	} else {
		actualWinner = 1.0
		actualLoser = 0.0
	}

	// Update ratings: R' = R + K * (actual - expected)
	winnerRating.Rating += e.config.KFactor * (actualWinner - expectedWinner)
	loserRating.Rating += e.config.KFactor * (actualLoser - expectedLoser)

	// Update statistics
	winnerRating.Comparisons++
	loserRating.Comparisons++
	if feedback.Tie {
		winnerRating.Ties++
		loserRating.Ties++
	} else {
		winnerRating.Wins++
		loserRating.Losses++
	}

	setRating(feedback.WinnerModel, winnerRating)
	setRating(feedback.LoserModel, loserRating)
}

// getRatingsForCandidates retrieves ratings for all candidate models
func (e *EloSelector) getRatingsForCandidates(decisionName string, candidates []config.ModelRef) []*ModelRating {
	ratings := make([]*ModelRating, 0, len(candidates))

	for _, c := range candidates {
		var rating *ModelRating

		// Try category-specific rating first
		if e.config.CategoryWeighted && decisionName != "" {
			rating = e.getCategoryRating(decisionName, c.Model)
		}

		// Fall back to global rating
		if rating == nil {
			rating = e.getGlobalRating(c.Model)
		}

		// Create default rating if not found
		if rating == nil {
			rating = &ModelRating{
				Model:  c.Model,
				Rating: e.config.InitialRating,
			}
		}

		ratings = append(ratings, rating)
	}

	return ratings
}

// getGlobalRating retrieves the global rating for a model
func (e *EloSelector) getGlobalRating(model string) *ModelRating {
	e.globalMu.RLock()
	defer e.globalMu.RUnlock()
	return e.globalRatings[model]
}

// setGlobalRating sets the global rating for a model
func (e *EloSelector) setGlobalRating(model string, rating *ModelRating) {
	e.globalMu.Lock()
	defer e.globalMu.Unlock()
	e.globalRatings[model] = rating
}

// getCategoryRating retrieves the category-specific rating for a model
func (e *EloSelector) getCategoryRating(category, model string) *ModelRating {
	e.categoryMu.RLock()
	defer e.categoryMu.RUnlock()
	if catRatings, ok := e.categoryRatings[category]; ok {
		return catRatings[model]
	}
	return nil
}

// setCategoryRating sets the category-specific rating for a model
func (e *EloSelector) setCategoryRating(category, model string, rating *ModelRating) {
	e.categoryMu.Lock()
	defer e.categoryMu.Unlock()
	if e.categoryRatings[category] == nil {
		e.categoryRatings[category] = make(map[string]*ModelRating)
	}
	e.categoryRatings[category][model] = rating
}

// applyCostAdjustment adjusts scores based on model costs
func (e *EloSelector) applyCostAdjustment(scores map[string]float64, costWeight float64) {
	e.costMu.RLock()
	defer e.costMu.RUnlock()

	if len(e.modelCosts) == 0 {
		return
	}

	// Find min and max costs for normalization
	minCost, maxCost := math.MaxFloat64, 0.0
	for model := range scores {
		if cost, ok := e.modelCosts[model]; ok {
			if cost < minCost {
				minCost = cost
			}
			if cost > maxCost {
				maxCost = cost
			}
		}
	}

	if maxCost == minCost {
		return // All same cost, no adjustment needed
	}

	// Adjust scores: cheaper models get bonus
	for model := range scores {
		if cost, ok := e.modelCosts[model]; ok {
			// Normalize cost to 0-1 (0 = cheapest, 1 = most expensive)
			normalizedCost := (cost - minCost) / (maxCost - minCost)
			// Cost penalty: cheaper models get higher bonus
			costBonus := (1.0 - normalizedCost) * costWeight * e.config.CostScalingFactor
			scores[model] *= (1.0 + costBonus)
		}
	}
}

// calculateConfidence returns confidence based on rating stability
func (e *EloSelector) calculateConfidence(rating *ModelRating) float64 {
	if rating == nil {
		return 0.5
	}

	// Confidence increases with more comparisons
	// Sigmoid function: 1 / (1 + e^(-k*(x-threshold)))
	k := 0.2 // Steepness
	threshold := float64(e.config.MinComparisons)
	confidence := 1.0 / (1.0 + math.Exp(-k*(float64(rating.Comparisons)-threshold)))

	return confidence
}

// GetLeaderboard returns models sorted by rating (for debugging/monitoring)
func (e *EloSelector) GetLeaderboard(category string) []*ModelRating {
	var ratings []*ModelRating

	if category != "" && e.config.CategoryWeighted {
		e.categoryMu.RLock()
		if catRatings, ok := e.categoryRatings[category]; ok {
			for _, r := range catRatings {
				ratings = append(ratings, r)
			}
		}
		e.categoryMu.RUnlock()
	} else {
		e.globalMu.RLock()
		for _, r := range e.globalRatings {
			ratings = append(ratings, r)
		}
		e.globalMu.RUnlock()
	}

	// Sort by rating descending
	sort.Slice(ratings, func(i, j int) bool {
		return ratings[i].Rating > ratings[j].Rating
	})

	return ratings
}
