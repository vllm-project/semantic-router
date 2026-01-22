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
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// HybridConfig configures the Hybrid selector that combines multiple methods
// Based on arXiv:2404.14618 - Hybrid LLM: Cost-Efficient Quality-Aware Query Routing
type HybridConfig struct {
	// EloWeight is the weight for Elo rating contribution (0-1)
	EloWeight float64 `yaml:"elo_weight"`

	// RouterDCWeight is the weight for embedding similarity contribution (0-1)
	RouterDCWeight float64 `yaml:"router_dc_weight"`

	// AutoMixWeight is the weight for POMDP value contribution (0-1)
	AutoMixWeight float64 `yaml:"automix_weight"`

	// CostWeight is the weight for cost consideration (0-1)
	CostWeight float64 `yaml:"cost_weight"`

	// QualityGapThreshold triggers escalation to larger models
	QualityGapThreshold float64 `yaml:"quality_gap_threshold"`

	// UseMLP enables MLP-based quality gap prediction (advanced)
	UseMLP bool `yaml:"use_mlp"`

	// NormalizeScores normalizes component scores before combination
	NormalizeScores bool `yaml:"normalize_scores"`
}

// DefaultHybridConfig returns the default Hybrid configuration
func DefaultHybridConfig() *HybridConfig {
	return &HybridConfig{
		EloWeight:           0.3,
		RouterDCWeight:      0.3,
		AutoMixWeight:       0.2,
		CostWeight:          0.2,
		QualityGapThreshold: 0.1,
		UseMLP:              false,
		NormalizeScores:     true,
	}
}

// HybridSelector combines multiple selection methods for robust routing
// It uses weighted combination of Elo ratings, embedding similarity,
// and POMDP values, with optional cost-aware optimization.
type HybridSelector struct {
	config *HybridConfig

	// Component selectors
	eloSelector      *EloSelector
	routerDCSelector *RouterDCSelector
	autoMixSelector  *AutoMixSelector

	// Model costs for cost-aware selection
	modelCosts map[string]float64
}

// NewHybridSelector creates a new Hybrid selector
func NewHybridSelector(cfg *HybridConfig) *HybridSelector {
	if cfg == nil {
		cfg = DefaultHybridConfig()
	}

	return &HybridSelector{
		config:           cfg,
		eloSelector:      NewEloSelector(DefaultEloConfig()),
		routerDCSelector: NewRouterDCSelector(DefaultRouterDCConfig()),
		autoMixSelector:  NewAutoMixSelector(DefaultAutoMixConfig()),
		modelCosts:       make(map[string]float64),
	}
}

// NewHybridSelectorWithComponents creates a Hybrid selector with custom components
func NewHybridSelectorWithComponents(
	cfg *HybridConfig,
	elo *EloSelector,
	routerDC *RouterDCSelector,
	autoMix *AutoMixSelector,
) *HybridSelector {
	if cfg == nil {
		cfg = DefaultHybridConfig()
	}

	return &HybridSelector{
		config:           cfg,
		eloSelector:      elo,
		routerDCSelector: routerDC,
		autoMixSelector:  autoMix,
		modelCosts:       make(map[string]float64),
	}
}

// Method returns the selection method type
func (h *HybridSelector) Method() SelectionMethod {
	return MethodHybrid
}

// SetEloSelector sets the Elo component
func (h *HybridSelector) SetEloSelector(elo *EloSelector) {
	h.eloSelector = elo
}

// SetRouterDCSelector sets the RouterDC component
func (h *HybridSelector) SetRouterDCSelector(routerDC *RouterDCSelector) {
	h.routerDCSelector = routerDC
}

// SetAutoMixSelector sets the AutoMix component
func (h *HybridSelector) SetAutoMixSelector(autoMix *AutoMixSelector) {
	h.autoMixSelector = autoMix
}

// SetModelCost sets the cost for a model
func (h *HybridSelector) SetModelCost(model string, cost float64) {
	h.modelCosts[model] = cost
}

// Select chooses the best model by combining multiple selection methods
func (h *HybridSelector) Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
	if len(selCtx.CandidateModels) == 0 {
		return nil, fmt.Errorf("no candidate models provided")
	}

	// Collect scores from each component
	componentScores := make(map[string]map[string]float64)
	var componentResults []*SelectionResult

	// Get Elo scores
	if h.eloSelector != nil && h.config.EloWeight > 0 {
		result, err := h.eloSelector.Select(ctx, selCtx)
		if err == nil && result != nil {
			componentScores["elo"] = result.AllScores
			componentResults = append(componentResults, result)
		}
	}

	// Get RouterDC scores
	if h.routerDCSelector != nil && h.config.RouterDCWeight > 0 {
		result, err := h.routerDCSelector.Select(ctx, selCtx)
		if err == nil && result != nil {
			componentScores["router_dc"] = result.AllScores
			componentResults = append(componentResults, result)
		}
	}

	// Get AutoMix scores
	if h.autoMixSelector != nil && h.config.AutoMixWeight > 0 {
		result, err := h.autoMixSelector.Select(ctx, selCtx)
		if err == nil && result != nil {
			componentScores["automix"] = result.AllScores
			componentResults = append(componentResults, result)
		}
	}

	// Normalize scores if enabled
	if h.config.NormalizeScores {
		for component, scores := range componentScores {
			componentScores[component] = h.normalizeScores(scores)
		}
	}

	// Combine scores with weights
	combinedScores := h.combineScores(componentScores, selCtx.CandidateModels)

	// Apply cost adjustment
	if h.config.CostWeight > 0 {
		h.applyCostAdjustment(combinedScores, selCtx.CostWeight)
	}

	logging.Infof("[HybridSelector] Combining scores (weights: elo=%.2f, dc=%.2f, am=%.2f, cost=%.2f):",
		h.config.EloWeight, h.config.RouterDCWeight, h.config.AutoMixWeight, h.config.CostWeight)
	for _, model := range selCtx.CandidateModels {
		var eloScore, dcScore, amScore float64
		if scores, ok := componentScores["elo"]; ok {
			eloScore = scores[model.Model]
		}
		if scores, ok := componentScores["router_dc"]; ok {
			dcScore = scores[model.Model]
		}
		if scores, ok := componentScores["automix"]; ok {
			amScore = scores[model.Model]
		}
		logging.Infof("[HybridSelector]   %s: elo=%.4f, dc=%.4f, am=%.4f → combined=%.4f",
			model.Model, eloScore, dcScore, amScore, combinedScores[model.Model])
	}

	// Find best model
	var bestModel *config.ModelRef
	var bestScore float64

	for i := range selCtx.CandidateModels {
		model := &selCtx.CandidateModels[i]
		score := combinedScores[model.Model]

		if score > bestScore || bestModel == nil {
			bestScore = score
			bestModel = model
		}
	}

	if bestModel == nil {
		return nil, fmt.Errorf("could not select a model")
	}

	// Calculate confidence from component agreement
	confidence := h.calculateConfidence(componentResults, bestModel.Model)

	// Record component agreement metric for evolution tracking
	if len(componentResults) > 1 {
		agreementRatio := float64(0)
		for _, r := range componentResults {
			if r.SelectedModel == bestModel.Model {
				agreementRatio++
			}
		}
		agreementRatio /= float64(len(componentResults))
		RecordComponentAgreement(agreementRatio)
	}

	// Build reasoning
	reasoning := h.buildReasoning(componentScores, bestModel.Model)

	logging.Infof("[HybridSelector] Selected model %s (score=%.4f, confidence=%.2f, components=%d)",
		bestModel.Model, bestScore, confidence, len(componentResults))

	return &SelectionResult{
		SelectedModel: bestModel.Model,
		LoRAName:      bestModel.LoRAName,
		Score:         bestScore,
		Confidence:    confidence,
		Method:        MethodHybrid,
		Reasoning:     reasoning,
		AllScores:     combinedScores,
	}, nil
}

// UpdateFeedback propagates feedback to all component selectors
func (h *HybridSelector) UpdateFeedback(ctx context.Context, feedback *Feedback) error {
	var errs []error

	if h.eloSelector != nil {
		if err := h.eloSelector.UpdateFeedback(ctx, feedback); err != nil {
			errs = append(errs, fmt.Errorf("elo: %w", err))
		}
	}

	if h.routerDCSelector != nil {
		if err := h.routerDCSelector.UpdateFeedback(ctx, feedback); err != nil {
			errs = append(errs, fmt.Errorf("router_dc: %w", err))
		}
	}

	if h.autoMixSelector != nil {
		if err := h.autoMixSelector.UpdateFeedback(ctx, feedback); err != nil {
			errs = append(errs, fmt.Errorf("automix: %w", err))
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("feedback update errors: %v", errs)
	}

	logging.Debugf("[HybridSelector] Propagated feedback to %d components", 3)
	return nil
}

// combineScores combines scores from all components with weights
func (h *HybridSelector) combineScores(componentScores map[string]map[string]float64, candidates []config.ModelRef) map[string]float64 {
	result := make(map[string]float64)

	// Initialize with zeros
	for _, c := range candidates {
		result[c.Model] = 0.0
	}

	// Weight mapping
	weights := map[string]float64{
		"elo":       h.config.EloWeight,
		"router_dc": h.config.RouterDCWeight,
		"automix":   h.config.AutoMixWeight,
	}

	// Calculate total weight for normalization
	totalWeight := 0.0
	for component, scores := range componentScores {
		if len(scores) > 0 {
			totalWeight += weights[component]
		}
	}

	if totalWeight == 0 {
		// No component scores available, use uniform
		for model := range result {
			result[model] = 1.0 / float64(len(candidates))
		}
		return result
	}

	// Weighted combination
	for component, scores := range componentScores {
		weight := weights[component]
		for model, score := range scores {
			result[model] += (weight / totalWeight) * score
		}
	}

	return result
}

// normalizeScores normalizes scores to [0, 1] range using min-max normalization
func (h *HybridSelector) normalizeScores(scores map[string]float64) map[string]float64 {
	if len(scores) == 0 {
		return scores
	}

	minScore := math.Inf(1)
	maxScore := math.Inf(-1)

	for _, s := range scores {
		if s < minScore {
			minScore = s
		}
		if s > maxScore {
			maxScore = s
		}
	}

	// Avoid division by zero
	if maxScore == minScore {
		result := make(map[string]float64)
		for model := range scores {
			result[model] = 0.5
		}
		return result
	}

	result := make(map[string]float64)
	for model, s := range scores {
		result[model] = (s - minScore) / (maxScore - minScore)
	}

	return result
}

// applyCostAdjustment applies cost-based score adjustment
func (h *HybridSelector) applyCostAdjustment(scores map[string]float64, costWeight float64) {
	if len(h.modelCosts) == 0 || costWeight <= 0 {
		return
	}

	// Find min and max costs
	minCost, maxCost := math.MaxFloat64, 0.0
	for model := range scores {
		if cost, ok := h.modelCosts[model]; ok {
			if cost < minCost {
				minCost = cost
			}
			if cost > maxCost {
				maxCost = cost
			}
		}
	}

	if maxCost == minCost {
		return
	}

	// Adjust scores: cheaper models get bonus
	for model := range scores {
		if cost, ok := h.modelCosts[model]; ok {
			normalizedCost := (cost - minCost) / (maxCost - minCost)
			costBonus := (1.0 - normalizedCost) * costWeight * h.config.CostWeight
			scores[model] *= (1.0 + costBonus)
		}
	}
}

// calculateConfidence calculates confidence based on component agreement
func (h *HybridSelector) calculateConfidence(results []*SelectionResult, selectedModel string) float64 {
	if len(results) == 0 {
		return 0.5
	}

	// Count how many components agree on the selected model
	agreements := 0
	totalConfidence := 0.0

	for _, r := range results {
		if r.SelectedModel == selectedModel {
			agreements++
		}
		totalConfidence += r.Confidence
	}

	// Agreement ratio
	agreementRatio := float64(agreements) / float64(len(results))

	// Average component confidence
	avgConfidence := totalConfidence / float64(len(results))

	// Combine: higher agreement and confidence = more confident
	return (agreementRatio + avgConfidence) / 2.0
}

// buildReasoning creates a human-readable explanation
func (h *HybridSelector) buildReasoning(componentScores map[string]map[string]float64, selectedModel string) string {
	parts := []string{}

	if scores, ok := componentScores["elo"]; ok {
		if score, ok := scores[selectedModel]; ok {
			parts = append(parts, fmt.Sprintf("Elo=%.3f", score))
		}
	}

	if scores, ok := componentScores["router_dc"]; ok {
		if score, ok := scores[selectedModel]; ok {
			parts = append(parts, fmt.Sprintf("RouterDC=%.3f", score))
		}
	}

	if scores, ok := componentScores["automix"]; ok {
		if score, ok := scores[selectedModel]; ok {
			parts = append(parts, fmt.Sprintf("AutoMix=%.3f", score))
		}
	}

	weightsStr := fmt.Sprintf("weights=[elo:%.2f, dc:%.2f, am:%.2f, cost:%.2f]",
		h.config.EloWeight, h.config.RouterDCWeight, h.config.AutoMixWeight, h.config.CostWeight)

	if len(parts) > 0 {
		return fmt.Sprintf("Hybrid combination: [%s], %s", strings.Join(parts, " "), weightsStr)
	}
	return fmt.Sprintf("Hybrid selection with %s", weightsStr)
}

// InitializeFromConfig initializes all component selectors from configuration
func (h *HybridSelector) InitializeFromConfig(modelConfig map[string]config.ModelParams, categories []config.Category) {
	if h.eloSelector != nil {
		h.eloSelector.InitializeFromConfig(modelConfig, categories)
	}

	if h.autoMixSelector != nil {
		h.autoMixSelector.InitializeFromConfig(modelConfig)
	}

	// Set costs from config
	for model, params := range modelConfig {
		if params.Pricing.PromptPer1M > 0 {
			h.modelCosts[model] = params.Pricing.PromptPer1M
		}
	}

	logging.Infof("[HybridSelector] Initialized from config with %d models", len(modelConfig))
}
