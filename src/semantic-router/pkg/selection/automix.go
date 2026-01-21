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
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// AutoMixConfig configures the AutoMix POMDP-based selector
// Based on arXiv:2310.12963 - Automatically Mixing Language Models
//
// NOTE: This is a PRE-SELECTION implementation of AutoMix concepts.
// The original paper describes a CASCADED EXECUTION approach where:
//  1. Start with the smallest/cheapest model
//  2. Execute the query and perform self-verification
//  3. If confidence is below threshold, escalate to a larger model
//  4. Repeat until confidence is acceptable or max escalations reached
//
// Our implementation applies AutoMix PRINCIPLES to pre-selection:
//   - We estimate which model is most likely to succeed based on learned capabilities
//   - We optimize the cost-quality tradeoff using POMDP value functions
//   - Feedback updates improve the selection over time
//
// For true cascaded execution with self-verification, the looper package
// would need to be extended to support multi-stage inference with confidence
// checks between stages. This is planned for a future enhancement.
type AutoMixConfig struct {
	// VerificationThreshold is the confidence threshold for self-verification
	// Responses below this threshold trigger escalation (default: 0.7)
	VerificationThreshold float64 `yaml:"verification_threshold"`

	// MaxEscalations limits how many times to escalate (default: 2)
	MaxEscalations int `yaml:"max_escalations"`

	// CostAwareRouting enables cost-quality tradeoff optimization
	CostAwareRouting bool `yaml:"cost_aware_routing"`

	// CostQualityTradeoff controls balance (0 = pure quality, 1 = pure cost)
	CostQualityTradeoff float64 `yaml:"cost_quality_tradeoff"`

	// DiscountFactor for POMDP value iteration (gamma, default: 0.95)
	DiscountFactor float64 `yaml:"discount_factor"`

	// UseLogprobVerification uses logprobs for confidence estimation
	UseLogprobVerification bool `yaml:"use_logprob_verification"`
}

// DefaultAutoMixConfig returns the default AutoMix configuration
func DefaultAutoMixConfig() *AutoMixConfig {
	return &AutoMixConfig{
		VerificationThreshold:  0.7,
		MaxEscalations:         2,
		CostAwareRouting:       true,
		CostQualityTradeoff:    0.3,
		DiscountFactor:         0.95,
		UseLogprobVerification: true,
	}
}

// ModelCapability stores learned model capabilities for POMDP states
type ModelCapability struct {
	Model             string  `json:"model"`
	ParamSize         float64 `json:"param_size"`          // Model size in billions of parameters
	Cost              float64 `json:"cost"`                // Cost per 1M tokens
	AvgQuality        float64 `json:"avg_quality"`         // Learned average quality score
	VerificationProb  float64 `json:"verification_prob"`   // Probability of passing self-verification
	EscalationReward  float64 `json:"escalation_reward"`   // Expected reward from escalation
	QuerySuccessCount int     `json:"query_success_count"` // Successful queries
	QueryTotalCount   int     `json:"query_total_count"`   // Total queries
}

// AutoMixSelector implements POMDP-based cascaded model selection
// The algorithm routes to smaller models first and escalates based on
// self-verification confidence, optimizing the cost-quality tradeoff.
type AutoMixSelector struct {
	config *AutoMixConfig

	// Model capabilities indexed by model name
	capabilities map[string]*ModelCapability
	capMu        sync.RWMutex

	// POMDP value function V(s) for each model
	valueFunction map[string]float64
	valueMu       sync.RWMutex

	// Transition probabilities P(s'|s,a) for escalation decisions
	transitionProbs map[string]map[string]float64
}

// NewAutoMixSelector creates a new AutoMix-based selector
func NewAutoMixSelector(cfg *AutoMixConfig) *AutoMixSelector {
	if cfg == nil {
		cfg = DefaultAutoMixConfig()
	}
	return &AutoMixSelector{
		config:          cfg,
		capabilities:    make(map[string]*ModelCapability),
		valueFunction:   make(map[string]float64),
		transitionProbs: make(map[string]map[string]float64),
	}
}

// Method returns the selection method type
func (a *AutoMixSelector) Method() SelectionMethod {
	return MethodAutoMix
}

// InitializeFromConfig sets up model capabilities from configuration
func (a *AutoMixSelector) InitializeFromConfig(modelConfig map[string]config.ModelParams) {
	a.capMu.Lock()
	defer a.capMu.Unlock()

	for model, params := range modelConfig {
		// Use configured quality score if available, otherwise default to 0.8
		qualityScore := params.QualityScore
		if qualityScore <= 0 || qualityScore > 1.0 {
			qualityScore = 0.8 // Default quality estimate
		}

		cap := &ModelCapability{
			Model:            model,
			Cost:             params.Pricing.PromptPer1M,
			AvgQuality:       qualityScore,
			VerificationProb: 0.7,                        // Default verification probability
			ParamSize:        a.estimateParamSize(model), // Estimate from model name
		}
		a.capabilities[model] = cap

		// Initialize value function (higher for larger/better models)
		a.valueMu.Lock()
		a.valueFunction[model] = cap.ParamSize / 100.0 // Normalize
		a.valueMu.Unlock()
	}

	logging.Infof("[AutoMix] Initialized capabilities for %d models", len(a.capabilities))
}

// Select chooses the best model using POMDP-based cost-quality optimization
func (a *AutoMixSelector) Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
	if len(selCtx.CandidateModels) == 0 {
		return nil, fmt.Errorf("no candidate models provided")
	}

	// Sort candidates by cost (cheaper first for cascaded routing)
	sortedCandidates := a.sortByCost(selCtx.CandidateModels)

	// Calculate expected value for each model using POMDP
	allScores := make(map[string]float64)
	a.capMu.RLock()
	a.valueMu.RLock()
	defer a.capMu.RUnlock()
	defer a.valueMu.RUnlock()

	logging.Infof("[AutoMix] Evaluating %d candidates (tradeoff=%.2f):",
		len(sortedCandidates), a.config.CostQualityTradeoff)
	for _, model := range sortedCandidates {
		modelName := model.Model
		score := a.computeExpectedValue(modelName, selCtx)
		allScores[modelName] = score
		if cap, ok := a.capabilities[modelName]; ok {
			logging.Infof("[AutoMix]   %s: cost=$%.2f, quality=%.2f, value=%.4f",
				modelName, cap.Cost, cap.AvgQuality, score)
		} else {
			logging.Infof("[AutoMix]   %s: value=%.4f (no capability data)", modelName, score)
		}
	}

	// Find optimal starting model (not necessarily the best, but best value)
	var selectedModel *config.ModelRef
	var selectedScore float64
	var reasoning string

	if a.config.CostAwareRouting {
		// Cost-aware: select model with best value considering cost
		selectedModel, selectedScore, reasoning = a.selectCostAware(sortedCandidates, allScores, selCtx)
	} else {
		// Quality-only: select model with highest expected quality
		selectedModel, selectedScore, reasoning = a.selectQualityOnly(sortedCandidates, allScores)
	}

	if selectedModel == nil {
		return nil, fmt.Errorf("could not select a model")
	}

	// Calculate confidence based on verification probability
	confidence := a.getVerificationProbability(selectedModel.Model)

	// Record AutoMix-specific metrics for evolution tracking
	for _, model := range sortedCandidates {
		if cap, ok := a.capabilities[model.Model]; ok {
			RecordAutoMixCapability(model.Model, cap.VerificationProb, cap.AvgQuality,
				cap.QuerySuccessCount, cap.QueryTotalCount)
		}
	}

	logging.Infof("[AutoMix] Selected model %s (score=%.4f, confidence=%.2f, cost-aware=%v)",
		selectedModel.Model, selectedScore, confidence, a.config.CostAwareRouting)

	return &SelectionResult{
		SelectedModel: selectedModel.Model,
		LoRAName:      selectedModel.LoRAName,
		Score:         selectedScore,
		Confidence:    confidence,
		Method:        MethodAutoMix,
		Reasoning:     reasoning,
		AllScores:     allScores,
	}, nil
}

// UpdateFeedback updates POMDP model based on verification outcomes
func (a *AutoMixSelector) UpdateFeedback(ctx context.Context, feedback *Feedback) error {
	if feedback.WinnerModel == "" {
		return fmt.Errorf("winner model is required")
	}

	a.capMu.Lock()
	defer a.capMu.Unlock()

	// Update winner model capabilities
	if cap, ok := a.capabilities[feedback.WinnerModel]; ok {
		cap.QuerySuccessCount++
		cap.QueryTotalCount++

		// Update verification probability with exponential moving average
		alpha := 0.1 // Learning rate
		cap.VerificationProb = cap.VerificationProb*(1-alpha) + 1.0*alpha
		cap.AvgQuality = cap.AvgQuality*(1-alpha) + 1.0*alpha

		logging.Debugf("[AutoMix] Updated winner %s: verification_prob=%.3f, quality=%.3f",
			feedback.WinnerModel, cap.VerificationProb, cap.AvgQuality)
	}

	// Update loser model capabilities (if this was a comparison)
	if feedback.LoserModel != "" && !feedback.Tie {
		if cap, ok := a.capabilities[feedback.LoserModel]; ok {
			cap.QueryTotalCount++

			alpha := 0.1
			cap.VerificationProb = cap.VerificationProb*(1-alpha) + 0.0*alpha
			cap.AvgQuality = cap.AvgQuality*(1-alpha) + 0.0*alpha

			logging.Debugf("[AutoMix] Updated loser %s: verification_prob=%.3f, quality=%.3f",
				feedback.LoserModel, cap.VerificationProb, cap.AvgQuality)
		}
	}

	// Run value iteration to update POMDP values (we already hold capMu.Lock)
	a.updateValueFunctionLocked()

	// Record updated metrics after feedback
	if cap, ok := a.capabilities[feedback.WinnerModel]; ok {
		RecordAutoMixCapability(feedback.WinnerModel, cap.VerificationProb, cap.AvgQuality,
			cap.QuerySuccessCount, cap.QueryTotalCount)
	}
	if feedback.LoserModel != "" {
		if cap, ok := a.capabilities[feedback.LoserModel]; ok {
			RecordAutoMixCapability(feedback.LoserModel, cap.VerificationProb, cap.AvgQuality,
				cap.QuerySuccessCount, cap.QueryTotalCount)
		}
	}

	return nil
}

// computeExpectedValue calculates the expected value of using a model
// V(model) = R(model) + γ * E[V(s') | escalation possible]
func (a *AutoMixSelector) computeExpectedValue(model string, selCtx *SelectionContext) float64 {
	cap := a.capabilities[model]
	if cap == nil {
		return 0.5 // Default value for unknown models
	}

	// Immediate reward: quality
	quality := cap.AvgQuality

	// Cost penalty (normalized)
	costPenalty := 0.0
	if a.config.CostAwareRouting && cap.Cost > 0 {
		// Normalize cost to 0-1 range (assuming max cost is ~$10/1M tokens)
		normalizedCost := cap.Cost / 10.0
		costPenalty = normalizedCost * a.config.CostQualityTradeoff
	}

	// Expected value from potential escalation
	verificationProb := cap.VerificationProb
	escalationValue := 0.0

	if verificationProb < a.config.VerificationThreshold {
		// Model likely needs escalation - consider value of larger models
		escalationValue = a.config.DiscountFactor * cap.EscalationReward
	}

	// Combine: value = quality - cost_penalty + escalation_value
	value := quality - costPenalty + escalationValue*(1-verificationProb)

	return value
}

// selectCostAware selects model optimizing cost-quality tradeoff
func (a *AutoMixSelector) selectCostAware(candidates []config.ModelRef, scores map[string]float64, selCtx *SelectionContext) (*config.ModelRef, float64, string) {
	var bestModel *config.ModelRef
	bestValue := math.Inf(-1)

	for i := range candidates {
		model := &candidates[i]
		score := scores[model.Model]

		cap := a.capabilities[model.Model]
		if cap == nil {
			continue
		}

		// Calculate cost-adjusted value
		costFactor := 1.0
		if cap.Cost > 0 {
			// Prefer cheaper models when cost weight is high
			costFactor = 1.0 / (1.0 + cap.Cost*selCtx.CostWeight)
		}

		value := score * costFactor

		// Prefer models above verification threshold
		if cap.VerificationProb >= a.config.VerificationThreshold {
			value *= 1.1 // 10% bonus for likely-to-succeed models
		}

		if value > bestValue {
			bestValue = value
			bestModel = model
		}
	}

	if bestModel == nil && len(candidates) > 0 {
		bestModel = &candidates[0]
		bestValue = scores[bestModel.Model]
	}

	reasoning := fmt.Sprintf("Cost-aware POMDP selection (tradeoff=%.2f, discount=%.2f)",
		a.config.CostQualityTradeoff, a.config.DiscountFactor)

	return bestModel, bestValue, reasoning
}

// selectQualityOnly selects the highest quality model regardless of cost
func (a *AutoMixSelector) selectQualityOnly(candidates []config.ModelRef, scores map[string]float64) (*config.ModelRef, float64, string) {
	var bestModel *config.ModelRef
	var bestScore float64

	for i := range candidates {
		model := &candidates[i]
		score := scores[model.Model]

		if score > bestScore || bestModel == nil {
			bestScore = score
			bestModel = model
		}
	}

	reasoning := fmt.Sprintf("Quality-only POMDP selection (threshold=%.2f)",
		a.config.VerificationThreshold)

	return bestModel, bestScore, reasoning
}

// sortByCost sorts models by cost (ascending)
func (a *AutoMixSelector) sortByCost(models []config.ModelRef) []config.ModelRef {
	sorted := make([]config.ModelRef, len(models))
	copy(sorted, models)

	a.capMu.RLock()
	defer a.capMu.RUnlock()

	sort.Slice(sorted, func(i, j int) bool {
		capI := a.capabilities[sorted[i].Model]
		capJ := a.capabilities[sorted[j].Model]

		costI := 0.0
		costJ := 0.0
		if capI != nil {
			costI = capI.Cost
		}
		if capJ != nil {
			costJ = capJ.Cost
		}

		return costI < costJ
	})

	return sorted
}

// getVerificationProbability returns the learned verification probability
func (a *AutoMixSelector) getVerificationProbability(model string) float64 {
	a.capMu.RLock()
	defer a.capMu.RUnlock()

	if cap, ok := a.capabilities[model]; ok {
		return cap.VerificationProb
	}
	return 0.7 // Default
}

// updateValueFunctionLocked performs one iteration of POMDP value update
// NOTE: Caller MUST hold capMu lock (read or write) before calling this
func (a *AutoMixSelector) updateValueFunctionLocked() {
	a.valueMu.Lock()
	defer a.valueMu.Unlock()

	// Simple value iteration: V(s) = R(s) + γ * max_a E[V(s')]
	for model, cap := range a.capabilities {
		// Current reward
		reward := cap.AvgQuality

		// Expected future value (from escalation)
		futureValue := 0.0
		if cap.VerificationProb < a.config.VerificationThreshold {
			// Calculate expected value of escalation
			for otherModel, otherCap := range a.capabilities {
				if otherCap.ParamSize > cap.ParamSize {
					// Larger model could be escalation target
					transitionProb := (1 - cap.VerificationProb) * 0.5 // Simplified
					futureValue += transitionProb * a.valueFunction[otherModel]
				}
			}
		}

		// Update value
		a.valueFunction[model] = reward + a.config.DiscountFactor*futureValue

		// Update escalation reward for capability
		cap.EscalationReward = futureValue
	}
}

// estimateParamSize estimates model size from name
func (a *AutoMixSelector) estimateParamSize(model string) float64 {
	// Extract size from common naming patterns (7b, 13b, 70b, etc.)
	sizes := []struct {
		pattern string
		size    float64
	}{
		{"405b", 405.0},
		{"70b", 70.0},
		{"72b", 72.0},
		{"34b", 34.0},
		{"32b", 32.0},
		{"14b", 14.0},
		{"13b", 13.0},
		{"8b", 8.0},
		{"7b", 7.0},
		{"3b", 3.0},
		{"1.8b", 1.8},
		{"1.5b", 1.5},
		{"0.5b", 0.5},
	}

	modelLower := strings.ToLower(model)
	for _, s := range sizes {
		if strings.Contains(modelLower, s.pattern) {
			return s.size
		}
	}

	return 7.0 // Default assumption
}

// GetCapabilities returns all model capabilities (for debugging)
func (a *AutoMixSelector) GetCapabilities() map[string]*ModelCapability {
	a.capMu.RLock()
	defer a.capMu.RUnlock()

	result := make(map[string]*ModelCapability)
	for k, v := range a.capabilities {
		capCopy := *v
		result[k] = &capCopy
	}
	return result
}

// SetCapability directly sets a model's capability
func (a *AutoMixSelector) SetCapability(model string, cap *ModelCapability) {
	a.capMu.Lock()
	defer a.capMu.Unlock()
	a.capabilities[model] = cap
}
