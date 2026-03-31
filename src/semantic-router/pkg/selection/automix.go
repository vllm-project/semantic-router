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
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// AutoMixConfig configures the AutoMix POMDP-based selector
// Based on arXiv:2310.12963 - Automatically Mixing Language Models (NeurIPS 2024)
//
// AutoMix implements a 3-step cascaded routing approach:
//  1. Generate: Start with the smallest/cheapest model
//  2. Self-Verify: Same model verifies answer via entailment check
//  3. Route: If confidence below threshold, escalate to larger model
//
// Key technical contributions from the paper:
//   - Few-shot self-verification framed as entailment
//   - POMDP-based router with belief states over model performance
//   - Particle filtering for belief updates
//   - IBC (Incremental Benefit per Cost) metric for evaluation
//
// This implementation supports both:
//   - Pre-selection mode: Estimate best model upfront
//   - Cascaded mode: Verify and escalate (requires looper integration)
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

	// EnableSelfVerification enables LLM-based self-verification (paper method)
	// When true, uses entailment-based verification prompt
	EnableSelfVerification bool `yaml:"enable_self_verification"`

	// VerificationSamples is the number of samples for confidence estimation (k in paper)
	VerificationSamples int `yaml:"verification_samples"`

	// VerificationTemperature for sampling during self-verification
	VerificationTemperature float64 `yaml:"verification_temperature"`

	// UsePOMDPRouter enables full POMDP-based routing with belief updates
	// When false, uses simpler threshold-based routing
	UsePOMDPRouter bool `yaml:"use_pomdp_router"`

	// BeliefParticles is the number of particles for belief representation
	BeliefParticles int `yaml:"belief_particles"`

	// CostLambda is the tradeoff parameter in R = Performance - λ × Cost
	CostLambda float64 `yaml:"cost_lambda"`

	// Self-Verification Server Configuration

	// VerifierServerURL is the URL of the AutoMix self-verification server
	// This server performs entailment-based answer verification
	VerifierServerURL string `yaml:"verifier_server_url"`

	// EnableCascade enables the full cascade execution mode
	// When true, Select() returns the smallest model first, and the cascade
	// is managed via SelectWithVerification() which includes verification
	EnableCascade bool `yaml:"enable_cascade"`
}

// DefaultAutoMixConfig returns the default AutoMix configuration
func DefaultAutoMixConfig() *AutoMixConfig {
	return &AutoMixConfig{
		VerificationThreshold:   0.7,
		MaxEscalations:          2,
		CostAwareRouting:        true,
		CostQualityTradeoff:     0.3,
		DiscountFactor:          0.95,
		UseLogprobVerification:  true,
		EnableSelfVerification:  false, // Requires LLM endpoint
		VerificationSamples:     5,     // k=5 as in paper
		VerificationTemperature: 0.7,
		UsePOMDPRouter:          true,
		BeliefParticles:         100,
		CostLambda:              0.5, // Balance performance and cost
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

	// AdaOps POMDP solver with particle filtering (full paper implementation)
	adaOpsSolver *AdaOpsSolver

	// Self-verification client for LLM-based verification
	// Requires external server: src/training/rl_model_selection/automix_verifier.py
	verifierClient *AutoMixVerifierClient
}

// NewAutoMixSelector creates a new AutoMix-based selector
func NewAutoMixSelector(cfg *AutoMixConfig) *AutoMixSelector {
	if cfg == nil {
		cfg = DefaultAutoMixConfig()
	}

	selector := &AutoMixSelector{
		config:          cfg,
		capabilities:    make(map[string]*ModelCapability),
		valueFunction:   make(map[string]float64),
		transitionProbs: make(map[string]map[string]float64),
	}

	// Initialize AdaOps solver if POMDP routing is enabled
	if cfg.UsePOMDPRouter {
		selector.adaOpsSolver = NewAdaOpsSolver(
			cfg.BeliefParticles,
			cfg.CostLambda,
			cfg.DiscountFactor,
		)
		logging.Infof("[AutoMix] Initialized AdaOps POMDP solver with %d particles", cfg.BeliefParticles)
	}

	// Initialize self-verification client if configured
	// Requires external server: src/training/rl_model_selection/automix_verifier.py
	if cfg.EnableSelfVerification && cfg.VerifierServerURL != "" {
		selector.verifierClient = NewAutoMixVerifierClient(cfg.VerifierServerURL)
		logging.Infof("[AutoMix] Self-verification enabled, server: %s", cfg.VerifierServerURL)

		// Verify connectivity
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := selector.verifierClient.HealthCheck(ctx); err != nil {
			logging.Warnf("[AutoMix] Verifier server not reachable: %v (will retry on use)", err)
		} else {
			logging.Infof("[AutoMix] Verifier server connected successfully")
		}
	} else if cfg.EnableSelfVerification {
		logging.Warnf("[AutoMix] Self-verification enabled but no server URL configured")
	}

	return selector
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

		paramSize := a.estimateParamSize(model)
		cap := &ModelCapability{
			Model:            model,
			Cost:             params.Pricing.PromptPer1M,
			AvgQuality:       qualityScore,
			VerificationProb: 0.7,       // Default verification probability
			ParamSize:        paramSize, // Estimate from model name
		}
		a.capabilities[model] = cap

		// Register with AdaOps solver
		if a.adaOpsSolver != nil {
			a.adaOpsSolver.RegisterModel(model, params.Pricing.PromptPer1M, paramSize)
		}

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

		// Update AdaOps belief state with observation
		if a.adaOpsSolver != nil {
			a.adaOpsSolver.UpdateBelief(feedback.WinnerModel, 1.0) // Winner = high performance
		}

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

			// Update AdaOps belief state with observation
			if a.adaOpsSolver != nil {
				a.adaOpsSolver.UpdateBelief(feedback.LoserModel, 0.0) // Loser = low performance
			}

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

// VerificationResult contains the result of self-verification
type VerificationResult struct {
	Confidence     float64 `json:"confidence"`
	ShouldEscalate bool    `json:"should_escalate"`
	NextModel      string  `json:"next_model,omitempty"`
	Reasoning      string  `json:"reasoning"`
}

// VerifyAnswer implements the AutoMix self-verification step
// Given a question and answer from a model, this uses entailment-based
// verification to estimate answer confidence and recommend escalation.
// This implements Section 3.2 of arXiv:2310.12963
// Requires external server: src/training/rl_model_selection/automix_verifier.py
func (a *AutoMixSelector) VerifyAnswer(ctx context.Context, question, answer, currentModel string, escalationChain []string) (*VerificationResult, error) {
	if a.verifierClient == nil {
		return nil, fmt.Errorf("verifier client not initialized - set verifier_server_url in config")
	}

	// Call the verification server
	verifyResp, err := a.verifierClient.Verify(ctx, question, answer, "", a.config.VerificationThreshold)
	if err != nil {
		return nil, fmt.Errorf("verification failed: %w", err)
	}

	result := &VerificationResult{
		Confidence:     verifyResp.Confidence,
		ShouldEscalate: verifyResp.ShouldEscalate,
	}

	// Build reasoning
	result.Reasoning = fmt.Sprintf("Self-verification confidence: %.2f (threshold: %.2f). %d/%d samples verified.",
		verifyResp.Confidence, a.config.VerificationThreshold,
		verifyResp.VerifiedCount, verifyResp.TotalSamples)

	// If escalation recommended, find the next model in the chain
	if result.ShouldEscalate && len(escalationChain) > 0 {
		// Find current model in chain
		currentIdx := -1
		for i, m := range escalationChain {
			if m == currentModel {
				currentIdx = i
				break
			}
		}

		// Get next larger model
		if currentIdx >= 0 && currentIdx < len(escalationChain)-1 {
			result.NextModel = escalationChain[currentIdx+1]
			result.Reasoning += fmt.Sprintf(" Recommending escalation to %s.", result.NextModel)
		} else {
			result.ShouldEscalate = false
			result.Reasoning += " No larger model available for escalation."
		}
	}

	logging.Infof("[AutoMix] Verification result: confidence=%.2f, escalate=%v, next=%s",
		result.Confidence, result.ShouldEscalate, result.NextModel)

	// Update model capability based on verification outcome
	a.capMu.Lock()
	if cap, ok := a.capabilities[currentModel]; ok {
		alpha := 0.1 // Learning rate
		if verifyResp.Confidence >= a.config.VerificationThreshold {
			// Passed verification
			cap.VerificationProb = cap.VerificationProb*(1-alpha) + 1.0*alpha
		} else {
			// Failed verification
			cap.VerificationProb = cap.VerificationProb*(1-alpha) + 0.0*alpha
		}
	}
	a.capMu.Unlock()

	return result, nil
}

// GetEscalationChain returns models sorted by capability (smallest to largest)
// This is the escalation path for cascaded routing
func (a *AutoMixSelector) GetEscalationChain(candidates []config.ModelRef) []string {
	// Sort by param size (smallest first)
	sorted := make([]config.ModelRef, len(candidates))
	copy(sorted, candidates)

	sort.Slice(sorted, func(i, j int) bool {
		sizeI := a.estimateParamSize(sorted[i].Model)
		sizeJ := a.estimateParamSize(sorted[j].Model)
		return sizeI < sizeJ
	})

	chain := make([]string, len(sorted))
	for i, m := range sorted {
		chain[i] = m.Model
	}

	return chain
}

// SelectWithCascadeState stores the state for cascaded execution
type SelectWithCascadeState struct {
	CurrentModel     string              `json:"current_model"`
	EscalationChain  []string            `json:"escalation_chain"`
	EscalationCount  int                 `json:"escalation_count"`
	MaxEscalations   int                 `json:"max_escalations"`
	Question         string              `json:"question"`
	LastVerification *VerificationResult `json:"last_verification,omitempty"`
}

// InitializeCascade prepares the cascade state for a new request
// Returns the first (smallest) model to try
func (a *AutoMixSelector) InitializeCascade(ctx context.Context, selCtx *SelectionContext) (*SelectWithCascadeState, *SelectionResult, error) {
	if len(selCtx.CandidateModels) == 0 {
		return nil, nil, fmt.Errorf("no candidate models provided")
	}

	// Get escalation chain (smallest to largest)
	chain := a.GetEscalationChain(selCtx.CandidateModels)

	// Start with the smallest model
	firstModel := chain[0]
	var modelRef *config.ModelRef
	for i := range selCtx.CandidateModels {
		if selCtx.CandidateModels[i].Model == firstModel {
			modelRef = &selCtx.CandidateModels[i]
			break
		}
	}

	if modelRef == nil {
		return nil, nil, fmt.Errorf("failed to find first model in candidates")
	}

	state := &SelectWithCascadeState{
		CurrentModel:    firstModel,
		EscalationChain: chain,
		EscalationCount: 0,
		MaxEscalations:  a.config.MaxEscalations,
	}

	logging.Infof("[AutoMix] Initialized cascade: chain=%v, starting with %s", chain, firstModel)

	return state, &SelectionResult{
		SelectedModel: modelRef.Model,
		LoRAName:      modelRef.LoRAName,
		Score:         0.5,
		Confidence:    0.5,
		Method:        MethodAutoMix,
		Reasoning:     fmt.Sprintf("AutoMix cascade: starting with smallest model %s", firstModel),
	}, nil
}

// ContinueCascade processes the answer from the current model and decides
// whether to escalate or return the final result
func (a *AutoMixSelector) ContinueCascade(ctx context.Context, state *SelectWithCascadeState, answer string, selCtx *SelectionContext) (*SelectionResult, bool, error) {
	if a.verifierClient == nil {
		// No verifier - accept the answer
		logging.Warnf("[AutoMix] No verifier configured, accepting answer from %s", state.CurrentModel)
		return &SelectionResult{
			SelectedModel: state.CurrentModel,
			Score:         0.8,
			Confidence:    0.8,
			Method:        MethodAutoMix,
			Reasoning:     "No verifier configured, accepting answer",
		}, true, nil
	}

	// Verify the answer
	verifyResult, err := a.VerifyAnswer(ctx, state.Question, answer, state.CurrentModel, state.EscalationChain)
	if err != nil {
		logging.Warnf("[AutoMix] Verification failed: %v, accepting answer", err)
		return &SelectionResult{
			SelectedModel: state.CurrentModel,
			Score:         0.6,
			Confidence:    0.6,
			Method:        MethodAutoMix,
			Reasoning:     fmt.Sprintf("Verification failed: %v", err),
		}, true, nil
	}

	state.LastVerification = verifyResult

	// Check if we should escalate
	if verifyResult.ShouldEscalate && state.EscalationCount < state.MaxEscalations && verifyResult.NextModel != "" {
		// Escalate to next model
		state.CurrentModel = verifyResult.NextModel
		state.EscalationCount++

		// Find the model ref
		var modelRef *config.ModelRef
		for i := range selCtx.CandidateModels {
			if selCtx.CandidateModels[i].Model == verifyResult.NextModel {
				modelRef = &selCtx.CandidateModels[i]
				break
			}
		}

		if modelRef == nil {
			return nil, false, fmt.Errorf("escalation model %s not in candidates", verifyResult.NextModel)
		}

		logging.Infof("[AutoMix] Escalating to %s (confidence was %.2f, escalation %d/%d)",
			verifyResult.NextModel, verifyResult.Confidence, state.EscalationCount, state.MaxEscalations)

		return &SelectionResult{
			SelectedModel: modelRef.Model,
			LoRAName:      modelRef.LoRAName,
			Score:         0.5,
			Confidence:    verifyResult.Confidence,
			Method:        MethodAutoMix,
			Reasoning:     verifyResult.Reasoning,
		}, false, nil // false = not done, continue cascade
	}

	// Accept the answer
	logging.Infof("[AutoMix] Accepting answer from %s (confidence=%.2f, escalations=%d)",
		state.CurrentModel, verifyResult.Confidence, state.EscalationCount)

	return &SelectionResult{
		SelectedModel: state.CurrentModel,
		Score:         verifyResult.Confidence,
		Confidence:    verifyResult.Confidence,
		Method:        MethodAutoMix,
		Reasoning:     verifyResult.Reasoning,
	}, true, nil // true = done
}
