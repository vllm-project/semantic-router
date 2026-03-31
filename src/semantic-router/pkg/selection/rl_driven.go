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
	"math/rand"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// RLDrivenConfig configures the RL-driven model selector
// Based on Router-R1 (arXiv:2506.09033) - RL multi-round routing with reward structure
//
// Router-R1 Reward Components:
//   - Format Reward: -1 for incorrect format, 0 for correct
//   - Outcome Reward: Based on Exact Match with ground truth
//   - Cost Reward: Inversely proportional to model size × output tokens
//
// Overall Reward: R = R_format + (1-α)*R_outcome + α*R_cost
// where α controls performance-cost tradeoff
type RLDrivenConfig struct {
	// ExplorationRate controls initial exploration (higher = more exploration)
	// Range: 0.0-1.0, default: 0.3
	ExplorationRate float64 `yaml:"exploration_rate"`

	// ExplorationDecay reduces exploration over time (per 100 selections)
	// Range: 0.0-1.0, default: 0.99 (1% decay per 100 selections)
	ExplorationDecay float64 `yaml:"exploration_decay"`

	// MinExploration is the minimum exploration rate to maintain
	// Range: 0.0-1.0, default: 0.05
	MinExploration float64 `yaml:"min_exploration"`

	// UseThompsonSampling enables Thompson Sampling for exploration/exploitation
	// When false, uses epsilon-greedy with ExplorationRate as epsilon
	UseThompsonSampling bool `yaml:"use_thompson_sampling"`

	// EnablePersonalization enables per-user preference tracking
	// When true, maintains separate preference models per user
	EnablePersonalization bool `yaml:"enable_personalization"`

	// PersonalizationBlend controls blend between global and user-specific preferences
	// Range: 0.0-1.0, where 1.0 = fully personalized, 0.0 = fully global
	PersonalizationBlend float64 `yaml:"personalization_blend"`

	// SessionContextWeight controls influence of within-session feedback
	// Higher values give more weight to recent within-session performance
	SessionContextWeight float64 `yaml:"session_context_weight"`

	// ImplicitFeedbackWeight controls weight of auto-detected feedback signals
	// Range: 0.0-1.0, default: 0.5 (implicit feedback counts half as much)
	ImplicitFeedbackWeight float64 `yaml:"implicit_feedback_weight"`

	// CostAwareness enables cost-aware exploration (prefer cheaper models for exploration)
	CostAwareness bool `yaml:"cost_awareness"`

	// CostWeight controls cost influence when CostAwareness is enabled
	CostWeight float64 `yaml:"cost_weight"`

	// StoragePath is the file path for persisting RL state (optional)
	StoragePath string `yaml:"storage_path,omitempty"`

	// AutoSaveInterval is the interval for automatic saves (default: 30s)
	AutoSaveInterval string `yaml:"auto_save_interval,omitempty"`

	// Router-R1 Reward Configuration (arXiv:2506.09033)

	// UseRouterR1Rewards enables Router-R1 style reward computation
	UseRouterR1Rewards bool `yaml:"use_router_r1_rewards"`

	// CostRewardAlpha controls performance-cost tradeoff in reward
	// R = R_format + (1-α)*R_outcome + α*R_cost
	// Range: 0.0-1.0, where 0.0 = pure outcome, 1.0 = pure cost
	CostRewardAlpha float64 `yaml:"cost_reward_alpha"`

	// FormatRewardPenalty is the penalty for incorrect response format
	FormatRewardPenalty float64 `yaml:"format_reward_penalty"`

	// ModelCostPerToken maps model names to cost per output token
	// Used for computing cost rewards
	ModelCostPerToken map[string]float64 `yaml:"model_cost_per_token,omitempty"`

	// Router-R1 LLM-as-Router Configuration (arXiv:2506.09033 Section 3.1)

	// EnableLLMRouting enables LLM-based routing using the Router-R1 approach
	// When enabled, an LLM will analyze the query and select the optimal model
	// This implements the "think" and "route" actions from the paper
	EnableLLMRouting bool `yaml:"enable_llm_routing"`

	// RouterR1ServerURL is the URL of the Router-R1 LLM server
	// This server should expose /route and /health endpoints
	RouterR1ServerURL string `yaml:"router_r1_server_url"`

	// LLMRoutingFallback controls behavior when LLM routing fails
	// Options: "thompson" (fall back to Thompson Sampling), "error" (return error)
	LLMRoutingFallback string `yaml:"llm_routing_fallback"`

	// EnableMultiRoundAggregation enables Router-R1 multi-round routing
	// When enabled, the router may query multiple models and aggregate responses
	// This implements the aggregation strategy from the paper
	EnableMultiRoundAggregation bool `yaml:"enable_multi_round_aggregation"`

	// MaxAggregationRounds is the maximum number of models to query in multi-round mode
	MaxAggregationRounds int `yaml:"max_aggregation_rounds"`
}

// DefaultRLDrivenConfig returns the default RL-driven configuration
func DefaultRLDrivenConfig() *RLDrivenConfig {
	return &RLDrivenConfig{
		ExplorationRate:        0.3,
		ExplorationDecay:       0.99,
		MinExploration:         0.05,
		UseThompsonSampling:    true,
		EnablePersonalization:  true,
		PersonalizationBlend:   0.7,
		SessionContextWeight:   0.3,
		ImplicitFeedbackWeight: 0.5,
		CostAwareness:          true, // Enable by default for Router-R1
		CostWeight:             0.2,
		// Router-R1 defaults
		UseRouterR1Rewards:  true,
		CostRewardAlpha:     0.3, // 30% cost, 70% outcome
		FormatRewardPenalty: -1.0,
	}
}

// RouterR1Reward computes the Router-R1 style reward
// R = R_format + (1-α)*R_outcome + α*R_cost
func (c *RLDrivenConfig) ComputeRouterR1Reward(
	formatCorrect bool,
	outcomeScore float64, // 0.0-1.0, where 1.0 = exact match
	modelCost float64, // Cost per output token
	outputTokens int,
) float64 {
	// Format reward
	formatReward := 0.0
	if !formatCorrect {
		formatReward = c.FormatRewardPenalty
		// If format is wrong, nullify other rewards (hierarchical reward)
		return formatReward
	}

	// Outcome reward (already 0-1)
	outcomeReward := outcomeScore

	// Cost reward: inversely proportional to cost
	// Normalize to 0-1 range (assuming max cost is $10/1M tokens)
	normalizedCost := modelCost * float64(outputTokens) / 10.0
	costReward := 1.0 - normalizedCost
	if costReward < 0 {
		costReward = 0
	}

	// Combine with alpha
	alpha := c.CostRewardAlpha
	totalReward := formatReward + (1-alpha)*outcomeReward + alpha*costReward

	return totalReward
}

// BetaDistribution represents a Beta distribution for Thompson Sampling
// Used to model uncertainty about a model's true win probability
type BetaDistribution struct {
	// Alpha is the number of successes + 1 (prior)
	Alpha float64 `json:"alpha"`

	// Beta is the number of failures + 1 (prior)
	Beta float64 `json:"beta"`
}

// Sample draws a sample from the Beta distribution
// Uses the gamma distribution method: X ~ Beta(a,b) = Gamma(a,1) / (Gamma(a,1) + Gamma(b,1))
func (b *BetaDistribution) Sample(rng *rand.Rand) float64 {
	// Handle edge cases
	if b.Alpha <= 0 {
		b.Alpha = 1.0
	}
	if b.Beta <= 0 {
		b.Beta = 1.0
	}

	x := gammaVariate(rng, b.Alpha)
	y := gammaVariate(rng, b.Beta)

	if x+y == 0 {
		return 0.5
	}
	return x / (x + y)
}

// Mean returns the mean of the Beta distribution
func (b *BetaDistribution) Mean() float64 {
	if b.Alpha+b.Beta == 0 {
		return 0.5
	}
	return b.Alpha / (b.Alpha + b.Beta)
}

// Variance returns the variance of the Beta distribution
func (b *BetaDistribution) Variance() float64 {
	sum := b.Alpha + b.Beta
	if sum == 0 {
		return 0
	}
	return (b.Alpha * b.Beta) / (sum * sum * (sum + 1))
}

// gammaVariate generates a random variate from Gamma(alpha, 1) distribution
// Uses Marsaglia and Tsang's method for alpha >= 1
func gammaVariate(rng *rand.Rand, alpha float64) float64 {
	if alpha < 1 {
		// For alpha < 1, use the transformation: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
		return gammaVariate(rng, alpha+1) * math.Pow(rng.Float64(), 1/alpha)
	}

	d := alpha - 1.0/3.0
	c := 1.0 / math.Sqrt(9.0*d)

	for {
		var x, v float64
		for {
			x = rng.NormFloat64()
			v = 1.0 + c*x
			if v > 0 {
				break
			}
		}

		v = v * v * v
		u := rng.Float64()

		if u < 1.0-0.0331*(x*x)*(x*x) {
			return d * v
		}

		if math.Log(u) < 0.5*x*x+d*(1.0-v+math.Log(v)) {
			return d * v
		}
	}
}

// ModelPreference stores per-model preference data for RL learning
type ModelPreference struct {
	// Model is the model name
	Model string `json:"model"`

	// Distribution is the Beta distribution for Thompson Sampling
	Distribution BetaDistribution `json:"distribution"`

	// TotalInteractions is the total number of times this model was selected
	TotalInteractions int `json:"total_interactions"`

	// ExplicitFeedbackCount is the count of explicit (API) feedback received
	ExplicitFeedbackCount int `json:"explicit_feedback_count"`

	// ImplicitFeedbackCount is the count of implicit (signal-detected) feedback
	ImplicitFeedbackCount int `json:"implicit_feedback_count"`

	// LastUpdated is when this preference was last updated
	LastUpdated time.Time `json:"last_updated"`
}

// RLDrivenSelector implements reinforcement learning-based model selection
// It uses Thompson Sampling to balance exploration and exploitation while
// learning user preferences over time.
//
// When EnableLLMRouting is true, this selector implements the full Router-R1
// approach from arXiv:2506.09033, using an LLM to analyze queries and make
// intelligent routing decisions with "think" and "route" actions.
type RLDrivenSelector struct {
	config *RLDrivenConfig

	// rng is the random number generator (thread-safe via mutex)
	rng *rand.Rand

	// Global preferences (shared across all users)
	globalPreferences map[string]*ModelPreference
	globalMu          sync.RWMutex

	// Per-user preferences (user_id -> model -> preference)
	userPreferences map[string]map[string]*ModelPreference
	userMu          sync.RWMutex

	// Per-category preferences (category -> model -> preference)
	categoryPreferences map[string]map[string]*ModelPreference
	categoryMu          sync.RWMutex

	// Session context (session_id -> model -> recent performance)
	sessionContext map[string]map[string]*SessionModelStats
	sessionMu      sync.RWMutex

	// Model costs for cost-aware exploration
	modelCosts map[string]float64
	costMu     sync.RWMutex

	// Selection count for exploration decay
	selectionCount int64
	countMu        sync.Mutex

	// Storage backend for persistence
	storage EloStorage

	// Router-R1 LLM client for intelligent routing
	// Requires external Router-R1 server (see src/training/rl_model_selection/router_r1_server.py)
	routerR1Client *RouterR1Client
}

// SessionModelStats tracks within-session model performance
type SessionModelStats struct {
	Model       string    `json:"model"`
	Selections  int       `json:"selections"`
	Successes   int       `json:"successes"`
	LastUpdated time.Time `json:"last_updated"`
}

// NewRLDrivenSelector creates a new RL-driven selector
func NewRLDrivenSelector(cfg *RLDrivenConfig) *RLDrivenSelector {
	if cfg == nil {
		cfg = DefaultRLDrivenConfig()
	}

	selector := &RLDrivenSelector{
		config:              cfg,
		rng:                 rand.New(rand.NewSource(time.Now().UnixNano())),
		globalPreferences:   make(map[string]*ModelPreference),
		userPreferences:     make(map[string]map[string]*ModelPreference),
		categoryPreferences: make(map[string]map[string]*ModelPreference),
		sessionContext:      make(map[string]map[string]*SessionModelStats),
		modelCosts:          make(map[string]float64),
	}

	// Initialize storage if path is configured
	if cfg.StoragePath != "" {
		storage, err := NewFileEloStorage(cfg.StoragePath)
		if err != nil {
			logging.Errorf("[RLDrivenSelector] Failed to initialize storage: %v", err)
		} else {
			selector.storage = storage

			// Load existing preferences from storage
			if err := selector.loadFromStorage(); err != nil {
				logging.Warnf("[RLDrivenSelector] Failed to load preferences from storage: %v", err)
			}

			// Start auto-save
			interval := 30 * time.Second
			if cfg.AutoSaveInterval != "" {
				if parsed, err := time.ParseDuration(cfg.AutoSaveInterval); err == nil {
					interval = parsed
				}
			}

			storage.StartAutoSave(interval, selector.getAllPreferencesForStorage)
			logging.Infof("[RLDrivenSelector] Storage initialized with auto-save interval: %v", interval)
		}
	}

	// Initialize Router-R1 LLM client if configured
	// Requires external server: src/training/rl_model_selection/router_r1_server.py
	if cfg.EnableLLMRouting && cfg.RouterR1ServerURL != "" {
		selector.routerR1Client = NewRouterR1Client(cfg.RouterR1ServerURL)
		logging.Infof("[RLDrivenSelector] Router-R1 LLM routing enabled, server: %s", cfg.RouterR1ServerURL)

		// Verify connectivity
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := selector.routerR1Client.HealthCheck(ctx); err != nil {
			logging.Warnf("[RLDrivenSelector] Router-R1 server not reachable: %v (will retry on use)", err)
		} else {
			logging.Infof("[RLDrivenSelector] Router-R1 server connected successfully")
		}
	} else if cfg.EnableLLMRouting {
		logging.Warnf("[RLDrivenSelector] Router-R1 LLM routing enabled but no server URL configured")
	}

	return selector
}

// Method returns the selection method type
func (r *RLDrivenSelector) Method() SelectionMethod {
	return MethodRLDriven
}

// SetModelCost sets the cost for a model (for cost-aware exploration)
func (r *RLDrivenSelector) SetModelCost(model string, costPer1M float64) {
	r.costMu.Lock()
	defer r.costMu.Unlock()
	r.modelCosts[model] = costPer1M
}

// InitializeFromConfig sets up initial preferences from model configuration
func (r *RLDrivenSelector) InitializeFromConfig(modelConfig map[string]config.ModelParams, categories []config.Category) {
	r.globalMu.Lock()
	defer r.globalMu.Unlock()

	// Initialize global preferences for all models with uniform priors
	for model := range modelConfig {
		if _, exists := r.globalPreferences[model]; !exists {
			r.globalPreferences[model] = &ModelPreference{
				Model: model,
				Distribution: BetaDistribution{
					Alpha: 1.0, // Uniform prior
					Beta:  1.0,
				},
				LastUpdated: time.Now(),
			}
		}
	}

	// Set costs from config
	r.costMu.Lock()
	for model, params := range modelConfig {
		if params.Pricing.PromptPer1M > 0 {
			r.modelCosts[model] = params.Pricing.PromptPer1M
		}
	}
	r.costMu.Unlock()

	// Initialize category preferences from ModelScores if available
	r.categoryMu.Lock()
	for _, category := range categories {
		if r.categoryPreferences[category.Name] == nil {
			r.categoryPreferences[category.Name] = make(map[string]*ModelPreference)
		}
		for _, ms := range category.ModelScores {
			// Convert static scores to Beta distribution parameters
			// Higher score = more alpha (successes)
			alpha := 1.0 + ms.Score*10.0 // Scale score to meaningful prior
			beta := 1.0 + (1.0-ms.Score)*10.0
			r.categoryPreferences[category.Name][ms.Model] = &ModelPreference{
				Model: ms.Model,
				Distribution: BetaDistribution{
					Alpha: alpha,
					Beta:  beta,
				},
				LastUpdated: time.Now(),
			}
		}
	}
	r.categoryMu.Unlock()

	logging.Infof("[RLDrivenSelector] Initialized with %d models, %d categories",
		len(modelConfig), len(categories))
}

// Select chooses the best model using Thompson Sampling or Router-R1 LLM routing
// When EnableLLMRouting is configured, the selector first attempts to use an LLM
// to analyze the query and make an intelligent routing decision (Router-R1 approach).
// If LLM routing fails or is disabled, falls back to Thompson Sampling.
func (r *RLDrivenSelector) Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
	if len(selCtx.CandidateModels) == 0 {
		return nil, fmt.Errorf("no candidate models provided")
	}

	// Try Router-R1 LLM routing first if enabled
	if r.config.EnableLLMRouting && r.routerR1Client != nil {
		result, err := r.selectWithRouterR1(ctx, selCtx)
		if err == nil {
			return result, nil
		}
		// Log the error and fall back
		logging.Warnf("[RLDrivenSelector] Router-R1 LLM routing failed: %v, falling back to Thompson Sampling", err)
		if r.config.LLMRoutingFallback == "error" {
			return nil, fmt.Errorf("Router-R1 LLM routing failed: %w", err)
		}
	}

	// Increment selection count for exploration decay
	r.countMu.Lock()
	r.selectionCount++
	selectionNum := r.selectionCount
	r.countMu.Unlock()

	// Get preferences for each candidate
	preferences := r.getPreferencesForCandidates(selCtx)

	// Log candidate evaluation
	logging.Infof("[RLDrivenSelector] Evaluating %d candidates (user=%s, session=%s, selection=%d):",
		len(selCtx.CandidateModels), selCtx.UserID, selCtx.SessionID, selectionNum)

	allScores := make(map[string]float64)
	var selectedModel *config.ModelRef
	var selectedScore float64
	var sampledValues map[string]float64

	if r.config.UseThompsonSampling {
		// Thompson Sampling: sample from each model's Beta distribution
		sampledValues = make(map[string]float64)

		for _, pref := range preferences {
			sample := pref.Distribution.Sample(r.rng)

			// Apply cost adjustment if enabled
			if r.config.CostAwareness && r.config.CostWeight > 0 {
				sample = r.applyCostBonus(pref.Model, sample)
			}

			// Apply session context if available
			if selCtx.SessionID != "" && r.config.SessionContextWeight > 0 {
				sample = r.applySessionContext(selCtx.SessionID, pref.Model, sample)
			}

			sampledValues[pref.Model] = sample
			allScores[pref.Model] = sample

			logging.Infof("[RLDrivenSelector]   %s: alpha=%.2f, beta=%.2f, mean=%.3f, sampled=%.3f",
				pref.Model, pref.Distribution.Alpha, pref.Distribution.Beta,
				pref.Distribution.Mean(), sample)
		}

		// Select model with highest sampled value
		for i := range selCtx.CandidateModels {
			model := &selCtx.CandidateModels[i]
			score := sampledValues[model.Model]
			if score > selectedScore || selectedModel == nil {
				selectedScore = score
				selectedModel = model
			}
		}
	} else {
		// Epsilon-greedy exploration
		explorationRate := r.getCurrentExplorationRate(selectionNum)

		if r.rng.Float64() < explorationRate {
			// Explore: select random model (prefer cheaper if cost-aware)
			selectedModel = r.selectRandomModel(selCtx.CandidateModels)
			selectedScore = 0.5
			logging.Infof("[RLDrivenSelector] Exploring with random selection")
		} else {
			// Exploit: select model with highest mean
			for _, pref := range preferences {
				mean := pref.Distribution.Mean()
				allScores[pref.Model] = mean
			}

			for i := range selCtx.CandidateModels {
				model := &selCtx.CandidateModels[i]
				score := allScores[model.Model]
				if score > selectedScore || selectedModel == nil {
					selectedScore = score
					selectedModel = model
				}
			}
		}
	}

	if selectedModel == nil {
		return nil, fmt.Errorf("could not select a model")
	}

	// Calculate confidence based on distribution certainty
	pref := r.getPreference(selCtx, selectedModel.Model)
	confidence := r.calculateConfidence(pref)

	// Build reasoning
	reasoning := r.buildReasoning(selCtx, selectedModel.Model, preferences, sampledValues)

	logging.Infof("[RLDrivenSelector] Selected %s (score=%.4f, confidence=%.2f)",
		selectedModel.Model, selectedScore, confidence)

	// Record metrics
	RecordRLSelection(selectedModel.Model, selCtx.DecisionName, selCtx.UserID, selectedScore)

	return &SelectionResult{
		SelectedModel: selectedModel.Model,
		LoRAName:      selectedModel.LoRAName,
		Score:         selectedScore,
		Confidence:    confidence,
		Method:        MethodRLDriven,
		Reasoning:     reasoning,
		AllScores:     allScores,
	}, nil
}

// selectWithRouterR1 implements the Router-R1 LLM-as-Router approach
// This method uses an LLM to analyze the query and make intelligent routing decisions.
// The LLM performs "think" (analyze query complexity, requirements) and "route"
// (select the optimal model) actions as described in arXiv:2506.09033.
// Requires external server: src/training/rl_model_selection/router_r1_server.py
func (r *RLDrivenSelector) selectWithRouterR1(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
	if r.routerR1Client == nil {
		return nil, fmt.Errorf("Router-R1 client not initialized - set router_r1_server_url in config")
	}

	// Extract query from context - build a descriptive query including available models
	modelNames := make([]string, 0, len(selCtx.CandidateModels))
	for _, m := range selCtx.CandidateModels {
		modelNames = append(modelNames, m.Model)
	}
	query := fmt.Sprintf("Task: %s\nAvailable models: %s", selCtx.DecisionName, strings.Join(modelNames, ", "))

	// Call the Router-R1 LLM server
	response, err := r.routerR1Client.Route(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("Router-R1 routing failed: %w", err)
	}

	// Validate that the selected model is in our candidate list
	var selectedModel *config.ModelRef
	for i := range selCtx.CandidateModels {
		if selCtx.CandidateModels[i].Model == response.SelectedModel {
			selectedModel = &selCtx.CandidateModels[i]
			break
		}
	}

	if selectedModel == nil {
		// LLM selected an invalid model, try to find a partial match
		for i := range selCtx.CandidateModels {
			if strings.Contains(selCtx.CandidateModels[i].Model, response.SelectedModel) ||
				strings.Contains(response.SelectedModel, selCtx.CandidateModels[i].Model) {
				selectedModel = &selCtx.CandidateModels[i]
				logging.Warnf("[RLDrivenSelector] Router-R1 partial match: %s -> %s",
					response.SelectedModel, selectedModel.Model)
				break
			}
		}
	}

	if selectedModel == nil {
		return nil, fmt.Errorf("Router-R1 selected invalid model: %s (not in candidates)", response.SelectedModel)
	}

	// Build reasoning from LLM response
	reasoning := fmt.Sprintf("Router-R1 LLM Analysis:\n%s\n\nDecision: %s",
		response.Thinking, response.SelectedModel)

	// Increment selection count
	r.countMu.Lock()
	r.selectionCount++
	r.countMu.Unlock()

	logging.Infof("[RLDrivenSelector] Router-R1 selected %s (thinking: %s)",
		selectedModel.Model, truncateString(response.Thinking, 100))

	// Record metrics
	RecordRLSelection(selectedModel.Model, selCtx.DecisionName, selCtx.UserID, 1.0)

	return &SelectionResult{
		SelectedModel: selectedModel.Model,
		LoRAName:      selectedModel.LoRAName,
		Score:         1.0, // LLM routing is deterministic
		Confidence:    0.9, // High confidence in LLM decision
		Method:        MethodRLDriven,
		Reasoning:     reasoning,
		AllScores:     map[string]float64{selectedModel.Model: 1.0},
	}, nil
}

// truncateString truncates a string to the specified length
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// MultiRoundResult contains the result of multi-round model selection
type MultiRoundResult struct {
	SelectedModels []string           `json:"selected_models"`
	Scores         map[string]float64 `json:"scores"`
	Reasoning      string             `json:"reasoning"`
}

// SelectMultiRound implements Router-R1 multi-round routing
// Instead of selecting a single model, it selects multiple models
// to query in parallel, with responses to be aggregated.
// This implements Section 3.3 of arXiv:2506.09033
func (r *RLDrivenSelector) SelectMultiRound(ctx context.Context, selCtx *SelectionContext) (*MultiRoundResult, error) {
	if !r.config.EnableMultiRoundAggregation {
		return nil, fmt.Errorf("multi-round aggregation not enabled")
	}

	if len(selCtx.CandidateModels) == 0 {
		return nil, fmt.Errorf("no candidate models provided")
	}

	maxRounds := r.config.MaxAggregationRounds
	if maxRounds <= 0 {
		maxRounds = 3 // Default
	}
	if maxRounds > len(selCtx.CandidateModels) {
		maxRounds = len(selCtx.CandidateModels)
	}

	// Get preferences for each candidate
	preferences := r.getPreferencesForCandidates(selCtx)

	// Sort by expected value (Thompson Sampling mean + exploration bonus)
	type modelScore struct {
		model string
		score float64
	}
	scores := make([]modelScore, 0, len(preferences))
	allScores := make(map[string]float64)

	for _, pref := range preferences {
		// Sample from distribution for exploration
		sample := pref.Distribution.Sample(r.rng)

		// Apply cost awareness
		if r.config.CostAwareness && r.config.CostWeight > 0 {
			sample = r.applyCostBonus(pref.Model, sample)
		}

		scores = append(scores, modelScore{model: pref.Model, score: sample})
		allScores[pref.Model] = sample
	}

	// Sort by score descending
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	// Select top models
	selectedModels := make([]string, 0, maxRounds)
	for i := 0; i < maxRounds && i < len(scores); i++ {
		selectedModels = append(selectedModels, scores[i].model)
	}

	reasoning := fmt.Sprintf("Multi-round aggregation: selected %d models based on Thompson Sampling", len(selectedModels))

	logging.Infof("[RLDrivenSelector] Multi-round selection: %v", selectedModels)

	return &MultiRoundResult{
		SelectedModels: selectedModels,
		Scores:         allScores,
		Reasoning:      reasoning,
	}, nil
}

// AggregateResponses aggregates responses from multiple models
// This implements the response aggregation from Router-R1
// Options: voting, confidence-weighted, best-of-n
func (r *RLDrivenSelector) AggregateResponses(responses map[string]string, scores map[string]float64) (string, string, error) {
	if len(responses) == 0 {
		return "", "", fmt.Errorf("no responses to aggregate")
	}

	// Confidence-weighted selection: pick response from highest-scored model
	var bestModel string
	var bestScore float64
	for model := range responses {
		score := scores[model]
		if score > bestScore || bestModel == "" {
			bestScore = score
			bestModel = model
		}
	}

	response := responses[bestModel]
	reasoning := fmt.Sprintf("Aggregated from %d models, selected %s (score=%.3f)", len(responses), bestModel, bestScore)

	return response, reasoning, nil
}

// UpdateFeedback updates the RL model based on user feedback
func (r *RLDrivenSelector) UpdateFeedback(ctx context.Context, feedback *Feedback) error {
	if feedback.WinnerModel == "" && feedback.LoserModel == "" {
		return fmt.Errorf("either winner_model or loser_model is required")
	}

	// Determine feedback weight based on type
	weight := 1.0
	if feedback.FeedbackType != "" {
		// Implicit feedback gets weighted by ImplicitFeedbackWeight
		weight = r.config.ImplicitFeedbackWeight

		// Further scale by confidence if provided
		if feedback.Confidence > 0 {
			weight *= feedback.Confidence
		}
	}

	// Update global preferences
	r.updatePreference(feedback, weight, r.getGlobalPreference, r.setGlobalPreference)

	// Update user preferences if personalization enabled and UserID provided
	if r.config.EnablePersonalization && feedback.UserID != "" {
		r.updatePreference(feedback, weight,
			func(model string) *ModelPreference {
				return r.getUserPreference(feedback.UserID, model)
			},
			func(model string, pref *ModelPreference) {
				r.setUserPreference(feedback.UserID, model, pref)
			})
	}

	// Update category preferences if DecisionName provided
	if feedback.DecisionName != "" {
		r.updatePreference(feedback, weight,
			func(model string) *ModelPreference {
				return r.getCategoryPreference(feedback.DecisionName, model)
			},
			func(model string, pref *ModelPreference) {
				r.setCategoryPreference(feedback.DecisionName, model, pref)
			})
	}

	// Update session context if SessionID provided
	if feedback.SessionID != "" {
		r.updateSessionContext(feedback)
	}

	logging.Infof("[RLDrivenSelector] Feedback updated: winner=%s, loser=%s, user=%s, weight=%.2f",
		feedback.WinnerModel, feedback.LoserModel, feedback.UserID, weight)

	// Mark storage as dirty
	if r.storage != nil {
		if fileStorage, ok := r.storage.(*FileEloStorage); ok {
			fileStorage.MarkDirty()
		}
	}

	// Record feedback metrics
	RecordRLFeedback(feedback.WinnerModel, feedback.LoserModel, feedback.DecisionName, feedback.UserID)

	return nil
}

// updatePreference performs the Beta distribution update based on feedback
func (r *RLDrivenSelector) updatePreference(
	feedback *Feedback,
	weight float64,
	getPref func(string) *ModelPreference,
	setPref func(string, *ModelPreference),
) {
	now := time.Now()

	// Handle winner model
	if feedback.WinnerModel != "" {
		winnerPref := getPref(feedback.WinnerModel)
		if winnerPref == nil {
			winnerPref = &ModelPreference{
				Model: feedback.WinnerModel,
				Distribution: BetaDistribution{
					Alpha: 1.0,
					Beta:  1.0,
				},
			}
		}

		// Update Beta distribution: add weighted success
		if feedback.Tie {
			// Tie: add 0.5 to both alpha and beta
			winnerPref.Distribution.Alpha += 0.5 * weight
			winnerPref.Distribution.Beta += 0.5 * weight
		} else {
			// Win: add to alpha (success)
			winnerPref.Distribution.Alpha += weight
		}

		winnerPref.TotalInteractions++
		if feedback.FeedbackType != "" {
			winnerPref.ImplicitFeedbackCount++
		} else {
			winnerPref.ExplicitFeedbackCount++
		}
		winnerPref.LastUpdated = now

		setPref(feedback.WinnerModel, winnerPref)
	}

	// Handle loser model
	if feedback.LoserModel != "" {
		loserPref := getPref(feedback.LoserModel)
		if loserPref == nil {
			loserPref = &ModelPreference{
				Model: feedback.LoserModel,
				Distribution: BetaDistribution{
					Alpha: 1.0,
					Beta:  1.0,
				},
			}
		}

		// Update Beta distribution: add weighted failure
		if feedback.Tie {
			// Tie: add 0.5 to both alpha and beta
			loserPref.Distribution.Alpha += 0.5 * weight
			loserPref.Distribution.Beta += 0.5 * weight
		} else {
			// Loss: add to beta (failure)
			loserPref.Distribution.Beta += weight
		}

		loserPref.TotalInteractions++
		if feedback.FeedbackType != "" {
			loserPref.ImplicitFeedbackCount++
		} else {
			loserPref.ExplicitFeedbackCount++
		}
		loserPref.LastUpdated = now

		setPref(feedback.LoserModel, loserPref)
	}
}

// getPreferencesForCandidates retrieves preferences for all candidate models
func (r *RLDrivenSelector) getPreferencesForCandidates(selCtx *SelectionContext) []*ModelPreference {
	preferences := make([]*ModelPreference, 0, len(selCtx.CandidateModels))

	for _, c := range selCtx.CandidateModels {
		pref := r.getPreference(selCtx, c.Model)
		preferences = append(preferences, pref)
	}

	return preferences
}

// getPreference retrieves the blended preference for a model
func (r *RLDrivenSelector) getPreference(selCtx *SelectionContext, model string) *ModelPreference {
	// Start with global preference
	globalPref := r.getGlobalPreference(model)
	if globalPref == nil {
		globalPref = &ModelPreference{
			Model: model,
			Distribution: BetaDistribution{
				Alpha: 1.0,
				Beta:  1.0,
			},
		}
	}

	// If personalization disabled or no user, return global
	if !r.config.EnablePersonalization || selCtx.UserID == "" {
		// Try category preference if available
		if selCtx.DecisionName != "" {
			catPref := r.getCategoryPreference(selCtx.DecisionName, model)
			if catPref != nil {
				return r.blendPreferences(globalPref, catPref, 0.5)
			}
		}
		return globalPref
	}

	// Blend global and user preferences
	userPref := r.getUserPreference(selCtx.UserID, model)
	if userPref == nil {
		userPref = globalPref
	}

	blended := r.blendPreferences(globalPref, userPref, r.config.PersonalizationBlend)

	// Also blend with category if available
	if selCtx.DecisionName != "" {
		catPref := r.getCategoryPreference(selCtx.DecisionName, model)
		if catPref != nil {
			blended = r.blendPreferences(blended, catPref, 0.3)
		}
	}

	return blended
}

// blendPreferences combines two preferences with a blend factor
func (r *RLDrivenSelector) blendPreferences(pref1, pref2 *ModelPreference, blend float64) *ModelPreference {
	// Blend the Beta distributions by weighted sum of parameters
	alpha := (1-blend)*pref1.Distribution.Alpha + blend*pref2.Distribution.Alpha
	beta := (1-blend)*pref1.Distribution.Beta + blend*pref2.Distribution.Beta

	return &ModelPreference{
		Model: pref1.Model,
		Distribution: BetaDistribution{
			Alpha: alpha,
			Beta:  beta,
		},
		TotalInteractions:     pref1.TotalInteractions + pref2.TotalInteractions,
		ExplicitFeedbackCount: pref1.ExplicitFeedbackCount + pref2.ExplicitFeedbackCount,
		ImplicitFeedbackCount: pref1.ImplicitFeedbackCount + pref2.ImplicitFeedbackCount,
		LastUpdated:           pref2.LastUpdated,
	}
}

// Global preference accessors
func (r *RLDrivenSelector) getGlobalPreference(model string) *ModelPreference {
	r.globalMu.RLock()
	defer r.globalMu.RUnlock()
	return r.globalPreferences[model]
}

func (r *RLDrivenSelector) setGlobalPreference(model string, pref *ModelPreference) {
	r.globalMu.Lock()
	defer r.globalMu.Unlock()
	r.globalPreferences[model] = pref
}

// User preference accessors
func (r *RLDrivenSelector) getUserPreference(userID, model string) *ModelPreference {
	r.userMu.RLock()
	defer r.userMu.RUnlock()
	if userPrefs, ok := r.userPreferences[userID]; ok {
		return userPrefs[model]
	}
	return nil
}

func (r *RLDrivenSelector) setUserPreference(userID, model string, pref *ModelPreference) {
	r.userMu.Lock()
	defer r.userMu.Unlock()
	if r.userPreferences[userID] == nil {
		r.userPreferences[userID] = make(map[string]*ModelPreference)
	}
	r.userPreferences[userID][model] = pref
}

// Category preference accessors
func (r *RLDrivenSelector) getCategoryPreference(category, model string) *ModelPreference {
	r.categoryMu.RLock()
	defer r.categoryMu.RUnlock()
	if catPrefs, ok := r.categoryPreferences[category]; ok {
		return catPrefs[model]
	}
	return nil
}

func (r *RLDrivenSelector) setCategoryPreference(category, model string, pref *ModelPreference) {
	r.categoryMu.Lock()
	defer r.categoryMu.Unlock()
	if r.categoryPreferences[category] == nil {
		r.categoryPreferences[category] = make(map[string]*ModelPreference)
	}
	r.categoryPreferences[category][model] = pref
}

// Session context methods
func (r *RLDrivenSelector) updateSessionContext(feedback *Feedback) {
	r.sessionMu.Lock()
	defer r.sessionMu.Unlock()

	if r.sessionContext[feedback.SessionID] == nil {
		r.sessionContext[feedback.SessionID] = make(map[string]*SessionModelStats)
	}

	// Update winner stats
	if feedback.WinnerModel != "" {
		stats := r.sessionContext[feedback.SessionID][feedback.WinnerModel]
		if stats == nil {
			stats = &SessionModelStats{Model: feedback.WinnerModel}
		}
		stats.Selections++
		if !feedback.Tie {
			stats.Successes++
		}
		stats.LastUpdated = time.Now()
		r.sessionContext[feedback.SessionID][feedback.WinnerModel] = stats
	}

	// Update loser stats
	if feedback.LoserModel != "" {
		stats := r.sessionContext[feedback.SessionID][feedback.LoserModel]
		if stats == nil {
			stats = &SessionModelStats{Model: feedback.LoserModel}
		}
		stats.Selections++
		// Don't increment successes for loser
		stats.LastUpdated = time.Now()
		r.sessionContext[feedback.SessionID][feedback.LoserModel] = stats
	}
}

func (r *RLDrivenSelector) applySessionContext(sessionID, model string, score float64) float64 {
	r.sessionMu.RLock()
	defer r.sessionMu.RUnlock()

	if sessionStats, ok := r.sessionContext[sessionID]; ok {
		if stats, ok := sessionStats[model]; ok {
			if stats.Selections > 0 {
				// Calculate session win rate
				sessionWinRate := float64(stats.Successes) / float64(stats.Selections)
				// Blend with current score
				return (1-r.config.SessionContextWeight)*score + r.config.SessionContextWeight*sessionWinRate
			}
		}
	}
	return score
}

// Cost-aware exploration
func (r *RLDrivenSelector) applyCostBonus(model string, score float64) float64 {
	r.costMu.RLock()
	defer r.costMu.RUnlock()

	if len(r.modelCosts) == 0 {
		return score
	}

	cost, ok := r.modelCosts[model]
	if !ok {
		return score
	}

	// Find min and max costs
	minCost, maxCost := math.MaxFloat64, 0.0
	for _, c := range r.modelCosts {
		if c < minCost {
			minCost = c
		}
		if c > maxCost {
			maxCost = c
		}
	}

	if maxCost == minCost {
		return score
	}

	// Normalize cost to 0-1 and apply bonus to cheaper models
	normalizedCost := (cost - minCost) / (maxCost - minCost)
	costBonus := (1.0 - normalizedCost) * r.config.CostWeight

	return score * (1.0 + costBonus)
}

// selectRandomModel selects a random model, with cost-aware preference if enabled
func (r *RLDrivenSelector) selectRandomModel(candidates []config.ModelRef) *config.ModelRef {
	if !r.config.CostAwareness || len(r.modelCosts) == 0 {
		// Uniform random
		idx := r.rng.Intn(len(candidates))
		return &candidates[idx]
	}

	// Cost-aware: prefer cheaper models for exploration
	r.costMu.RLock()
	defer r.costMu.RUnlock()

	// Sort candidates by cost (ascending)
	sorted := make([]config.ModelRef, len(candidates))
	copy(sorted, candidates)
	sort.Slice(sorted, func(i, j int) bool {
		costI := r.modelCosts[sorted[i].Model]
		costJ := r.modelCosts[sorted[j].Model]
		return costI < costJ
	})

	// Weight towards cheaper models (first third has 50% chance)
	idx := 0
	if len(sorted) > 2 {
		if r.rng.Float64() < 0.5 {
			// Pick from cheaper third
			idx = r.rng.Intn(len(sorted) / 3)
		} else {
			// Pick uniformly from rest
			idx = r.rng.Intn(len(sorted))
		}
	} else {
		idx = r.rng.Intn(len(sorted))
	}

	return &sorted[idx]
}

// getCurrentExplorationRate returns the current exploration rate with decay
func (r *RLDrivenSelector) getCurrentExplorationRate(selectionNum int64) float64 {
	// Decay exploration rate over time
	decayFactor := math.Pow(r.config.ExplorationDecay, float64(selectionNum)/100.0)
	rate := r.config.ExplorationRate * decayFactor

	// Apply minimum exploration
	if rate < r.config.MinExploration {
		rate = r.config.MinExploration
	}

	return rate
}

// calculateConfidence calculates confidence based on Beta distribution certainty
func (r *RLDrivenSelector) calculateConfidence(pref *ModelPreference) float64 {
	if pref == nil {
		return 0.5
	}

	// Confidence increases with more data (lower variance)
	// Use 1 - variance as a proxy (variance max is 0.25 at alpha=beta=1)
	variance := pref.Distribution.Variance()
	confidence := 1.0 - 4*variance // Scale to 0-1

	// Also consider total interactions
	interactionFactor := 1.0 / (1.0 + math.Exp(-0.1*(float64(pref.TotalInteractions)-10)))
	confidence = (confidence + interactionFactor) / 2.0

	return math.Min(1.0, math.Max(0.0, confidence))
}

// buildReasoning creates a human-readable explanation
func (r *RLDrivenSelector) buildReasoning(selCtx *SelectionContext, selectedModel string, preferences []*ModelPreference, sampledValues map[string]float64) string {
	var parts []string

	// Find the selected model's preference
	var selectedPref *ModelPreference
	for _, p := range preferences {
		if p.Model == selectedModel {
			selectedPref = p
			break
		}
	}

	if r.config.UseThompsonSampling && sampledValues != nil {
		parts = append(parts, fmt.Sprintf("Thompson Sampling: sampled=%.3f", sampledValues[selectedModel]))
	}

	if selectedPref != nil {
		parts = append(parts, fmt.Sprintf("Beta(%.1f, %.1f)", selectedPref.Distribution.Alpha, selectedPref.Distribution.Beta))
		parts = append(parts, fmt.Sprintf("mean=%.3f", selectedPref.Distribution.Mean()))
		if selectedPref.TotalInteractions > 0 {
			parts = append(parts, fmt.Sprintf("interactions=%d", selectedPref.TotalInteractions))
		}
	}

	if selCtx.UserID != "" && r.config.EnablePersonalization {
		parts = append(parts, fmt.Sprintf("personalized for user=%s", selCtx.UserID))
	}

	if len(parts) == 0 {
		return "RL-driven selection with Thompson Sampling"
	}

	return fmt.Sprintf("RL-driven: [%s]", joinStrings(parts, ", "))
}

// joinStrings joins strings with a separator
func joinStrings(strs []string, sep string) string {
	if len(strs) == 0 {
		return ""
	}
	result := strs[0]
	for i := 1; i < len(strs); i++ {
		result += sep + strs[i]
	}
	return result
}

// Storage methods

func (r *RLDrivenSelector) loadFromStorage() error {
	if r.storage == nil {
		return nil
	}

	// Load all ratings and convert to preferences
	allRatings, err := r.storage.LoadAllRatings()
	if err != nil {
		return err
	}

	r.globalMu.Lock()
	r.categoryMu.Lock()
	defer r.globalMu.Unlock()
	defer r.categoryMu.Unlock()

	for category, ratings := range allRatings {
		for model, rating := range ratings {
			// Convert Elo rating to Beta distribution
			// Higher Elo = higher alpha
			winRate := (rating.Rating - 1000) / 1000 // Normalize to ~0-1
			if winRate < 0.1 {
				winRate = 0.1
			}
			if winRate > 0.9 {
				winRate = 0.9
			}

			totalComparisons := float64(rating.Comparisons)
			if totalComparisons < 2 {
				totalComparisons = 2
			}

			pref := &ModelPreference{
				Model: model,
				Distribution: BetaDistribution{
					Alpha: 1 + winRate*totalComparisons,
					Beta:  1 + (1-winRate)*totalComparisons,
				},
				TotalInteractions: rating.Comparisons,
				LastUpdated:       time.Now(),
			}

			if category == "_global" {
				r.globalPreferences[model] = pref
			} else {
				if r.categoryPreferences[category] == nil {
					r.categoryPreferences[category] = make(map[string]*ModelPreference)
				}
				r.categoryPreferences[category][model] = pref
			}
		}
	}

	logging.Infof("[RLDrivenSelector] Loaded preferences from storage")
	return nil
}

func (r *RLDrivenSelector) getAllPreferencesForStorage() map[string]map[string]*ModelRating {
	result := make(map[string]map[string]*ModelRating)

	r.globalMu.RLock()
	if len(r.globalPreferences) > 0 {
		result["_global"] = make(map[string]*ModelRating)
		for model, pref := range r.globalPreferences {
			result["_global"][model] = r.preferenceToRating(pref)
		}
	}
	r.globalMu.RUnlock()

	r.categoryMu.RLock()
	for cat, prefs := range r.categoryPreferences {
		result[cat] = make(map[string]*ModelRating)
		for model, pref := range prefs {
			result[cat][model] = r.preferenceToRating(pref)
		}
	}
	r.categoryMu.RUnlock()

	return result
}

func (r *RLDrivenSelector) preferenceToRating(pref *ModelPreference) *ModelRating {
	// Convert Beta distribution mean to Elo-like rating
	mean := pref.Distribution.Mean()
	rating := 1000 + mean*1000 // Scale to ~1000-2000

	// Estimate wins/losses from alpha/beta
	wins := int(pref.Distribution.Alpha - 1)
	losses := int(pref.Distribution.Beta - 1)
	if wins < 0 {
		wins = 0
	}
	if losses < 0 {
		losses = 0
	}

	return &ModelRating{
		Model:       pref.Model,
		Rating:      rating,
		Comparisons: pref.TotalInteractions,
		Wins:        wins,
		Losses:      losses,
	}
}

// Close stops storage operations
func (r *RLDrivenSelector) Close() error {
	if r.storage != nil {
		return r.storage.Close()
	}
	return nil
}

// GetLeaderboard returns models sorted by win probability
func (r *RLDrivenSelector) GetLeaderboard(category string) []*ModelPreference {
	var preferences []*ModelPreference

	if category != "" {
		r.categoryMu.RLock()
		if catPrefs, ok := r.categoryPreferences[category]; ok {
			for _, p := range catPrefs {
				preferences = append(preferences, p)
			}
		}
		r.categoryMu.RUnlock()
	} else {
		r.globalMu.RLock()
		for _, p := range r.globalPreferences {
			preferences = append(preferences, p)
		}
		r.globalMu.RUnlock()
	}

	// Sort by mean descending
	sort.Slice(preferences, func(i, j int) bool {
		return preferences[i].Distribution.Mean() > preferences[j].Distribution.Mean()
	})

	return preferences
}

// GetUserLeaderboard returns models sorted by user-specific preferences
func (r *RLDrivenSelector) GetUserLeaderboard(userID string) []*ModelPreference {
	var preferences []*ModelPreference

	r.userMu.RLock()
	if userPrefs, ok := r.userPreferences[userID]; ok {
		for _, p := range userPrefs {
			preferences = append(preferences, p)
		}
	}
	r.userMu.RUnlock()

	// Sort by mean descending
	sort.Slice(preferences, func(i, j int) bool {
		return preferences[i].Distribution.Mean() > preferences[j].Distribution.Mean()
	})

	return preferences
}

// GetDebugState returns the current state for debugging
func (r *RLDrivenSelector) GetDebugState(userID string) map[string]interface{} {
	state := map[string]interface{}{
		"config": map[string]interface{}{
			"use_thompson_sampling":  r.config.UseThompsonSampling,
			"exploration_rate":       r.config.ExplorationRate,
			"enable_personalization": r.config.EnablePersonalization,
			"use_router_r1_rewards":  r.config.UseRouterR1Rewards,
		},
	}

	// Global preferences
	globalPrefs := make(map[string]interface{})
	r.globalMu.RLock()
	for model, pref := range r.globalPreferences {
		globalPrefs[model] = map[string]interface{}{
			"alpha":        pref.Distribution.Alpha,
			"beta":         pref.Distribution.Beta,
			"mean":         pref.Distribution.Mean(),
			"variance":     pref.Distribution.Variance(),
			"interactions": pref.TotalInteractions,
		}
	}
	r.globalMu.RUnlock()
	state["global_preferences"] = globalPrefs

	// User-specific preferences if requested
	if userID != "" {
		userPrefs := make(map[string]interface{})
		r.userMu.RLock()
		if prefs, ok := r.userPreferences[userID]; ok {
			for model, pref := range prefs {
				userPrefs[model] = map[string]interface{}{
					"alpha":        pref.Distribution.Alpha,
					"beta":         pref.Distribution.Beta,
					"mean":         pref.Distribution.Mean(),
					"variance":     pref.Distribution.Variance(),
					"interactions": pref.TotalInteractions,
				}
			}
		}
		r.userMu.RUnlock()
		state["user_preferences"] = userPrefs
	}

	// All users summary
	r.userMu.RLock()
	userCount := len(r.userPreferences)
	allUsers := make([]string, 0, len(r.userPreferences))
	for uid := range r.userPreferences {
		allUsers = append(allUsers, uid)
	}
	r.userMu.RUnlock()
	state["total_users"] = userCount
	state["users"] = allUsers

	return state
}
