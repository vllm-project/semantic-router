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

package looper

import (
	"context"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ConfidenceLooper implements confidence model selection.
// It tries smaller models first and escalates to larger models if confidence is low.
// Models are ordered by their param_size in ModelParams (e.g., "10b", "5b", "100m").
type ConfidenceLooper struct {
	*BaseLooper
}

// NewConfidenceLooper creates a new ConfidenceLooper instance
func NewConfidenceLooper(cfg *config.LooperConfig) *ConfidenceLooper {
	return &ConfidenceLooper{
		BaseLooper: NewBaseLooper(cfg),
	}
}

// parseParamSize parses a param_size string (e.g., "10b", "5b", "100m") into a comparable integer
// Returns the number of parameters in millions (e.g., "10b" -> 10000, "100m" -> 100)
func parseParamSize(size string) int64 {
	if size == "" {
		return 0
	}

	size = strings.ToLower(strings.TrimSpace(size))

	// Match pattern like "10b", "5.5b", "100m", "500k"
	re := regexp.MustCompile(`^([0-9.]+)([bBmMkK]?)$`)
	matches := re.FindStringSubmatch(size)
	if len(matches) < 2 {
		return 0
	}

	numStr := matches[1]
	unit := ""
	if len(matches) >= 3 {
		unit = strings.ToLower(matches[2])
	}

	num, err := strconv.ParseFloat(numStr, 64)
	if err != nil {
		return 0
	}

	// Convert to millions for comparison
	switch unit {
	case "b": // billions
		return int64(num * 1000)
	case "m": // millions
		return int64(num)
	case "k": // thousands
		return int64(num / 1000)
	default: // assume billions if no unit
		return int64(num * 1000)
	}
}

// sortModelRefsBySize sorts ModelRefs by their param_size (from ModelParams) in ascending order (smallest first)
// modelParams maps model names to their configuration (including param_size)
func sortModelRefsBySize(refs []config.ModelRef, modelParams map[string]config.ModelParams) []config.ModelRef {
	// Create a copy to avoid modifying the original
	sorted := make([]config.ModelRef, len(refs))
	copy(sorted, refs)

	// Helper function to get param_size for a model ref
	getParamSize := func(ref config.ModelRef) string {
		modelName := ref.Model
		if modelParams != nil {
			if params, ok := modelParams[modelName]; ok {
				return params.ParamSize
			}
		}
		return ""
	}

	sort.SliceStable(sorted, func(i, j int) bool {
		sizeI := parseParamSize(getParamSize(sorted[i]))
		sizeJ := parseParamSize(getParamSize(sorted[j]))
		return sizeI < sizeJ
	})

	return sorted
}

// sortModelRefsByCost sorts ModelRefs by their pricing (cheapest first)
// Uses prompt_per_1m from ModelParams.Pricing
func sortModelRefsByCost(refs []config.ModelRef, modelParams map[string]config.ModelParams) []config.ModelRef {
	sorted := make([]config.ModelRef, len(refs))
	copy(sorted, refs)

	// Helper function to get cost for a model ref
	getCost := func(ref config.ModelRef) float64 {
		if modelParams != nil {
			if params, ok := modelParams[ref.Model]; ok {
				return params.Pricing.PromptPer1M
			}
		}
		return math.MaxFloat64 // Unknown cost goes last
	}

	sort.SliceStable(sorted, func(i, j int) bool {
		return getCost(sorted[i]) < getCost(sorted[j])
	})

	return sorted
}

// sortModelRefsByAutoMix sorts ModelRefs using POMDP-inspired cost-quality optimization
// Models are scored by: value = (1 - tradeoff) * quality + tradeoff * (1 - normalized_cost)
// Lower tradeoff values favor quality; higher values favor cost savings
func sortModelRefsByAutoMix(refs []config.ModelRef, modelParams map[string]config.ModelParams, tradeoff float64) []config.ModelRef {
	sorted := make([]config.ModelRef, len(refs))
	copy(sorted, refs)

	// First, compute min/max cost for normalization
	minCost, maxCost := math.MaxFloat64, 0.0
	for _, ref := range refs {
		if modelParams != nil {
			if params, ok := modelParams[ref.Model]; ok {
				cost := params.Pricing.PromptPer1M
				if cost > 0 {
					if cost < minCost {
						minCost = cost
					}
					if cost > maxCost {
						maxCost = cost
					}
				}
			}
		}
	}

	// Prevent division by zero
	costRange := maxCost - minCost
	if costRange <= 0 {
		costRange = 1.0
	}

	// Helper to compute AutoMix value for a model
	getValue := func(ref config.ModelRef) float64 {
		quality := 0.5  // Default quality estimate
		costScore := 0.5 // Default cost score (mid-range)

		if modelParams != nil {
			if params, ok := modelParams[ref.Model]; ok {
				// Estimate quality from param_size (larger = higher quality)
				size := parseParamSize(params.ParamSize)
				if size > 0 {
					// Normalize size: assume 1B-70B range maps to 0.3-1.0 quality
					quality = 0.3 + 0.7*math.Min(float64(size)/70000, 1.0)
				}

				// Normalize cost: 0 = most expensive, 1 = cheapest
				cost := params.Pricing.PromptPer1M
				if cost > 0 && costRange > 0 {
					costScore = 1.0 - (cost-minCost)/costRange
				}
			}
		}

		// POMDP-inspired value function:
		// value = (1 - tradeoff) * quality + tradeoff * costScore
		// When tradeoff = 0: pure quality ordering
		// When tradeoff = 1: pure cost ordering (cheapest first)
		// When tradeoff = 0.3: favor quality but consider cost
		value := (1-tradeoff)*quality + tradeoff*costScore
		return value
	}

	// Sort by value ascending (start with lower-value/cheaper models for cascading)
	// This matches AutoMix behavior: try cheaper/smaller models first
	sort.SliceStable(sorted, func(i, j int) bool {
		// For cascading: we want to try "worse" (cheaper/smaller) models first
		// So we sort by value ASCENDING to start with lower-value options
		return getValue(sorted[i]) < getValue(sorted[j])
	})

	return sorted
}

// getEscalationOrder returns the configured escalation order, defaulting to "size"
func getEscalationOrder(cfg *config.ConfidenceAlgorithmConfig) string {
	if cfg == nil || cfg.EscalationOrder == "" {
		return "size"
	}
	return cfg.EscalationOrder
}

// getCostQualityTradeoff returns the configured tradeoff, defaulting to 0.3
func getCostQualityTradeoff(cfg *config.ConfidenceAlgorithmConfig) float64 {
	if cfg == nil || cfg.CostQualityTradeoff <= 0 {
		return 0.3
	}
	return cfg.CostQualityTradeoff
}

// ConfidenceEvaluator evaluates model response confidence based on configured method
type ConfidenceEvaluator struct {
	Method        string  // "avg_logprob", "margin", or "hybrid"
	Threshold     float64 // Threshold for the chosen method
	LogprobWeight float64 // Weight for logprob in hybrid mode
	MarginWeight  float64 // Weight for margin in hybrid mode
}

// NewConfidenceEvaluator creates a confidence evaluator from algorithm config
func NewConfidenceEvaluator(cfg *config.ConfidenceAlgorithmConfig) *ConfidenceEvaluator {
	eval := &ConfidenceEvaluator{
		Method:        "avg_logprob", // Default method
		Threshold:     -1.0,          // Default threshold for avg_logprob
		LogprobWeight: 0.5,
		MarginWeight:  0.5,
	}

	if cfg == nil {
		return eval
	}

	// Set method
	if cfg.ConfidenceMethod != "" {
		eval.Method = cfg.ConfidenceMethod
	}

	// Set threshold based on method
	if cfg.Threshold != 0 {
		eval.Threshold = cfg.Threshold
	} else {
		// Set sensible defaults based on method
		switch eval.Method {
		case "avg_logprob":
			eval.Threshold = -1.0 // Very permissive
		case "margin":
			eval.Threshold = 0.5 // Moderate confidence
		case "hybrid":
			eval.Threshold = 0.5 // Normalized score
		}
	}

	// Set hybrid weights
	if cfg.HybridWeights != nil {
		if cfg.HybridWeights.LogprobWeight > 0 {
			eval.LogprobWeight = cfg.HybridWeights.LogprobWeight
		}
		if cfg.HybridWeights.MarginWeight > 0 {
			eval.MarginWeight = cfg.HybridWeights.MarginWeight
		}
	}

	return eval
}

// normalizeLogprob converts avg_logprob to 0-1 range
// Input range: typically -10 to 0 (closer to 0 = more confident)
// Output range: 0 to 1 (1 = most confident)
func normalizeLogprob(avgLogprob float64) float64 {
	// Map -3 to 0 -> 0 to 1 (values below -3 are clamped to 0)
	normalized := (avgLogprob + 3.0) / 3.0
	if normalized < 0 {
		normalized = 0
	}
	if normalized > 1 {
		normalized = 1
	}
	return normalized
}

// normalizeMargin converts margin to 0-1 range
// Input range: typically 0 to 10+ (higher = more confident)
// Output range: 0 to 1 (1 = most confident)
func normalizeMargin(margin float64) float64 {
	// Use sigmoid-like transformation for smoother mapping
	// margin=0 -> 0, margin=2 -> ~0.67, margin=5 -> ~0.91, margin=10 -> ~0.99
	// Formula: 1 - exp(-margin/3)
	if margin <= 0 {
		return 0
	}
	normalized := 1.0 - math.Exp(-margin/3.0)
	if normalized > 1 {
		normalized = 1
	}
	return normalized
}

// Evaluate checks if the response meets the confidence threshold
// All methods return normalized confidence in 0-1 range (1 = most confident)
// Returns (confidence_score, meets_threshold)
func (e *ConfidenceEvaluator) Evaluate(resp *ModelResponse) (float64, bool) {
	switch e.Method {
	case "margin":
		// Use average margin between top-1 and top-2 logprobs
		// Normalized to 0-1 range
		confidence := normalizeMargin(resp.AverageMargin)
		return confidence, confidence >= e.Threshold

	case "hybrid":
		// Combine both methods with weights
		normalizedLogprob := normalizeLogprob(resp.AverageLogprob)
		normalizedMargin := normalizeMargin(resp.AverageMargin)
		confidence := e.LogprobWeight*normalizedLogprob + e.MarginWeight*normalizedMargin
		return confidence, confidence >= e.Threshold

	default: // "avg_logprob"
		// Use average logprob across all tokens
		// Normalized to 0-1 range
		confidence := normalizeLogprob(resp.AverageLogprob)
		return confidence, confidence >= e.Threshold
	}
}

// NeedsLogprobs returns whether this evaluator needs logprobs enabled
func (e *ConfidenceEvaluator) NeedsLogprobs() bool {
	return true // All methods need logprobs
}

// NeedsTopLogprobs returns the number of top_logprobs needed (0 if not needed)
func (e *ConfidenceEvaluator) NeedsTopLogprobs() int {
	switch e.Method {
	case "margin", "hybrid":
		return 2 // Need at least 2 for margin calculation
	default:
		return 0 // avg_logprob doesn't need top_logprobs
	}
}

// Execute implements the confidence algorithm:
// 1. Sort models by param_size in ascending order (smallest first)
// 2. Try smallest model first
// 3. If confidence is below threshold, try next larger model
// 4. Continue until confidence is acceptable or all models tried
func (l *ConfidenceLooper) Execute(ctx context.Context, req *Request) (*Response, error) {
	if len(req.ModelRefs) == 0 {
		return nil, fmt.Errorf("no models configured")
	}

	// Get config from algorithm
	onError := "skip"
	var sizeAwareCfg *config.ConfidenceAlgorithmConfig
	if req.Algorithm != nil && req.Algorithm.Confidence != nil {
		sizeAwareCfg = req.Algorithm.Confidence
		if sizeAwareCfg.OnError != "" {
			onError = sizeAwareCfg.OnError
		}
	}

	// Create confidence evaluator based on config
	evaluator := NewConfidenceEvaluator(sizeAwareCfg)

	// Configure logprobs based on evaluator needs
	logprobsCfg := &LogprobsConfig{
		Enabled:     evaluator.NeedsLogprobs(),
		TopLogprobs: evaluator.NeedsTopLogprobs(),
	}

	// Sort models based on configured escalation order
	escalationOrder := getEscalationOrder(sizeAwareCfg)
	var sortedRefs []config.ModelRef

	switch escalationOrder {
	case "cost":
		// AutoMix-style: order by pricing (cheapest first)
		sortedRefs = sortModelRefsByCost(req.ModelRefs, req.ModelParams)
		logging.Infof("[ConfidenceLooper] Using cost-based escalation order (cheapest first)")
	case "automix":
		// POMDP-optimized: cost-quality tradeoff
		tradeoff := getCostQualityTradeoff(sizeAwareCfg)
		sortedRefs = sortModelRefsByAutoMix(req.ModelRefs, req.ModelParams, tradeoff)
		logging.Infof("[ConfidenceLooper] Using AutoMix escalation order (tradeoff=%.2f)", tradeoff)
	default:
		// Default: order by param_size (smallest first)
		sortedRefs = sortModelRefsBySize(req.ModelRefs, req.ModelParams)
		logging.Infof("[ConfidenceLooper] Using size-based escalation order (smallest first)")
	}

	logging.Infof("[ConfidenceLooper] Starting with %d models, method=%s, threshold=%.4f, on_error=%s, streaming=%v, escalation=%s",
		len(sortedRefs), evaluator.Method, evaluator.Threshold, onError, req.IsStreaming, escalationOrder)

	// Helper to get param_size for logging
	getParamSize := func(modelName string) string {
		if req.ModelParams != nil {
			if params, ok := req.ModelParams[modelName]; ok {
				return params.ParamSize
			}
		}
		return ""
	}

	// Log the sorted order
	for i, ref := range sortedRefs {
		modelName := ref.Model
		if ref.LoRAName != "" {
			modelName = ref.LoRAName
		}
		logging.Debugf("[ConfidenceLooper] Model order[%d]: %s (param_size=%s)", i, modelName, getParamSize(ref.Model))
	}

	var lastResponse *ModelResponse
	var modelsUsed []string
	iteration := 0

	for _, modelRef := range sortedRefs {
		iteration++
		modelName := modelRef.Model
		if modelRef.LoRAName != "" {
			modelName = modelRef.LoRAName
		}

		// Get access key from model params
		accessKey := ""
		if req.ModelParams != nil {
			if params, ok := req.ModelParams[modelRef.Model]; ok {
				accessKey = params.AccessKey
			}
		}

		logging.Infof("[ConfidenceLooper] Trying model: %s (iteration=%d)", modelName, iteration)

		resp, err := l.client.CallModel(ctx, req.OriginalRequest, modelName, false, iteration, logprobsCfg, accessKey)
		if err != nil {
			logging.Errorf("[ConfidenceLooper] Model %s failed: %v", modelName, err)
			if onError == "fail" {
				return nil, fmt.Errorf("model %s failed: %w", modelName, err)
			}
			continue
		}

		lastResponse = resp
		modelsUsed = append(modelsUsed, modelName)

		// Evaluate confidence using configured method
		confidence, meetsThreshold := evaluator.Evaluate(resp)

		logging.Infof("[ConfidenceLooper] Model %s: confidence=%.4f (method=%s), threshold=%.4f, meets=%v",
			modelName, confidence, evaluator.Method, evaluator.Threshold, meetsThreshold)

		if meetsThreshold {
			logging.Infof("[ConfidenceLooper] Confidence acceptable, using model %s", modelName)
			break
		}

		logging.Infof("[ConfidenceLooper] Confidence below threshold, trying next model")
	}

	if lastResponse == nil {
		return nil, fmt.Errorf("all models failed")
	}

	// Format the final response (only include the last response, not aggregated)
	agg := &AggregatedResponse{
		Models:          modelsUsed,
		Responses:       []*ModelResponse{lastResponse},
		CombinedContent: lastResponse.Content,
		FinalModel:      lastResponse.Model,
		AverageLogprob:  lastResponse.AverageLogprob,
	}

	if req.IsStreaming {
		return l.formatConfidenceStreamingResponse(agg, modelsUsed, iteration)
	}
	return l.formatConfidenceJSONResponse(agg, modelsUsed, iteration)
}

// formatConfidenceJSONResponse creates response with confidence algorithm type
func (l *ConfidenceLooper) formatConfidenceJSONResponse(agg *AggregatedResponse, modelsUsed []string, iterations int) (*Response, error) {
	resp, err := l.formatJSONResponse(agg, modelsUsed, iterations)
	if err != nil {
		return nil, err
	}
	resp.AlgorithmType = "confidence"
	return resp, nil
}

// formatConfidenceStreamingResponse creates response with confidence algorithm type
func (l *ConfidenceLooper) formatConfidenceStreamingResponse(agg *AggregatedResponse, modelsUsed []string, iterations int) (*Response, error) {
	resp, err := l.formatStreamingResponse(agg, modelsUsed, iterations)
	if err != nil {
		return nil, err
	}
	resp.AlgorithmType = "confidence"
	return resp, nil
}
