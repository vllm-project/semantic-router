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
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// SelfVerificationPrompt is the prompt template for AutoMix self-verification
// The model evaluates its own answer and provides a confidence score
const SelfVerificationPrompt = `You are evaluating the quality of an AI assistant's response.

Original Question: %s

AI's Response: %s

Rate the quality and correctness of this response on a scale of 0.0 to 1.0:
- 1.0 = Completely correct, comprehensive, and well-explained
- 0.8 = Mostly correct with minor issues
- 0.6 = Partially correct but missing important details
- 0.4 = Has some correct elements but significant errors
- 0.2 = Mostly incorrect or irrelevant
- 0.0 = Completely wrong or harmful

Respond with ONLY a JSON object in this format:
{"confidence": 0.X, "reason": "brief explanation"}
`

// SelfVerificationResult represents the parsed result of self-verification
type SelfVerificationResult struct {
	Confidence float64 `json:"confidence"`
	Reason     string  `json:"reason"`
}

// selfVerificationExecution separates call accounting from verification
// success. A completed model response remains paid evidence even when its
// verification payload cannot be parsed.
type selfVerificationExecution struct {
	Confidence float64
	Accepted   bool
	Response   *ModelResponse
	Attempted  bool
}

// parseSelfVerification parses the model's self-verification response
func parseSelfVerification(response string) (*SelfVerificationResult, error) {
	// Try to extract JSON from the response
	response = strings.TrimSpace(response)

	// Find JSON object in response
	startIdx := strings.Index(response, "{")
	endIdx := strings.LastIndex(response, "}")
	if startIdx == -1 || endIdx == -1 || endIdx < startIdx {
		// Try to extract just a number
		re := regexp.MustCompile(`([0-9]+\.?[0-9]*)`)
		matches := re.FindStringSubmatch(response)
		if len(matches) >= 2 {
			confidence, err := strconv.ParseFloat(matches[1], 64)
			if err == nil && confidence >= 0 && confidence <= 1 {
				return &SelfVerificationResult{Confidence: confidence, Reason: "parsed from numeric response"}, nil
			}
		}
		return nil, fmt.Errorf("no valid JSON or confidence value found in response")
	}

	jsonStr := response[startIdx : endIdx+1]
	var result SelfVerificationResult
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		return nil, fmt.Errorf("failed to parse self-verification JSON: %w", err)
	}

	// Clamp confidence to valid range
	if result.Confidence < 0 {
		result.Confidence = 0
	}
	if result.Confidence > 1 {
		result.Confidence = 1
	}

	return &result, nil
}

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
		quality := 0.5   // Default quality estimate
		costScore := 0.5 // Default cost score (mid-range)

		if modelParams != nil {
			if params, ok := modelParams[ref.Model]; ok {
				// Use configured QualityScore if available
				if params.QualityScore > 0 && params.QualityScore <= 1.0 {
					quality = params.QualityScore
				} else {
					// Fallback: estimate quality from param_size (larger = higher quality)
					size := parseParamSize(params.ParamSize)
					if size > 0 {
						// Normalize size: assume 1B-70B range maps to 0.3-1.0 quality
						quality = 0.3 + 0.7*math.Min(float64(size)/70000, 1.0)
					}
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

// MethodAutoMixEntailment is the confidence method name for paper-faithful
// AutoMix self-verification (arXiv:2310.12963 §3.2). Unlike "self_verify",
// which prompts the generation model to grade its own answer, this method
// delegates verification to a separate entailment verifier reached over HTTP
// via selection.AutoMixVerifierClient (reference server:
// src/training/model_selection/rl_model_selection/automix_verifier.py).
const MethodAutoMixEntailment = "automix_entailment"

// ConfidenceEvaluator evaluates model response confidence based on configured method
type ConfidenceEvaluator struct {
	Method        string  // "avg_logprob", "margin", "hybrid", "self_verify", or "automix_entailment"
	Threshold     float64 // Threshold for the chosen method
	LogprobWeight float64 // Weight for logprob in hybrid mode
	MarginWeight  float64 // Weight for margin in hybrid mode
	TokenFilter   string  // "all", "tool_call_args" — selects which tokens feed confidence

	// VerifierServerURL and VerifierTimeoutSeconds carry the AutoMix
	// entailment verifier configuration when Method == "automix_entailment".
	// Empty/zero values when the method is anything else.
	VerifierServerURL      string
	VerifierTimeoutSeconds int
	MaxResponseBytes       int64
}

// NewConfidenceEvaluator creates a confidence evaluator from algorithm config
func NewConfidenceEvaluator(cfg *config.ConfidenceAlgorithmConfig) *ConfidenceEvaluator {
	eval := &ConfidenceEvaluator{
		Method:           "avg_logprob", // Default method
		Threshold:        -1.0,          // Default threshold for avg_logprob
		LogprobWeight:    0.5,
		MarginWeight:     0.5,
		MaxResponseBytes: config.DefaultMaxResponseBytes,
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
		case "self_verify":
			eval.Threshold = 0.7 // AutoMix paper default
		case MethodAutoMixEntailment:
			eval.Threshold = 0.7 // AutoMix paper default (k-sample entailment)
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

	if cfg.TokenFilter != "" {
		eval.TokenFilter = cfg.TokenFilter
	}

	eval.VerifierServerURL = cfg.VerifierServerURL
	eval.VerifierTimeoutSeconds = cfg.VerifierTimeoutSeconds
	if cfg.MaxResponseBytes > 0 {
		eval.MaxResponseBytes = cfg.MaxResponseBytes
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

// Evaluate checks if the response meets the confidence threshold.
// When a TokenFilter is active and filtered metrics are available, those are
// used instead of the all-token averages.
// All methods return normalized confidence in 0-1 range (1 = most confident).
// Logprob-based methods reject responses without token logprobs so the float64
// zero value cannot be mistaken for a perfectly confident average logprob of 0.
func (e *ConfidenceEvaluator) Evaluate(resp *ModelResponse) (float64, bool) {
	confidence, accepted, _ := e.evaluate(resp)
	return confidence, accepted
}

func (e *ConfidenceEvaluator) evaluate(resp *ModelResponse) (float64, bool, error) {
	if resp == nil {
		return 0, false, fmt.Errorf("confidence response is nil")
	}
	if e.NeedsLogprobs() && len(resp.Logprobs) == 0 {
		return 0, false, fmt.Errorf("backend returned no token logprobs required by confidence method %q", e.Method)
	}
	if (e.Method == "margin" || e.Method == "hybrid") && !resp.MarginEvidenceComplete {
		return 0, false, fmt.Errorf(
			"backend returned no complete top-logprob alternatives required by confidence method %q",
			e.Method,
		)
	}

	avgLP := resp.AverageLogprob
	avgM := resp.AverageMargin

	useFiltered := e.TokenFilter != "" && e.TokenFilter != "all"
	if useFiltered && resp.FilteredTokenCount > 0 {
		avgLP = resp.FilteredAverageLogprob
		avgM = resp.FilteredAverageMargin
	}

	switch e.Method {
	case "margin":
		confidence := normalizeMargin(avgM)
		return confidence, confidence >= e.Threshold, nil

	case "hybrid":
		normalizedLogprob := normalizeLogprob(avgLP)
		normalizedMargin := normalizeMargin(avgM)
		confidence := e.LogprobWeight*normalizedLogprob + e.MarginWeight*normalizedMargin
		return confidence, confidence >= e.Threshold, nil

	default: // "avg_logprob"
		confidence := normalizeLogprob(avgLP)
		return confidence, confidence >= e.Threshold, nil
	}
}

// NeedsLogprobs returns whether this evaluator needs logprobs enabled
func (e *ConfidenceEvaluator) NeedsLogprobs() bool {
	// self_verify and automix_entailment use external verification signals
	// rather than logprobs from the generation call.
	switch e.Method {
	case "self_verify", MethodAutoMixEntailment:
		return false
	}
	return true
}

// IsSelfVerify returns true if using AutoMix self-verification method
func (e *ConfidenceEvaluator) IsSelfVerify() bool {
	return e.Method == "self_verify"
}

// IsAutoMixEntailment returns true when the evaluator delegates verification
// to an external AutoMix entailment server (arXiv:2310.12963 §3.2).
func (e *ConfidenceEvaluator) IsAutoMixEntailment() bool {
	return e.Method == MethodAutoMixEntailment
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

func confidenceModelCallStreaming(clientStreaming bool, evaluator *ConfidenceEvaluator) bool {
	return clientStreaming && !evaluator.NeedsLogprobs()
}

// Execute implements the confidence algorithm:
// 1. Sort models by param_size in ascending order (smallest first)
// 2. Try smallest model first
// 3. If confidence is below threshold, try next larger model
// 4. Continue until confidence is acceptable or all models tried
// Iterations counts actual backend attempts, including failed dispatches and
// the second model call used by self_verify.
func (l *ConfidenceLooper) Execute(ctx context.Context, req *Request) (*Response, error) {
	if len(req.ModelRefs) == 0 {
		return nil, fmt.Errorf("no models configured")
	}

	// Set decision name in client for header transmission
	l.client.SetDecisionName(req.DecisionName)

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
	case config.ConfidenceEscalationOrderDeclared:
		sortedRefs = append([]config.ModelRef(nil), req.ModelRefs...)
		logging.ComponentDebugEvent("looper", "confidence_escalation_order_selected", map[string]interface{}{
			"looper":           "confidence",
			"decision":         req.DecisionName,
			"escalation_order": escalationOrder,
			"strategy":         "declared",
		})
	case "cost":
		// AutoMix-style: order by pricing (cheapest first)
		sortedRefs = sortModelRefsByCost(req.ModelRefs, req.ModelParams)
		logging.ComponentDebugEvent("looper", "confidence_escalation_order_selected", map[string]interface{}{
			"looper":           "confidence",
			"decision":         req.DecisionName,
			"escalation_order": escalationOrder,
			"strategy":         "cost",
		})
	case "automix":
		// POMDP-optimized: cost-quality tradeoff
		tradeoff := getCostQualityTradeoff(sizeAwareCfg)
		sortedRefs = sortModelRefsByAutoMix(req.ModelRefs, req.ModelParams, tradeoff)
		logging.ComponentDebugEvent("looper", "confidence_escalation_order_selected", map[string]interface{}{
			"looper":           "confidence",
			"decision":         req.DecisionName,
			"escalation_order": escalationOrder,
			"strategy":         "automix",
			"tradeoff":         tradeoff,
		})
	default:
		// Default: order by param_size (smallest first)
		sortedRefs = sortModelRefsBySize(req.ModelRefs, req.ModelParams)
		logging.ComponentDebugEvent("looper", "confidence_escalation_order_selected", map[string]interface{}{
			"looper":           "confidence",
			"decision":         req.DecisionName,
			"escalation_order": escalationOrder,
			"strategy":         "size",
		})
	}

	tokenFilterLabel := "all"
	if evaluator.TokenFilter != "" {
		tokenFilterLabel = evaluator.TokenFilter
	}
	logging.ComponentEvent("looper", "execution_started", map[string]interface{}{
		"looper":           "confidence",
		"decision":         req.DecisionName,
		"candidate_models": len(sortedRefs),
		"method":           evaluator.Method,
		"threshold":        evaluator.Threshold,
		"token_filter":     tokenFilterLabel,
		"on_error":         onError,
		"streaming":        req.IsStreaming,
		"escalation_order": escalationOrder,
	})

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
		logging.ComponentDebugEvent("looper", "confidence_model_order", map[string]interface{}{
			"looper":     "confidence",
			"decision":   req.DecisionName,
			"index":      i,
			"model_ref":  modelName,
			"param_size": getParamSize(ref.Model),
		})
	}

	var lastResponse *ModelResponse
	var allResponses []*ModelResponse
	var modelsUsed []string
	var lastEvaluationErr error
	attempts := 0
	partialExecutionError := func(cause error) error {
		return newConfidencePartialExecutionError(cause, allResponses, modelsUsed, attempts)
	}

	for _, modelRef := range sortedRefs {
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

		logging.ComponentDebugEvent("looper", "model_dispatch_started", map[string]interface{}{
			"looper":    "confidence",
			"decision":  req.DecisionName,
			"model_ref": modelName,
			"iteration": attempts + 1,
		})

		attempts++
		resp, err := l.client.CallModel(
			ctx,
			req.OriginalRequest,
			modelName,
			confidenceModelCallStreaming(req.IsStreaming, evaluator),
			attempts,
			logprobsCfg,
			accessKey,
		)
		if err != nil {
			logging.ComponentWarnEvent("looper", "model_dispatch_failed", map[string]interface{}{
				"looper":    "confidence",
				"decision":  req.DecisionName,
				"model_ref": modelName,
				"iteration": attempts,
				"error":     err.Error(),
			})
			if onError == "fail" {
				return nil, partialExecutionError(fmt.Errorf("model %s failed: %w", modelName, err))
			}
			continue
		}

		// Apply token filter before confidence evaluation (e.g., exclude JSON boilerplate)
		ApplyTokenFilter(resp, evaluator.TokenFilter)
		// The call has already consumed backend capacity even if its confidence
		// evidence is unusable. Preserve every paid attempt for usage and replay.
		allResponses = append(allResponses, resp)
		modelsUsed = append(modelsUsed, modelName)

		var confidence float64
		var meetsThreshold bool

		// Evaluate confidence using configured method
		switch {
		case evaluator.IsAutoMixEntailment():
			// AutoMix entailment cascade (arXiv:2310.12963 §3.2): delegate
			// verification to an external few-shot entailment server.
			var verifyErr error
			confidence, meetsThreshold, verifyErr = l.performAutoMixEntailment(ctx, req, evaluator, modelName, resp.Content)
			if verifyErr != nil {
				logging.ComponentWarnEvent("looper", "automix_entailment_failed", map[string]interface{}{
					"looper":    "confidence",
					"decision":  req.DecisionName,
					"model_ref": modelName,
					"error":     verifyErr.Error(),
				})
				if onError == "fail" {
					return nil, partialExecutionError(fmt.Errorf(
						"automix_entailment verification for %s failed: %w",
						modelName,
						verifyErr,
					))
				}
			}
			logging.ComponentDebugEvent("looper", "automix_entailment_completed", map[string]interface{}{
				"looper":     "confidence",
				"decision":   req.DecisionName,
				"model_ref":  modelName,
				"confidence": confidence,
				"threshold":  evaluator.Threshold,
				"accepted":   meetsThreshold,
			})
		case evaluator.IsSelfVerify():
			// AutoMix self-verification: ask the model to evaluate its own answer
			verificationIteration := attempts + 1
			verification, verifyErr := l.performSelfVerification(
				ctx,
				req,
				modelName,
				resp.Content,
				accessKey,
				evaluator.Threshold,
				verificationIteration,
			)
			if verification.Attempted {
				attempts++
				modelsUsed = append(modelsUsed, modelName)
			}
			if verification.Response != nil {
				allResponses = append(allResponses, verification.Response)
			}
			if verifyErr != nil {
				lastEvaluationErr = fmt.Errorf("self verification for model %q failed: %w", modelName, verifyErr)
				logging.ComponentWarnEvent("looper", "self_verification_failed", map[string]interface{}{
					"looper":    "confidence",
					"decision":  req.DecisionName,
					"model_ref": modelName,
					"iteration": verificationIteration,
					"attempted": verification.Attempted,
					"error":     verifyErr.Error(),
				})
				if onError == "fail" {
					return nil, partialExecutionError(lastEvaluationErr)
				}
				continue
			}
			confidence = verification.Confidence
			meetsThreshold = verification.Accepted
			logging.ComponentDebugEvent("looper", "self_verification_completed", map[string]interface{}{
				"looper":     "confidence",
				"decision":   req.DecisionName,
				"model_ref":  modelName,
				"confidence": confidence,
				"threshold":  evaluator.Threshold,
				"accepted":   meetsThreshold,
			})
		default:
			var evaluateErr error
			confidence, meetsThreshold, evaluateErr = evaluator.evaluate(resp)
			if evaluateErr != nil {
				lastEvaluationErr = fmt.Errorf("confidence evaluation for model %q failed: %w", modelName, evaluateErr)
				logging.ComponentWarnEvent("looper", "confidence_evaluation_failed", map[string]interface{}{
					"looper":    "confidence",
					"decision":  req.DecisionName,
					"model_ref": modelName,
					"iteration": attempts,
					"method":    evaluator.Method,
					"error":     evaluateErr.Error(),
				})
				if onError == "fail" {
					return nil, partialExecutionError(lastEvaluationErr)
				}
				continue
			}
			if evaluator.TokenFilter != "" && evaluator.TokenFilter != "all" && resp.FilteredTokenCount > 0 {
				logging.ComponentDebugEvent("looper", "confidence_evaluated", map[string]interface{}{
					"looper":             "confidence",
					"decision":           req.DecisionName,
					"model_ref":          modelName,
					"confidence":         confidence,
					"method":             evaluator.Method,
					"token_filter":       evaluator.TokenFilter,
					"threshold":          evaluator.Threshold,
					"accepted":           meetsThreshold,
					"unfiltered_logprob": resp.AverageLogprob,
				})
			} else {
				logging.ComponentDebugEvent("looper", "confidence_evaluated", map[string]interface{}{
					"looper":     "confidence",
					"decision":   req.DecisionName,
					"model_ref":  modelName,
					"confidence": confidence,
					"method":     evaluator.Method,
					"threshold":  evaluator.Threshold,
					"accepted":   meetsThreshold,
				})
			}
		}

		lastResponse = resp

		if meetsThreshold {
			logging.ComponentEvent("looper", "execution_completed", map[string]interface{}{
				"looper":         "confidence",
				"decision":       req.DecisionName,
				"models_used":    modelsUsed,
				"iterations":     attempts,
				"selected_model": modelName,
				"reason":         "threshold_met",
			})
			break
		}

		logging.ComponentDebugEvent("looper", "confidence_threshold_not_met", map[string]interface{}{
			"looper":     "confidence",
			"decision":   req.DecisionName,
			"model_ref":  modelName,
			"confidence": confidence,
			"threshold":  evaluator.Threshold,
		})
	}

	if lastResponse == nil {
		if lastEvaluationErr != nil {
			return nil, partialExecutionError(fmt.Errorf("all models failed: %w", lastEvaluationErr))
		}
		return nil, partialExecutionError(fmt.Errorf("all models failed"))
	}

	// Publish only the last confidence-evaluated response. Usage and models-used
	// retain every paid attempt, including a skipped response with missing
	// logprob evidence.
	agg := &AggregatedResponse{
		Models:          modelsUsed,
		Responses:       []*ModelResponse{lastResponse},
		UsageResponses:  allResponses,
		CombinedContent: lastResponse.Content,
		FinalModel:      lastResponse.Model,
		AverageLogprob:  lastResponse.AverageLogprob,
		HasToolCalls:    lastResponse.HasToolCalls,
	}

	if req.IsStreaming {
		return l.formatConfidenceStreamingResponse(
			agg,
			modelsUsed,
			attempts,
			confidenceStreamUsageRequested(req),
		)
	}
	return l.formatConfidenceJSONResponse(agg, modelsUsed, attempts)
}

func confidenceStreamUsageRequested(req *Request) bool {
	return req != nil && req.OriginalRequest != nil &&
		req.OriginalRequest.StreamOptions.IncludeUsage.Or(false)
}

// performSelfVerification implements AutoMix self-verification
// The model evaluates its own answer and returns a confidence score
// This is the "True AutoMix Cascading" from the paper (arXiv:2310.12963)
func (l *ConfidenceLooper) performSelfVerification(
	ctx context.Context,
	req *Request,
	modelName string,
	responseContent string,
	accessKey string,
	threshold float64,
	iteration int,
) (selfVerificationExecution, error) {
	// Extract original question from the request
	originalQuestion := l.extractQuestionFromRequest(req.OriginalRequest)
	if originalQuestion == "" {
		return selfVerificationExecution{}, fmt.Errorf("could not extract user question from request")
	}

	// Build self-verification prompt
	verificationPrompt := fmt.Sprintf(SelfVerificationPrompt, originalQuestion, responseContent)

	// Create a new request for self-verification
	verifyRequest := l.buildSelfVerificationRequest(verificationPrompt)
	if verifyRequest == nil {
		return selfVerificationExecution{}, fmt.Errorf("could not build self-verification request")
	}

	logging.ComponentDebugEvent("looper", "self_verification_started", map[string]interface{}{
		"looper":    "confidence",
		"model_ref": modelName,
	})

	// Call the same model to evaluate its answer
	verifyResp, err := l.client.CallModel(ctx, verifyRequest, modelName, false, iteration, nil, accessKey)
	if err != nil {
		return selfVerificationExecution{Attempted: true}, fmt.Errorf("verifier model call failed: %w", err)
	}

	// Parse the self-verification result
	result, err := parseSelfVerification(verifyResp.Content)
	if err != nil {
		return selfVerificationExecution{
			Response:  verifyResp,
			Attempted: true,
		}, fmt.Errorf("could not parse verifier result: %w", err)
	}

	logging.ComponentDebugEvent("looper", "self_verification_result_parsed", map[string]interface{}{
		"looper":     "confidence",
		"model_ref":  modelName,
		"confidence": result.Confidence,
		"reason_len": len(result.Reason),
	})

	return selfVerificationExecution{
		Confidence: result.Confidence,
		Accepted:   result.Confidence >= threshold,
		Response:   verifyResp,
		Attempted:  true,
	}, nil
}

// extractQuestionFromRequest extracts the user's question from the original request
// Uses JSON marshaling for robust extraction across SDK versions
func (l *ConfidenceLooper) extractQuestionFromRequest(originalRequest *openai.ChatCompletionNewParams) string {
	if originalRequest == nil {
		return ""
	}

	// Marshal to JSON and parse to extract messages
	data, err := json.Marshal(originalRequest)
	if err != nil {
		return ""
	}

	var reqMap map[string]interface{}
	if err := json.Unmarshal(data, &reqMap); err != nil {
		return ""
	}

	messages, ok := reqMap["messages"].([]interface{})
	if !ok || len(messages) == 0 {
		return ""
	}

	// Find the last user message
	for i := len(messages) - 1; i >= 0; i-- {
		msg, ok := messages[i].(map[string]interface{})
		if !ok {
			continue
		}
		role, _ := msg["role"].(string)
		if role == "user" {
			// Content can be a string or array of parts
			switch content := msg["content"].(type) {
			case string:
				return content
			case []interface{}:
				// Array of content parts
				for _, part := range content {
					if partMap, ok := part.(map[string]interface{}); ok {
						if partMap["type"] == "text" {
							if text, ok := partMap["text"].(string); ok {
								return text
							}
						}
					}
				}
			}
		}
	}

	return ""
}

// buildSelfVerificationRequest creates a new request for self-verification
// Returns a new ChatCompletionNewParams for the verification call
func (l *ConfidenceLooper) buildSelfVerificationRequest(verificationPrompt string) *openai.ChatCompletionNewParams {
	// Build request via JSON for SDK compatibility
	verifyReqData := map[string]interface{}{
		"model": "auto",
		"messages": []map[string]string{
			{
				"role":    "user",
				"content": verificationPrompt,
			},
		},
		"max_tokens":  256,
		"temperature": 0.1,
	}

	data, err := json.Marshal(verifyReqData)
	if err != nil {
		logging.Errorf("[SelfVerify] Failed to marshal verification request: %v", err)
		return nil
	}

	var params openai.ChatCompletionNewParams
	if err := json.Unmarshal(data, &params); err != nil {
		logging.Errorf("[SelfVerify] Failed to unmarshal verification request: %v", err)
		return nil
	}

	return &params
}

// automixVerifierCache memoises one selection.AutoMixVerifierClient per
// (server_url, timeout_seconds, max_response_bytes) tuple so repeated requests
// share a single http.Client (and its underlying connection pool) instead of
// reconstructing one per evaluation.
var automixVerifierCache sync.Map

type automixVerifierCacheKey struct {
	url              string
	timeoutSeconds   int
	maxResponseBytes int64
}

func getAutoMixVerifierClient(serverURL string, timeoutSeconds int, maxResponseBytes int64) *selection.AutoMixVerifierClient {
	if maxResponseBytes <= 0 {
		maxResponseBytes = config.DefaultMaxResponseBytes
	}
	key := automixVerifierCacheKey{url: serverURL, timeoutSeconds: timeoutSeconds, maxResponseBytes: maxResponseBytes}
	if existing, ok := automixVerifierCache.Load(key); ok {
		return existing.(*selection.AutoMixVerifierClient)
	}
	client := selection.NewAutoMixVerifierClient(serverURL)
	client.SetMaxResponseBytes(maxResponseBytes)
	if timeoutSeconds > 0 {
		client.SetTimeout(time.Duration(timeoutSeconds) * time.Second)
	}
	actual, _ := automixVerifierCache.LoadOrStore(key, client)
	return actual.(*selection.AutoMixVerifierClient)
}

// performAutoMixEntailment runs the AutoMix paper-faithful self-verification
// step (arXiv:2310.12963 §3.2) for one candidate model: it sends the original
// question and the candidate's answer to an external entailment verifier and
// returns (confidence, accepted, err). A non-nil err signals an unrecoverable
// verifier failure; the caller decides how to react via the looper's on_error
// policy. Missing or malformed configuration is reported as a hard error so it
// fails loudly at the first request rather than silently degrading.
func (l *ConfidenceLooper) performAutoMixEntailment(
	ctx context.Context,
	req *Request,
	evaluator *ConfidenceEvaluator,
	modelName string,
	responseContent string,
) (float64, bool, error) {
	if evaluator.VerifierServerURL == "" {
		return 0, false, fmt.Errorf("confidence_method=%s requires verifier_server_url", MethodAutoMixEntailment)
	}

	question := l.extractQuestionFromRequest(req.OriginalRequest)
	if question == "" {
		return 0, false, fmt.Errorf("could not extract user question from request")
	}

	client := getAutoMixVerifierClient(evaluator.VerifierServerURL, evaluator.VerifierTimeoutSeconds, evaluator.MaxResponseBytes)

	logging.ComponentDebugEvent("looper", "automix_entailment_started", map[string]interface{}{
		"looper":     "confidence",
		"decision":   req.DecisionName,
		"model_ref":  modelName,
		"server_url": evaluator.VerifierServerURL,
	})

	verifyResp, err := client.Verify(ctx, question, responseContent, "", evaluator.Threshold)
	if err != nil {
		return 0, false, fmt.Errorf("verifier call failed: %w", err)
	}

	accepted := verifyResp.Confidence >= evaluator.Threshold
	return verifyResp.Confidence, accepted, nil
}
