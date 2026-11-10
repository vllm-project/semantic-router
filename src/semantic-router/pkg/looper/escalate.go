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

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// SelfVerificationPrompt is the prompt template for self_verify confidence:
// the model rates its own answer.
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

// AnswerChecker asks a System One model about one answer to a request: the
// answer_check preset (is it correct and complete?) and the refusal preset
// (did the assistant refuse?). Each returns the probability of its outcome;
// Refusal also returns the threshold at which the preset flags a refusal.
type AnswerChecker interface {
	CheckAnswer(ctx context.Context, request, answer string) (float64, error)
	Refusal(ctx context.Context, request, answer string) (probability, threshold float64, err error)
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

// EscalateLooper implements the escalate executor on Chat: it calls the
// decision's models one at a time, in modelRefs order or by size or cost,
// until one answer is accepted.
type EscalateLooper struct {
	*BaseLooper
}

// NewEscalateLooper creates a new EscalateLooper instance
func NewEscalateLooper(cfg *config.LooperConfig) *EscalateLooper {
	return newEscalateLooper(cfg, ownClient(NewClient(cfg)))
}

func newEscalateLooper(cfg *config.LooperConfig, binding clientBinding) *EscalateLooper {
	return &EscalateLooper{
		BaseLooper: newBaseLooper(cfg, binding),
	}
}

// paramSizePattern matches the leading count of a param_size, such as "10b",
// "1.6T" or "397B (17B active)".
var paramSizePattern = regexp.MustCompile(`^~?([0-9]+(?:\.[0-9]+)?)\s*([kmbt]?)`)

// parseParamSize parses a param_size string by its leading count into millions
// of parameters (e.g., "10b" -> 10000, "397B (17B active)" -> 397000, "1.6T" ->
// 1600000). A size it cannot read returns math.MaxFloat64, so it sorts last.
func parseParamSize(size string) float64 {
	matches := paramSizePattern.FindStringSubmatch(strings.ToLower(strings.TrimSpace(size)))
	if matches == nil {
		return math.MaxFloat64
	}

	num, err := strconv.ParseFloat(matches[1], 64)
	if err != nil {
		return math.MaxFloat64
	}

	// Convert to millions for comparison
	switch matches[2] {
	case "t": // trillions
		return num * 1000000
	case "m": // millions
		return num
	case "k": // thousands
		return num / 1000
	default: // billions, also when no unit is given
		return num * 1000
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
			if params, ok := modelParams[ref.Model]; ok && params.Pricing != (config.ModelPricing{}) {
				return params.Pricing.PromptPer1M
			}
		}
		return math.MaxFloat64 // Unknown or unset cost goes last
	}

	sort.SliceStable(sorted, func(i, j int) bool {
		return getCost(sorted[i]) < getCost(sorted[j])
	})

	return sorted
}

// escalationOrder returns the models in the order the escalation calls them.
func escalationOrder(req *Request, order string) []config.ModelRef {
	switch order {
	case config.EscalateOrderSize:
		return sortModelRefsBySize(req.ModelRefs, req.ModelParams)
	case config.EscalateOrderCost:
		return sortModelRefsByCost(req.ModelRefs, req.ModelParams)
	default:
		return append([]config.ModelRef(nil), req.ModelRefs...)
	}
}

// ConfidenceEvaluator evaluates model response confidence based on configured method
type ConfidenceEvaluator struct {
	Method        string  // "avg_logprob", "margin", "hybrid" or "self_verify"
	Threshold     float64 // Threshold for the chosen method
	LogprobWeight float64 // Weight for logprob in hybrid mode
	MarginWeight  float64 // Weight for margin in hybrid mode
	TokenFilter   string  // "all", "tool_call_args" — selects which tokens feed confidence
}

// NewConfidenceEvaluator creates a confidence evaluator from an escalation's
// accept.confidence.
func NewConfidenceEvaluator(cfg *config.EscalateConfidence) *ConfidenceEvaluator {
	eval := &ConfidenceEvaluator{
		Method:        cfg.EffectiveMethod(),
		LogprobWeight: 0.5,
		MarginWeight:  0.5,
	}
	if cfg == nil {
		return eval
	}
	eval.TokenFilter = cfg.TokenFilter
	if cfg.GTE != nil {
		eval.Threshold = *cfg.GTE
	}
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
	if (e.Method == config.ConfidenceMethodMargin || e.Method == config.ConfidenceMethodHybrid) && !resp.MarginEvidenceComplete {
		return 0, false, fmt.Errorf(
			"backend returned no complete top-logprob alternatives required by confidence method %q",
			e.Method,
		)
	}

	avgLP := resp.AverageLogprob
	avgM := resp.AverageMargin

	useFiltered := e.TokenFilter != "" && e.TokenFilter != config.ConfidenceTokenFilterAll
	if useFiltered && resp.FilteredTokenCount > 0 {
		avgLP = resp.FilteredAverageLogprob
		avgM = resp.FilteredAverageMargin
	}

	switch e.Method {
	case config.ConfidenceMethodMargin:
		confidence := normalizeMargin(avgM)
		return confidence, confidence >= e.Threshold, nil

	case config.ConfidenceMethodHybrid:
		normalizedLogprob := normalizeLogprob(avgLP)
		normalizedMargin := normalizeMargin(avgM)
		confidence := e.LogprobWeight*normalizedLogprob + e.MarginWeight*normalizedMargin
		return confidence, confidence >= e.Threshold, nil

	default: // avg_logprob
		confidence := normalizeLogprob(avgLP)
		return confidence, confidence >= e.Threshold, nil
	}
}

// NeedsLogprobs returns whether this evaluator needs logprobs enabled
func (e *ConfidenceEvaluator) NeedsLogprobs() bool {
	return e.Method != config.ConfidenceMethodSelfVerify
}

// IsSelfVerify returns true if the model rates its own answer.
func (e *ConfidenceEvaluator) IsSelfVerify() bool {
	return e.Method == config.ConfidenceMethodSelfVerify
}

// NeedsTopLogprobs returns the number of top_logprobs needed (0 if not needed)
func (e *ConfidenceEvaluator) NeedsTopLogprobs() int {
	switch e.Method {
	case config.ConfidenceMethodMargin, config.ConfidenceMethodHybrid:
		return 2 // Need at least 2 for margin calculation
	default:
		return 0 // avg_logprob doesn't need top_logprobs
	}
}

// escalateAcceptance judges each answer of a Chat escalation by its one
// accept check: the answer's confidence, the decision model's answer_check,
// or the refusal preset.
type escalateAcceptance struct {
	confidence  *ConfidenceEvaluator
	answerCheck *config.EscalateAnswerCheck
	refusal     bool
}

func newEscalateAcceptance(accept *config.EscalateAccept) escalateAcceptance {
	switch {
	case accept == nil:
		return escalateAcceptance{confidence: NewConfidenceEvaluator(nil)}
	case accept.SystemOne != nil:
		return escalateAcceptance{answerCheck: accept.SystemOne}
	case accept.Refusal != nil:
		return escalateAcceptance{refusal: true}
	default:
		return escalateAcceptance{confidence: NewConfidenceEvaluator(accept.Confidence)}
	}
}

// kind names the check in logs and attempt traces.
func (a escalateAcceptance) kind() string {
	switch {
	case a.answerCheck != nil:
		return "answer_check"
	case a.refusal:
		return "refusal"
	default:
		return a.confidence.Method
	}
}

func (a escalateAcceptance) logprobs() *LogprobsConfig {
	if a.confidence == nil {
		return &LogprobsConfig{}
	}
	return &LogprobsConfig{Enabled: a.confidence.NeedsLogprobs(), TopLogprobs: a.confidence.NeedsTopLogprobs()}
}

// streamsCandidates reports whether a candidate call may stream to the
// client: only when no token logprobs need reading from it.
func (a escalateAcceptance) streamsCandidates(clientStreaming bool) bool {
	return clientStreaming && a.confidence != nil && !a.confidence.NeedsLogprobs()
}

// escalateRun is one Chat escalation's accounting: every paid attempt and the
// last answer whose check ran.
type escalateRun struct {
	ctx                context.Context
	req                *Request
	acceptance         escalateAcceptance
	attempts           int
	allResponses       []*ModelResponse
	modelsUsed         []string
	lastResponse       *ModelResponse
	lastAttemptOrdinal int
	lastErr            error
}

func (run *escalateRun) partialExecutionError(cause error) error {
	trace := ExecutionTrace{Version: ExecutionTraceVersion}
	if tracker := attemptTrackerFromContext(run.ctx); tracker != nil {
		trace = tracker.snapshot()
	}
	return newEscalatePartialExecutionError(cause, run.allResponses, run.modelsUsed, run.attempts, trace)
}

// Execute implements escalate on Chat:
// 1. Order the models: modelRefs order, or smallest or cheapest first
// 2. Call the next model and judge its answer by the accept check
// 3. Stop at the first accepted answer
// 4. Without one, serve the last judged answer (on_error: last_attempt) or
// fail the request (on_error: fail)
// Iterations counts actual backend attempts, including failed dispatches and
// the second model call used by self_verify.
func (l *EscalateLooper) Execute(ctx context.Context, req *Request) (*Response, error) {
	if len(req.ModelRefs) == 0 {
		return nil, fmt.Errorf("no models configured")
	}
	var escalate config.EscalateAlgorithmConfig
	onError := config.EscalateOnErrorLastAttempt
	if req.Algorithm != nil {
		if req.Algorithm.Escalate != nil {
			escalate = *req.Algorithm.Escalate
		}
		onError = req.Algorithm.EscalateOnError()
	}
	run := &escalateRun{ctx: ctx, req: req, acceptance: newEscalateAcceptance(escalate.Accept)}
	if (run.acceptance.answerCheck != nil || run.acceptance.refusal) && req.AnswerChecks == nil {
		return nil, fmt.Errorf("escalate accept %s requires a System One answer checker", run.acceptance.kind())
	}
	sortedRefs := escalationOrder(req, escalate.Order)
	logging.ComponentEvent("looper", "execution_started", map[string]interface{}{
		"looper":           config.DecisionAlgorithmEscalate,
		"decision":         req.DecisionName,
		"candidate_models": len(sortedRefs),
		"accept":           run.acceptance.kind(),
		"on_error":         onError,
		"streaming":        req.IsStreaming,
		"order":            escalate.Order,
	})

	accepted := false
	for _, modelRef := range sortedRefs {
		if l.escalateTo(run, modelRef) {
			accepted = true
			break
		}
	}

	if run.lastResponse == nil {
		if run.lastErr != nil {
			return nil, run.partialExecutionError(fmt.Errorf("all models failed: %w", run.lastErr))
		}
		return nil, run.partialExecutionError(fmt.Errorf("all models failed"))
	}
	if !accepted && onError == config.EscalateOnErrorFail {
		return nil, run.partialExecutionError(fmt.Errorf("no model's answer was accepted by %s", run.acceptance.kind()))
	}
	if tracker := attemptTrackerFromContext(ctx); tracker != nil {
		tracker.markSelected(run.lastAttemptOrdinal)
	}

	// Publish only the last judged response. Usage and models-used retain
	// every paid attempt, including a response whose check could not run.
	agg := &AggregatedResponse{
		Models:          run.modelsUsed,
		Responses:       []*ModelResponse{run.lastResponse},
		UsageResponses:  run.allResponses,
		CombinedContent: run.lastResponse.Content,
		FinalModel:      run.lastResponse.Model,
		AverageLogprob:  run.lastResponse.AverageLogprob,
		HasToolCalls:    run.lastResponse.HasToolCalls,
	}

	var response *Response
	var err error
	if req.IsStreaming {
		response, err = l.formatEscalateStreamingResponse(agg, run.modelsUsed, run.attempts, escalateStreamUsageRequested(req))
	} else {
		response, err = l.formatEscalateJSONResponse(agg, run.modelsUsed, run.attempts)
	}
	if err != nil {
		return nil, run.partialExecutionError(err)
	}
	return response, nil
}

// escalateTo calls one model and judges its answer; it reports whether the
// answer was accepted. A failed call or a check that cannot run moves on.
func (l *EscalateLooper) escalateTo(run *escalateRun, modelRef config.ModelRef) bool {
	req := run.req
	modelName := modelRef.Model
	if modelRef.LoRAName != "" {
		modelName = modelRef.LoRAName
	}
	accessKey := ""
	if params, ok := req.ModelParams[modelRef.Model]; ok {
		accessKey = params.AccessKey
	}

	run.attempts++
	resp, attempt, err := l.startConfidenceModelAttempt(
		run.ctx,
		req,
		req.OriginalRequest,
		modelName,
		"candidate",
		"generator",
		llmprotocol.TrustedStageCandidate,
		run.acceptance.streamsCandidates(req.IsStreaming),
		run.attempts,
		run.acceptance.logprobs(),
		accessKey,
	)
	if err != nil {
		logging.ComponentWarnEvent("looper", "model_dispatch_failed", map[string]interface{}{
			"looper":    config.DecisionAlgorithmEscalate,
			"decision":  req.DecisionName,
			"model_ref": modelName,
			"iteration": run.attempts,
			"error":     err.Error(),
		})
		run.lastErr = fmt.Errorf("model %s failed: %w", modelName, err)
		return false
	}

	if run.acceptance.confidence != nil {
		// Exclude tokens such as JSON boilerplate before confidence is read.
		ApplyTokenFilter(resp, run.acceptance.confidence.TokenFilter)
	}
	// The call has already consumed backend capacity even if its answer
	// cannot be judged. Preserve every paid attempt for usage and replay.
	run.allResponses = append(run.allResponses, resp)
	run.modelsUsed = append(run.modelsUsed, modelName)

	score, threshold, accepted, checkErr := l.judgeAnswer(run, modelName, accessKey, resp)
	if checkErr != nil {
		usable := false
		run.lastErr = fmt.Errorf("%s for model %q failed: %w", run.acceptance.kind(), modelName, checkErr)
		logging.ComponentWarnEvent("looper", "escalate_check_failed", map[string]interface{}{
			"looper":    config.DecisionAlgorithmEscalate,
			"decision":  req.DecisionName,
			"model_ref": modelName,
			"iteration": run.attempts,
			"accept":    run.acceptance.kind(),
			"error":     checkErr.Error(),
		})
		if attempt != nil {
			attempt.finish(attemptResult{response: resp, reason: AttemptReasonUnusable, usable: &usable})
		}
		return false
	}
	logging.ComponentDebugEvent("looper", "escalate_answer_judged", map[string]interface{}{
		"looper":    config.DecisionAlgorithmEscalate,
		"decision":  req.DecisionName,
		"model_ref": modelName,
		"accept":    run.acceptance.kind(),
		"score":     score,
		"threshold": threshold,
		"accepted":  accepted,
	})
	if attempt != nil {
		usable := true
		reason := AttemptReasonThresholdNotMet
		if accepted {
			reason = AttemptReasonThresholdMet
		}
		attempt.finish(attemptResult{
			response: resp, reason: reason, usable: &usable,
			accepted: &accepted, score: &score, threshold: &threshold,
			verifierType: run.acceptance.kind(),
		})
		run.lastAttemptOrdinal = attempt.ordinal
	}
	run.lastResponse = resp
	if accepted {
		logging.ComponentEvent("looper", "execution_completed", map[string]interface{}{
			"looper":         config.DecisionAlgorithmEscalate,
			"decision":       req.DecisionName,
			"models_used":    run.modelsUsed,
			"iterations":     run.attempts,
			"selected_model": modelName,
			"reason":         "accepted",
		})
	}
	return accepted
}

// judgeAnswer applies the accept check to one answer: its score, the
// threshold the score is compared with, and whether it is accepted.
func (l *EscalateLooper) judgeAnswer(run *escalateRun, modelName, accessKey string, resp *ModelResponse) (float64, float64, bool, error) {
	req, acceptance := run.req, run.acceptance
	switch {
	case acceptance.answerCheck != nil:
		question := extractQuestionFromRequest(req.OriginalRequest)
		if question == "" {
			return 0, 0, false, fmt.Errorf("could not extract user question from request")
		}
		probability, err := req.AnswerChecks.CheckAnswer(run.ctx, question, judgedAnswer(resp))
		if err != nil {
			return 0, 0, false, err
		}
		threshold := *acceptance.answerCheck.GTE
		return probability, threshold, probability >= threshold, nil
	case acceptance.refusal:
		question := extractQuestionFromRequest(req.OriginalRequest)
		if question == "" {
			return 0, 0, false, fmt.Errorf("could not extract user question from request")
		}
		probability, threshold, err := req.AnswerChecks.Refusal(run.ctx, question, judgedAnswer(resp))
		if err != nil {
			return 0, 0, false, err
		}
		return probability, threshold, probability < threshold, nil
	case acceptance.confidence.IsSelfVerify():
		verificationIteration := run.attempts + 1
		verification, err := l.performSelfVerification(run.ctx, req, modelName, judgedAnswer(resp), accessKey, acceptance.confidence.Threshold, verificationIteration)
		if verification.Attempted {
			run.attempts++
			run.modelsUsed = append(run.modelsUsed, modelName)
		}
		if verification.Response != nil {
			run.allResponses = append(run.allResponses, verification.Response)
		}
		if err != nil {
			return 0, 0, false, err
		}
		return verification.Confidence, acceptance.confidence.Threshold, verification.Accepted, nil
	default:
		confidence, accepted, err := acceptance.confidence.evaluate(resp)
		return confidence, acceptance.confidence.Threshold, accepted, err
	}
}

// judgedAnswer is the answer an accept check judges: its text, then each tool
// call it makes as the tool's name and arguments, so a call is not judged as
// an empty answer.
func judgedAnswer(resp *ModelResponse) string {
	answer := resp.Content
	if resp.Parsed != nil && len(resp.Parsed.Choices) > 0 {
		for _, call := range resp.Parsed.Choices[0].Message.ToolCalls {
			answer = strings.TrimSpace(answer + "\n" + call.Function.Name + " " + call.Function.Arguments)
		}
	}
	return answer
}

func escalateStreamUsageRequested(req *Request) bool {
	return req != nil && req.OriginalRequest != nil &&
		req.OriginalRequest.StreamOptions.IncludeUsage.Or(false)
}

// performSelfVerification asks the model that answered to rate its own
// answer and returns that rating as the answer's confidence.
func (l *EscalateLooper) performSelfVerification(
	ctx context.Context,
	req *Request,
	modelName string,
	responseContent string,
	accessKey string,
	threshold float64,
	iteration int,
) (selfVerificationExecution, error) {
	// Extract original question from the request
	originalQuestion := extractQuestionFromRequest(req.OriginalRequest)
	if originalQuestion == "" {
		return selfVerificationExecution{}, fmt.Errorf("could not extract user question from request")
	}

	// Build self-verification prompt
	verificationPrompt := fmt.Sprintf(SelfVerificationPrompt, originalQuestion, responseContent)

	// Create a new request for self-verification
	verifyRequest := buildSelfVerificationRequest(verificationPrompt)
	if verifyRequest == nil {
		return selfVerificationExecution{}, fmt.Errorf("could not build self-verification request")
	}

	logging.ComponentDebugEvent("looper", "self_verification_started", map[string]interface{}{
		"looper":    config.DecisionAlgorithmEscalate,
		"model_ref": modelName,
	})

	// Call the same model to evaluate its answer
	verifyResp, attempt, err := l.startConfidenceModelAttempt(
		ctx, req, verifyRequest, modelName, "self_verifier", "verifier", llmprotocol.TrustedStageVerifier, false, iteration, nil, accessKey,
	)
	if err != nil {
		return selfVerificationExecution{Attempted: true}, fmt.Errorf("verifier model call failed: %w", err)
	}

	// Parse the self-verification result
	result, err := parseSelfVerification(verifyResp.Content)
	if err != nil {
		if attempt != nil {
			usable := false
			attempt.finish(attemptResult{response: verifyResp, err: err, reason: AttemptReasonInvalidResponse, usable: &usable})
		}
		return selfVerificationExecution{
			Response:  verifyResp,
			Attempted: true,
		}, fmt.Errorf("could not parse verifier result: %w", err)
	}
	accepted := result.Confidence >= threshold
	if attempt != nil {
		usable := true
		attemptReason := AttemptReasonThresholdNotMet
		if accepted {
			attemptReason = AttemptReasonThresholdMet
		}
		attempt.finish(attemptResult{
			response: verifyResp, reason: attemptReason, usable: &usable,
			accepted: &accepted, score: &result.Confidence, threshold: &threshold,
			verifierType: config.ConfidenceMethodSelfVerify,
		})
	}

	logging.ComponentDebugEvent("looper", "self_verification_result_parsed", map[string]interface{}{
		"looper":     config.DecisionAlgorithmEscalate,
		"model_ref":  modelName,
		"confidence": result.Confidence,
		"reason_len": len(result.Reason),
	})

	return selfVerificationExecution{
		Confidence: result.Confidence,
		Accepted:   accepted,
		Response:   verifyResp,
		Attempted:  true,
	}, nil
}

// extractQuestionFromRequest extracts the user's question from the original request
// Uses JSON marshaling for robust extraction across SDK versions
func extractQuestionFromRequest(originalRequest *openai.ChatCompletionNewParams) string {
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
func buildSelfVerificationRequest(verificationPrompt string) *openai.ChatCompletionNewParams {
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
