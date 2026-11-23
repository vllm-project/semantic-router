package selection

import (
	"context"
	"math"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selectiontrace"
)

// CandidateAnswer is what the estimating model answered about one
// candidate: the probability that it answers the request correctly
// (candidate_success) and the probability of each output_length band. A
// candidate the model could not estimate carries its status and reason
// instead.
type CandidateAnswer struct {
	Status             selectiontrace.EstimateStatus
	Reason             string
	SuccessProbability float64
	OutputLength       map[string]float64
}

// CandidateEstimator asks the estimating deployment about each candidate of
// a request, each its own state of one call, and returns one answer per
// candidate in order. It never fails the selection: a candidate it could not
// estimate carries its status.
type CandidateEstimator func(ctx context.Context, request string, candidates []config.ModelRef) []CandidateAnswer

// MultiFactorEstimates turns on per-request candidate estimates. In observe
// mode, the only mode, they are recorded in the selection trace and the
// selection is unchanged.
type MultiFactorEstimates struct {
	Mode       string
	Deployment string
	Estimate   CandidateEstimator
}

// estimateCandidates asks for the estimates of the candidates within the SLO
// ceilings; nil when the decision turns no estimates on.
func (s *MultiFactorSelector) estimateCandidates(
	ctx context.Context,
	selCtx *SelectionContext,
	candidates []config.ModelRef,
) *selectiontrace.CandidateEstimates {
	estimates := s.config.Estimates
	if estimates == nil || estimates.Estimate == nil || len(candidates) == 0 {
		return nil
	}
	answers := estimates.Estimate(ctx, selCtx.Query, candidates)
	recorded := &selectiontrace.CandidateEstimates{
		Mode:       estimates.Mode,
		Candidates: make([]selectiontrace.CandidateEstimate, len(candidates)),
	}
	for index, candidate := range candidates {
		answer := CandidateAnswer{Status: selectiontrace.EstimateUnavailable, Reason: "missing_answer"}
		if index < len(answers) {
			answer = answers[index]
		}
		recorded.Candidates[index] = s.candidateEstimate(candidate, answer, estimates.Deployment, selCtx)
	}
	return recorded
}

// candidateEstimate types one answer: the success probability as answered,
// the expected output tokens bounded by the candidate's output capacity, and
// the request's cost at those tokens.
func (s *MultiFactorSelector) candidateEstimate(
	candidate config.ModelRef,
	answer CandidateAnswer,
	deployment string,
	selCtx *SelectionContext,
) selectiontrace.CandidateEstimate {
	estimate := selectiontrace.CandidateEstimate{
		Candidate:  traceCandidate(candidate),
		Status:     answer.Status,
		Reason:     answer.Reason,
		Source:     selectiontrace.EstimateSourceSystemOne,
		Deployment: deployment,
	}
	unavailable := func(reason string) selectiontrace.CandidateEstimate {
		estimate.Status, estimate.Reason = selectiontrace.EstimateUnavailable, reason
		return estimate
	}
	switch {
	case answer.Status == "":
		return unavailable(answer.Reason)
	case answer.Status != selectiontrace.EstimateAvailable:
		return estimate
	case math.IsNaN(answer.SuccessProbability) || answer.SuccessProbability < 0 || answer.SuccessProbability > 1:
		return unavailable("invalid_success_probability")
	}
	expected, ok := presets.ExpectedOutputTokens(answer.OutputLength, s.outputCapacity(candidate.Model, selCtx))
	if !ok {
		return unavailable("invalid_output_length")
	}
	probability, tokens := answer.SuccessProbability, int(math.Round(expected))
	estimate.Reason = ""
	estimate.SuccessProbability, estimate.ExpectedOutputTokens = &probability, &tokens
	if cost, currency, priced := s.estimatedCost(candidate.Model, tokens, selCtx); priced {
		estimate.EstimatedCost, estimate.Currency = &cost, currency
	}
	return estimate
}

// outputCapacity is the most a candidate can write for this request: the
// least of the request's output budget, the provider-rendered budget and the
// model's maximum output; zero when none is known.
func (s *MultiFactorSelector) outputCapacity(model string, selCtx *SelectionContext) int {
	capacity := 0
	bound := func(limit int) {
		if limit > 0 && (capacity == 0 || limit < capacity) {
			capacity = limit
		}
	}
	if selCtx != nil {
		bound(selCtx.ExpectedOutputTokens)
		if demand, ok := selCtx.CandidateDemands[model]; ok && demand.MaxOutputTokens != nil {
			bound(int(min(*demand.MaxOutputTokens, math.MaxInt32)))
		}
	}
	if params, ok := s.modelParams[model]; ok {
		bound(params.MaxOutputTokens)
	}
	return capacity
}

// estimatedCost prices the request's input tokens, as multi_factor's cost
// factor counts them, and the expected output tokens at the model's rates.
func (s *MultiFactorSelector) estimatedCost(model string, outputTokens int, selCtx *SelectionContext) (float64, string, bool) {
	params, ok := s.modelParams[model]
	if !ok || !modelPricingConfigured(params.Pricing) {
		return 0, "", false
	}
	inputTokens := 0
	if selCtx != nil {
		inputTokens = max(selCtx.InputTokens, 0)
		if demand, ok := selCtx.CandidateDemands[model]; ok {
			inputTokens = max(demand.InputTokens, 0)
		}
	}
	cost := (float64(inputTokens)*params.Pricing.PromptPer1M + float64(outputTokens)*params.Pricing.CompletionPer1M) / 1_000_000
	return cost, params.Pricing.Currency, true
}

// withEstimates records the estimates in a selection's trace, beside the
// objective's stages when it has them.
func withEstimates(trace *selectiontrace.MultiFactorObjective, estimates *selectiontrace.CandidateEstimates) *selectiontrace.MultiFactorObjective {
	if estimates == nil {
		return trace
	}
	if trace == nil {
		trace = &selectiontrace.MultiFactorObjective{}
	}
	trace.Estimates = estimates
	return trace
}

// resultWithEstimates records the estimates in a no-candidate policy's
// result.
func resultWithEstimates(result *SelectionResult, estimates *selectiontrace.CandidateEstimates) *SelectionResult {
	if result != nil {
		result.MultiFactor = withEstimates(result.MultiFactor, estimates)
	}
	return result
}
