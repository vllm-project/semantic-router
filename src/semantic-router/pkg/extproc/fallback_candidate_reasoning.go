package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// applyCandidateReasoning writes a candidate's own modelRefs reasoning onto its
// request. A model without a reasoning family gets the request's reasoning
// untouched, as the first attempt's dispatch leaves it for such a model: the
// Router has no control to write for it, and clearing the mode alone would
// leave a client's thinking display behind, which no backend accepts.
func (r *OpenAIRouter) applyCandidateReasoning(
	ctx *RequestContext,
	request *llmprotocol.Request,
	candidateRef *config.ModelRef,
	dispatch *providerDispatch,
	useReasoning bool,
) {
	if r.getModelReasoningFamily(dispatch.effectiveBackendModel()) == nil {
		return
	}
	if !useReasoning {
		request.ReasoningEffort, request.ReasoningMode, request.ReasoningBudgetTokens = "", "", nil
		return
	}
	candidateModel := candidateModelIdentity(*candidateRef)
	request.ReasoningEffort = candidateRef.ReasoningEffort
	if request.ReasoningEffort == "" {
		request.ReasoningEffort = r.getReasoningEffort(ctx.decisionForCandidate(candidateModel), candidateModel)
	}
	request.ReasoningMode, request.ReasoningBudgetTokens = r.candidateReasoningMode(
		ctx, candidateRef, dispatch.targetFormat, request.ReasoningBudgetTokens,
	)
}

// candidateReasoningMode is the reasoning mode, and the thinking budget that
// goes with it, of a candidate with reasoning on. Only the Messages wire
// carries a mode, and enabled thinking there needs a budget, so a budgetless
// enabled mode becomes adaptive, as for the primary. Chat Completions and
// Responses carry none: their provider adapter turns use_reasoning into the
// backend's own control, and a mode left on the request fails validation
// before the call.
func (r *OpenAIRouter) candidateReasoningMode(
	ctx *RequestContext,
	candidateRef *config.ModelRef,
	target llmprotocol.WireFormat,
	budget *int64,
) (llmprotocol.ReasoningMode, *int64) {
	if target != llmprotocol.AnthropicMessagesV1 {
		return "", budget
	}
	mode := candidateRef.ReasoningMode
	if mode == "" {
		candidateModel := candidateModelIdentity(*candidateRef)
		mode = r.getReasoningMode(ctx.decisionForCandidate(candidateModel), candidateModel, true)
	}
	if mode == "" {
		return "", budget
	}
	return messagesThinkingBudget(target, llmprotocol.ReasoningMode(mode), budget)
}
