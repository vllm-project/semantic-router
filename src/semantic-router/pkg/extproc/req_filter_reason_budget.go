package extproc

import (
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// reasoningOffForOutputBudget is the trace reason of reasoning the Router
// turned off because the request's output limit was too small for it.
const reasoningOffForOutputBudget = "output_budget"

// chatTemplateReasoningSwitches are the chat_template_kwargs members that
// open-model chat templates read to turn thinking on or off or set its effort.
var chatTemplateReasoningSwitches = []string{"enable_thinking", "thinking", "reasoning_effort"}

// clientStatesReasoningPreference reports whether a client request says how
// the model should reason: Messages thinking; a Chat reasoning_effort,
// reasoning_budget_tokens or chat template thinking switch; a Responses
// reasoning effort or summary. A Messages output_config.effort sets the effort
// of the whole answer, not whether the model thinks.
func clientStatesReasoningPreference(request llmprotocol.Request) bool {
	if request.ReasoningMode != "" || request.ReasoningBudgetTokens != nil {
		return true
	}
	if request.Trusted.SourceFormat == llmprotocol.AnthropicMessagesV1 {
		return false
	}
	if request.ReasoningEffort != "" || request.ReasoningSummary != "" {
		return true
	}
	var kwargs map[string]json.RawMessage
	if json.Unmarshal(request.ChatTemplateKwargs, &kwargs) != nil {
		return false
	}
	for _, name := range chatTemplateReasoningSwitches {
		if _, set := kwargs[name]; set {
			return true
		}
	}
	return false
}

// unstatedReasoningOutputLimit is the output limit of a client request that
// states no reasoning preference, read before the Router sets either. It is
// nil when the client stated a preference or set no limit.
func unstatedReasoningOutputLimit(request llmprotocol.Request) *int64 {
	if request.Sampling.MaxOutputTokens == nil || clientStatesReasoningPreference(request) {
		return nil
	}
	limit := *request.Sampling.MaxOutputTokens
	return &limit
}

// outputBudgetTurnsReasoningOff reports whether model runs this request with
// its reasoning off instead of the reasoning its decision chose. A backend
// counts reasoning against the output limit, so a request that states no
// reasoning preference and sets a limit below
// providers.defaults.reasoning_min_output_tokens would end with no answer.
func (r *OpenAIRouter) outputBudgetTurnsReasoningOff(ctx *RequestContext, model string) bool {
	if r == nil || r.Config == nil || ctx == nil || ctx.unstatedReasoningOutputLimit == nil {
		return false
	}
	minimum := int64(r.Config.ReasoningMinOutputTokens)
	return minimum > 0 && *ctx.unstatedReasoningOutputLimit < minimum && r.canTurnReasoningOff(model)
}

// canTurnReasoningOff reports whether model's reasoning family has an off
// control that every one of its Provider bindings can send. The rules for
// requests that state no reasoning preference change only such a model.
func (r *OpenAIRouter) canTurnReasoningOff(model string) bool {
	return anthropicBackendCanDisableReasoning(r.getModelReasoningFamily(model)) &&
		r.Config.CanProjectReasoningOff(model)
}

// unstatedReasoningOffReason is the trace reason for which model runs a
// request that states no reasoning preference with its reasoning off instead
// of the reasoning its decision chose, or "" when it keeps that reasoning.
func (r *OpenAIRouter) unstatedReasoningOffReason(ctx *RequestContext, model string) string {
	switch {
	case r.outputBudgetTurnsReasoningOff(ctx, model):
		return reasoningOffForOutputBudget
	case r.stopSequencesTurnReasoningOff(ctx, model):
		return reasoningOffForStopSequences
	default:
		return ""
	}
}
