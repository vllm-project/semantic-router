package extproc

import (
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// reasoningOffForStopSequences is the trace reason of reasoning the Router
// turned off because the request sets stop sequences.
const reasoningOffForStopSequences = "stop_sequences"

// unstatedReasoningStopSequences reports whether a client request that states
// no reasoning preference sets stop sequences: Chat stop or Messages
// stop_sequences. Responses has none.
func unstatedReasoningStopSequences(request llmprotocol.Request) bool {
	return len(request.Sampling.Stop) > 0 && !clientStatesReasoningPreference(request)
}

// stopSequencesTurnReasoningOff reports whether model runs this request with
// its reasoning off instead of the reasoning its decision chose. Backends such
// as vLLM match stop sequences against the reasoning as well as the answer, so
// a stop sequence the reasoning writes would end the request before any
// answer. A decision whose request_params plugin blocks stop sends none.
func (r *OpenAIRouter) stopSequencesTurnReasoningOff(ctx *RequestContext, model string) bool {
	if r == nil || r.Config == nil || ctx == nil || !ctx.unstatedReasoningStopSequences {
		return false
	}
	return !decisionBlocksStop(ctx.VSRSelectedDecision) && r.canTurnReasoningOff(model)
}

func decisionBlocksStop(decision *config.Decision) bool {
	if decision == nil {
		return false
	}
	params := decision.GetRequestParamsConfig()
	return params != nil && slices.ContainsFunc(params.BlockedParams, func(field string) bool {
		return strings.TrimSpace(field) == "stop"
	})
}
