package extproc

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/outputtokens"
)

func snapshotClientMaxOutputTokens(request llmprotocol.Request, ctx *RequestContext) {
	if ctx == nil || ctx.LooperRequest {
		return
	}
	ctx.ClientMaxOutputTokens = outputtokens.Clone(request.Sampling.MaxOutputTokens)
}

// applyLooperHopOutputTokenBounds copies the ceilings the Router put on the
// typed hop. Client headers are ignored: a request cannot pose as a hop or
// widen the selected model's bound.
func applyLooperHopOutputTokenBounds(ctx *RequestContext) {
	if ctx == nil || !ctx.LooperRequest {
		return
	}
	ctx.ClientMaxOutputTokens = nil
	ctx.AlgorithmStageMaxOutputTokens = nil
	if ctx.Hop == nil {
		return
	}
	ctx.ClientMaxOutputTokens = positiveOutputTokenBound(ctx.Hop.ClientMaxOutputTokens)
	ctx.AlgorithmStageMaxOutputTokens = positiveOutputTokenBound(ctx.Hop.StageMaxOutputTokens)
}

func positiveOutputTokenBound(value *int64) *int64 {
	if value == nil || *value < 1 {
		return nil
	}
	return outputtokens.Clone(value)
}

// prepareAutomaticDispatchWithOutputLimit renders automatic output first, then
// composes the strictest ceiling. Rendered capacity is an upper bound and
// cannot replace a stricter ModelRef, plugin, client, stage, or ledger limit.
func (r *OpenAIRouter) prepareAutomaticDispatchWithOutputLimit(
	ctx *RequestContext,
	request *llmprotocol.Request,
	dispatch *providerDispatch,
) error {
	if err := r.prepareAutomaticDispatch(ctx, request, dispatch); err != nil {
		return err
	}
	changed, err := r.applyDispatchOutputTokenLimit(request, dispatch, ctx)
	if changed && request != nil {
		request.Generation++
	}
	return err
}

func (r *OpenAIRouter) applyDispatchOutputTokenLimit(
	request *llmprotocol.Request,
	dispatch *providerDispatch,
	ctx *RequestContext,
) (bool, error) {
	if request == nil || ctx == nil {
		return false, nil
	}
	sources := r.outputTokenLimitSources(dispatch, ctx)
	if request.Sampling.AutomaticOutput {
		sources.Automatic = outputtokens.Clone(request.Sampling.MaxOutputTokens)
	}
	result := outputtokens.Compose(sources)
	if outputTokenLimitBlocked(ctx) && sources.Client == nil && result.Effective == nil {
		result.Fallback = outputtokens.FallbackBlockedParam
	}
	if !wireFormatSupportsOutputTokenLimit(dispatch) && result.Effective != nil {
		result.Fallback = outputtokens.FallbackCodecUnsupported
		result.Effective = nil
		result.Source = ""
	}
	if result.Effective == nil && result.Fallback == "" {
		// No composed ceiling and no fallback that must clear one. Keep an
		// already-encoded hop or caller bound instead of treating silence as a wipe.
		ctx.EffectiveMaxOutputTokens = outputtokens.Clone(request.Sampling.MaxOutputTokens)
		ctx.EffectiveMaxOutputTokensSource = ""
		ctx.EffectiveMaxOutputTokensFallback = ""
		return false, nil
	}
	previous := outputtokens.Clone(request.Sampling.MaxOutputTokens)
	request.Sampling.MaxOutputTokens = outputtokens.Clone(result.Effective)
	ctx.EffectiveMaxOutputTokens = outputtokens.Clone(result.Effective)
	ctx.EffectiveMaxOutputTokensSource = result.Source
	ctx.EffectiveMaxOutputTokensFallback = result.Fallback
	if sources.Plugin != nil && result.Source == outputtokens.SourcePlugin {
		decisionKey := ""
		if ctx.VSRSelectedDecision != nil {
			decisionKey = config.RoutingDecisionKey(ctx.Routing.RecipeName(), ctx.VSRSelectedDecision.Name)
		}
		metrics.RecordMaxTokensCapped(decisionKey)
	}
	changed := !int64PointersEqual(previous, request.Sampling.MaxOutputTokens)
	if responsesOutputTokenLimitUnsupported(dispatch, result.Effective) {
		return changed, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"unsupported_responses_max_output_tokens",
			fmt.Sprintf("composed output token limit %d is below the Responses API minimum of %d", *result.Effective, outputtokens.ResponsesMinOutputTokens),
			nil,
		)
	}
	return changed, nil
}

func (r *OpenAIRouter) outputTokenLimitSources(
	dispatch *providerDispatch,
	ctx *RequestContext,
) outputtokens.Sources {
	sources := outputtokens.Sources{
		Client:         outputtokens.Clone(ctx.ClientMaxOutputTokens),
		AlgorithmStage: outputtokens.Clone(ctx.AlgorithmStageMaxOutputTokens),
		Ledger:         ctx.OutputTokenLedger,
	}
	if outputTokenLimitBlocked(ctx) {
		sources.Client = nil
	}
	if ctx.VSRSelectedDecision != nil {
		if params := ctx.VSRSelectedDecision.GetRequestParamsConfig(); params != nil {
			sources.Plugin = outputtokens.FromInt(params.MaxTokensLimit)
		}
		if dispatch != nil {
			sources.ModelRef = modelRefMaxCompletionTokens(r, ctx.VSRSelectedDecision, dispatch.logicalModel)
		}
	}
	return sources
}

func outputTokenLimitBlocked(ctx *RequestContext) bool {
	if ctx == nil {
		return false
	}
	return decisionBlocksOutputTokenLimit(ctx.VSRSelectedDecision)
}

func decisionBlocksOutputTokenLimit(decision *config.Decision) bool {
	if decision == nil {
		return false
	}
	params := decision.GetRequestParamsConfig()
	if params == nil {
		return false
	}
	for _, field := range params.BlockedParams {
		switch strings.TrimSpace(field) {
		case "max_tokens", "max_completion_tokens", "max_output_tokens":
			return true
		}
	}
	return false
}

func modelRefMaxCompletionTokens(router *OpenAIRouter, decision *config.Decision, modelName string) *int64 {
	if decision == nil || modelName == "" {
		return nil
	}
	for _, ref := range decision.ModelRefs {
		if !selectedModelRefMatches(router, ref, modelName) {
			continue
		}
		return outputtokens.FromInt(ref.MaxCompletionTokens)
	}
	return nil
}

// selectedModelRefMatches distinguishes a LoRA adapter from the bare base it
// belongs to. An adapter matches only its own name; a base reference matches
// that base, including a provider alias, and not a sibling adapter.
func selectedModelRefMatches(router *OpenAIRouter, ref config.ModelRef, modelName string) bool {
	if router != nil {
		return router.modelRefMatchesCandidate(ref, modelName)
	}
	return candidateModelIdentity(ref) == modelName
}

func wireFormatSupportsOutputTokenLimit(dispatch *providerDispatch) bool {
	if dispatch == nil {
		return true
	}
	switch dispatch.targetFormat {
	case llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1:
		return true
	default:
		return false
	}
}

func responsesOutputTokenLimitUnsupported(dispatch *providerDispatch, effective *int64) bool {
	if dispatch == nil || effective == nil || dispatch.targetFormat != llmprotocol.OpenAIResponsesV1 {
		return false
	}
	return *effective < outputtokens.ResponsesMinOutputTokens
}

func int64PointersEqual(left, right *int64) bool {
	if left == nil || right == nil {
		return left == right
	}
	return *left == *right
}
