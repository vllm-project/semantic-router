package extproc

import (
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/outputtokens"
)

func snapshotClientMaxOutputTokens(request llmprotocol.Request, ctx *RequestContext) {
	if ctx == nil {
		return
	}
	ctx.ClientMaxOutputTokens = outputtokens.Clone(request.Sampling.MaxOutputTokens)
}

func parseLooperOutputTokenBoundHeaders(ctx *RequestContext) {
	if ctx == nil || !ctx.LooperRequest {
		return
	}
	if value, ok := parsePositiveInt64Header(ctx, headers.VSRLooperClientMaxOutputTokens); ok {
		ctx.ClientMaxOutputTokens = &value
	}
	if value, ok := parsePositiveInt64Header(ctx, headers.VSRLooperStageMaxOutputTokens); ok {
		ctx.AlgorithmStageMaxOutputTokens = &value
	}
}

func parsePositiveInt64Header(ctx *RequestContext, name string) (int64, bool) {
	raw := strings.TrimSpace(headerValueCI(ctx, name))
	if raw == "" {
		return 0, false
	}
	value, err := strconv.ParseInt(raw, 10, 64)
	if err != nil || value < 1 {
		return 0, false
	}
	return value, true
}

func (r *OpenAIRouter) applyDispatchOutputTokenLimit(
	request *llmprotocol.Request,
	dispatch *providerDispatch,
	ctx *RequestContext,
) bool {
	if request == nil || ctx == nil {
		return false
	}
	sources := r.outputTokenLimitSources(request, dispatch, ctx)
	result := outputtokens.Compose(sources)
	if outputTokenLimitBlocked(ctx) && sources.Client == nil && result.Effective == nil {
		result.Fallback = outputtokens.FallbackBlockedParam
	}
	if !wireFormatSupportsOutputTokenLimit(dispatch) && result.Effective != nil {
		result.Fallback = outputtokens.FallbackCodecUnsupported
		result.Effective = nil
		result.Source = ""
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
	return !int64PointersEqual(previous, request.Sampling.MaxOutputTokens)
}

func (r *OpenAIRouter) outputTokenLimitSources(
	request *llmprotocol.Request,
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
	} else if sources.Client == nil && request != nil {
		sources.Client = outputtokens.Clone(request.Sampling.MaxOutputTokens)
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
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return false
	}
	params := ctx.VSRSelectedDecision.GetRequestParamsConfig()
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
		matchesModel := ref.Model == modelName || ref.LoRAName == modelName
		if router != nil && router.Config != nil {
			matchesModel = router.Config.ModelNameMatches(ref.Model, modelName) || ref.LoRAName == modelName
		}
		if !matchesModel {
			continue
		}
		return outputtokens.FromInt(ref.MaxCompletionTokens)
	}
	return nil
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

func int64PointersEqual(left, right *int64) bool {
	if left == nil || right == nil {
		return left == right
	}
	return *left == *right
}
