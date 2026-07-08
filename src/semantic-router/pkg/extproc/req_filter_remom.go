package extproc

import (
	"fmt"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (r *OpenAIRouter) handleDirectReMoMExecution(
	openAIRequest *openai.ChatCompletionNewParams,
	originalModel string,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	if r.Config == nil || !r.Config.Looper.IsEnabled() {
		return r.createErrorResponse(500, "ReMoM execution requires global.integrations.looper.endpoint"), nil
	}

	ctx.RequestModel = originalModel
	decision, status, err := r.resolveDirectReMoMDecision(ctx)
	if err != nil {
		return r.createErrorResponse(status, err.Error()), nil
	}
	ctx.VSRSelectedDecision = decision
	ctx.VSRSelectedDecisionName = decision.Name
	ctx.VSRSelectionMethod = "remom"

	return r.handleLooperExecution(ctx.TraceContext, openAIRequest, decision, ctx)
}

func (r *OpenAIRouter) resolveDirectReMoMDecision(ctx *RequestContext) (*config.Decision, int, error) {
	if isReMoMDecision(ctx.VSRSelectedDecision) {
		return ctx.VSRSelectedDecision, 0, nil
	}
	if ctx.VSRSelectedDecision != nil {
		return nil, 400, fmt.Errorf("no eligible ReMoM decision matched for model %q", ctx.RequestModel)
	}
	if decision := r.defaultLooperDecisionByAlgorithm("remom"); decision != nil {
		return decision, 0, nil
	}
	return nil, 400, fmt.Errorf("no eligible ReMoM decision matched for model %q", ctx.RequestModel)
}

func isReMoMDecision(decision *config.Decision) bool {
	return decision != nil && decision.Algorithm != nil && decision.Algorithm.Type == "remom"
}
