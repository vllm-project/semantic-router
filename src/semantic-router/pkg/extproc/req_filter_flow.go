package extproc

import (
	"fmt"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (r *OpenAIRouter) handleDirectFlowExecution(
	openAIRequest *openai.ChatCompletionNewParams,
	originalModel string,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	if r.Config == nil || !r.Config.Looper.IsEnabled() {
		return r.createErrorResponse(500, "Flow execution requires global.integrations.looper.endpoint"), nil
	}

	ctx.RequestModel = originalModel
	decision, status, err := r.resolveDirectFlowDecision(ctx)
	if err != nil {
		return r.createErrorResponse(status, err.Error()), nil
	}
	ctx.VSRSelectedDecision = decision
	ctx.VSRSelectedDecisionName = decision.Name
	ctx.VSRSelectionMethod = "workflows"

	return r.handleLooperExecution(ctx.TraceContext, openAIRequest, decision, ctx)
}

func (r *OpenAIRouter) resolveDirectFlowDecision(ctx *RequestContext) (*config.Decision, int, error) {
	if isFlowDecision(ctx.VSRSelectedDecision) {
		return ctx.VSRSelectedDecision, 0, nil
	}
	if ctx.VSRSelectedDecision != nil {
		return nil, 400, fmt.Errorf("no eligible Flow decision matched for model %q", ctx.RequestModel)
	}
	if decision := r.defaultLooperDecisionByAlgorithm("workflows"); decision != nil {
		return decision, 0, nil
	}
	return nil, 400, fmt.Errorf("no eligible Flow decision matched for model %q", ctx.RequestModel)
}

func isFlowDecision(decision *config.Decision) bool {
	return decision != nil && decision.Algorithm != nil && decision.Algorithm.Type == "workflows"
}
