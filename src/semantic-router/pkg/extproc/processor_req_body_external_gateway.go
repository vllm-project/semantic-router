package extproc

import (
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// usesExternalGatewayDispatch reports whether a concrete model is declared as
// protocol metadata only; see config.RouterConfig.UsesExternalGatewayDispatch.
func (r *OpenAIRouter) usesExternalGatewayDispatch(model string) bool {
	return r != nil && r.Config.UsesExternalGatewayDispatch(model)
}

func (r *OpenAIRouter) handleExternalGatewayModelRouting(
	request *llmprotocol.Request,
	model string,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	targetFormat, err := wireFormatForModel(r.Config.GetModelAPIFormat(model))
	if err != nil {
		return nil, err
	}
	dispatch := &providerDispatch{
		logicalModel:  model,
		upstreamModel: model,
		targetFormat:  targetFormat,
	}
	ctx.VSRSelectedModel = ""
	ctx.VSRSelectedDecisionName = ""
	ctx.VSRSelectedDecision = nil
	ctx.VSRSelectedDecisionConfidence = 0
	ctx.VSRSelectedDecisionConfidenceScored = false
	ctx.VSRSelectionMethod = ""
	ctx.VSRSelectionReasoning = ""
	ctx.VSRSelectionTrace = nil
	ctx.VSREligibleModelRefs = nil
	ctx.VSRPolicyEligibleModelRefs = nil
	ctx.VSRSelectedCandidate = nil
	ctx.pendingSessionDecision = nil
	ctx.VSRReasoningMode = "off"

	changed, err := r.prepareProviderRequest(request, dispatch, ctx)
	if err != nil {
		return nil, err
	}
	if changed {
		request.Generation++
	}
	ctx.TargetFormat = dispatch.targetFormat
	ctx.SemanticRequest = request

	ctx.RequestModel = model

	state := &routeHeaderState{removeHeaders: []string{"content-length"}}
	// Rewriting the protocol endpoint is part of request conversion. Envoy
	// retains the already-selected route because this mode never clears the
	// route cache, so upstream ownership remains with the external gateway.
	setProviderRequestPath(&state.setHeaders, nil, targetFormat)
	response := buildRequestBodyContinueResponse(state, nil, false)

	logging.ComponentDebugEvent("extproc", "external_gateway_dispatch_prepared", map[string]interface{}{
		"request_id":  ctx.RequestID,
		"model":       model,
		"wire_format": targetFormat,
	})
	return r.finalizeProviderDispatchResponse(dispatch, response, ctx)
}
