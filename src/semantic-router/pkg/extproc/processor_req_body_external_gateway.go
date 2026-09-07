package extproc

import (
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// usesExternalGatewayDispatch reports whether a concrete model is declared as
// protocol metadata only. In this mode the external gateway owns the route,
// backend, and credentials; Semantic Router owns only protocol conversion and
// Responses object state.
func (r *OpenAIRouter) usesExternalGatewayDispatch(model string) bool {
	if r == nil || r.Config == nil || r.Config.ModelSelection.Enabled || len(r.Config.Listeners) != 0 {
		return false
	}
	model = strings.TrimSpace(model)
	if model == "" || len(r.Config.GetEndpointsForModel(model)) != 0 {
		return false
	}
	_, configured := r.Config.ModelConfig[model]
	return configured
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
	ctx.VSRSelectionMethod = ""
	ctx.VSRSelectionReasoning = ""
	ctx.VSREligibleModelRefs = nil
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
