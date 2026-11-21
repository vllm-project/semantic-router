package extproc

import (
	"fmt"
	"net/http"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// listenerModelRejection answers a request whose model the listener it
// arrived on does not accept, before any signal, cache or decision runs: 403
// model_not_allowed for a model the Router serves elsewhere, and 404
// model_not_found for one it does not serve at all. The model is the one
// request decoding parsed, which routing uses too.
func (r *OpenAIRouter) listenerModelRejection(model string, ctx *RequestContext) *ext_proc.ProcessingResponse {
	if r.Config != nil && r.Config.IsSystemOneBackend(model) {
		if _, chat := r.Config.ResolveEntrypoint(config.ChatAPI, model); !chat {
			message := "This model serves the System One API. Use /v1/systemone."
			if ctx != nil {
				ctx.ImmediateProtocolError = llmprotocol.NewError(llmprotocol.ErrorPermission, "model_api_mismatch", message, nil)
			}
			return r.createErrorResponse(http.StatusBadRequest, message)
		}
	}
	if ctx == nil || ctx.ListenerModels == nil || ctx.ListenerModels.Allows(model) {
		return nil
	}
	if !r.servesModel(model) {
		return r.unavailableModelResponse(model, ctx)
	}
	logging.ComponentDebugEvent("extproc", "listener_model_not_allowed", map[string]interface{}{
		"request_id": ctx.RequestID,
		"model":      model,
	})
	message := fmt.Sprintf("The model %q is not available on this endpoint.", model)
	ctx.ImmediateProtocolError = llmprotocol.NewError(llmprotocol.ErrorPermission, "model_not_allowed", message, nil)
	return r.createErrorResponse(http.StatusForbidden, message)
}

// maxRequestedModelBytes bounds the client's model name a routed request
// keeps for its receipts.
const maxRequestedModelBytes = 256

// routeUnservedModel is the model a request is routed as: the one it names,
// or the listener API's unserved_models route_to when the API routes models
// it does not accept or the Router does not serve. The name the client gave
// is kept for the record and the trace only; no signal reads it. A System
// One model keeps its own answer.
func (r *OpenAIRouter) routeUnservedModel(model string, ctx *RequestContext) string {
	if route := r.unservedModelRoute(model, ctx); route == model {
		return model
	}
	ctx.RequestedModel = truncateShadowText(model, maxRequestedModelBytes)
	if ctx.SemanticRequest != nil {
		ctx.SemanticRequest.Model = ctx.UnservedModelRoute
	}
	logging.ComponentDebugEvent("extproc", "unserved_model_routed", map[string]interface{}{
		"request_id": ctx.RequestID,
		"route_to":   ctx.UnservedModelRoute,
	})
	return ctx.UnservedModelRoute
}

// unservedModelRoute is the model routeUnservedModel routes a request naming
// model as, without recording the route.
func (r *OpenAIRouter) unservedModelRoute(model string, ctx *RequestContext) string {
	if ctx == nil || ctx.UnservedModelRoute == "" || model == "" || model == ctx.UnservedModelRoute {
		return model
	}
	if r.Config != nil && r.Config.IsSystemOneBackend(model) {
		return model
	}
	if (ctx.ListenerModels == nil || ctx.ListenerModels.Allows(model)) && r.servesModel(model) {
		return model
	}
	return ctx.UnservedModelRoute
}

// servesModel reports whether a request naming model reaches anything the
// Router serves, on any listener: an entrypoint of either API, a model with a
// backend or one of its LoRA adapters, a model an external gateway serves, or
// a System One backend.
func (r *OpenAIRouter) servesModel(model string) bool {
	if r == nil || r.Config == nil {
		return false
	}
	_, systemOne := r.Config.ResolveEntrypoint(config.SystemOneAPI, model)
	return systemOne || r.Config.IsEntrypointModelName(model) || r.Config.RoutesModelToBackend(model) ||
		r.Config.UsesExternalGatewayDispatch(model) || r.Config.IsSystemOneBackend(model)
}
