package extproc

import (
	"fmt"
	"net/http"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// listenerModelRejection answers a request whose model the listener it
// arrived on does not accept, before any signal, cache or decision runs. The
// model is the one request decoding parsed, which routing uses too.
func (r *OpenAIRouter) listenerModelRejection(model string, ctx *RequestContext) *ext_proc.ProcessingResponse {
	if ctx == nil || ctx.ListenerModels == nil || ctx.ListenerModels.Allows(model) {
		return nil
	}
	logging.ComponentDebugEvent("extproc", "listener_model_not_allowed", map[string]interface{}{
		"request_id": ctx.RequestID,
		"model":      model,
	})
	message := fmt.Sprintf("The model %q is not available on this endpoint.", model)
	ctx.ImmediateProtocolError = llmprotocol.NewError(llmprotocol.ErrorPermission, "model_not_allowed", message, nil)
	return r.createErrorResponse(http.StatusForbidden, message)
}
