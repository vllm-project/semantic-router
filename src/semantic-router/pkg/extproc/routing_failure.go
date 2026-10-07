package extproc

import (
	"net/http"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// routingFailure is why the Router could not route a request, as its client
// sees it: the HTTP status, and a stable code and short message in the
// client's error envelope. The Router's own reason, with the request details,
// goes only to its log, under the request id.
type routingFailure struct {
	status   int
	category llmprotocol.ErrorCategory
	code     string
	message  string
}

// These are the only routing failures a client can see, and the Router API
// reference documents every one. A candidate-budget rejection keeps the code
// and message of its selection.RequestBudgetError.
var (
	routingFailureModelNotFound = routingFailure{
		status: http.StatusBadRequest, category: llmprotocol.ErrorInvalidRequest,
		code: "model_not_found", message: "the requested model is not available",
	}
	routingFailureNoRoute = routingFailure{
		status: http.StatusBadRequest, category: llmprotocol.ErrorInvalidRequest,
		code: "no_route", message: "no route matched the request",
	}
	routingFailureContextLength = routingFailure{
		status: http.StatusUnprocessableEntity, category: llmprotocol.ErrorInvalidRequest,
		code: "context_length_exceeded", message: "the request exceeds the context window of the models that can serve it",
	}
	routingFailureDecisionUnresolved = routingFailure{
		status: http.StatusServiceUnavailable, category: llmprotocol.ErrorInternal,
		code: "decision_unresolved", message: "the routing decision could not be resolved",
	}
	routingFailureNoEligibleModel = routingFailure{
		status: http.StatusServiceUnavailable, category: llmprotocol.ErrorInternal,
		code: "no_eligible_model", message: "no model is eligible to serve the request",
	}
)

// routingFailureResponse answers a request with failure. The client's error
// envelope is rendered from it when the reply is encoded for the client.
func (r *OpenAIRouter) routingFailureResponse(ctx *RequestContext, failure routingFailure) *ext_proc.ProcessingResponse {
	if ctx != nil {
		ctx.ImmediateProtocolError = llmprotocol.NewError(failure.category, failure.code, failure.message, nil)
	}
	return r.createErrorResponse(failure.status, failure.message)
}

// logRoutingFailure records why a request was not routed, under the request
// id and with the code its client receives.
func logRoutingFailure(ctx *RequestContext, event, code string, cause error) {
	requestID := ""
	if ctx != nil {
		requestID = ctx.RequestID
	}
	logging.ComponentWarnEvent("extproc", event, map[string]interface{}{
		"request_id": requestID,
		"code":       code,
		"error":      cause.Error(),
	})
}
