package extproc

import (
	"encoding/json"
	"net/http"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// listenerPathRejection applies the API grant of the listener the gateway
// named in x-vsr-listener to the path the client sent, as the standalone
// gateway does at ingress: 404 with a detail naming the listener for a path
// it does not serve. It runs before the request-body phase rewrites the path
// to the provider's, so the grant never reads the provider path. A gateway
// that names no listener gets no grant; one that names a listener the
// Router's configuration does not have gets 503.
func (r *OpenAIRouter) listenerPathRejection(method, target string, ctx *RequestContext) *ext_proc.ProcessingResponse {
	if ctx == nil || ctx.GatewayListener == "" {
		return nil
	}
	listener, ok := r.Config.ListenerByName(ctx.GatewayListener)
	if !ok {
		logging.ComponentWarnEvent("extproc", "gateway_listener_unknown", map[string]interface{}{
			"request_id": ctx.RequestID,
			"listener":   ctx.GatewayListener,
		})
		return r.createErrorResponse(http.StatusServiceUnavailable, "The router is not serving this listener.")
	}
	path, _, _ := strings.Cut(target, "?")
	path = listener.APIPath(path)
	// Every listener answers the model listing, as in standalone mode.
	if method == http.MethodGet && path == "/v1/models" {
		return nil
	}
	_, detail, served := listener.PathGrant(path)
	if served {
		return nil
	}
	logging.ComponentDebugEvent("extproc", "listener_path_not_served", map[string]interface{}{
		"request_id": ctx.RequestID,
		"listener":   listener.Name,
		"path":       path,
	})
	body, err := json.Marshal(map[string]string{"detail": detail})
	if err != nil {
		return r.createErrorResponse(http.StatusInternalServerError, "Internal server error")
	}
	// The body is the standalone gateway's, not an LLM API error to re-encode.
	ctx.ImmediateResponseEncoded = true
	return createImmediateJSONResponse(http.StatusNotFound, body)
}
