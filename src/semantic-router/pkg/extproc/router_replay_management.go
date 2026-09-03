package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

// HandleReplayRequest adapts the existing Router Replay serializer to the
// transport-neutral contract consumed by the authenticated management API.
func (r *OpenAIRouter) HandleReplayRequest(
	method string,
	requestTarget string,
) (routerruntime.ReplayResponse, bool) {
	response := r.handleRouterReplayAPI(method, requestTarget)
	if response == nil {
		return routerruntime.ReplayResponse{}, false
	}
	immediate := response.GetImmediateResponse()
	if immediate == nil || immediate.Status == nil {
		return routerruntime.ReplayResponse{
			StatusCode: 500,
			Body:       []byte(`{"error":{"message":"Internal server error","type":"internal_error","code":500}}`),
		}, true
	}
	return routerruntime.ReplayResponse{
		StatusCode: int(immediate.Status.Code),
		Body:       append([]byte(nil), immediate.Body...),
	}, true
}

// AcquireLease keeps the replay store alive while a management request is
// reading it. The router generation already owns this lease through its
// learning runtime, so replay and outcome APIs share one retirement barrier.
func (r *OpenAIRouter) AcquireLease() (func(), bool) {
	if r == nil {
		return nil, false
	}
	return r.routerLearningRuntimeState().AcquireLease()
}

// isRouterReplayRequestTarget intentionally matches the whole reserved prefix
// so variants cannot escape the public-listener deny rule.
func isRouterReplayRequestTarget(requestTarget string) bool {
	path, _, _ := strings.Cut(requestTarget, "?")
	return strings.HasPrefix(path, routerReplayAPIBasePath)
}
