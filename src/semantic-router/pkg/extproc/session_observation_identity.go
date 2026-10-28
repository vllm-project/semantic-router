package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// sessionObservationAgentComponent keeps a thread key from colliding with a
// one-component session key or a two-component session/conversation key.
const sessionObservationAgentComponent = "agent"

// configuredAgentID returns the value of identity.headers.agent when that
// header name is configured and the request carries it. An unset header name
// has no default, so requests without the setting keep their existing key.
func (r *OpenAIRouter) configuredAgentID(ctx *RequestContext) string {
	if r == nil || r.Config == nil || ctx == nil {
		return ""
	}
	headerName := r.Config.RouterLearning.Protection.HeaderName("agent")
	if headerName == "" {
		return ""
	}
	return strings.TrimSpace(headerValueCI(ctx, headerName))
}

// configuredAgentObservationKey identifies previous-model and idle observations
// for one thread. It is empty unless both the configured session value and the
// configured agent value are present. Protection's hold key is unchanged, and
// this key is not the KV address key.
func (r *OpenAIRouter) configuredAgentObservationKey(ctx *RequestContext) string {
	if r == nil || r.Config == nil || ctx == nil || requestBypassesRouting(ctx) {
		return ""
	}
	agentID := r.configuredAgentID(ctx)
	if agentID == "" {
		return ""
	}
	sessionID := strings.TrimSpace(headerValueCI(ctx, r.Config.RouterLearning.Protection.HeaderName("session")))
	if sessionID == "" {
		return ""
	}
	return sessiontelemetry.RoutingSessionKey(ctx.Routing.RecipeName(), sessionID, sessionObservationAgentComponent, agentID)
}

// bindThreadObservationKey caches the thread observation key for package
// functions that record or read previous-model and idle state. An empty result
// leaves those functions on routingSessionStateKey.
func (r *OpenAIRouter) bindThreadObservationKey(ctx *RequestContext) {
	if r == nil || ctx == nil {
		return
	}
	ctx.threadObservationKey = r.configuredAgentObservationKey(ctx)
}

// modelObservationKey is the previous-model and idle key for this request.
func modelObservationKey(ctx *RequestContext) string {
	if ctx != nil && ctx.threadObservationKey != "" {
		return ctx.threadObservationKey
	}
	return routingSessionStateKey(ctx)
}
