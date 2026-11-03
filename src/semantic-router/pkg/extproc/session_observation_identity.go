package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// requestIdentity is the configured session, conversation, and agent parsed
// once from the request headers.
type requestIdentity struct {
	SessionID      string
	ConversationID string
	AgentID        string
}

// ensureRequestIdentity parses the configured learning identity headers once.
// Later reads, including previous-model observations and Replay, use the stored
// value instead of reading the headers again.
func (r *OpenAIRouter) ensureRequestIdentity(ctx *RequestContext) requestIdentity {
	if ctx == nil {
		return requestIdentity{}
	}
	if ctx.requestIdentityReady {
		return ctx.requestIdentity
	}
	headers := config.RouterLearningIdentityHeadersConfig{}
	if r != nil && r.Config != nil {
		headers = r.Config.RouterLearning.EffectiveIdentity().Headers
	}
	ctx.requestIdentity = requestIdentity{
		SessionID:      strings.TrimSpace(headerValueCI(ctx, headers.SessionHeader())),
		ConversationID: strings.TrimSpace(headerValueCI(ctx, headers.ConversationHeader())),
		AgentID:        strings.TrimSpace(headerValueCI(ctx, headers.AgentHeader())),
	}
	ctx.requestIdentityReady = true
	return ctx.requestIdentity
}

// modelObservationKey is the previous-model and idle key for this request.
// A configured session and agent stay together. Every other request keeps the
// routing session key, including KV affinity. Callers that run before identity
// is resolved only see that session key, so ingress resolves identity first.
func modelObservationKey(ctx *RequestContext) string {
	if ctx == nil || requestBypassesRouting(ctx) {
		return ""
	}
	if ctx.requestIdentity.SessionID != "" && ctx.requestIdentity.AgentID != "" {
		return sessiontelemetry.RoutingSessionKey(
			ctx.Routing.RecipeName(),
			ctx.requestIdentity.SessionID,
			"agent",
			ctx.requestIdentity.AgentID,
		)
	}
	return routingSessionStateKey(ctx)
}

// selectionObservationKey resolves identity, then returns the key selection and
// previous-model memory share. When that key is empty, selection keeps the
// session id derived from the request itself.
func (r *OpenAIRouter) selectionObservationKey(ctx *RequestContext, sessionID string) string {
	r.ensureRequestIdentity(ctx)
	if key := modelObservationKey(ctx); key != "" {
		return key
	}
	var recipe config.RecipeName
	if ctx != nil {
		recipe = ctx.Routing.RecipeName()
	}
	return sessiontelemetry.RoutingSessionKey(recipe, sessionID)
}
