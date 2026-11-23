package extproc

import (
	"net/http"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/clientauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// An entrypoint with client_auth admits a request only with one of its keys,
// presented as Authorization: Bearer <key>. Admission runs once the request
// names its target and before any work done for it, so a refused request
// selects no recipe, reads no stored history, runs no signal, cache or
// translation, and reaches no backend.

const (
	clientKeyCode      = "invalid_api_key" // #nosec G101 -- error code, not a credential
	clientKeyMessage   = "Authentication failed. The model takes an API key, which the request must present as Authorization: Bearer <key>."
	clientKeyChallenge = `Bearer realm="vllm-semantic-router"`
	supersededCode     = "configuration_changed"
	supersededMessage  = "The Router's configuration changed while the request arrived, and it now authenticates the model differently. Retry the request."
)

// clientAdmissionRefusal answers a request for model that its client may not
// make. An in-process hop serves a request already admitted.
func (r *OpenAIRouter) clientAdmissionRefusal(ctx *RequestContext, model string) *ext_proc.ProcessingResponse {
	if ctx.Hop != nil {
		return nil
	}
	served, active := r.clientPolicies()
	switch clientauth.Admit(served, active, config.ChatAPI, model, headerValueCI(ctx, headers.Authorization)) {
	case clientauth.Unauthorized:
		logging.ComponentDebugEvent("extproc", "client_key_refused", map[string]interface{}{
			"request_id": ctx.RequestID, "model": model,
		})
		ctx.ImmediateProtocolError = llmprotocol.NewError(llmprotocol.ErrorAuthentication, clientKeyCode, clientKeyMessage, nil)
		response := r.createErrorResponse(http.StatusUnauthorized, clientKeyMessage)
		appendImmediateResponseHeader(response, "WWW-Authenticate", clientKeyChallenge)
		return response
	case clientauth.Superseded:
		return r.supersededResponse(ctx)
	}
	return nil
}

// skippedRequestRefusal answers a request that took the skip-processing
// opt-out, which only a configuration without client keys honors, once the
// configuration active when its body arrives has some.
func (r *OpenAIRouter) skippedRequestRefusal(ctx *RequestContext) *ext_proc.ProcessingResponse {
	if _, active := r.clientPolicies(); !active.AnyRequired() {
		return nil
	}
	return r.supersededResponse(ctx)
}

func (r *OpenAIRouter) supersededResponse(ctx *RequestContext) *ext_proc.ProcessingResponse {
	ctx.ImmediateProtocolError = llmprotocol.NewError(llmprotocol.ErrorConflict, supersededCode, supersededMessage, nil)
	return r.createErrorResponse(http.StatusConflict, supersededMessage)
}

// clientKeysRequired reports whether an entrypoint of the router's
// configuration takes a client key.
func (r *OpenAIRouter) clientKeysRequired() bool {
	served, _ := r.clientPolicies()
	return served.AnyRequired()
}

// clientPolicies returns the client policy of the configuration the router
// serves and that of the configuration active now. They differ once a reload
// activates another configuration while a request arrives.
func (r *OpenAIRouter) clientPolicies() (served, active *clientauth.Policy) {
	if r == nil {
		return nil, nil
	}
	r.routerLearningMu.Lock()
	generation := r.generation
	r.routerLearningMu.Unlock()
	if generation == nil {
		return nil, nil
	}
	served = snapshotClientAuth(generation.snapshot)
	return served, generation.service.activeClientAuth(served)
}

// ActiveClientAuth returns the client policy of the configuration active now,
// which differs from that of the leased snapshot once a reload activates
// another configuration during the lease.
func (l *Lease) ActiveClientAuth() *clientauth.Policy {
	return l.service.activeClientAuth(snapshotClientAuth(l.Snapshot))
}

// activeClientAuth returns the client policy of the generation rs serves now,
// or fallback when there is none.
func (rs *RouterService) activeClientAuth(fallback *clientauth.Policy) *clientauth.Policy {
	if rs == nil {
		return fallback
	}
	generation := rs.current.Load()
	if generation == nil {
		return fallback
	}
	return snapshotClientAuth(generation.snapshot)
}

func snapshotClientAuth(snapshot *configsnapshot.Snapshot) *clientauth.Policy {
	if snapshot == nil {
		return nil
	}
	return snapshot.ClientAuth()
}
