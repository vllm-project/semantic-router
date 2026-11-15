package extproc

import (
	"context"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/principal"
)

// The delegated credential is the requesting user's credential a deployment
// authenticator attaches to an authenticated request (issue #2286). It is a
// trusted authentication output like the identity headers: the gateway in
// front strips client-supplied copies, and a standalone listener drops them
// unless it trusts identity headers. The Router takes it off the request at
// the header phase, keeps it typed on the request context and on the
// principal of the calls made for the request (package principal), removes
// the header from every upstream request, and presents the value only to an
// upstream whose auth is credential: delegated, under that upstream's auth
// header. It never stands in for a static key, and no Router-own call can
// present it.

// delegatedCredentialMessage is the client-visible reason; it never carries a
// credential value. delegatedCredentialCode is its protocol error code.
const (
	delegatedCredentialMessage = "Authentication failed. The selected upstream requires a delegated credential, which the request does not carry."
	delegatedCredentialCode    = "delegated_credential_missing" // #nosec G101 -- error code, not a credential
)

func (r *OpenAIRouter) delegatedCredentialConfig() config.DelegatedCredentialConfig {
	if r == nil || r.Config == nil {
		return config.DelegatedCredentialConfig{}
	}
	return r.Config.Authz.DelegatedCredential
}

// delegatedCredentialHeadersForRemoval names the headers no upstream request
// may carry: the default header always, and the configured one when it
// differs. Removing the default even when the contract is disabled keeps the
// reserved x-vsr-* name from reaching an upstream.
func (r *OpenAIRouter) delegatedCredentialHeadersForRemoval() []string {
	names := []string{headers.VSRDelegatedCredential}
	cfg := r.delegatedCredentialConfig()
	if cfg.Enabled {
		if configured := cfg.GetHeader(); configured != headers.VSRDelegatedCredential {
			names = append(names, configured)
		}
	}
	return names
}

// appendDelegatedCredentialRemoval adds a renamed credential header to a
// header-phase mutation so every branch after the header phase,
// skip-processing included, sends the upstream a request without it. The
// default name is already one of the headers every provider-bound request
// drops (headers.KeptFromProviders).
func (r *OpenAIRouter) appendDelegatedCredentialRemoval(mutation *ext_proc.HeaderMutation) {
	if mutation == nil {
		return
	}
	for _, name := range r.delegatedCredentialHeadersForRemoval() {
		if name != headers.VSRDelegatedCredential {
			mutation.RemoveHeaders = append(mutation.RemoveHeaders, name)
		}
	}
}

// captureDelegatedCredential moves the credential off ctx.Headers onto the
// typed request context, and makes every call made with the request's
// context one made for it, with the client headers deployments forward. A
// hop never reads either from headers: its context derives from the client
// request's (see executeLooperGraph), so it shares that request's principal,
// and nothing a hop request carries can pose as a credential. The value is
// kept only when the request also carries the trusted principal the same
// authenticator sets; a credential without an authentication result is
// dropped.
func (r *OpenAIRouter) captureDelegatedCredential(ctx *RequestContext) {
	if ctx == nil {
		return
	}
	cfg := r.delegatedCredentialConfig()
	for _, name := range r.delegatedCredentialHeadersForRemoval() {
		if !cfg.Enabled || ctx.Hop != nil {
			removeHeaderValueCI(ctx, name)
			continue
		}
		value := strings.TrimSpace(headerValueCI(ctx, name))
		removeHeaderValueCI(ctx, name)
		if value == "" || name != cfg.GetHeader() {
			continue
		}
		if strings.TrimSpace(headerValueCI(ctx, r.Config.Authz.Identity.GetUserIDHeader())) == "" {
			logging.ComponentWarnEvent("extproc", "delegated_credential_without_principal_dropped", map[string]interface{}{
				"request_id": ctx.RequestID,
				"header":     name,
			})
			continue
		}
		ctx.delegatedCredential = value
	}
	if ctx.TraceContext == nil {
		ctx.TraceContext = context.Background()
	}
	if ctx.Hop != nil {
		ctx.delegatedCredential, _ = principal.RequestCredential(ctx.TraceContext)
		return
	}
	ctx.TraceContext = principal.ForRequestWithHeaders(ctx.TraceContext, ctx.delegatedCredential, r.deploymentForwardedHeaders(ctx))
}

// deploymentForwardedHeaders are the client's values of the request headers
// some deployment forwards, by lower-case name: what the calls made for the
// request may copy (connector.DeploymentAuthorizer). The values are copied
// once, so those calls never read the request's header map.
func (r *OpenAIRouter) deploymentForwardedHeaders(ctx *RequestContext) map[string]string {
	if r == nil {
		return nil
	}
	var forwarded map[string]string
	for _, name := range r.Config.DeploymentForwardedHeaders() {
		if value := headerValueCI(ctx, name); value != "" {
			if forwarded == nil {
				forwarded = map[string]string{}
			}
			forwarded[name] = value
		}
	}
	return forwarded
}

// requestDelegatedCredential is the credential a delegated upstream may
// receive for the request.
func requestDelegatedCredential(ctx *RequestContext) (string, bool) {
	if ctx == nil || ctx.delegatedCredential == "" {
		return "", false
	}
	return ctx.delegatedCredential, true
}

// delegatedCredentialMissing answers the client 401 before a call to a
// delegated upstream that the request cannot pay for.
func (r *OpenAIRouter) delegatedCredentialMissing(ctx *RequestContext) *ext_proc.ProcessingResponse {
	// The protocol contract replaces a bare 401 with a generic message; the
	// typed error keeps the client-visible reason.
	ctx.ImmediateProtocolError = llmprotocol.NewError(
		llmprotocol.ErrorAuthentication, delegatedCredentialCode, delegatedCredentialMessage, nil,
	)
	return r.createErrorResponse(401, delegatedCredentialMessage)
}

// delegatedCredentialRefusal answers 401 once a call made for the request was
// refused because the request carries no delegated credential, as a decision
// model's is, whose failure a signal would otherwise absorb.
func (r *OpenAIRouter) delegatedCredentialRefusal(ctx *RequestContext) *ext_proc.ProcessingResponse {
	if ctx == nil || !principal.Refused(ctx.TraceContext) {
		return nil
	}
	logging.ComponentWarnEvent("extproc", "delegated_credential_missing", map[string]interface{}{
		"request_id": ctx.RequestID,
	})
	return r.delegatedCredentialMissing(ctx)
}
