package extproc

import (
	"context"
	"errors"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// The gateway credential is the dedicated credential a deployment
// authenticator attaches to an authenticated request for a trusted upstream
// gateway (issue #2286). It is a trusted authentication output like the
// identity headers: the gateway in front strips client-supplied copies, and a
// standalone listener drops them unless it trusts identity headers. The Router
// takes it off the request at the header phase, keeps it typed on the request
// context, removes the header from every upstream request, and presents the
// value only to a backend marked trusted_gateway, under that backend's auth
// header. It never stands in for a static key, and no router-initiated call
// can use it.

// errTrustedGatewayCredentialMissing fails a dispatch to a trusted gateway
// whose request carries no trusted gateway credential.
var errTrustedGatewayCredentialMissing = errors.New("the selected backend requires a trusted gateway credential, which the request does not carry")

// errTrustedGatewayRouterInitiated refuses the credential to calls the Router
// makes on its own behalf; only the client's request may spend it.
var errTrustedGatewayRouterInitiated = errors.New("the selected backend requires the request's trusted gateway credential, which a router-initiated call cannot use")

// trustedGatewayCredentialMessage is the client-visible reason; it never
// carries a credential value. trustedGatewayCredentialCode is its protocol
// error code.
const (
	trustedGatewayCredentialMessage = "Authentication failed. The selected backend requires a trusted gateway credential."
	trustedGatewayCredentialCode    = "trusted_gateway_credential_missing"
)

func (r *OpenAIRouter) gatewayCredentialConfig() config.GatewayCredentialConfig {
	if r == nil || r.Config == nil {
		return config.GatewayCredentialConfig{}
	}
	return r.Config.Authz.GatewayCredential
}

// gatewayCredentialHeadersForRemoval names the headers no upstream request may
// carry: the default header always, and the configured one when it differs.
// Removing the default even when the contract is disabled keeps the reserved
// x-vsr-* name from reaching a backend.
func (r *OpenAIRouter) gatewayCredentialHeadersForRemoval() []string {
	names := []string{headers.VSRGatewayCredential}
	cfg := r.gatewayCredentialConfig()
	if cfg.Enabled {
		if configured := cfg.GetHeader(); configured != headers.VSRGatewayCredential {
			names = append(names, configured)
		}
	}
	return names
}

// appendGatewayCredentialRemoval adds the credential headers to a header-phase
// mutation so every branch after the header phase, skip-processing included,
// sends the upstream a request without them.
func (r *OpenAIRouter) appendGatewayCredentialRemoval(mutation *ext_proc.HeaderMutation) {
	if mutation == nil {
		return
	}
	mutation.RemoveHeaders = append(mutation.RemoveHeaders, r.gatewayCredentialHeadersForRemoval()...)
}

// captureGatewayCredential moves the credential off ctx.Headers onto the
// typed request context. A hop never reads it from headers: it inherits the
// client request's credential through its context (see withGatewayCredential),
// so nothing a hop request carries can pose as one. The value is kept only
// when the request also carries the trusted principal the same authenticator
// sets; a credential without an authentication result is dropped.
func (r *OpenAIRouter) captureGatewayCredential(ctx *RequestContext) {
	if ctx == nil {
		return
	}
	cfg := r.gatewayCredentialConfig()
	for _, name := range r.gatewayCredentialHeadersForRemoval() {
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
			logging.ComponentWarnEvent("extproc", "gateway_credential_without_principal_dropped", map[string]interface{}{
				"request_id": ctx.RequestID,
				"header":     name,
			})
			continue
		}
		ctx.gatewayCredential = value
	}
}

// trustedGatewayCredential is the credential a trusted gateway may receive.
func trustedGatewayCredential(ctx *RequestContext) (string, bool) {
	if ctx == nil || ctx.gatewayCredential == "" {
		return "", false
	}
	return ctx.gatewayCredential, true
}

type gatewayCredentialKey struct{}

// withGatewayCredential hands the client request's credential to the hops
// run on its behalf, typed and in process, never through a header.
func withGatewayCredential(ctx context.Context, reqCtx *RequestContext) context.Context {
	credential, ok := trustedGatewayCredential(reqCtx)
	if !ok || ctx == nil {
		return ctx
	}
	return context.WithValue(ctx, gatewayCredentialKey{}, credential)
}

// gatewayCredentialFrom returns the credential a hop's context inherited.
func gatewayCredentialFrom(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	credential, _ := ctx.Value(gatewayCredentialKey{}).(string)
	return credential
}
