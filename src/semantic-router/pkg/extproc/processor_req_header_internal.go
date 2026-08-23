package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/internalauth"
)

var looperInternalContextHeaders = []string{
	headers.VSRInternalAuth,
	headers.VSRLooperRequest,
	headers.VSRLooperIteration,
	headers.VSRLooperDecision,
	headers.VSRFusionDepth,
	headers.VSRSelectedRecipe,
	headers.VSRRoutingNamespace,
	headers.VSRRoutingQuotaPartition,
	headers.VSRRoutingPublication,
	headers.VSRRoutingRuntimeEpoch,
	headers.VSRRoutingSnapshotRevision,
	headers.VSRRoutingDigest,
}

// gatewayReservedContextHeaders are identity or transport-control names that
// external callers may attempt to spoof. They are removed from RequestContext
// before routing and from every upstream request.
var gatewayReservedContextHeaders = []string{
	"x-vllm-sr-api-key-id",
	"x-vllm-sr-user-id",
	"x-vllm-sr-team-id",
	"x-authz-user-id",
	"x-authz-user-groups",
	"x-authz-team-id",
	"x-authz-tenant-id",
}

func scrubUntrustedIdentityHeaders(ctx *RequestContext) {
	for _, header := range gatewayReservedContextHeaders {
		removeHeaderValueCI(ctx, header)
	}
}

func authenticateLooperRequestContext(ctx *RequestContext) {
	if ctx == nil {
		return
	}

	markerPresent := strings.EqualFold(
		strings.TrimSpace(headerValueCI(ctx, headers.VSRLooperRequest)),
		"true",
	)
	ctx.LooperRequest = markerPresent &&
		internalauth.Authenticate(headerValueCI(ctx, headers.VSRInternalAuth))

	// The credential is only needed while authenticating the captured context.
	removeHeaderValueCI(ctx, headers.VSRInternalAuth)
	if ctx.LooperRequest {
		return
	}

	// Treat unauthenticated internal context as a normal external request. The
	// recipe and decision hints must not influence routing or plugin execution.
	for _, header := range looperInternalContextHeaders {
		removeHeaderValueCI(ctx, header)
	}
}

func removeHeaderValueCI(ctx *RequestContext, canonical string) {
	if ctx == nil || canonical == "" {
		return
	}
	for key := range ctx.Headers {
		if strings.EqualFold(key, canonical) {
			delete(ctx.Headers, key)
		}
	}
}

func looperInternalHeadersForRemoval() []string {
	return append([]string(nil), looperInternalContextHeaders...)
}

func upstreamInternalHeadersForRemoval(stripAuthorization bool) []string {
	names := make([]string, 0, len(looperInternalContextHeaders)+len(gatewayReservedContextHeaders)+1)
	names = append(names, looperInternalContextHeaders...)
	names = append(names, gatewayReservedContextHeaders...)
	if stripAuthorization {
		names = append(names, "authorization")
	}
	return names
}
