package extproc

import (
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

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

func buildLooperInternalHeaderRemovalMutation() *ext_proc.HeaderMutation {
	return &ext_proc.HeaderMutation{
		RemoveHeaders: looperInternalHeadersForRemoval(),
	}
}
