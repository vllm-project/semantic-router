package looper

import (
	"context"
	"fmt"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/internalauth"
)

// setInternalRequestHeaders attaches authenticated routing context for the
// in-process extproc hop. These values are consumed and removed before the
// physical model backend is invoked.
func setInternalRequestHeaders(
	header http.Header,
	ctx context.Context,
	options CallOptions,
) {
	header.Set(headers.VSRInternalAuth, internalauth.Token())
	header.Set(headers.VSRLooperRequest, "true")
	header.Set(headers.VSRLooperIteration, fmt.Sprintf("%d", options.Iteration))
	if options.FusionDepth > 0 {
		header.Set(headers.VSRFusionDepth, fmt.Sprintf("%d", options.FusionDepth))
	}
	if recipe := routingRecipeFromContext(ctx); recipe != "" {
		header.Set(headers.VSRSelectedRecipe, string(recipe))
	}
	if options.DecisionName != "" {
		header.Set(headers.VSRLooperDecision, options.DecisionName)
	}
}

func (c *Client) requestHeaders(
	ctx context.Context,
	target ModelTarget,
	options CallOptions,
) http.Header {
	header := make(http.Header, len(c.headers)+5)
	header.Set("Content-Type", "application/json")
	for name, value := range c.headers {
		header.Set(name, value)
	}
	if target.AccessKey != "" {
		header.Set("Authorization", "Bearer "+target.AccessKey)
	}
	setInternalRequestHeaders(header, ctx, options)
	return header
}
