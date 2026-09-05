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
func (c *Client) setInternalRequestHeaders(
	header http.Header,
	ctx context.Context,
	iteration int,
	decisionName string,
	fusionDepth int,
) {
	header.Set(headers.VSRInternalAuth, internalauth.Token())
	header.Set(headers.VSRLooperRequest, "true")
	header.Set(headers.VSRLooperIteration, fmt.Sprintf("%d", iteration))
	if fusionDepth > 0 {
		header.Set(headers.VSRFusionDepth, fmt.Sprintf("%d", fusionDepth))
	}
	if recipe := routingRecipeFromContext(ctx); recipe != "" {
		header.Set(headers.VSRSelectedRecipe, string(recipe))
	}
	if decisionName != "" {
		header.Set(headers.VSRLooperDecision, decisionName)
	}
}

func (c *Client) requestHeaders(
	ctx context.Context,
	iteration int,
	decisionName string,
	fusionDepth int,
	accessKey string,
) http.Header {
	header := make(http.Header, len(c.headers)+5)
	header.Set("Content-Type", "application/json")
	for name, value := range c.headers {
		header.Set(name, value)
	}
	if accessKey != "" {
		header.Set("Authorization", "Bearer "+accessKey)
	}
	c.setInternalRequestHeaders(header, ctx, iteration, decisionName, fusionDepth)
	return header
}
