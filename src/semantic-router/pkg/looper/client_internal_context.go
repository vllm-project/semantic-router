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
	request *http.Request,
	ctx context.Context,
	iteration int,
) {
	request.Header.Set(headers.VSRInternalAuth, internalauth.Token())
	// Carry the caller's original Authorization on a dedicated internal header,
	// never on Authorization, which may already hold the backend's static access
	// key. On the authenticated internal leg extproc treats this as the sole
	// source of caller identity for backends that opt into
	// forward_authorization_header, and strips it before the upstream.
	if c.inboundAuthorization != "" {
		request.Header.Set(headers.VSRInboundAuthorization, c.inboundAuthorization)
	}
	request.Header.Set(headers.VSRLooperRequest, "true")
	request.Header.Set(headers.VSRLooperIteration, fmt.Sprintf("%d", iteration))
	if c.fusionDepth > 0 {
		request.Header.Set(headers.VSRFusionDepth, fmt.Sprintf("%d", c.fusionDepth))
	}
	if recipe := routingRecipeFromContext(ctx); recipe != "" {
		request.Header.Set(headers.VSRSelectedRecipe, string(recipe))
	}
	if c.decisionName != "" {
		request.Header.Set(headers.VSRLooperDecision, c.decisionName)
	}
}
