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
