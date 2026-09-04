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
	decisionName string,
	fusionDepth int,
) {
	request.Header.Set(headers.VSRInternalAuth, internalauth.Token())
	request.Header.Set(headers.VSRLooperRequest, "true")
	request.Header.Set(headers.VSRLooperIteration, fmt.Sprintf("%d", iteration))
	if fusionDepth > 0 {
		request.Header.Set(headers.VSRFusionDepth, fmt.Sprintf("%d", fusionDepth))
	}
	if recipe := routingRecipeFromContext(ctx); recipe != "" {
		request.Header.Set(headers.VSRSelectedRecipe, string(recipe))
	}
	if decisionName != "" {
		request.Header.Set(headers.VSRLooperDecision, decisionName)
	}
}
