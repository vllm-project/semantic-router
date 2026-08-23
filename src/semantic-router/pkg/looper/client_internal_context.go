package looper

import (
	"context"
	"fmt"
	"net/http"
	"strconv"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/internalauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
)

// setInternalRequestHeaders attaches authenticated routing context for the
// in-process extproc hop. These values are consumed and removed before the
// physical model backend is invoked.
func (c *Client) setInternalRequestHeaders(
	request *http.Request,
	ctx context.Context,
	iteration int,
	dispatchGrant string,
	requestID string,
) {
	request.Header.Set(headers.VSRInternalAuth, internalauth.Token())
	request.Header.Set(headers.VSRLooperRequest, "true")
	request.Header.Set(headers.VSRLooperIteration, fmt.Sprintf("%d", iteration))
	if c.fusionDepth > 0 {
		request.Header.Set(headers.VSRFusionDepth, fmt.Sprintf("%d", c.fusionDepth))
	}
	if dispatchGrant != "" {
		request.Header.Set(headers.VSRDispatchGrant, dispatchGrant)
	}
	if requestID != "" {
		request.Header.Set(headers.RequestID, requestID)
	}
	if recipe := routingRecipeFromContext(ctx); recipe != "" {
		request.Header.Set(headers.VSRSelectedRecipe, string(recipe))
	}
	if c.decisionName != "" {
		request.Header.Set(headers.VSRLooperDecision, c.decisionName)
	}
	if generation, ok := routingcontext.GenerationFrom(ctx); ok {
		request.Header.Set(headers.VSRRoutingNamespace, generation.NamespaceID)
		request.Header.Set(headers.VSRRoutingQuotaPartition, generation.QuotaPartition)
		request.Header.Set(headers.VSRRoutingPublication, generation.PublicationID)
		request.Header.Set(headers.VSRRoutingRuntimeEpoch, strconv.FormatUint(generation.RuntimeEpoch, 10))
		request.Header.Set(headers.VSRRoutingSnapshotRevision, strconv.FormatInt(generation.SnapshotRevision, 10))
		request.Header.Set(headers.VSRRoutingDigest, generation.RoutingDigest)
	}
}
