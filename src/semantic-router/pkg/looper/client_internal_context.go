package looper

import (
	"context"
	"fmt"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/internalauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

// setInternalRequestHeaders attaches authenticated routing context for the
// in-process extproc hop. These values are consumed and removed before the
// physical model backend is invoked.
func (c *Client) setInternalRequestHeaders(
	header http.Header,
	ctx context.Context,
	options CallOptions,
) {
	header.Set(headers.VSRInternalAuth, internalauth.Token())
	header.Set(headers.VSRLooperRequest, "true")
	header.Set(headers.VSRLooperIteration, fmt.Sprintf("%d", options.Iteration))
	depth := options.FusionDepth
	if depth <= 0 {
		depth = fusionDepthFromContext(ctx)
	}
	if depth > 0 {
		header.Set(headers.VSRFusionDepth, fmt.Sprintf("%d", depth))
	}
	if recipe := routingRecipeFromContext(ctx); recipe != "" {
		header.Set(headers.VSRSelectedRecipe, string(recipe))
	}
	if options.DecisionName != "" {
		header.Set(headers.VSRLooperDecision, options.DecisionName)
	}
	setOptionalInt64Header(header, headers.VSRLooperClientMaxOutputTokens, options.ClientMaxOutputTokens)
	setOptionalInt64Header(header, headers.VSRLooperStageMaxOutputTokens, options.StageMaxOutputTokens)
}

func setOptionalInt64Header(header http.Header, name string, value *int64) {
	if value == nil || *value < 1 {
		return
	}
	header.Set(name, fmt.Sprintf("%d", *value))
}

func (c *Client) requestHeaders(
	ctx context.Context,
	options CallOptions,
	accessKey string,
) http.Header {
	header := make(http.Header, len(c.headers)+7)
	header.Set("Content-Type", "application/json")
	for name, value := range c.headers {
		header.Set(name, value)
	}
	traceHeaders := make(map[string]string)
	tracing.InjectTraceContext(ctx, traceHeaders)
	for name, value := range traceHeaders {
		header.Set(name, value)
	}
	if accessKey != "" {
		header.Set("Authorization", "Bearer "+accessKey)
	}
	c.setInternalRequestHeaders(header, ctx, options)
	return header
}
