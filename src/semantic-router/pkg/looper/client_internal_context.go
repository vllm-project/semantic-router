package looper

import (
	"context"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

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
	traceHeaders := make(map[string]string)
	tracing.InjectTraceContext(ctx, traceHeaders)
	for name, value := range traceHeaders {
		header.Set(name, value)
	}
	if target.AccessKey != "" {
		header.Set("Authorization", "Bearer "+target.AccessKey)
	}
	return header
}
