package extproc

import (
	"context"
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

// handleRequestHeaders processes the request headers.
func (r *OpenAIRouter) handleRequestHeaders(v *ext_proc.ProcessingRequest_RequestHeaders, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	ctx.StartTime = time.Now()

	span := startRequestHeaderSpan(v, ctx)
	defer span.End()

	method, path := captureRequestHeaders(v, ctx, r.skipProcessingEnabled())

	setRequestHeaderSpanAttributes(span, ctx, method, path)

	// Honor x-vsr-skip-processing as early as possible: once captured we bypass
	// every router-side header check (replay API, validation, response-API
	// translation) and emit a plain CONTINUE so the request flows through.
	// Streaming detection still runs because the same flag drives mode selection
	// for downstream filters and is cheap; the body and response handlers will
	// also short-circuit in the no-op path.
	if ctx.SkipProcessing {
		detectStreamingExpectation(ctx)
		return newContinueRequestHeadersResponse(), nil
	}

	if replayResp := r.handleRouterReplayAPI(method, path); replayResp != nil {
		return replayResp, nil
	}

	detectStreamingExpectation(ctx)
	if modelsResp, err := r.handleModelsRequestHeaders(method, path); err != nil || modelsResp != nil {
		return modelsResp, err
	}
	if responseAPIResp, err := r.handleResponseAPIRequestHeaders(method, path, ctx); err != nil || responseAPIResp != nil {
		return responseAPIResp, err
	}
	if validationResp := r.validateRequestHeaders(method, path); validationResp != nil {
		return validationResp, nil
	}
	return newContinueRequestHeadersResponse(), nil
}

func startRequestHeaderSpan(
	v *ext_proc.ProcessingRequest_RequestHeaders,
	ctx *RequestContext,
) trace.Span {
	baseCtx := context.Background()
	headerMap := make(map[string]string, len(v.RequestHeaders.Headers.Headers))
	for _, header := range v.RequestHeaders.Headers.Headers {
		headerMap[header.Key] = extractHeaderValue(header)
	}

	ctx.TraceContext = tracing.ExtractTraceContext(baseCtx, headerMap)
	spanCtx, span := tracing.StartSpan(
		ctx.TraceContext,
		tracing.SpanRequestReceived,
		trace.WithSpanKind(trace.SpanKindServer),
	)
	ctx.TraceContext = spanCtx
	return span
}

func captureRequestHeaders(
	v *ext_proc.ProcessingRequest_RequestHeaders,
	ctx *RequestContext,
	skipProcessingGateEnabled bool,
) (string, string) {
	requestHeaders := v.RequestHeaders.Headers
	for _, header := range requestHeaders.Headers {
		headerValue := extractHeaderValue(header)
		ctx.Headers[header.Key] = headerValue

		// HTTP/2 lowercases header names, but we accept either case for both the
		// internal looper marker and the external skip-processing opt-out so
		// upstream filters do not have to worry about casing.
		lowerKey := strings.ToLower(header.Key)
		if lowerKey == headers.RequestID {
			ctx.RequestID = headerValue
		}
		if lowerKey == headers.VSRLooperRequest && headerValue == "true" {
			ctx.LooperRequest = true
		}
		// The x-vsr-skip-processing opt-out is gated by the deployment-level
		// global.router.skip_processing.enabled flag. When disabled (the
		// default), the header is ignored entirely so an unauthenticated
		// upstream caller cannot bypass router policy by injecting it.
		if skipProcessingGateEnabled &&
			lowerKey == headers.VSRSkipProcessing &&
			strings.EqualFold(strings.TrimSpace(headerValue), "true") {
			ctx.SkipProcessing = true
		}
	}

	method := ctx.Headers[":method"]
	path := ctx.Headers[":path"]
	logging.ComponentDebugEvent("extproc", "request_headers_captured", map[string]interface{}{
		"request_id":      ctx.RequestID,
		"method":          method,
		"path":            path,
		"header_count":    len(requestHeaders.Headers),
		"looper_request":  ctx.LooperRequest,
		"skip_processing": ctx.SkipProcessing,
	})

	return method, path
}

func setRequestHeaderSpanAttributes(
	span trace.Span,
	ctx *RequestContext,
	method string,
	path string,
) {
	if ctx.RequestID != "" {
		tracing.SetSpanAttributes(
			span,
			attribute.String(tracing.AttrRequestID, ctx.RequestID),
		)
	}

	tracing.SetSpanAttributes(
		span,
		attribute.String(tracing.AttrHTTPMethod, method),
		attribute.String(tracing.AttrHTTPPath, path),
	)
}

func detectStreamingExpectation(ctx *RequestContext) {
	accept, ok := ctx.Headers["accept"]
	if !ok {
		return
	}

	if strings.Contains(strings.ToLower(accept), "text/event-stream") {
		ctx.ExpectStreamingResponse = true
		logging.ComponentDebugEvent("extproc", "streaming_expectation_detected", map[string]interface{}{
			"request_id": ctx.RequestID,
			"source":     "accept_header",
		})
	}
}

func extractHeaderValue(header interface {
	GetValue() string
	GetRawValue() []byte
},
) string {
	headerValue := header.GetValue()
	if headerValue == "" && len(header.GetRawValue()) > 0 {
		return string(header.GetRawValue())
	}
	return headerValue
}

func newContinueRequestHeadersResponse() *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestHeaders{
			RequestHeaders: &ext_proc.HeadersResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
				},
			},
		},
	}
}
