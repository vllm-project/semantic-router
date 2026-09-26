package extproc

import (
	"context"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
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

	method, path := captureRequestHeaders(v, ctx, r.skipProcessingEnabled())
	setRequestHeaderSpanAttributes(span, ctx, method, path)
	if rejected := r.benchmarkConfigPrecondition(ctx); rejected != nil {
		return rejected, nil
	}

	detectSourceFormat(path, ctx)
	applyHeaderPassThroughPolicy(ctx)

	// Router Replay contains captured request, response, and tool data. It is a
	// management API only: the public inference listener must fail closed even
	// when a caller supplies the otherwise valid skip-processing opt-out.
	if isRouterReplayRequestTarget(path) {
		return r.createErrorResponse(404, "endpoint not found"), nil
	}

	// Honor x-vsr-skip-processing as early as possible: once captured we bypass
	// every router-side header check (replay API, validation, response-API
	// translation) and emit a plain CONTINUE so the request flows through.
	// Streaming detection still runs because the same flag drives mode selection
	// for downstream filters and is cheap; the body and response handlers will
	// also short-circuit in the no-op path.
	if ctx.SkipProcessing {
		detectStreamingExpectation(ctx)
		mutation := buildLooperInternalHeaderRemovalMutation()
		mutation.RemoveHeaders = append(mutation.RemoveHeaders, headers.SelectedModel)
		if isAzureOpenAIPath(path) {
			mutation.RemoveHeaders = append(mutation.RemoveHeaders, azureAPIKeyHeader)
		}
		response := newContinueRequestHeadersResponse(mutation)
		if headerValueCI(ctx, headers.SelectedModel) != "" {
			// A caller-supplied selected-model header may have selected a provider
			// route before ext_proc ran. Skip-processing does not materialize a
			// provider path, so remove the untrusted selector and re-evaluate onto
			// the default route, which owns the ingress path-prefix rewrite.
			response.GetRequestHeaders().GetResponse().ClearRouteCache = true
		}
		return response, nil
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
	mutation := buildIdentityEncodingRequestMutation()
	if isAzureOpenAIPath(path) {
		// The Azure client key authenticates to the Router, never to a provider.
		mutation.RemoveHeaders = append(mutation.RemoveHeaders, azureAPIKeyHeader)
	}
	return newContinueRequestHeadersResponse(mutation), nil
}

func startRequestHeaderSpan(
	v *ext_proc.ProcessingRequest_RequestHeaders,
	ctx *RequestContext,
) trace.Span {
	baseCtx := ctx.TraceContext
	if baseCtx == nil {
		baseCtx = context.Background()
	}
	headerMap := make(map[string]string, len(v.RequestHeaders.Headers.Headers))
	for _, header := range v.RequestHeaders.Headers.Headers {
		headerMap[header.Key] = extractHeaderValue(header)
	}

	ctx.TraceContext = tracing.ExtractTraceContext(baseCtx, headerMap)
	spanCtx, span := tracing.StartSpan(
		ctx.TraceContext,
		tracing.SpanRequest,
		trace.WithSpanKind(trace.SpanKindServer),
	)
	ctx.TraceContext = spanCtx
	ctx.RequestSpan = span
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

		// HTTP/2 lowercases header names, but we accept either case for the
		// external skip-processing opt-out so upstream filters do not have to
		// worry about casing. Internal request authentication runs after all
		// headers have been captured so header order cannot affect validation.
		lowerKey := strings.ToLower(header.Key)
		if lowerKey == headers.RequestID {
			ctx.RequestID = headerValue
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
	authenticateLooperRequestContext(ctx)

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
	route, kind := requestTraceRoute(path)
	if kind == "inference" && ctx.LooperRequest {
		kind = "inference_internal"
	}
	ctx.TraceTrafficKind = kind
	switch method {
	case "GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS":
	default:
		method = "OTHER"
	}
	if ctx.RequestID != "" {
		tracing.SetSpanAttributes(
			span,
			attribute.String(tracing.AttrRequestID, ctx.RequestID),
		)
	}

	tracing.SetSpanAttributes(
		span,
		attribute.String(tracing.AttrHTTPMethod, method),
		attribute.String(tracing.AttrHTTPPath, route),
		attribute.String("http.route", route),
		attribute.String("traffic.kind", kind),
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

func buildIdentityEncodingRequestMutation() *ext_proc.HeaderMutation {
	return &ext_proc.HeaderMutation{
		SetHeaders: []*core.HeaderValueOption{{
			Header: &core.HeaderValue{
				Key:   "accept-encoding",
				Value: "identity",
			},
		}},
		RemoveHeaders: looperInternalHeadersForRemoval(),
	}
}

// hopByHopDropList is the set of HTTP framing headers we strip from
// ctx.Headers before any downstream filter or body-phase routing sees
// them. Envoy already strips most of these from the request before
// extproc receives it; we re-apply the policy as defense-in-depth and
// to make the contract explicit in code.
var hopByHopDropList = []string{
	"host",
	"content-length",
	"connection",
	"keep-alive",
	"proxy-connection",
	"transfer-encoding",
	"upgrade",
	"te",
	"trailer",
	"expect",
}

// applyHeaderPassThroughPolicy enforces the request-header pass-through
// contract by stripping transport framing from the semantic request view.
// Provider headers are supplied by the selected provider profile rather than
// copied from an untrusted client request.
func applyHeaderPassThroughPolicy(ctx *RequestContext) {
	if ctx == nil || ctx.Headers == nil {
		return
	}

	for _, name := range hopByHopDropList {
		delete(ctx.Headers, name)
	}
}

func newContinueRequestHeadersResponse(headerMutation ...*ext_proc.HeaderMutation) *ext_proc.ProcessingResponse {
	var mutation *ext_proc.HeaderMutation
	if len(headerMutation) > 0 {
		mutation = headerMutation[0]
	}
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestHeaders{
			RequestHeaders: &ext_proc.HeadersResponse{
				Response: &ext_proc.CommonResponse{
					Status:         ext_proc.CommonResponse_CONTINUE,
					HeaderMutation: mutation,
				},
			},
		},
	}
}
