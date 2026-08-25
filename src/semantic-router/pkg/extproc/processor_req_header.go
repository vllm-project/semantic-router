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
	defer span.End()

	method, path := captureRequestHeaders(v, ctx)

	setRequestHeaderSpanAttributes(span, ctx, method, path)
	detectSourceFormat(path, ctx)
	applyHeaderPassThroughPolicy(ctx)
	if accessResponse := r.bindInferenceAuthentication(ctx); accessResponse != nil {
		return accessResponse, nil
	}
	if outcomeResponse := r.handleOutcomeFeedbackRequestHeaders(method, path, ctx); outcomeResponse != nil {
		return outcomeResponse, nil
	}

	// Router Replay contains captured request, response, and tool data. It is a
	// management API only: the public inference listener must fail closed.
	if isRouterReplayRequestTarget(path) {
		return r.createErrorResponse(404, "endpoint not found"), nil
	}

	detectStreamingExpectation(ctx)
	if modelsResp, err := r.handleModelsRequestHeaders(method, path, ctx); err != nil || modelsResp != nil {
		return modelsResp, err
	}
	if responseAPIResp, err := r.handleResponseAPIRequestHeaders(method, path, ctx); err != nil || responseAPIResp != nil {
		return responseAPIResp, err
	}
	if validationResp := r.validateRequestHeaders(method, path); validationResp != nil {
		return validationResp, nil
	}
	return newContinueRequestHeadersResponse(buildIdentityEncodingRequestMutation(r.nativeAccessEnabled())), nil
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
		tracing.SpanRequestReceived,
		trace.WithSpanKind(trace.SpanKindServer),
	)
	ctx.TraceContext = spanCtx
	return span
}

func captureRequestHeaders(
	v *ext_proc.ProcessingRequest_RequestHeaders,
	ctx *RequestContext,
) (string, string) {
	requestHeaders := v.RequestHeaders.Headers
	for _, header := range requestHeaders.Headers {
		headerValue := extractHeaderValue(header)
		ctx.Headers[header.Key] = headerValue

		lowerKey := strings.ToLower(header.Key)
		if lowerKey == headers.RequestID {
			ctx.RequestID = headerValue
		}
	}
	authenticateLooperRequestContext(ctx)
	scrubUntrustedIdentityHeaders(ctx)

	method := ctx.Headers[":method"]
	path := ctx.Headers[":path"]
	logging.ComponentDebugEvent("extproc", "request_headers_captured", map[string]interface{}{
		"request_id":     ctx.RequestID,
		"method":         method,
		"path":           path,
		"header_count":   len(requestHeaders.Headers),
		"looper_request": ctx.LooperRequest,
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

func buildIdentityEncodingRequestMutation(stripAuthorization bool) *ext_proc.HeaderMutation {
	return &ext_proc.HeaderMutation{
		SetHeaders: []*core.HeaderValueOption{{
			Header: &core.HeaderValue{
				Key:   "accept-encoding",
				Value: "identity",
			},
		}},
		RemoveHeaders: upstreamInternalHeadersForRemoval(stripAuthorization),
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
// contract. Client protocol headers are not provider connection settings;
// BackendInvoker obtains those only from the compiled Integration.
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
