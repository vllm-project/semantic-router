package extproc

import (
	"context"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ir"
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
	detectClientProtocol(path, ctx)
	applyHeaderPassThroughPolicy(ctx)

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
	return newContinueRequestHeadersResponse(buildIdentityEncodingRequestMutation()), nil
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

func buildIdentityEncodingRequestMutation() *ext_proc.HeaderMutation {
	return &ext_proc.HeaderMutation{
		SetHeaders: []*core.HeaderValueOption{{
			Header: &core.HeaderValue{
				Key:   "accept-encoding",
				Value: "identity",
			},
		}},
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

// anthropicPassThroughHeader names a header captured from an Anthropic
// inbound request so the body-phase routing step can layer the value
// under the provider-profile pin. Stored on IRExtensions; downstream
// reads decide whether to forward. `anthropicPassThroughHeaders` is
// the canonical list: both capture (request-header phase) and forward
// (request-body routing phase via appendCapturedPassThroughHeaders)
// iterate it, so adding a new header only requires one entry here.
type anthropicPassThroughHeader struct {
	name   string
	assign func(ext *ir.IRExtensions, value string)
	read   func(ext *ir.IRExtensions) string
}

var anthropicPassThroughHeaders = []anthropicPassThroughHeader{
	{
		name:   "anthropic-version",
		assign: func(ext *ir.IRExtensions, v string) { ext.InboundAnthropicVersion = v },
		read:   func(ext *ir.IRExtensions) string { return ext.InboundAnthropicVersion },
	},
	{
		name:   "anthropic-beta",
		assign: func(ext *ir.IRExtensions, v string) { ext.InboundAnthropicBeta = v },
		read:   func(ext *ir.IRExtensions) string { return ext.InboundAnthropicBeta },
	},
	{
		name:   "anthropic-dangerous-direct-browser-access",
		assign: func(ext *ir.IRExtensions, v string) { ext.InboundDangerousDirectBrowserAccess = v },
		read:   func(ext *ir.IRExtensions) string { return ext.InboundDangerousDirectBrowserAccess },
	},
}

// applyHeaderPassThroughPolicy enforces the request-header pass-through
// contract: hop-by-hop framing headers are stripped from ctx.Headers
// (defense-in-depth — Envoy already filters most), and on Anthropic
// ingress the named pass-through headers are captured into
// IRExtensions so the body-phase routing step can layer them under any
// provider-profile pin.
//
// KEEP-by-default for everything else: ctx.Headers retains the inbound
// view of any header not in the drop list. The body-phase routing step
// decides what to forward to the upstream.
func applyHeaderPassThroughPolicy(ctx *RequestContext) {
	if ctx == nil || ctx.Headers == nil {
		return
	}

	for _, name := range hopByHopDropList {
		delete(ctx.Headers, name)
	}

	if ctx.ClientProtocol != config.ClientProtocolAnthropic {
		return
	}
	capturePassThroughHeaders(ctx)
}

// capturePassThroughHeaders records the inbound values of the named
// Anthropic pass-through headers into IRExtensions. The PR2 inbound
// parser may already have allocated IRExtensions when this runs (it
// runs at body-parse time, after the header phase); both call sites
// tolerate the other running first.
func capturePassThroughHeaders(ctx *RequestContext) {
	captured := make(map[string]string, len(anthropicPassThroughHeaders))
	for _, h := range anthropicPassThroughHeaders {
		if v := strings.TrimSpace(headerValueCI(ctx, h.name)); v != "" {
			captured[h.name] = v
		}
	}
	if len(captured) == 0 {
		return
	}

	if ctx.IRExtensions == nil {
		ctx.IRExtensions = &ir.IRExtensions{
			SourceProtocol: ctx.ClientProtocol,
		}
	}
	for _, h := range anthropicPassThroughHeaders {
		if v, ok := captured[h.name]; ok {
			h.assign(ctx.IRExtensions, v)
		}
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
