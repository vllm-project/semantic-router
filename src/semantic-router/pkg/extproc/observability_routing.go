package extproc

import (
	"context"
	"errors"
	"io"
	"math"
	"net/url"
	"sort"
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	grpcCodes "google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

func observeRoutingIdentity(ctx *RequestContext, entrypoint string) {
	entrypoint = strings.TrimSpace(entrypoint)
	attrs := []attribute.KeyValue{
		attribute.String(tracing.AttrEntrypoint, entrypoint),
		attribute.String(tracing.AttrRecipe, string(ctx.Routing.RecipeName())),
	}
	ctx.TraceContext = tracing.WithRoutingAttributes(ctx.TraceContext, attrs...)
	tracing.SetSpanAttributes(ctx.RequestSpan, attrs...)
	if ctx.learningPreview == nil && ctx.RequestSpan != nil {
		metrics.RecordEntrypointResolution(entrypoint, string(ctx.Routing.RecipeName()))
	}
}

func observeDecisionIdentity(ctx *RequestContext, span trace.Span, decision *config.Decision) {
	attrs := []attribute.KeyValue{attribute.String(tracing.AttrDecisionName, decision.Name)}
	if decision.Algorithm != nil {
		attrs = append(attrs, attribute.String(tracing.AttrAlgorithm, decision.Algorithm.Type))
	}
	ctx.TraceContext = tracing.WithRoutingAttributes(ctx.TraceContext, attrs...)
	tracing.SetSpanAttributes(span, attrs...)
	tracing.SetSpanAttributes(ctx.RequestSpan, attrs...)
}

func observeRoutingStage(ctx *RequestContext, stage string, started time.Time) {
	if ctx.Routing.SelectedRecipe() != nil && ctx.learningPreview == nil && ctx.RequestSpan != nil {
		metrics.ObserveRoutingStage(string(ctx.Routing.RecipeName()), stage, time.Since(started).Seconds())
	}
}

// Signal evidence contains only actual finite numeric results. Presence flags
// distinguish missing evidence from a legitimate zero; configured keys and the
// number of events are bounded independently of user input.
func observeSignalEvidence(ctx *RequestContext, span trace.Span) {
	keys := make(map[string]struct{}, len(ctx.VSRSignalValues)+len(ctx.VSRSignalConfidences))
	for key := range ctx.VSRSignalValues {
		keys[key] = struct{}{}
	}
	for key := range ctx.VSRSignalConfidences {
		keys[key] = struct{}{}
	}
	names := make([]string, 0, len(keys))
	for key := range keys {
		names = append(names, key)
	}
	sort.Strings(names)
	const maxEvents = 128
	recorded := 0
	omitted := 0
	for _, key := range names {
		value, hasValue := ctx.VSRSignalValues[key]
		confidence, hasConfidence := ctx.VSRSignalConfidences[key]
		hasValue = hasValue && !math.IsNaN(value) && !math.IsInf(value, 0)
		hasConfidence = hasConfidence && !math.IsNaN(confidence) && !math.IsInf(confidence, 0)
		if !hasValue && !hasConfidence {
			continue
		}
		if recorded == maxEvents {
			omitted++
			continue
		}
		name := []rune(key)
		if len(name) > 256 {
			name = name[:256]
		}
		attrs := []attribute.KeyValue{
			attribute.String("signal.key", string(name)),
			attribute.Bool("signal.value_present", hasValue),
			attribute.Bool("signal.confidence_present", hasConfidence),
		}
		if hasValue {
			attrs = append(attrs, attribute.Float64("signal.value", value))
		}
		if hasConfidence {
			attrs = append(attrs, attribute.Float64("signal.confidence", confidence))
		}
		span.AddEvent("routing.signal.evidence", trace.WithAttributes(attrs...))
		recorded++
	}
	tracing.SetSpanAttributes(span, attribute.Int("signal.evidence_omitted", omitted))
}

// Projection values come from the evaluated recipe. Do not substitute a
// synthetic confidence, nor include input text, request identifiers or headers.
func observeProjectionResults(ctx *RequestContext, span trace.Span) {
	tracing.SetSpanAttributes(span, attribute.StringSlice("projection.matched_outputs", ctx.VSRMatchedProjection))
	names := make([]string, 0, len(ctx.VSRProjectionScores))
	for name := range ctx.VSRProjectionScores {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		value := ctx.VSRProjectionScores[name]
		if math.IsNaN(value) || math.IsInf(value, 0) {
			continue
		}
		span.AddEvent("routing.projection.score", trace.WithAttributes(
			attribute.String("projection.name", name), attribute.Float64("projection.value", value)))
		if ctx.Routing.SelectedRecipe() != nil && ctx.learningPreview == nil && ctx.RequestSpan != nil {
			metrics.ObserveProjectionScore(string(ctx.Routing.RecipeName()), name, value)
		}
	}
}

func observeAlgorithmSelection(ctx *RequestContext, span trace.Span, model string, err error) {
	attrs := []attribute.KeyValue{
		attribute.String(tracing.AttrSelectedModel, model),
		attribute.String("routing.selection_method", ctx.VSRSelectionMethod),
	}
	tracing.SetSpanAttributes(span, attrs...)
	tracing.SetSpanAttributes(ctx.RequestSpan, attribute.String("routing.selection_candidate", model))
	if err != nil {
		span.SetStatus(codes.Error, "model_selection_failed")
		return
	}
	if model != "" && ctx.Routing.SelectedRecipe() != nil && ctx.learningPreview == nil && ctx.RequestSpan != nil {
		metrics.RecordRecipeSelection(string(ctx.Routing.RecipeName()), ctx.VSRSelectedDecision.Name, ctx.VSRSelectionMethod, model)
	}
}

// Process owns these spans. A response header is not the end of a streamed
// provider request, and cancellation/error paths may never receive a body.
func finishRequestTrace(ctx *RequestContext, err error) {
	if ctx == nil {
		return
	}
	if err == nil {
		err = ctx.TraceReceiveError
	}
	statusCode := ctx.UpstreamStatusCode
	if ctx.TraceStatusCode != 0 {
		statusCode = ctx.TraceStatusCode
	}
	if ctx.RequestSpan != nil {
		seconds := 0.0
		if !ctx.StartTime.IsZero() {
			seconds = time.Since(ctx.StartTime).Seconds()
		}
		outcome := requestTraceOutcome(ctx, err, statusCode)
		metrics.RecordRequestOutcome(ctx.TraceTrafficKind, outcome, seconds)
		tracing.SetSpanAttributes(ctx.RequestSpan, attribute.String("request.outcome", outcome))
	}
	// A looper may replace the initial selection candidate with a synthesis or
	// fallback model. Only a successful terminal response asserts its final model.
	if err == nil && !ctx.StreamingAborted && statusCode >= 200 && statusCode < 300 && ctx.VSRSelectedModel != "" {
		tracing.SetSpanAttributes(ctx.RequestSpan, attribute.String(tracing.AttrSelectedModel, ctx.VSRSelectedModel))
	}
	for _, boundary := range []struct {
		span trace.Span
		code int
	}{{ctx.UpstreamSpan, ctx.UpstreamStatusCode}, {ctx.RequestSpan, statusCode}} {
		span := boundary.span
		if span == nil {
			continue
		}
		tracing.SetSpanAttributes(span,
			attribute.Int("http.status_code", boundary.code),
			attribute.Bool("response.streaming", ctx.IsStreamingResponse),
			attribute.Bool("response.streaming_complete", ctx.StreamingComplete))
		if err != nil {
			reason := "request_failed"
			if errors.Is(err, context.Canceled) || status.Code(err) == grpcCodes.Canceled {
				reason = "request_canceled"
			} else if errors.Is(err, context.DeadlineExceeded) || status.Code(err) == grpcCodes.DeadlineExceeded {
				reason = "request_deadline_exceeded"
			} else if errors.Is(err, io.ErrUnexpectedEOF) {
				reason = "stream_ended_before_terminal_response"
			}
			span.SetStatus(codes.Error, reason)
		} else if ctx.StreamingAborted {
			span.SetStatus(codes.Error, "stream_ended_before_completion")
		} else if boundary.code >= 400 {
			span.SetStatus(codes.Error, "request_failed")
		}
		span.End()
	}
	ctx.UpstreamSpan = nil
	ctx.RequestSpan = nil
}

func finishImmediateResponseTrace(ctx *RequestContext, response *ext_proc.ProcessingResponse) {
	if response == nil || response.GetImmediateResponse() == nil {
		return
	}
	ctx.TraceStatusCode = int(response.GetImmediateResponse().GetStatus().GetCode())
	finishRequestTrace(ctx, nil)
}

func requestTraceOutcome(ctx *RequestContext, err error, code int) string {
	switch {
	case errors.Is(err, context.Canceled) || status.Code(err) == grpcCodes.Canceled:
		return "canceled"
	case errors.Is(err, context.DeadlineExceeded) || status.Code(err) == grpcCodes.DeadlineExceeded:
		return "timeout"
	case ctx.StreamingAborted || errors.Is(err, io.ErrUnexpectedEOF):
		return "incomplete"
	case err != nil:
		return "error"
	case code >= 500:
		return "server_error"
	case code >= 400:
		return "client_error"
	case code >= 200 && code < 400:
		return "success"
	default:
		return "incomplete"
	}
}

// Routes are bounded templates, never raw paths containing queries, credentials
// or response/model identifiers. Catalog polling remains distinguishable from inference.
func requestTraceRoute(raw string) (route, kind string) {
	parsed, err := url.ParseRequestURI(raw)
	if err != nil {
		return "/{unmatched}", "other"
	}
	switch path := parsed.Path; path {
	case "/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages":
		return path, "inference"
	case "/v1/models":
		return path, "catalog"
	case "/health", "/healthz", "/ready", "/readyz", "/metrics":
		return path, "health"
	default:
		if strings.HasPrefix(path, "/v1/responses/") {
			if strings.HasSuffix(path, "/input_items") {
				return "/v1/responses/{response_id}/input_items", "response_object"
			}
			return "/v1/responses/{response_id}", "response_object"
		}
		if strings.HasPrefix(path, "/v1/models/") {
			return "/v1/models/{model}", "catalog"
		}
		return "/{unmatched}", "other"
	}
}
