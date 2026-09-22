package tracing

import (
	"context"
	"fmt"
	"strings"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.4.0"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// TracingConfig holds the tracing configuration
type TracingConfig struct {
	Enabled               bool
	Provider              string
	ExporterType          string
	ExporterEndpoint      string
	ExporterInsecure      bool
	SamplingType          string
	SamplingRate          float64
	ServiceName           string
	ServiceVersion        string
	DeploymentEnvironment string
}

var (
	tracerProvider *sdktrace.TracerProvider
	tracer         trace.Tracer
)

// InitTracing initializes the OpenTelemetry tracing provider
func InitTracing(ctx context.Context, cfg TracingConfig) error {
	if !cfg.Enabled {
		return nil
	}

	// Create resource with service information
	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceNameKey.String(cfg.ServiceName),
			semconv.ServiceVersionKey.String(cfg.ServiceVersion),
			semconv.DeploymentEnvironmentKey.String(cfg.DeploymentEnvironment),
		),
	)
	if err != nil {
		return fmt.Errorf("failed to create resource: %w", err)
	}

	// Create exporter based on configuration
	var exporter sdktrace.SpanExporter
	switch cfg.ExporterType {
	case "otlp":
		exporter, err = createOTLPExporter(ctx, cfg)
		if err != nil {
			return fmt.Errorf("failed to create OTLP exporter: %w", err)
		}
	case "stdout":
		exporter, err = stdouttrace.New(
			stdouttrace.WithPrettyPrint(),
		)
		if err != nil {
			return fmt.Errorf("failed to create stdout exporter: %w", err)
		}
	default:
		return fmt.Errorf("unsupported exporter type: %s", cfg.ExporterType)
	}

	sampler := samplerFromConfig(cfg)

	// Create tracer provider
	tracerProvider = sdktrace.NewTracerProvider(
		sdktrace.WithResource(res),
		sdktrace.WithBatcher(observedExporter{exporter}),
		sdktrace.WithSampler(sampler),
	)

	// Set global tracer provider
	otel.SetTracerProvider(tracerProvider)

	// Set global propagator for trace context propagation
	otel.SetTextMapPropagator(DefaultPropagator())

	// Create named tracer for the router
	tracer = tracerProvider.Tracer("semantic-router")

	return nil
}

// DefaultPropagator is the text-map propagator InitTracing installs
// globally: W3C trace context plus baggage. Baggage members a client sends
// are therefore extracted into the request's trace context, which is why
// outbound calls to less trusted targets must inject with
// InjectSpanContextToSlice rather than the global propagator.
func DefaultPropagator() propagation.TextMapPropagator {
	return propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	)
}

func samplerFromConfig(cfg TracingConfig) sdktrace.Sampler {
	switch samplingType := strings.ToLower(strings.TrimSpace(cfg.SamplingType)); samplingType {
	case "always_on":
		return sdktrace.AlwaysSample()
	case "always_off":
		return sdktrace.NeverSample()
	case "probabilistic", "traceidratio", "trace_id_ratio":
		return sdktrace.TraceIDRatioBased(cfg.SamplingRate)
	case "":
		return sdktrace.AlwaysSample()
	default:
		logging.ComponentWarnEvent("tracing", "tracing_sampling_type_unknown", map[string]interface{}{
			"sampling_type": cfg.SamplingType,
			"fallback":      "always_on",
		})
		return sdktrace.AlwaysSample()
	}
}

// createOTLPExporter creates an OTLP gRPC exporter
func createOTLPExporter(ctx context.Context, cfg TracingConfig) (sdktrace.SpanExporter, error) {
	opts := []otlptracegrpc.Option{
		otlptracegrpc.WithEndpoint(cfg.ExporterEndpoint),
	}

	if cfg.ExporterInsecure {
		opts = append(opts, otlptracegrpc.WithTLSCredentials(insecure.NewCredentials()))
	}

	// Create exporter with timeout context for initialization
	// Note: We don't use WithBlock() to allow the exporter to connect asynchronously
	// This prevents blocking on startup if the collector is temporarily unavailable
	ctxWithTimeout, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	return otlptracegrpc.New(ctxWithTimeout, opts...)
}

// ShutdownTracing gracefully shuts down the tracing provider
func ShutdownTracing(ctx context.Context) error {
	if tracerProvider != nil {
		return tracerProvider.Shutdown(ctx)
	}
	return nil
}

// GetTracer returns the global tracer instance
func GetTracer() trace.Tracer {
	if tracer == nil {
		// Return noop tracer if tracing is not initialized
		return otel.Tracer("semantic-router")
	}
	return tracer
}

// StartSpan starts a new span with the given name and options
func StartSpan(ctx context.Context, spanName string, opts ...trace.SpanStartOption) (context.Context, trace.Span) {
	// Handle nil context by using background context
	if ctx == nil {
		ctx = context.Background()
	}

	if attrs, ok := ctx.Value(routingAttributesKey{}).([]attribute.KeyValue); ok {
		opts = append([]trace.SpanStartOption{trace.WithAttributes(attrs...)}, opts...)
	}
	if tracer == nil {
		// Return noop tracer if tracing is not initialized
		return otel.Tracer("semantic-router").Start(ctx, spanName, opts...)
	}
	return tracer.Start(ctx, spanName, opts...)
}

// SetSpanAttributes sets attributes on a span if it exists
func SetSpanAttributes(span trace.Span, attrs ...attribute.KeyValue) {
	if span != nil {
		span.SetAttributes(attrs...)
	}
}

// RecordError records a bounded reason, never provider error text or user content.
func RecordError(span trace.Span, reason string) {
	if span != nil {
		span.AddEvent("error", trace.WithAttributes(attribute.String("error.reason", reason)))
		span.SetStatus(codes.Error, reason)
	}
}

// EndSignalSpan ends a signal span with matched rules and confidence
func EndSignalSpan(span trace.Span, matchedRules []string, confidence float64, latencyMs int64, scored bool) {
	if span == nil {
		return
	}

	if len(matchedRules) > 0 {
		SetSpanAttributes(span,
			attribute.StringSlice(AttrSignalMatchedRules, matchedRules),
			attribute.Int64(AttrSignalLatencyMs, latencyMs))
	} else {
		SetSpanAttributes(span,
			attribute.Int64(AttrSignalLatencyMs, latencyMs))
	}

	available := scored
	SetSpanAttributes(span, attribute.Bool(AttrSignalConfidence+"_available", available))
	if available && len(matchedRules) > 0 {
		SetSpanAttributes(span, attribute.Float64(AttrSignalConfidence, confidence))
	}
	span.End()
}

// EndDecisionSpan ends a decision span with evaluation results
func EndDecisionSpan(span trace.Span, confidence float64, matchedRules []string, strategy string, scored bool) {
	if span == nil {
		return
	}

	available := scored
	if available {
		SetSpanAttributes(span, attribute.Float64(AttrDecisionConfidence, confidence))
	}
	SetSpanAttributes(span,
		attribute.Bool(AttrDecisionConfidence+"_available", available),
		attribute.StringSlice(AttrDecisionMatchedRules, matchedRules),
		attribute.String(AttrDecisionStrategy, strategy))

	span.End()
}

// StartPluginSpan starts a new span for plugin execution with standard attributes
// pluginType: the type of plugin (e.g., "pii", "jailbreak", "system_prompt", "semantic-cache")
// decisionName: the decision name this plugin is associated with
// Returns the new context and span
func StartPluginSpan(ctx context.Context, pluginType string, decisionName string) (context.Context, trace.Span) {
	spanCtx, span := StartSpan(ctx, SpanPluginExecution)

	// Set standard plugin attributes
	SetSpanAttributes(span,
		attribute.String(AttrPluginType, pluginType),
		attribute.String(AttrPluginDecision, decisionName))

	return spanCtx, span
}

// EndPluginSpan ends a plugin span with status and latency
// status: "success", "error", "blocked", "skipped", etc.
// latencyMs: execution time in milliseconds
// result: optional result string (e.g., "pii_detected", "jailbreak_blocked", "cache_hit")
func EndPluginSpan(span trace.Span, status string, latencyMs int64, result string) {
	if span == nil {
		return
	}

	attrs := []attribute.KeyValue{
		attribute.String(AttrPluginStatus, status),
		attribute.Int64(AttrPluginLatency, latencyMs),
	}

	if result != "" {
		attrs = append(attrs, attribute.String(AttrPluginResult, result))
	}

	SetSpanAttributes(span, attrs...)
	if status == "error" {
		span.SetStatus(codes.Error, "plugin_failed")
	}
	span.End()
}

// Attribute keys emitted by the current entrypoint/recipe routing pipeline.
const (
	// Request metadata
	AttrRequestID  = "request.id"
	AttrHTTPMethod = "http.method"
	AttrHTTPPath   = "http.path"

	// Signal layer attributes
	AttrSignalMatchedRules = "signal.matched_rules"
	AttrSignalConfidence   = "signal.confidence"
	AttrSignalLatencyMs    = "signal.latency_ms"

	// Decision layer attributes
	AttrDecisionName         = "decision.name"
	AttrDecisionConfidence   = "decision.confidence"
	AttrDecisionMatchedRules = "decision.matched_rules"
	AttrDecisionStrategy     = "decision.strategy"

	// Plugin layer attributes
	AttrPluginType     = "plugin.type"
	AttrPluginDecision = "plugin.decision"
	AttrPluginStatus   = "plugin.status"
	AttrPluginLatency  = "plugin.latency_ms"
	AttrPluginResult   = "plugin.result"

	// Model layer attributes
	AttrModelName        = "model.name"
	AttrReasoningEnabled = "model.reasoning_enabled"
	AttrReasoningEffort  = "model.reasoning_effort"

	// Routing and plugin outcome attributes
	AttrRoutingReason           = "routing.reason"
	AttrOriginalModel           = "routing.original_model"
	AttrSelectedModel           = "routing.selected_model"
	AttrEndpointAddress         = "endpoint.address"
	AttrCacheHit                = "cache.hit"
	AttrCacheLookupTimeMs       = "cache.lookup_time_ms"
	AttrCacheWriteSkippedReason = "cache.write_skipped_reason"
)

// Current runtime phase spans; plugin and looper spans carry inherited recipe identity.
const (
	// Root span
	SpanRequest = "semantic_router.request"

	// Signal evaluation layer (Layer 1)
	SpanSignalEvaluation = "semantic_router.signal.evaluation"

	// Decision evaluation layer (Layer 2)
	SpanDecisionEvaluation = "semantic_router.decision.evaluation"

	// Plugin execution layer (Layer 3)
	SpanPluginExecution = "semantic_router.plugin.execution"

	// RAG (Retrieval-Augmented Generation) spans
	SpanRAGRetrieval = "semantic_router.rag.retrieval"

	// Model invocation layer (Layer 4)
	SpanUpstreamRequest = "semantic_router.upstream.request"
)
