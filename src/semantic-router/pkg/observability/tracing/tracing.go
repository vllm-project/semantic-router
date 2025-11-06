package tracing

import (
	"context"
	"fmt"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.4.0"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc/credentials/insecure"
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

	// Create sampler based on configuration
	var sampler sdktrace.Sampler
	switch cfg.SamplingType {
	case "always_on":
		sampler = sdktrace.AlwaysSample()
	case "always_off":
		sampler = sdktrace.NeverSample()
	case "probabilistic":
		sampler = sdktrace.TraceIDRatioBased(cfg.SamplingRate)
	default:
		sampler = sdktrace.AlwaysSample()
	}

	// Create tracer provider
	tracerProvider = sdktrace.NewTracerProvider(
		sdktrace.WithResource(res),
		sdktrace.WithBatcher(exporter),
		sdktrace.WithSampler(sampler),
	)

	// Set global tracer provider
	otel.SetTracerProvider(tracerProvider)

	// Set global propagator for trace context propagation
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	// Create named tracer for the router
	tracer = tracerProvider.Tracer("semantic-router")

	return nil
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

// RecordError records an error on a span if it exists
func RecordError(span trace.Span, err error) {
	if span != nil && err != nil {
		span.RecordError(err)
	}
}

// Span attribute keys following OpenInference conventions for LLM observability
const (
	// Request metadata
	AttrRequestID  = "request.id"
	AttrUserID     = "user.id"
	AttrSessionID  = "session.id"
	AttrHTTPMethod = "http.method"
	AttrHTTPPath   = "http.path"

	// Model information
	AttrModelName     = "model.name"
	AttrModelProvider = "model.provider"
	AttrModelVersion  = "model.version"

	// Classification
	AttrCategoryName       = "category.name"
	AttrCategoryConfidence = "category.confidence"
	AttrClassifierType     = "classifier.type"

	// Routing
	AttrRoutingStrategy = "routing.strategy"
	AttrRoutingReason   = "routing.reason"
	AttrOriginalModel   = "routing.original_model"
	AttrSelectedModel   = "routing.selected_model"
	AttrEndpointName    = "endpoint.name"
	AttrEndpointAddress = "endpoint.address"

	// Security
	AttrPIIDetected       = "pii.detected"
	AttrPIITypes          = "pii.types"
	AttrJailbreakDetected = "jailbreak.detected"
	AttrJailbreakType     = "jailbreak.type"
	AttrSecurityAction    = "security.action"

	// Performance
	AttrTokenCountPrompt     = "token.count.prompt"
	AttrTokenCountCompletion = "token.count.completion"
	AttrCacheHit             = "cache.hit"
	AttrCacheKey             = "cache.key"

	// Reasoning
	AttrReasoningEnabled = "reasoning.enabled"
	AttrReasoningEffort  = "reasoning.effort"
	AttrReasoningFamily  = "reasoning.family"

	// Tools
	AttrToolsSelected = "tools.selected"
	AttrToolsCount    = "tools.count"

	// Processing times
	AttrProcessingTimeMs         = "processing.time_ms"
	AttrClassificationTimeMs     = "classification.time_ms"
	AttrCacheLookupTimeMs        = "cache.lookup_time_ms"
	AttrPIIDetectionTimeMs       = "pii.detection_time_ms"
	AttrJailbreakDetectionTimeMs = "jailbreak.detection_time_ms"
)

// Span names for different operations
const (
	SpanRequestReceived       = "semantic_router.request.received"
	SpanClassification        = "semantic_router.classification"
	SpanPIIDetection          = "semantic_router.security.pii_detection"
	SpanJailbreakDetection    = "semantic_router.security.jailbreak_detection"
	SpanCacheLookup           = "semantic_router.cache.lookup"
	SpanRoutingDecision       = "semantic_router.routing.decision"
	SpanBackendSelection      = "semantic_router.backend.selection"
	SpanUpstreamRequest       = "semantic_router.upstream.request"
	SpanResponseProcessing    = "semantic_router.response.processing"
	SpanToolSelection         = "semantic_router.tools.selection"
	SpanSystemPromptInjection = "semantic_router.system_prompt.injection"
)
