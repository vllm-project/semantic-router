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

// StartSignalSpan starts a new span for signal evaluation
// signalType: the type of signal (e.g., "keyword", "embedding", "domain")
// Returns the new context and span
func StartSignalSpan(ctx context.Context, signalType string) (context.Context, trace.Span) {
	// Use specific signal span name based on type
	var spanName string
	switch signalType {
	case "keyword":
		spanName = SpanSignalKeyword
	case "embedding":
		spanName = SpanSignalEmbedding
	case "domain":
		spanName = SpanSignalDomain
	case "fact_check":
		spanName = SpanSignalFactCheck
	case "user_feedback":
		spanName = SpanSignalUserFeedback
	case "preference":
		spanName = SpanSignalPreference
	case "language":
		spanName = SpanSignalLanguage
	case "latency":
		spanName = SpanSignalLatency
	default:
		spanName = SpanSignalEvaluation
	}

	spanCtx, span := StartSpan(ctx, spanName)
	SetSpanAttributes(span, attribute.String(AttrSignalType, signalType))
	return spanCtx, span
}

// EndSignalSpan ends a signal span with matched rules and confidence
func EndSignalSpan(span trace.Span, matchedRules []string, confidence float64, latencyMs int64) {
	if span == nil {
		return
	}

	if len(matchedRules) > 0 {
		SetSpanAttributes(span,
			attribute.StringSlice(AttrSignalMatchedRules, matchedRules),
			attribute.Float64(AttrSignalConfidence, confidence),
			attribute.Int64(AttrSignalLatencyMs, latencyMs))
	} else {
		SetSpanAttributes(span,
			attribute.Int64(AttrSignalLatencyMs, latencyMs))
	}

	span.End()
}

// StartDecisionSpan starts a new span for decision evaluation
// decisionName: the name of the decision being evaluated
// Returns the new context and span
func StartDecisionSpan(ctx context.Context, decisionName string) (context.Context, trace.Span) {
	spanCtx, span := StartSpan(ctx, SpanDecisionEvaluation)
	SetSpanAttributes(span, attribute.String(AttrDecisionName, decisionName))
	return spanCtx, span
}

// EndDecisionSpan ends a decision span with evaluation results
func EndDecisionSpan(span trace.Span, confidence float64, matchedRules []string, strategy string) {
	if span == nil {
		return
	}

	SetSpanAttributes(span,
		attribute.Float64(AttrDecisionConfidence, confidence),
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
	span.End()
}

// Span attribute keys following the signal -> decision -> plugin -> model hierarchy
const (
	// Request metadata
	AttrRequestID  = "request.id"
	AttrUserID     = "user.id"
	AttrSessionID  = "session.id"
	AttrHTTPMethod = "http.method"
	AttrHTTPPath   = "http.path"

	// Signal layer attributes
	AttrSignalType         = "signal.type"
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
	AttrPluginEnabled  = "plugin.enabled"
	AttrPluginResult   = "plugin.result"

	// Model layer attributes
	AttrModelName            = "model.name"
	AttrModelProvider        = "model.provider"
	AttrModelVersion         = "model.version"
	AttrModelEndpoint        = "model.endpoint"
	AttrReasoningEnabled     = "model.reasoning_enabled"
	AttrReasoningEffort      = "model.reasoning_effort"
	AttrTokenCountPrompt     = "model.token_count_prompt"
	AttrTokenCountCompletion = "model.token_count_completion"

	// Legacy attributes (for backward compatibility, to be deprecated)
	AttrCategoryName             = "category.name"
	AttrCategoryConfidence       = "category.confidence"
	AttrClassifierType           = "classifier.type"
	AttrRoutingStrategy          = "routing.strategy"
	AttrRoutingReason            = "routing.reason"
	AttrOriginalModel            = "routing.original_model"
	AttrSelectedModel            = "routing.selected_model"
	AttrEndpointName             = "endpoint.name"
	AttrEndpointAddress          = "endpoint.address"
	AttrPIIDetected              = "pii.detected"
	AttrPIITypes                 = "pii.types"
	AttrJailbreakDetected        = "jailbreak.detected"
	AttrJailbreakType            = "jailbreak.type"
	AttrSecurityAction           = "security.action"
	AttrCacheHit                 = "cache.hit"
	AttrCacheKey                 = "cache.key"
	AttrReasoningFamily          = "reasoning.family"
	AttrToolsSelected            = "tools.selected"
	AttrToolsCount               = "tools.count"
	AttrProcessingTimeMs         = "processing.time_ms"
	AttrClassificationTimeMs     = "classification.time_ms"
	AttrCacheLookupTimeMs        = "cache.lookup_time_ms"
	AttrPIIDetectionTimeMs       = "pii.detection_time_ms"
	AttrJailbreakDetectionTimeMs = "jailbreak.detection_time_ms"
)

// Span names following the hierarchy: signal -> decision -> plugin -> model
const (
	// Root span
	SpanRequestReceived = "semantic_router.request.received"

	// Signal evaluation layer (Layer 1)
	SpanSignalEvaluation   = "semantic_router.signal.evaluation"
	SpanSignalKeyword      = "semantic_router.signal.keyword"
	SpanSignalEmbedding    = "semantic_router.signal.embedding"
	SpanSignalDomain       = "semantic_router.signal.domain"
	SpanSignalFactCheck    = "semantic_router.signal.fact_check"
	SpanSignalUserFeedback = "semantic_router.signal.user_feedback"
	SpanSignalPreference   = "semantic_router.signal.preference"
	SpanSignalLanguage     = "semantic_router.signal.language"
	SpanSignalLatency      = "semantic_router.signal.latency"

	// Decision evaluation layer (Layer 2)
	SpanDecisionEvaluation = "semantic_router.decision.evaluation"

	// Plugin execution layer (Layer 3)
	SpanPluginExecution = "semantic_router.plugin.execution"

	// RAG (Retrieval-Augmented Generation) spans
	SpanRAGRetrieval        = "semantic_router.rag.retrieval"
	SpanRAGContextInjection = "semantic_router.rag.context_injection"

	// Model invocation layer (Layer 4)
	SpanUpstreamRequest       = "semantic_router.upstream.request"
	SpanResponseProcessing    = "semantic_router.response.processing"
	SpanToolSelection         = "semantic_router.tools.selection"
	SpanSystemPromptInjection = "semantic_router.system_prompt.injection"

	// Legacy spans (deprecated - kept for backward compatibility during migration)
	SpanClassification     = "semantic_router.classification" // Use SpanSignalEvaluation instead
	SpanPIIDetection       = "semantic_router.security.pii_detection"
	SpanJailbreakDetection = "semantic_router.security.jailbreak_detection"
	SpanCacheLookup        = "semantic_router.cache.lookup"
	SpanRoutingDecision    = "semantic_router.routing.decision"
	SpanBackendSelection   = "semantic_router.backend.selection"
)
