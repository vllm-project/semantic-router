package observability

import (
	"context"
	"testing"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
)

func TestTracingConfiguration(t *testing.T) {
	tests := []struct {
		name    string
		cfg     TracingConfig
		wantErr bool
	}{
		{
			name: "disabled tracing",
			cfg: TracingConfig{
				Enabled: false,
			},
			wantErr: false,
		},
		{
			name: "stdout exporter",
			cfg: TracingConfig{
				Enabled:               true,
				Provider:              "opentelemetry",
				ExporterType:          "stdout",
				SamplingType:          "always_on",
				ServiceName:           "test-service",
				ServiceVersion:        "v1.0.0",
				DeploymentEnvironment: "test",
			},
			wantErr: false,
		},
		{
			name: "probabilistic sampling",
			cfg: TracingConfig{
				Enabled:               true,
				Provider:              "opentelemetry",
				ExporterType:          "stdout",
				SamplingType:          "probabilistic",
				SamplingRate:          0.5,
				ServiceName:           "test-service",
				ServiceVersion:        "v1.0.0",
				DeploymentEnvironment: "test",
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			err := InitTracing(ctx, tt.cfg)
			if (err != nil) != tt.wantErr {
				t.Errorf("InitTracing() error = %v, wantErr %v", err, tt.wantErr)
			}

			// Cleanup
			if err == nil {
				shutdownCtx := context.Background()
				_ = ShutdownTracing(shutdownCtx)
			}
		})
	}
}

func TestSpanCreation(t *testing.T) {
	// Initialize tracing with stdout exporter
	ctx := context.Background()
	cfg := TracingConfig{
		Enabled:               true,
		Provider:              "opentelemetry",
		ExporterType:          "stdout",
		SamplingType:          "always_on",
		ServiceName:           "test-service",
		ServiceVersion:        "v1.0.0",
		DeploymentEnvironment: "test",
	}

	err := InitTracing(ctx, cfg)
	if err != nil {
		t.Fatalf("Failed to initialize tracing: %v", err)
	}
	defer func() {
		shutdownCtx := context.Background()
		_ = ShutdownTracing(shutdownCtx)
	}()

	// Test span creation
	spanCtx, span := StartSpan(ctx, SpanRequestReceived)
	if span == nil {
		t.Fatal("StartSpan returned nil span")
	}

	// Test setting attributes
	SetSpanAttributes(span,
		attribute.String(AttrRequestID, "test-request-123"),
		attribute.String(AttrModelName, "gpt-4"),
	)

	// Test recording error
	testErr := context.Canceled
	RecordError(span, testErr)
	span.SetStatus(codes.Error, "test error")

	span.End()

	// Verify context was updated
	if spanCtx == nil {
		t.Fatal("StartSpan returned nil context")
	}
}

func TestTraceContextPropagation(t *testing.T) {
	// Initialize tracing
	ctx := context.Background()
	cfg := TracingConfig{
		Enabled:               true,
		Provider:              "opentelemetry",
		ExporterType:          "stdout",
		SamplingType:          "always_on",
		ServiceName:           "test-service",
		ServiceVersion:        "v1.0.0",
		DeploymentEnvironment: "test",
	}

	err := InitTracing(ctx, cfg)
	if err != nil {
		t.Fatalf("Failed to initialize tracing: %v", err)
	}
	defer func() {
		shutdownCtx := context.Background()
		_ = ShutdownTracing(shutdownCtx)
	}()

	// Create a span to establish trace context
	spanCtx, span := StartSpan(ctx, "test-span")
	defer span.End()

	// Test injection
	headers := make(map[string]string)
	InjectTraceContext(spanCtx, headers)

	// Verify trace context was injected
	if len(headers) == 0 {
		t.Error("InjectTraceContext did not inject any headers")
	}

	// Test extraction
	extractedCtx := ExtractTraceContext(ctx, headers)
	if extractedCtx == nil {
		t.Error("ExtractTraceContext returned nil context")
	}
}

func TestGetTracerWhenNotInitialized(t *testing.T) {
	// Don't initialize tracing
	tracer := GetTracer()
	if tracer == nil {
		t.Error("GetTracer returned nil when not initialized")
	}

	// Should return a noop tracer that doesn't panic
	ctx := context.Background()
	_, span := tracer.Start(ctx, "test-span")
	if span == nil {
		t.Error("Noop tracer returned nil span")
	}
	span.End()
}

func TestStartSpanWithNilContext(t *testing.T) {
	// Test that StartSpan handles nil context gracefully
	// This simulates the scenario where TraceContext may not be initialized
	ctx, span := StartSpan(nil, "test-span")
	if span == nil {
		t.Error("StartSpan returned nil span with nil context")
	}
	if ctx == nil {
		t.Error("StartSpan returned nil context")
	}
	span.End()
}

func TestSpanAttributeConstants(t *testing.T) {
	// Verify span name constants are defined
	spanNames := []string{
		SpanRequestReceived,
		SpanClassification,
		SpanPIIDetection,
		SpanJailbreakDetection,
		SpanCacheLookup,
		SpanRoutingDecision,
		SpanBackendSelection,
		SpanUpstreamRequest,
		SpanResponseProcessing,
		SpanToolSelection,
		SpanSystemPromptInjection,
	}

	for _, name := range spanNames {
		if name == "" {
			t.Errorf("Span name constant is empty")
		}
		if len(name) < 10 {
			t.Errorf("Span name %q is too short", name)
		}
	}

	// Verify attribute key constants are defined
	attrKeys := []string{
		AttrRequestID,
		AttrModelName,
		AttrCategoryName,
		AttrRoutingStrategy,
		AttrPIIDetected,
		AttrJailbreakDetected,
		AttrCacheHit,
		AttrReasoningEnabled,
	}

	for _, key := range attrKeys {
		if key == "" {
			t.Errorf("Attribute key constant is empty")
		}
	}
}
