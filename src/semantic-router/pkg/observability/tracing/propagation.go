package tracing

import (
	"context"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
)

// spanContextPropagator carries only the W3C trace-context headers,
// traceparent and tracestate. It is fixed rather than read from the global
// propagator so that baggage, which DefaultPropagator also carries, can never
// ride along with it.
var spanContextPropagator propagation.TextMapPropagator = propagation.TraceContext{}

// InjectTraceContext injects trace context into a map (e.g., HTTP headers)
func InjectTraceContext(ctx context.Context, headers map[string]string) {
	propagator := otel.GetTextMapPropagator()
	carrier := propagation.MapCarrier(headers)
	propagator.Inject(ctx, carrier)
}

// ExtractTraceContext extracts trace context from a map (e.g., HTTP headers)
func ExtractTraceContext(ctx context.Context, headers map[string]string) context.Context {
	propagator := otel.GetTextMapPropagator()
	carrier := propagation.MapCarrier(headers)
	return propagator.Extract(ctx, carrier)
}

// InjectTraceContextToSlice injects trace context into a slice of key-value pairs
func InjectTraceContextToSlice(ctx context.Context) [][2]string {
	headers := make(map[string]string)
	InjectTraceContext(ctx, headers)
	return headerPairs(headers)
}

// InjectSpanContextToSlice injects only the span context for ctx, the W3C
// traceparent and tracestate headers, into a slice of key-value pairs.
// Baggage is never included whatever the global propagator carries, so the
// result is safe to send to a target that must not see baggage members a
// client attached to the inbound request.
func InjectSpanContextToSlice(ctx context.Context) [][2]string {
	headers := make(map[string]string)
	spanContextPropagator.Inject(ctx, propagation.MapCarrier(headers))
	return headerPairs(headers)
}

// ExtractTraceContextFromSlice extracts trace context from a slice of key-value pairs
func ExtractTraceContextFromSlice(ctx context.Context, headers [][2]string) context.Context {
	headerMap := make(map[string]string, len(headers))
	for _, h := range headers {
		headerMap[h[0]] = h[1]
	}
	return ExtractTraceContext(ctx, headerMap)
}

func headerPairs(headers map[string]string) [][2]string {
	result := make([][2]string, 0, len(headers))
	for k, v := range headers {
		result = append(result, [2]string{k, v})
	}
	return result
}
