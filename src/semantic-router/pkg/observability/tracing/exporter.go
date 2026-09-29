package tracing

import (
	"context"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

var exportSpans = promauto.NewCounterVec(prometheus.CounterOpts{
	Name: "llm_trace_export_spans_total",
	Help: "Spans in completed exporter batches by result; does not count unsampled or SDK queue-dropped spans.",
}, []string{"result"})

type observedExporter struct{ sdktrace.SpanExporter }

// Count the actual export outcome, without retrying or changing SDK sampling,
// batching, queueing or shutdown behavior.
func (e observedExporter) ExportSpans(ctx context.Context, spans []sdktrace.ReadOnlySpan) error {
	err := e.SpanExporter.ExportSpans(ctx, spans)
	result := "success"
	if err != nil {
		result = "error"
	}
	exportSpans.WithLabelValues(result).Add(float64(len(spans)))
	return err
}
