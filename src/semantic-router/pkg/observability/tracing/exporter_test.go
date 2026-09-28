package tracing

import (
	"context"
	"errors"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

type failingExporter struct {
	tracetest.InMemoryExporter
	calls int
}

func (e *failingExporter) ExportSpans(context.Context, []sdktrace.ReadOnlySpan) error {
	e.calls++
	return errors.New("collector unavailable")
}

func TestObservedExporterCountsActualOutcomesWithoutRetry(t *testing.T) {
	failed := &failingExporter{}
	before := testutil.ToFloat64(exportSpans.WithLabelValues("error"))
	if err := (observedExporter{failed}).ExportSpans(t.Context(), make([]sdktrace.ReadOnlySpan, 3)); err == nil {
		t.Fatal("exporter error lost")
	}
	if failed.calls != 1 || testutil.ToFloat64(exportSpans.WithLabelValues("error")) != before+3 {
		t.Fatal("export failure must be counted exactly once, without retry")
	}
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(observedExporter{exporter}))
	before = testutil.ToFloat64(exportSpans.WithLabelValues("success"))
	_, span := provider.Tracer("fixture").Start(t.Context(), "fixture")
	span.End()
	if err := provider.Shutdown(t.Context()); err != nil {
		t.Fatal(err)
	}
	if testutil.ToFloat64(exportSpans.WithLabelValues("success")) != before+1 {
		t.Fatal("successful export not counted")
	}
}
