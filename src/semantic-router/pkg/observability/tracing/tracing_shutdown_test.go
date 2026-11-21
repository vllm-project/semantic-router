package tracing

import (
	"context"
	"net"
	"testing"
	"time"
)

// Stopping with spans queued for a collector that never answers takes at most
// the flush bound, not the exporter's 10 s retry per batch, and is no error.
func TestShutdownDoesNotWaitForAnUnreachableCollector(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	endpoint := listener.Addr().String()
	_ = listener.Close() // nothing listens there any more
	if err = InitTracing(context.Background(), TracingConfig{
		Enabled: true, ExporterType: "otlp", ExporterEndpoint: endpoint, ExporterInsecure: true, ServiceName: "test",
	}); err != nil {
		t.Fatal(err)
	}
	defer func() { tracerProvider, tracer = nil, nil }()
	_, span := StartSpan(context.Background(), SpanRequest)
	span.End()

	started := time.Now()
	if err = ShutdownTracing(context.Background()); err != nil {
		t.Fatalf("shutdown error = %v, want the unsent spans dropped", err)
	}
	if elapsed := time.Since(started); elapsed > shutdownFlushTimeout+time.Second {
		t.Fatalf("shutdown took %s with the collector unreachable, want at most %s", elapsed, shutdownFlushTimeout)
	}
}
