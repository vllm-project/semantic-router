package tracing

import (
	"context"
	"testing"

	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

func runTraceRecorder(t *testing.T) *tracetest.SpanRecorder {
	t.Helper()
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder), sdktrace.WithIDGenerator(RunIDGenerator()))
	previous := otel.GetTracerProvider()
	otel.SetTracerProvider(provider)
	emittedRunSpans.Clear()
	t.Cleanup(func() {
		_ = provider.Shutdown(context.Background())
		otel.SetTracerProvider(previous)
		emittedRunSpans.Clear()
	})
	return recorder
}

// Every call of a run without trace context lands in one trace: under the
// run's root span for the main agent, and under a subagent's span, a child
// of the agent that spawned it, for a subagent. The run and agent spans are
// emitted once. Another session is another trace, and a call that carries
// a trace keeps it.
func TestRunCallsShareOneTraceUnderTheirAgents(t *testing.T) {
	recorder := runTraceRecorder(t)
	request := func(ctx context.Context, run Run) sdktrace.ReadOnlySpan {
		_, span := otel.Tracer("test").Start(WithRunParent(ctx, run), SpanRequest)
		span.End()
		ended := recorder.Ended()
		return ended[len(ended)-1]
	}
	main := request(context.Background(), Run{SessionID: "run-1", Harness: "codex"})
	again := request(context.Background(), Run{SessionID: "run-1", Harness: "codex"})
	child := request(context.Background(), Run{SessionID: "run-1", AgentID: "a"})
	grandchild := request(context.Background(), Run{SessionID: "run-1", AgentID: "b", ParentAgentID: "a"})
	other := request(context.Background(), Run{SessionID: "run-2"})

	spans := map[string]sdktrace.ReadOnlySpan{}
	for _, span := range recorder.Ended() {
		if span.Name() != SpanRequest {
			if _, twice := spans[span.SpanContext().SpanID().String()]; twice {
				t.Fatalf("run span %s emitted twice", span.Name())
			}
			spans[span.SpanContext().SpanID().String()] = span
		}
	}
	if len(spans) != 4 {
		t.Fatalf("run and agent spans = %d, want root and two agents of run-1 and root of run-2", len(spans))
	}
	traceID := main.SpanContext().TraceID()
	for _, span := range []sdktrace.ReadOnlySpan{again, child, grandchild} {
		if span.SpanContext().TraceID() != traceID {
			t.Fatalf("call of run-1 in trace %s, want %s", span.SpanContext().TraceID(), traceID)
		}
	}
	if other.SpanContext().TraceID() == traceID {
		t.Fatal("another session shares run-1's trace")
	}
	parentOf := func(span sdktrace.ReadOnlySpan) sdktrace.ReadOnlySpan {
		parent, ok := spans[span.Parent().SpanID().String()]
		if !ok {
			t.Fatalf("%s has no emitted parent", span.Name())
		}
		return parent
	}
	root := parentOf(main)
	if root.Name() != SpanRun || root.Parent().IsValid() || parentOf(again) != root {
		t.Fatalf("main calls are not children of one run root: %s %v", root.Name(), root.Parent())
	}
	childAgent := parentOf(child)
	if childAgent.Name() != SpanAgent || parentOf(childAgent) != root {
		t.Fatal("subagent span is not a child of the run root")
	}
	if grandAgent := parentOf(grandchild); parentOf(grandAgent) != childAgent {
		t.Fatal("nested subagent span is not a child of the agent that spawned it")
	}

	incoming := trace.ContextWithRemoteSpanContext(context.Background(), trace.NewSpanContext(trace.SpanContextConfig{
		TraceID: trace.TraceID{1}, SpanID: trace.SpanID{2}, TraceFlags: trace.FlagsSampled,
	}))
	if kept := request(incoming, Run{SessionID: "run-1"}); kept.SpanContext().TraceID() != (trace.TraceID{1}) {
		t.Fatal("a call with trace context left its trace")
	}
}

// The provider InitTracing installs gives a run's root span the ids its
// calls are parented under; without them the root lands in a trace of its own.
func TestInitTracingGivesRunSpansTheirIDs(t *testing.T) {
	previous := otel.GetTracerProvider()
	if err := InitTracing(context.Background(), TracingConfig{
		Enabled: true, ExporterType: "stdout", SamplingType: "always_on", ServiceName: "run-test",
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = ShutdownTracing(context.Background())
		tracerProvider, tracer = nil, nil
		otel.SetTracerProvider(previous)
	})
	traceID, rootID := runSpanIDs("run-1")
	ctx := context.WithValue(context.Background(), runSpanIDsKey{}, runSpanIDsValue{traceID: traceID, spanID: rootID})
	_, span := GetTracer().Start(ctx, SpanRun)
	span.End()
	if got := span.SpanContext(); got.TraceID() != traceID || got.SpanID() != rootID {
		t.Fatalf("run root span = %s/%s, want %s/%s", got.TraceID(), got.SpanID(), traceID, rootID)
	}
}
