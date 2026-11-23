package tracing

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"sync"
	"sync/atomic"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// An agent run is one trace. A request whose harness sent no trace context
// is parented under spans the Router creates for its run: a root span for
// the session and one span per subagent, a child of the agent that spawned
// it. Their ids derive from the session and agent ids, so every Router
// process that serves the run puts its calls in the same trace.
const (
	SpanRun   = "semantic_router.run"
	SpanAgent = "semantic_router.agent"

	AttrSessionID     = "session.id"
	AttrAgentID       = "vsr.agent.id"
	AttrAgentParentID = "vsr.agent.parent_id"
	AttrAgentKind     = "vsr.agent.kind"
	AttrHarness       = "vsr.harness"
)

// maxRunSpans bounds the run and agent spans a process remembers having
// emitted; past it the set is cleared and a span may be emitted again.
const maxRunSpans = 100_000

// Run names the agent run a request belongs to. AgentID and ParentAgentID
// are the hashed agent ids Replay records, empty for the main agent.
type Run struct {
	SessionID     string
	Harness       string
	AgentID       string
	ParentAgentID string
	AgentKind     string
}

// Attributes are the run's span attributes.
func (r Run) Attributes() []attribute.KeyValue {
	attrs := []attribute.KeyValue{attribute.String(AttrSessionID, r.SessionID)}
	for _, attr := range []struct{ key, value string }{
		{AttrHarness, r.Harness}, {AttrAgentID, r.AgentID}, {AttrAgentParentID, r.ParentAgentID}, {AttrAgentKind, r.AgentKind},
	} {
		if attr.value != "" {
			attrs = append(attrs, attribute.String(attr.key, attr.value))
		}
	}
	return attrs
}

var (
	emittedRunSpans sync.Map
	emittedRunCount atomic.Int64
)

// WithRunParent parents what ctx starts next under its run's agent span,
// emitting the run's root and agent spans the first time this process sees
// them. A context that already carries a span, such as the harness's
// traceparent or a Looper hop's request, is returned unchanged, as is every
// context while no SDK tracer provider is installed.
func WithRunParent(ctx context.Context, run Run) context.Context {
	if run.SessionID == "" || trace.SpanContextFromContext(ctx).IsValid() {
		return ctx
	}
	if _, ok := otel.GetTracerProvider().(*sdktrace.TracerProvider); !ok {
		return ctx
	}
	traceID, rootID := runSpanIDs(run.SessionID)
	emitRunSpan(ctx, SpanRun, traceID, rootID, trace.SpanID{}, attribute.String(AttrSessionID, run.SessionID),
		attribute.String(AttrHarness, run.Harness))
	parent := rootID
	if run.AgentID != "" {
		parentID := rootID
		if run.ParentAgentID != "" {
			parentID = agentSpanID(run.SessionID, run.ParentAgentID)
		}
		parent = agentSpanID(run.SessionID, run.AgentID)
		emitRunSpan(ctx, SpanAgent, traceID, parent, parentID, run.Attributes()...)
	}
	return trace.ContextWithRemoteSpanContext(ctx, trace.NewSpanContext(trace.SpanContextConfig{
		TraceID: traceID, SpanID: parent, TraceFlags: trace.FlagsSampled,
	}))
}

func runSpanIDs(sessionID string) (trace.TraceID, trace.SpanID) {
	sum := sha256.Sum256([]byte("vsr-run\x00" + sessionID))
	var traceID trace.TraceID
	var spanID trace.SpanID
	copy(traceID[:], sum[:16])
	copy(spanID[:], sum[16:24])
	return traceID, spanID
}

func agentSpanID(sessionID, agentID string) trace.SpanID {
	sum := sha256.Sum256([]byte("vsr-run-agent\x00" + sessionID + "\x00" + agentID))
	var spanID trace.SpanID
	copy(spanID[:], sum[:8])
	return spanID
}

// emitRunSpan ends a zero-length span with the given ids under parentID, or
// as the trace's root when parentID is empty, once per process.
func emitRunSpan(ctx context.Context, name string, traceID trace.TraceID, spanID, parentID trace.SpanID, attrs ...attribute.KeyValue) {
	if _, seen := emittedRunSpans.LoadOrStore(spanID, struct{}{}); seen {
		return
	}
	if emittedRunCount.Add(1) > maxRunSpans {
		emittedRunSpans.Clear()
		emittedRunCount.Store(0)
	}
	parent := trace.ContextWithSpanContext(context.WithoutCancel(ctx), trace.SpanContext{})
	if parentID.IsValid() {
		parent = trace.ContextWithRemoteSpanContext(parent, trace.NewSpanContext(trace.SpanContextConfig{
			TraceID: traceID, SpanID: parentID, TraceFlags: trace.FlagsSampled,
		}))
	}
	parent = context.WithValue(parent, runSpanIDsKey{}, runSpanIDsValue{traceID: traceID, spanID: spanID})
	_, span := otel.Tracer("semantic-router").Start(parent, name,
		trace.WithSpanKind(trace.SpanKindInternal), trace.WithAttributes(attrs...))
	span.End()
}

type (
	runSpanIDsKey   struct{}
	runSpanIDsValue struct {
		traceID trace.TraceID
		spanID  trace.SpanID
	}
)

// RunIDGenerator returns the span id generator InitTracing installs: random
// ids, except for the run and agent spans, whose ids derive from the run.
func RunIDGenerator() sdktrace.IDGenerator { return runIDGenerator{} }

type runIDGenerator struct{}

func (runIDGenerator) NewIDs(ctx context.Context) (trace.TraceID, trace.SpanID) {
	if ids, ok := ctx.Value(runSpanIDsKey{}).(runSpanIDsValue); ok {
		return ids.traceID, ids.spanID
	}
	var traceID trace.TraceID
	for !traceID.IsValid() {
		_, _ = rand.Read(traceID[:])
	}
	return traceID, randomSpanID()
}

func (runIDGenerator) NewSpanID(ctx context.Context, traceID trace.TraceID) trace.SpanID {
	if ids, ok := ctx.Value(runSpanIDsKey{}).(runSpanIDsValue); ok && ids.traceID == traceID {
		return ids.spanID
	}
	return randomSpanID()
}

func randomSpanID() trace.SpanID {
	var spanID trace.SpanID
	for !spanID.IsValid() {
		_, _ = rand.Read(spanID[:])
	}
	return spanID
}
