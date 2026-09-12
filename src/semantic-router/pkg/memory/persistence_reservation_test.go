package memory

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/baggage"
	"go.opentelemetry.io/otel/trace"
)

func TestPersistenceReservationTimeoutRetainsPreparationCapacity(t *testing.T) {
	receipts := newOutcomeRecorder()
	runner := NewPersistenceRunner(20*time.Millisecond, 1, 1)
	first := runner.TryReserve(context.Background(), receipts.report)
	require.NotNil(t, first)
	defer first.Abort(PersistenceOutcome{}, nil)
	require.Eventually(t, func() bool { return len(runner.jobs) == 0 }, time.Second, time.Millisecond)
	second := runner.TryReserve(context.Background(), receipts.report)
	require.NotNil(t, second)
	defer second.Abort(PersistenceOutcome{}, nil)
	require.Eventually(t, func() bool { return receipts.count("timeout/persist_timeout") == 2 }, time.Second, time.Millisecond)
	require.Nil(t, runner.TryReserve(context.Background(), receipts.report))
	assert.Equal(t, 1, receipts.count("rejected/queue_full"))

	retired := make(chan error, 1)
	go func() { retired <- runner.RetireAndWait(time.Second) }()
	select {
	case <-runner.Done():
		t.Fatal("generation resources released while preparation still owns them")
	case <-time.After(10 * time.Millisecond):
	}
	ran := false
	first.Start(func(context.Context) (PersistenceOutcome, error) {
		ran = true
		return PersistenceOutcome{}, nil
	})
	second.Abort(PersistenceOutcome{Status: "skipped", Reason: "history_too_large"}, nil)
	require.NoError(t, <-retired)
	assert.False(t, ran, "expired preparation must never execute")
	assert.Equal(t, 2, receipts.count("timeout/persist_timeout"), "late preparation adds no terminal")
}

func TestPersistenceReservationAbortReleasesCapacity(t *testing.T) {
	receipts := newOutcomeRecorder()
	runner := NewPersistenceRunner(time.Second, 1, 1)
	reservation := runner.TryReserve(context.Background(), receipts.report)
	require.NotNil(t, reservation)
	reservation.Abort(PersistenceOutcome{Status: "skipped", Reason: "history_too_large"}, nil)
	reservation.Start(func(context.Context) (PersistenceOutcome, error) {
		t.Error("aborted reservation executed")
		return PersistenceOutcome{}, nil
	})
	require.NoError(t, runner.RetireAndWait(time.Second))
	assert.Equal(t, 1, receipts.count("skipped/history_too_large"))
	assert.Zero(t, receipts.count("completed/persisted"))
	assert.Nil(t, runner.TryReserve(context.Background(), receipts.report))
	assert.Equal(t, 1, receipts.count("rejected/shutting_down"))
}

func TestPersistenceRunnerDetachesOnlySpanContext(t *testing.T) {
	type privateKey struct{}
	state, err := trace.ParseTraceState("vendor=opaque")
	require.NoError(t, err)
	span := trace.NewSpanContext(trace.SpanContextConfig{
		TraceID: trace.TraceID{1}, SpanID: trace.SpanID{2},
		TraceFlags: trace.FlagsSampled, TraceState: state,
	})
	member, err := baggage.NewMember("token", "test-only-value")
	require.NoError(t, err)
	bag, err := baggage.New(member)
	require.NoError(t, err)
	inbound := trace.ContextWithSpanContext(context.Background(), span)
	inbound = baggage.ContextWithBaggage(inbound, bag)
	inbound = context.WithValue(inbound, privateKey{}, "request-owned")
	inbound, cancel := context.WithCancel(inbound)
	defer cancel()
	receipts := newOutcomeRecorder()
	runner := NewPersistenceRunner(time.Second, 1, 1)
	reservation := runner.TryReserve(inbound, receipts.report)
	require.NotNil(t, reservation)
	cancel()
	seen := make(chan context.Context, 1)
	reservation.Start(func(ctx context.Context) (PersistenceOutcome, error) {
		seen <- ctx
		return PersistenceOutcome{}, ctx.Err()
	})
	jobCtx := <-seen
	require.NoError(t, runner.RetireAndWait(time.Second))
	assert.True(t, span.Equal(trace.SpanContextFromContext(jobCtx)))
	assert.Empty(t, baggage.FromContext(jobCtx).Members())
	assert.Nil(t, jobCtx.Value(privateKey{}))
	assert.Equal(t, 1, receipts.count("completed/persisted"))
}
