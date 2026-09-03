package routerreplay

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestRecorderLifecycleRequiresExplicitTerminalTransition(t *testing.T) {
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recordID := mustAddLifecycleRecord(t, recorder, RoutingRecord{
		ID:        "replay-lifecycle-1",
		Timestamp: time.Now().UTC().Add(-25 * time.Millisecond),
		Decision:  "decision-a",
	})

	started, found := recorder.GetRecord(recordID)
	assertLifecycleStarted(t, started, found)
	if err := recorder.FinalizeLifecycle(recordID, LifecycleCompleted, "response_complete"); err != nil {
		t.Fatalf("failed to finalize replay: %v", err)
	}

	completed, found := recorder.GetRecord(recordID)
	assertLifecycleCompleted(t, completed, found)

	if err := recorder.FinalizeLifecycle(recordID, LifecycleFailed, "late_failure"); err != nil {
		t.Fatalf("idempotent finalization failed: %v", err)
	}
	unchanged, _ := recorder.GetRecord(recordID)
	assertLifecycleCompleted(t, unchanged, true)
}

func mustAddLifecycleRecord(t *testing.T, recorder *Recorder, record RoutingRecord) string {
	t.Helper()
	recordID, err := recorder.AddRecord(record)
	if err != nil {
		t.Fatalf("failed to add record: %v", err)
	}
	return recordID
}

func assertLifecycleStarted(t *testing.T, record RoutingRecord, found bool) {
	t.Helper()
	if !found || record.LifecycleState != LifecycleInProgress || record.EndedAt != nil {
		t.Fatalf("new replay lifecycle = %+v, want in progress without end time", record)
	}
}

func assertLifecycleCompleted(t *testing.T, record RoutingRecord, found bool) {
	t.Helper()
	if !found || record.LifecycleState != LifecycleCompleted || record.EndedAt == nil || record.DurationMS <= 0 ||
		record.TerminalReason != "response_complete" {
		t.Fatalf("completed lifecycle = %+v", record)
	}
}

func TestRecorderLifecycleSerializesCompetingTerminalTransitions(t *testing.T) {
	underlying := store.NewMemoryStore(10, 0)
	blocking := &blockingLifecycleStore{
		Storage: underlying,
		entered: make(chan struct{}),
		release: make(chan struct{}),
	}
	recorder := NewRecorder(blocking)
	recordID, err := recorder.AddRecord(RoutingRecord{
		ID:        "replay-lifecycle-race",
		Timestamp: time.Now().UTC().Add(-25 * time.Millisecond),
	})
	if err != nil {
		t.Fatalf("failed to add record: %v", err)
	}

	completedResult := make(chan error, 1)
	go func() {
		completedResult <- recorder.FinalizeLifecycle(recordID, LifecycleCompleted, "response_complete")
	}()
	<-blocking.entered
	abortedResult := make(chan error, 1)
	go func() {
		abortedResult <- recorder.FinalizeLifecycle(recordID, LifecycleAborted, "stream_closed")
	}()
	close(blocking.release)

	if err := <-completedResult; err != nil {
		t.Fatalf("completed transition failed: %v", err)
	}
	if err := <-abortedResult; err != nil {
		t.Fatalf("competing transition failed: %v", err)
	}
	record, found := recorder.GetRecord(recordID)
	if !found || record.LifecycleState != LifecycleCompleted || record.TerminalReason != "response_complete" {
		t.Fatalf("terminal lifecycle was overwritten: %+v", record)
	}
}

func TestRecorderLifecyclePropagatesStorageReadFailure(t *testing.T) {
	recorder := NewRecorder(&failingGetStore{Storage: store.NewMemoryStore(10, 0)})

	err := recorder.FinalizeLifecycle("replay-read-error", LifecycleCompleted, "response_complete")
	if err == nil || !strings.Contains(err.Error(), "replay storage unavailable") {
		t.Fatalf("FinalizeLifecycle() error = %v, want storage failure", err)
	}
}

func TestRecorderLifecycleBoundsStalledReplayBackend(t *testing.T) {
	recorder := NewRecorder(&stalledGetStore{Storage: store.NewMemoryStore(10, 0)})
	recorder.operationTimeout = 20 * time.Millisecond

	startedAt := time.Now()
	err := recorder.FinalizeLifecycle("replay-stalled-store", LifecycleCompleted, "response_complete")
	elapsed := time.Since(startedAt)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("FinalizeLifecycle() error = %v, want bounded deadline", err)
	}
	if elapsed > 500*time.Millisecond {
		t.Fatalf("FinalizeLifecycle() blocked for %s, want a bounded replay-store failure", elapsed)
	}
}

type blockingLifecycleStore struct {
	store.Storage
	entered chan struct{}
	release chan struct{}
}

type failingGetStore struct {
	store.Storage
}

type stalledGetStore struct {
	store.Storage
}

func (s *failingGetStore) Get(context.Context, string) (store.Record, bool, error) {
	return store.Record{}, false, fmt.Errorf("replay storage unavailable")
}

func (s *stalledGetStore) Get(ctx context.Context, _ string) (store.Record, bool, error) {
	<-ctx.Done()
	return store.Record{}, false, ctx.Err()
}

func (s *blockingLifecycleStore) UpdateLifecycle(
	ctx context.Context,
	id string,
	state string,
	endedAt time.Time,
	durationMS int64,
	reason string,
) error {
	close(s.entered)
	<-s.release
	return s.Storage.UpdateLifecycle(ctx, id, state, endedAt, durationMS, reason)
}
