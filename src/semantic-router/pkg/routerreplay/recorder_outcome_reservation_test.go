package routerreplay

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOutcomeReservationsSurviveSharedQueueSaturation(t *testing.T) {
	backend := &gatedOutcomeStore{entered: make(chan struct{}, 1), release: make(chan struct{})}
	first, second := NewRecorder(backend), NewRecorder(backend)
	first.outcomes = newOutcomeQueue(2, time.Second)
	second.outcomes = first.outcomes
	var once sync.Once
	release := func() { once.Do(func() { close(backend.release) }) }
	t.Cleanup(func() { release(); assert.NoError(t, first.DrainOutcomes()) })
	a := first.TryReserveOutcome("first")
	require.NotNil(t, a)
	a.Scheduled(Outcome{Verdict: "scheduled-a"})
	awaitOutcomeSignal(t, backend.entered)
	b := second.TryReserveOutcome("second")
	require.NotNil(t, b)
	metadata := map[string]string{"state": "original"}
	b.Scheduled(Outcome{Verdict: "scheduled-b", Metadata: metadata})
	metadata["state"] = "mutated"
	require.Nil(t, first.TryReserveOutcome("overflow"))
	for i := 0; i < 2; i++ {
		require.True(t, second.TryAppendOutcome("ordinary", Outcome{Verdict: "ordinary"}))
	}
	require.False(t, first.TryAppendOutcome("ordinary", Outcome{}))
	published := make(chan struct{})
	go func() {
		a.Finish(Outcome{Verdict: "completed-a"})
		b.Finish(Outcome{Verdict: "timeout-b"})
		a.Finish(Outcome{Verdict: "duplicate"})
		b.Scheduled(Outcome{Verdict: "late-scheduled"})
		close(published)
	}()
	awaitOutcomeSignal(t, published)
	require.Nil(t, second.TryReserveOutcome("not-written-yet"))
	release()
	require.NoError(t, first.DrainOutcomes())
	backend.mu.Lock()
	defer backend.mu.Unlock()
	var reserved []string
	for _, outcome := range backend.seen {
		if outcome.Verdict != "ordinary" {
			reserved = append(reserved, outcome.Verdict)
		}
		if outcome.Verdict == "scheduled-b" {
			assert.Equal(t, "original", outcome.Metadata["state"])
		}
	}
	assert.Equal(t, []string{"scheduled-a", "scheduled-b", "completed-a", "timeout-b"}, reserved)
}

func TestOutcomeReservationRetainsSlotAfterScheduledWrite(t *testing.T) {
	backend := &gatedOutcomeStore{entered: make(chan struct{}, 1), release: make(chan struct{})}
	close(backend.release)
	recorder := NewRecorder(backend)
	recorder.outcomes = newOutcomeQueue(1, time.Second)
	reservation := recorder.TryReserveOutcome("attempt")
	require.NotNil(t, reservation)
	reservation.Scheduled(Outcome{Verdict: "scheduled"})
	require.Eventually(t, func() bool {
		backend.mu.Lock()
		defer backend.mu.Unlock()
		return len(backend.seen) == 1
	}, time.Second, time.Millisecond)
	require.Nil(t, recorder.TryReserveOutcome("overflow"), "writing scheduled must not free the terminal slot")
	reservation.Finish(Outcome{Verdict: "completed"})
	require.NoError(t, recorder.DrainOutcomes())
	backend.mu.Lock()
	defer backend.mu.Unlock()
	require.Len(t, backend.seen, 2)
	assert.Equal(t, "completed", backend.seen[1].Verdict)
}

func TestOutcomeReservationKeepsStorageOwnedDuringDrain(t *testing.T) {
	backend := &gatedOutcomeStore{entered: make(chan struct{}, 1), release: make(chan struct{})}
	recorder := NewRecorder(backend)
	recorder.outcomes = newOutcomeQueue(1, 10*time.Millisecond)
	reservation := recorder.TryReserveOutcome("preparing")
	require.NotNil(t, reservation)
	require.ErrorIs(t, recorder.Close(), context.DeadlineExceeded)
	require.Nil(t, recorder.TryReserveOutcome("retired"))
	backend.mu.Lock()
	assert.False(t, backend.closed)
	backend.mu.Unlock()
	reservation.Finish(Outcome{Verdict: "cancelled"})
	awaitOutcomeSignal(t, recorder.outcomes.done)
	require.Eventually(t, func() bool {
		backend.mu.Lock()
		defer backend.mu.Unlock()
		return backend.closed
	}, time.Second, time.Millisecond)
}
