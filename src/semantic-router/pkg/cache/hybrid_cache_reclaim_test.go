//go:build !windows && cgo

package cache

import (
	"context"
	"sync/atomic"
	"testing"
	"time"
)

func TestReclaimIntervalForTTL(t *testing.T) {
	tests := []struct {
		name       string
		ttlSeconds int
		want       time.Duration
	}{
		{"one second floored", 1, minReclaimInterval},
		{"short ttl floored", 10, minReclaimInterval},
		{"exactly floor", 60, 60 * time.Second},
		{"above floor", 120, 120 * time.Second},
		{"long ttl", 3600, 3600 * time.Second},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := reclaimIntervalForTTL(tt.ttlSeconds); got != tt.want {
				t.Fatalf("reclaimIntervalForTTL(%d) = %v, want %v", tt.ttlSeconds, got, tt.want)
			}
		})
	}
}

func TestStartBackgroundReclaimNoopWhenTTLDisabled(t *testing.T) {
	// ttlSeconds <= 0: nothing ever expires, so no goroutine/ticker should start.
	h := &HybridCache{enabled: true, ttlSeconds: 0}
	h.startBackgroundReclaim()
	if h.stopReclaim != nil || h.reclaimTicker != nil {
		t.Fatalf("expected no reclaim goroutine when TTL disabled; stopReclaim=%v ticker=%v",
			h.stopReclaim, h.reclaimTicker)
	}
	// stop must be safe even though nothing started.
	h.stopBackgroundReclaim()
}

func TestStartBackgroundReclaimNoopWhenDisabled(t *testing.T) {
	h := &HybridCache{enabled: false, ttlSeconds: 600}
	h.startBackgroundReclaim()
	if h.stopReclaim != nil || h.reclaimTicker != nil {
		t.Fatalf("expected no reclaim goroutine when cache disabled; stopReclaim=%v ticker=%v",
			h.stopReclaim, h.reclaimTicker)
	}
}

func TestStartBackgroundReclaimDefaultsReclaimFn(t *testing.T) {
	// A long TTL means the real ticker never fires during the test, so the
	// nil milvusCache behind the default RebuildFromMilvus is never invoked.
	h := &HybridCache{enabled: true, ttlSeconds: 3600}
	h.startBackgroundReclaim()
	defer h.stopBackgroundReclaim()

	if h.reclaimFn == nil {
		t.Fatal("expected reclaimFn to default to RebuildFromMilvus")
	}
	if h.stopReclaim == nil || h.reclaimTicker == nil {
		t.Fatal("expected reclaim goroutine to be started for enabled cache with TTL")
	}
}

func TestBackgroundReclaimTicksThenStops(t *testing.T) {
	var calls int64
	h := &HybridCache{
		enabled:    true,
		ttlSeconds: 600,
		reclaimFn: func(context.Context) error {
			atomic.AddInt64(&calls, 1)
			return nil
		},
	}
	// Drive a fast ticker directly (bypassing the 60s floor) to exercise the loop.
	h.stopReclaim = make(chan struct{})
	h.reclaimTicker = time.NewTicker(5 * time.Millisecond)

	done := make(chan struct{})
	go func() {
		h.backgroundReclaim()
		close(done)
	}()

	time.Sleep(60 * time.Millisecond) // allow several ticks
	h.stopBackgroundReclaim()

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("backgroundReclaim did not exit after stop")
	}

	if atomic.LoadInt64(&calls) == 0 {
		t.Fatal("expected reclaimFn to be called at least once")
	}

	// After stop, no further calls.
	settled := atomic.LoadInt64(&calls)
	time.Sleep(30 * time.Millisecond)
	if got := atomic.LoadInt64(&calls); got != settled {
		t.Fatalf("reclaimFn called after stop: settled=%d got=%d", settled, got)
	}
}

func TestBackgroundReclaimSurvivesReclaimError(t *testing.T) {
	var calls int64
	h := &HybridCache{
		enabled:    true,
		ttlSeconds: 600,
		reclaimFn: func(context.Context) error {
			atomic.AddInt64(&calls, 1)
			return context.DeadlineExceeded // a failing pass must not kill the loop
		},
	}
	h.stopReclaim = make(chan struct{})
	h.reclaimTicker = time.NewTicker(5 * time.Millisecond)

	done := make(chan struct{})
	go func() {
		h.backgroundReclaim()
		close(done)
	}()

	time.Sleep(60 * time.Millisecond)
	h.stopBackgroundReclaim()

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("backgroundReclaim did not exit after stop")
	}

	if atomic.LoadInt64(&calls) < 2 {
		t.Fatalf("expected the loop to keep ticking after errors, got %d calls", atomic.LoadInt64(&calls))
	}
}

func TestStopBackgroundReclaimIdempotent(t *testing.T) {
	h := &HybridCache{enabled: true, ttlSeconds: 600}
	h.startBackgroundReclaim()
	// Multiple stops must not panic on a double channel close.
	h.stopBackgroundReclaim()
	h.stopBackgroundReclaim()
}

// TestStopBackgroundReclaimWaitsForInFlightReclaim locks in the property that
// stopBackgroundReclaim blocks until the goroutine has fully exited, including a
// reclaim pass that is already in flight. This is what makes Close()'s nil-out of
// the in-memory structures safe: by the time stop returns, no pass can still be
// about to take the write lock and touch them.
func TestStopBackgroundReclaimWaitsForInFlightReclaim(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	var finished atomic.Bool

	h := &HybridCache{
		enabled:    true,
		ttlSeconds: 600,
		reclaimFn: func(context.Context) error {
			close(started) // a pass is now in flight (only the first tick gets here)
			<-release      // simulate a slow RebuildFromMilvus still running
			finished.Store(true)
			return nil
		},
	}
	// Drive a fast ticker and wire reclaimDone the way startBackgroundReclaim does.
	h.stopReclaim = make(chan struct{})
	h.reclaimDone = make(chan struct{})
	h.reclaimTicker = time.NewTicker(5 * time.Millisecond)
	go func() {
		defer close(h.reclaimDone)
		h.backgroundReclaim()
	}()

	<-started // ensure a reclaim pass is in flight before we stop

	stopReturned := make(chan struct{})
	go func() {
		h.stopBackgroundReclaim()
		close(stopReturned)
	}()

	// stop must NOT return while a reclaim pass is still running.
	select {
	case <-stopReturned:
		t.Fatal("stopBackgroundReclaim returned before the in-flight reclaim pass finished")
	case <-time.After(50 * time.Millisecond):
	}

	// Let the in-flight pass complete; stop must now return promptly.
	close(release)
	select {
	case <-stopReturned:
	case <-time.After(time.Second):
		t.Fatal("stopBackgroundReclaim did not return after the in-flight reclaim pass finished")
	}
	if !finished.Load() {
		t.Fatal("expected the in-flight reclaim pass to complete before stop returned")
	}
}
