package extproc

import (
	"sync/atomic"
	"testing"
	"time"
)

// Regression for https://github.com/vllm-project/semantic-router/issues/1843.
//
// Before this change the four background goroutines listed in the
// issue had no panic recovery and a panic in any of them aborted the
// router process with no log line. goSafely is the helper added to
// extproc/safego.go; the tests below assert that it
//
//   1. runs the user function on a separate goroutine,
//   2. swallows any panic the user function emits, and
//   3. does not leak that panic to the parent goroutine.

func TestGoSafelyRunsTheFunction(t *testing.T) {
	var ran atomic.Int32
	done := make(chan struct{})
	goSafely("test_runs_fn", func() {
		ran.Add(1)
		close(done)
	})

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("goSafely did not run the user function within 2s")
	}

	if got := ran.Load(); got != 1 {
		t.Fatalf("goSafely should have run fn exactly once, got %d", got)
	}
}

func TestGoSafelyRecoversFromPanic(t *testing.T) {
	// If goSafely lets the panic propagate the test process aborts
	// with an unrecovered panic. The test PASSING is the assertion.
	finished := make(chan struct{})
	goSafely("test_panics", func() {
		defer close(finished)
		panic("simulated upstream model crash")
	})

	select {
	case <-finished:
	case <-time.After(2 * time.Second):
		t.Fatal("panicking goroutine never reached its defer")
	}

	// Give the deferred recover one extra tick to log + return.
	time.Sleep(20 * time.Millisecond)
}

func TestGoSafelyDoesNotBlockCaller(t *testing.T) {
	// The wrapper must launch its own goroutine; it must not run the
	// function inline in the caller's goroutine, otherwise a slow
	// model client would block the request handler.
	caller := make(chan struct{})
	stillBlocked := make(chan struct{})
	goSafely("test_async", func() {
		<-caller
		close(stillBlocked)
	})

	select {
	case <-stillBlocked:
		t.Fatal("goSafely should not block the caller until fn signals back")
	case <-time.After(50 * time.Millisecond):
		// Expected — the wrapped fn is waiting on `caller`.
	}

	close(caller)
	select {
	case <-stillBlocked:
	case <-time.After(2 * time.Second):
		t.Fatal("goSafely never let fn complete after the caller signal")
	}
}
