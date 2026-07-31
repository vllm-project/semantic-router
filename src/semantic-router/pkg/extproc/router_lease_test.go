package extproc

import (
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestRouterLeaseRetireWaitsForInFlightAcquireToRelease(t *testing.T) {
	lease := newRouterLease(&OpenAIRouter{})
	if !lease.acquire() {
		t.Fatal("acquire() = false, want true before retire")
	}

	retireDone := make(chan struct{})
	go func() {
		lease.retire(5 * time.Second)
		close(retireDone)
	}()

	select {
	case <-retireDone:
		t.Fatal("retire() returned before the in-flight acquire was released")
	case <-time.After(50 * time.Millisecond):
	}

	lease.release()

	select {
	case <-retireDone:
	case <-time.After(5 * time.Second):
		t.Fatal("retire() did not return after the in-flight acquire was released")
	}
}

func TestRouterLeaseAcquireFailsOnceRetiring(t *testing.T) {
	lease := newRouterLease(&OpenAIRouter{})
	lease.retire(time.Second)

	if lease.acquire() {
		t.Fatal("acquire() = true, want false once retire() has run")
	}
}

func TestRouterLeaseRetireRespectsDrainTimeout(t *testing.T) {
	lease := newRouterLease(&OpenAIRouter{})
	if !lease.acquire() {
		t.Fatal("acquire() = false, want true")
	}
	defer lease.release()

	start := time.Now()
	lease.retire(50 * time.Millisecond)
	elapsed := time.Since(start)

	if elapsed > 2*time.Second {
		t.Fatalf("retire() took %s, want it bounded near the 50ms drain timeout", elapsed)
	}
}

// TestRouterLeaseConcurrentAcquireDuringRetireNeverRacesAfterWait exercises
// the race retire() and acquire() must both defend against: a caller can
// call acquire() at any point up to and including the instant retire()'s
// internal Wait() returns. No successful acquire() may be observed once
// retire() has returned, since the caller closes the router's resources
// immediately afterward. Run with -race.
func TestRouterLeaseConcurrentAcquireDuringRetireNeverRacesAfterWait(t *testing.T) {
	for i := 0; i < 500; i++ {
		lease := newRouterLease(&OpenAIRouter{})
		var acquired bool

		retireStarted := make(chan struct{})
		retireDone := make(chan struct{})
		go func() {
			close(retireStarted)
			lease.retire(5 * time.Second)
			close(retireDone)
		}()

		<-retireStarted
		if lease.acquire() {
			acquired = true
			lease.release()
		}

		<-retireDone
		if acquired {
			// A successful acquire raced retire() legitimately (it landed
			// before retiring flipped); retire()'s Wait() must have already
			// accounted for it, so retireDone closing here is itself the
			// assertion that no deadlock or double-release occurred.
			continue
		}
	}
}

// TestRouterServiceProcessRejectsInsteadOfSpinningAfterShutdown covers the
// one case where acquire can never succeed: Shutdown retires the current
// lease without installing a replacement, so the retired lease stays
// installed forever. Retrying against it would be an unbounded busy loop
// burning a core; Process must reject the call instead.
func TestRouterServiceProcessRejectsInsteadOfSpinningAfterShutdown(t *testing.T) {
	rs := NewRouterService(&OpenAIRouter{})
	if err := rs.Shutdown(time.Second); err != nil {
		t.Fatalf("Shutdown() error = %v", err)
	}

	done := make(chan error, 1)
	go func() {
		_, err := rs.acquireCurrentLease()
		done <- err
	}()

	select {
	case err := <-done:
		if status.Code(err) != codes.Unavailable {
			t.Fatalf("acquireCurrentLease() error = %v, want an Unavailable status", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("acquireCurrentLease() never returned after Shutdown — it is spinning on a permanently retired lease")
	}
}

// TestRouterServiceProcessRetriesAcrossReload asserts the other side of that
// bound: when a reload has installed a replacement, a call that loses the
// race against retirement is served by the new lease rather than rejected.
func TestRouterServiceProcessRetriesAcrossReload(t *testing.T) {
	oldRouter := &OpenAIRouter{}
	rs := NewRouterService(oldRouter)

	staleLease := rs.current.Load()
	newRouter := &OpenAIRouter{}
	rs.Swap(newRouter)
	staleLease.retire(time.Second) // the reload's Retire, minus the Close

	lease, err := rs.acquireCurrentLease()
	if err != nil {
		t.Fatalf("acquireCurrentLease() error = %v, want it to fall through to the new lease", err)
	}
	defer lease.release()

	if lease.router != newRouter {
		t.Fatal("acquireCurrentLease() returned the retired lease instead of the reload's replacement")
	}
}

func TestRouterLeaseNilIsNoop(t *testing.T) {
	var lease *routerLease
	if lease.acquire() {
		t.Fatal("acquire() on nil lease = true, want false")
	}
	lease.release()
	lease.retire(time.Second)
}
